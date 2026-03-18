from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .module_probe import PUBLIC_KIMI_REPO_ID, REMOTE_CODE_FILENAMES
from .remote_code_support import ensure_hf_files, install_fla_backend, read_json

BASELINE_SMOKE_REPORT_KIND = "attnres.kimi.baseline_smoke_report"
BASELINE_SMOKE_REPORT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or report the Kimi public-checkpoint baseline smoke prerequisites.",
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--repo-id", default=PUBLIC_KIMI_REPO_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--prompt",
        default="Explain linear attention in one sentence.",
    )
    parser.add_argument(
        "--memory-overhead-gib",
        type=float,
        default=8.0,
        help="Conservative extra RAM requirement beyond raw checkpoint bytes before attempting a CPU smoke run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_smoke(
        artifact_dir=args.artifact_dir.resolve(),
        repo_id=args.repo_id,
        revision=args.revision,
        prompt=args.prompt,
        memory_overhead_gib=args.memory_overhead_gib,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote baseline smoke report to {args.output_path}")
    return 0


def run_smoke(
    artifact_dir: Path,
    repo_id: str,
    revision: str | None,
    prompt: str,
    memory_overhead_gib: float,
) -> dict[str, Any]:
    ensure_hf_files(artifact_dir, repo_id, REMOTE_CODE_FILENAMES, revision=revision)
    config = read_json(artifact_dir / "config.json")
    index = read_json(artifact_dir / "model.safetensors.index.json")
    unique_shards = sorted(set(index["weight_map"].values()))
    present_shards = sorted([shard for shard in unique_shards if (artifact_dir / shard).exists()])
    missing_shards = [shard for shard in unique_shards if shard not in present_shards]

    total_size_bytes = int(index["metadata"]["total_size"])
    total_ram_bytes = detect_total_ram_bytes()
    required_min_ram_bytes = total_size_bytes + int(memory_overhead_gib * (1024**3))

    report: dict[str, Any] = {
        "kind": BASELINE_SMOKE_REPORT_KIND,
        "version": BASELINE_SMOKE_REPORT_VERSION,
        "repo_id": repo_id,
        "revision": revision,
        "artifact_dir": str(artifact_dir),
        "model_type": config["model_type"],
        "dtype": config["dtype"],
        "num_hidden_layers": config["num_hidden_layers"],
        "checkpoint_total_size_bytes": total_size_bytes,
        "unique_shard_count": len(unique_shards),
        "present_shard_count": len(present_shards),
        "missing_shard_count": len(missing_shards),
        "host_total_ram_bytes": total_ram_bytes,
        "required_min_ram_bytes": required_min_ram_bytes,
        "device": "cpu",
        "prompt": prompt,
        "assumptions": [
            "smoke run targets the full public baseline checkpoint, not a selected-module probe",
            "CPU smoke requires the full local shard set before attempting model load",
            f"CPU smoke requires at least checkpoint bytes + {memory_overhead_gib:.1f} GiB of RAM before attempting model load",
            "missing full shards or insufficient RAM are reported as blocked instead of treated as test failures",
        ],
    }

    if missing_shards:
        report["status"] = "blocked_missing_full_checkpoint"
        report["reason"] = (
            f"artifact directory is missing {len(missing_shards)} of {len(unique_shards)} required shards"
        )
        report["missing_shards"] = missing_shards
        return report

    if total_ram_bytes is not None and total_ram_bytes < required_min_ram_bytes:
        report["status"] = "blocked_insufficient_ram"
        report["reason"] = (
            f"host total RAM {total_ram_bytes} is below conservative requirement {required_min_ram_bytes}"
        )
        return report

    start_load = time.perf_counter()
    fla_backend = install_fla_backend()
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True, revision=revision)
    dtype = torch.bfloat16 if config["dtype"] == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        artifact_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).eval()
    load_seconds = time.perf_counter() - start_load

    encoded = tokenizer(prompt, return_tensors="pt")
    start_forward = time.perf_counter()
    with torch.no_grad():
        logits = model(**encoded).logits[:, -1, :].to(dtype=torch.float32)
    forward_seconds = time.perf_counter() - start_forward
    values, indices = torch.topk(logits, k=5, dim=-1)
    tokens = []
    for token_id, logit in zip(indices[0].tolist(), values[0].tolist()):
        tokens.append(
            {
                "token_id": int(token_id),
                "token_text": tokenizer.decode([token_id]),
                "logit": float(logit),
            }
        )

    report["status"] = "passed"
    report["fla_backend"] = fla_backend
    report["load_seconds"] = load_seconds
    report["forward_seconds"] = forward_seconds
    report["top_next_tokens"] = tokens
    return report


def detect_total_ram_bytes() -> int | None:
    if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
        try:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (ValueError, OSError):
            pass
    if platform.system() == "Darwin":
        try:
            output = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                check=True,
                capture_output=True,
                text=True,
            )
            return int(output.stdout.strip())
        except Exception:
            return None
    return None


if __name__ == "__main__":
    raise SystemExit(main())
