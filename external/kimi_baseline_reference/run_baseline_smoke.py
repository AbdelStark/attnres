from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .module_probe import PUBLIC_KIMI_REPO_ID, REMOTE_CODE_FILENAMES
from .remote_code_support import (
    ensure_hf_files,
    install_fla_backend,
    load_remote_kimi_modules,
    read_json,
)

BASELINE_SMOKE_REPORT_KIND = "attnres.kimi.baseline_smoke_report"
BASELINE_SMOKE_REPORT_VERSION = 2
REMOTE_CODE_FINGERPRINT_FILENAMES = [
    "configuration_kimi.py",
    "modeling_kimi.py",
]


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
        "--execution-path",
        choices=["cpu_only", "cpu_disk_offload"],
        default="cpu_only",
        help="Execution path for the full public baseline smoke run.",
    )
    parser.add_argument(
        "--memory-overhead-gib",
        type=float,
        default=8.0,
        help="Conservative extra RAM requirement beyond raw checkpoint bytes before attempting a CPU smoke run.",
    )
    parser.add_argument(
        "--offload-dir",
        type=Path,
        default=None,
        help="Disk offload directory used by the cpu_disk_offload execution path.",
    )
    parser.add_argument(
        "--max-cpu-memory-gib",
        type=float,
        default=None,
        help="Explicit CPU memory budget for the cpu_disk_offload execution path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_smoke(
        artifact_dir=args.artifact_dir.resolve(),
        repo_id=args.repo_id,
        revision=args.revision,
        prompt=args.prompt,
        execution_path=args.execution_path,
        memory_overhead_gib=args.memory_overhead_gib,
        offload_dir=args.offload_dir.resolve() if args.offload_dir else None,
        max_cpu_memory_gib=args.max_cpu_memory_gib,
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
    execution_path: str,
    memory_overhead_gib: float,
    offload_dir: Path | None,
    max_cpu_memory_gib: float | None,
) -> dict[str, Any]:
    ensure_hf_files(
        artifact_dir,
        repo_id,
        REMOTE_CODE_FILENAMES + REMOTE_CODE_FINGERPRINT_FILENAMES,
        revision=revision,
    )
    config = read_json(artifact_dir / "config.json")
    index = read_json(artifact_dir / "model.safetensors.index.json")
    unique_shards = sorted(set(index["weight_map"].values()))
    present_shards = sorted([shard for shard in unique_shards if (artifact_dir / shard).exists()])
    missing_shards = [shard for shard in unique_shards if shard not in present_shards]

    total_size_bytes = int(index["metadata"]["total_size"])
    total_ram_bytes = detect_total_ram_bytes()
    required_min_ram_bytes = total_size_bytes + int(memory_overhead_gib * (1024**3))
    artifact_fingerprints = collect_artifact_fingerprints(
        artifact_dir=artifact_dir,
        present_shards=present_shards,
        remote_code_filenames=REMOTE_CODE_FINGERPRINT_FILENAMES,
    )

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
        "present_shards": present_shards,
        "host_total_ram_bytes": total_ram_bytes,
        "required_min_ram_bytes": required_min_ram_bytes,
        "host": detect_host_facts(),
        "device": "cpu",
        "execution_path": execution_path,
        "prompt": prompt,
        "artifact_fingerprints": artifact_fingerprints,
        "assumptions": [
            "smoke run targets the full public baseline checkpoint, not a selected-module probe",
            "full baseline smoke requires the complete local shard set before attempting model load",
            f"cpu_only requires at least checkpoint bytes + {memory_overhead_gib:.1f} GiB of RAM before attempting model load",
            "cpu_disk_offload is an equivalent execution path for lower-RAM hosts and records its explicit CPU-memory budget plus offload directory",
            "missing full shards or unmet execution-path prerequisites are reported as blocked instead of treated as test failures",
        ],
    }

    if missing_shards:
        report["status"] = "blocked_missing_full_checkpoint"
        report["reason"] = (
            f"artifact directory is missing {len(missing_shards)} of {len(unique_shards)} required shards"
        )
        report["missing_shards"] = missing_shards
        return report

    if execution_path == "cpu_only" and total_ram_bytes is not None and total_ram_bytes < required_min_ram_bytes:
        report["status"] = "blocked_insufficient_ram"
        report["reason"] = (
            f"host total RAM {total_ram_bytes} is below conservative requirement {required_min_ram_bytes}"
        )
        return report

    try:
        start_load = time.perf_counter()
        fla_backend = install_fla_backend()
        report["fla_backend"] = fla_backend
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True, revision=revision)
        dtype = torch.bfloat16 if config["dtype"] == "bfloat16" else torch.float32
        dtype_action = "preserve_artifact_dtype"
        if execution_path == "cpu_disk_offload" and dtype == torch.bfloat16:
            dtype = torch.float32
            dtype_action = "upcast_bfloat16_to_float32_for_cpu_offload"
        report["load_dtype"] = str(dtype).replace("torch.", "")
        report["dtype_action"] = dtype_action

        _, modeling_module, _ = load_remote_kimi_modules(artifact_dir)
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = modeling_module.eager_attention_forward
        report["attention_backend_override"] = {
            "forced_key": "flash_attention_2",
            "callable": "eager_attention_forward",
            "reason": "official Kimi remote code forces flash_attention_2 during full-model init; the CPU smoke path remaps that backend key to the remote module's eager attention implementation",
        }

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }
        if execution_path == "cpu_disk_offload":
            resolved_offload_dir = offload_dir or artifact_dir / "baseline-smoke-offload"
            resolved_offload_dir.mkdir(parents=True, exist_ok=True)
            resolved_max_cpu_memory_gib = resolve_max_cpu_memory_gib(
                host_total_ram_bytes=total_ram_bytes,
                explicit_max_cpu_memory_gib=max_cpu_memory_gib,
            )
            model_kwargs.update(
                {
                    "device_map": "auto",
                    "max_memory": {"cpu": format_gib_budget(resolved_max_cpu_memory_gib)},
                    "low_cpu_mem_usage": True,
                    "offload_folder": str(resolved_offload_dir),
                    "offload_state_dict": True,
                }
            )
            report["disk_offload"] = {
                "offload_dir": str(resolved_offload_dir),
                "max_cpu_memory_gib": resolved_max_cpu_memory_gib,
                "max_cpu_memory_bytes": gib_to_bytes(resolved_max_cpu_memory_gib),
            }

        model = AutoModelForCausalLM.from_pretrained(artifact_dir, **model_kwargs).eval()
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
        report["load_seconds"] = load_seconds
        report["forward_seconds"] = forward_seconds
        report["top_next_tokens"] = tokens
        report["model_device_map"] = getattr(model, "hf_device_map", None)
        return report
    except Exception as exc:
        report["status"] = "failed_runtime_error"
        report["reason"] = f"{type(exc).__name__}: {exc}"
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


def detect_host_facts() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_brand": detect_sysctl_value("machdep.cpu.brand_string"),
        "model": detect_sysctl_value("hw.model"),
        "os_product_version": detect_sysctl_value("kern.osproductversion"),
    }


def detect_sysctl_value(key: str) -> str | None:
    if platform.system() != "Darwin":
        return None
    try:
        output = subprocess.run(
            ["sysctl", "-n", key],
            check=True,
            capture_output=True,
            text=True,
        )
        return output.stdout.strip() or None
    except Exception:
        return None


def resolve_max_cpu_memory_gib(
    host_total_ram_bytes: int | None,
    explicit_max_cpu_memory_gib: float | None,
) -> float:
    if explicit_max_cpu_memory_gib is not None:
        return explicit_max_cpu_memory_gib
    if host_total_ram_bytes is None:
        return 32.0
    host_total_ram_gib = host_total_ram_bytes / (1024**3)
    return max(1.0, math.floor(host_total_ram_gib - 6.0))


def format_gib_budget(value_gib: float) -> str:
    rounded = round(value_gib)
    if abs(value_gib - rounded) < 1e-6:
        return f"{int(rounded)}GiB"
    return f"{value_gib:.1f}GiB"


def gib_to_bytes(value_gib: float) -> int:
    return int(value_gib * (1024**3))


def collect_artifact_fingerprints(
    artifact_dir: Path,
    present_shards: list[str],
    remote_code_filenames: list[str],
) -> dict[str, Any]:
    fingerprints: dict[str, Any] = {
        "config": fingerprint_file(artifact_dir / "config.json"),
        "index": fingerprint_file(artifact_dir / "model.safetensors.index.json"),
        "remote_code": {},
        "shards": [],
    }
    for filename in remote_code_filenames:
        path = artifact_dir / filename
        if path.exists():
            fingerprints["remote_code"][filename] = fingerprint_file(path)
    for shard_name in present_shards:
        shard_path = artifact_dir / shard_name
        entry = fingerprint_file(shard_path)
        entry["path"] = shard_name
        fingerprints["shards"].append(entry)
    return fingerprints


def fingerprint_file(path: Path) -> dict[str, Any]:
    return {
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
