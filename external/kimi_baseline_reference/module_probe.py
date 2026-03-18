from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

from .remote_code_support import (
    RemoteCodeError,
    ensure_hf_files,
    load_remote_kimi_modules,
    read_json,
)

KIMI_MODULE_PROBE_REQUEST_KIND = "attnres.kimi.module_probe_request"
KIMI_MODULE_PROBE_FIXTURE_KIND = "attnres.kimi.module_probe_fixture"
KIMI_MODULE_PROBE_VERSION = 1
KIMI_MODULE_PROBE_RUNTIME_DTYPE = "float32"
PUBLIC_KIMI_REPO_ID = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
REMOTE_CODE_FILENAMES = [
    "config.json",
    "model.safetensors.index.json",
    "configuration_kimi.py",
    "modeling_kimi.py",
]


class ModuleProbeError(RuntimeError):
    pass


def generate_fixture(
    artifact_dir: Path,
    request_path: Path,
    repo_id: str | None = None,
    revision: str | None = None,
    remote_code_dir: Path | None = None,
) -> tuple[dict[str, Any], str]:
    request = read_json(request_path)
    _validate_request_static(request)

    artifact_dir = artifact_dir.resolve()
    if remote_code_dir is None:
        remote_code_dir = artifact_dir

    if repo_id:
        ensure_hf_files(artifact_dir, repo_id, REMOTE_CODE_FILENAMES, revision=revision)

    config = read_json(artifact_dir / "config.json")
    validate_module_probe_artifact_config(config)
    _validate_artifact_matches_request(request["artifact"], config)

    request_tensor_names = sorted({name for probe in request["probes"] for name in tensor_names_for_target(probe["target"])})
    index = read_json(artifact_dir / "model.safetensors.index.json")
    required_shards = shard_names_for_tensors(index, request_tensor_names)
    if repo_id:
        ensure_hf_files(artifact_dir, repo_id, required_shards, revision=revision)

    kimi_config_cls, modeling_module, fla_backend = load_remote_kimi_modules(remote_code_dir)
    kimi_config = kimi_config_cls(**config)
    kimi_config._attn_implementation = "eager"
    torch.manual_seed(int(request["seed"]))

    probes = [
        generate_probe_case(
            artifact_dir=artifact_dir,
            config=config,
            kimi_config=kimi_config,
            modeling_module=modeling_module,
            probe=probe,
            index=index,
        )
        for probe in request["probes"]
    ]

    fixture = {
        "kind": KIMI_MODULE_PROBE_FIXTURE_KIND,
        "version": KIMI_MODULE_PROBE_VERSION,
        "seed": request["seed"],
        "artifact": request["artifact"],
        "tolerances": request["tolerances"],
        "probes": probes,
    }
    return fixture, fla_backend


def write_fixture(output_path: Path, fixture: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")


def generate_probe_case(
    artifact_dir: Path,
    config: dict[str, Any],
    kimi_config: Any,
    modeling_module: Any,
    probe: dict[str, Any],
    index: dict[str, Any],
) -> dict[str, Any]:
    _validate_probe_request(probe, config)
    tensor_names = tensor_names_for_target(probe["target"])
    payloads = load_named_tensors(artifact_dir, index, tensor_names)
    shard_paths = sorted({index["weight_map"][tensor_name] for tensor_name in tensor_names})
    fingerprint = {
        "tensor_names": tensor_names,
        "shard_paths": shard_paths,
        "tensor_fingerprint": fnv64_tensor_payloads(tensor_names, payloads),
    }
    input_tensor = tensor_from_json(probe["input"])

    target = probe["target"]
    kind = target["kind"]
    if kind == "kda_attention":
        layer_idx = int(target["layer_idx"])
        module = modeling_module.KimiDeltaAttention(kimi_config, layer_idx=layer_idx).eval()
        load_module_tensors(module, tensor_names, payloads, prefix=f"model.layers.{layer_idx}.self_attn.")
        with torch.no_grad():
            output = module(input_tensor, cache_params=None)
        decode_steps = generate_kda_decode_steps(module, input_tensor, kimi_config, layer_idx) if probe["compare_decode"] else []
    elif kind == "mla_attention":
        layer_idx = int(target["layer_idx"])
        module = modeling_module.KimiMLAAttention(kimi_config, layer_idx=layer_idx).eval()
        load_module_tensors(module, tensor_names, payloads, prefix=f"model.layers.{layer_idx}.self_attn.")
        with torch.no_grad():
            output = module(
                input_tensor,
                attention_mask=mla_attention_mask(
                    query_len=input_tensor.shape[1],
                    total_len=input_tensor.shape[1],
                    prefix_len=0,
                    device=input_tensor.device,
                ),
                past_key_values=None,
            )
        decode_steps = generate_mla_decode_steps(module, input_tensor, kimi_config, layer_idx, modeling_module) if probe["compare_decode"] else []
    elif kind == "final_norm":
        module = modeling_module.KimiRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"]).eval()
        load_module_state(module, {"weight": payloads["model.norm.weight"]})
        with torch.no_grad():
            output = module(input_tensor)
        decode_steps = []
    elif kind == "lm_head":
        module = torch.nn.Linear(config["hidden_size"], config["vocab_size"], bias=False).eval()
        load_module_state(module, {"weight": payloads["lm_head.weight"]})
        with torch.no_grad():
            output = module(input_tensor)
        decode_steps = []
    else:  # pragma: no cover - protected by validation
        raise ModuleProbeError(f"unsupported module probe target kind '{kind}'")

    return {
        "name": probe["name"],
        "target": probe["target"],
        "input": probe["input"],
        "output": tensor_json(output),
        "compare_decode": probe["compare_decode"],
        "decode_steps": decode_steps,
        "fingerprint": fingerprint,
    }


def generate_kda_decode_steps(module: Any, input_tensor: torch.Tensor, kimi_config: Any, layer_idx: int) -> list[dict[str, Any]]:
    cache = kimi_config is not None  # keep mypy happy
    del cache
    from_path = module.__class__.__module__
    modeling_module = __import__(from_path, fromlist=["KimiDynamicCache"])
    cache = modeling_module.KimiDynamicCache(kimi_config)

    seq_len = input_tensor.shape[1]
    steps: list[dict[str, Any]] = []
    for token_index in range(seq_len):
        token = input_tensor[:, token_index : token_index + 1, :]
        with torch.no_grad():
            output = module(token, cache_params=cache)
        q_state, k_state, v_state = cache.conv_states[layer_idx]
        steps.append(
            {
                "token_index": token_index,
                "output": tensor_json(output),
                "cache": {
                    "kind": "kda",
                    "processed_tokens": int(token_index + 1),
                    "q_conv_state": tensor_json(reshape_conv_state(q_state, kimi_config.linear_attn_config["num_heads"], kimi_config.linear_attn_config["head_dim"])),
                    "k_conv_state": tensor_json(reshape_conv_state(k_state, kimi_config.linear_attn_config["num_heads"], kimi_config.linear_attn_config["head_dim"])),
                    "v_conv_state": tensor_json(reshape_conv_state(v_state, kimi_config.linear_attn_config["num_heads"], kimi_config.v_head_dim)),
                    "recurrent_state": tensor_json(cache.recurrent_states[layer_idx]),
                },
            }
        )
    return steps


def generate_mla_decode_steps(
    module: Any,
    input_tensor: torch.Tensor,
    kimi_config: Any,
    layer_idx: int,
    modeling_module: Any,
) -> list[dict[str, Any]]:
    cache = modeling_module.KimiDynamicCache(kimi_config)
    seq_len = input_tensor.shape[1]
    steps: list[dict[str, Any]] = []
    for token_index in range(seq_len):
        token = input_tensor[:, token_index : token_index + 1, :]
        with torch.no_grad():
            output = module(
                token,
                attention_mask=mla_attention_mask(
                    query_len=1,
                    total_len=token_index + 1,
                    prefix_len=token_index,
                    device=token.device,
                ),
                past_key_values=cache,
            )
        steps.append(
            {
                "token_index": token_index,
                "output": tensor_json(output),
                "cache": {
                    "kind": "mla",
                    "processed_tokens": int(token_index + 1),
                    "keys": tensor_json(cache.key_cache[layer_idx]),
                    "values": tensor_json(cache.value_cache[layer_idx]),
                },
            }
        )
    return steps


def reshape_conv_state(state: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    if state.ndim == 4:
        return state.to(dtype=torch.float32)
    if state.ndim != 3:
        raise ModuleProbeError(f"unsupported KDA conv-state shape {tuple(state.shape)}")
    batch, hidden, history_len = state.shape
    if hidden != num_heads * head_dim:
        raise ModuleProbeError(
            f"KDA conv-state hidden dimension {hidden} does not match num_heads*head_dim={num_heads * head_dim}"
        )
    return state.to(dtype=torch.float32).view(batch, num_heads, head_dim, history_len).permute(0, 1, 3, 2).contiguous()


def mla_attention_mask(
    query_len: int,
    total_len: int,
    prefix_len: int,
    device: torch.device,
) -> torch.Tensor:
    query_positions = torch.arange(prefix_len, prefix_len + query_len, dtype=torch.int64, device=device).unsqueeze(1)
    key_positions = torch.arange(0, total_len, dtype=torch.int64, device=device).unsqueeze(0)
    disallowed = query_positions < key_positions
    mask = torch.zeros((query_len, total_len), dtype=torch.float32, device=device)
    mask = mask.masked_fill(disallowed, -1e9)
    return mask.view(1, 1, query_len, total_len)


def load_module_tensors(module: torch.nn.Module, tensor_names: list[str], payloads: dict[str, torch.Tensor], prefix: str) -> None:
    state = {}
    for tensor_name in tensor_names:
        if not tensor_name.startswith(prefix):
            raise ModuleProbeError(f"tensor '{tensor_name}' does not start with expected prefix '{prefix}'")
        state[tensor_name[len(prefix) :]] = payloads[tensor_name]
    load_module_state(module, state)


def load_module_state(module: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    module_state = module.state_dict()
    for key, value in state.items():
        if key not in module_state:
            raise ModuleProbeError(f"module state is missing key '{key}'")
        expected_shape = tuple(module_state[key].shape)
        actual_value = value
        actual_shape = tuple(actual_value.shape)
        if len(expected_shape) == 2 and actual_shape == (expected_shape[1], expected_shape[0]):
            actual_value = actual_value.transpose(0, 1).contiguous()
            actual_shape = tuple(actual_value.shape)
        if expected_shape != actual_shape:
            raise ModuleProbeError(
                f"module state key '{key}' expected shape {expected_shape}, got {actual_shape}"
            )
        module_state[key] = actual_value.to(dtype=torch.float32)
    missing = sorted(set(module_state.keys()) - set(state.keys()))
    if missing:
        raise ModuleProbeError(f"module state is missing required tensors: {missing}")
    module.load_state_dict(module_state, strict=True)


def load_named_tensors(
    artifact_dir: Path,
    index: dict[str, Any],
    tensor_names: list[str],
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    tensors_by_shard: dict[str, list[str]] = {}
    for tensor_name in tensor_names:
        shard_name = index["weight_map"].get(tensor_name)
        if not shard_name:
            raise ModuleProbeError(f"artifact index is missing tensor '{tensor_name}'")
        tensors_by_shard.setdefault(shard_name, []).append(tensor_name)

    for shard_name, shard_tensors in tensors_by_shard.items():
        shard_path = artifact_dir / shard_name
        if not shard_path.exists():
            raise ModuleProbeError(f"missing safetensors shard '{shard_path}'")
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            for tensor_name in shard_tensors:
                tensors[tensor_name] = shard.get_tensor(tensor_name).to(dtype=torch.float32).contiguous()
    return tensors


def shard_names_for_tensors(index: dict[str, Any], tensor_names: list[str]) -> list[str]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ModuleProbeError("artifact shard index is missing 'weight_map'")
    missing = [tensor_name for tensor_name in tensor_names if tensor_name not in weight_map]
    if missing:
        raise ModuleProbeError(f"artifact shard index is missing probe tensors: {missing}")
    return sorted({weight_map[tensor_name] for tensor_name in tensor_names})


def tensor_names_for_target(target: dict[str, Any]) -> list[str]:
    kind = target["kind"]
    if kind == "kda_attention":
        layer_idx = int(target["layer_idx"])
        return [
            f"model.layers.{layer_idx}.self_attn.q_proj.weight",
            f"model.layers.{layer_idx}.self_attn.k_proj.weight",
            f"model.layers.{layer_idx}.self_attn.v_proj.weight",
            f"model.layers.{layer_idx}.self_attn.q_conv1d.weight",
            f"model.layers.{layer_idx}.self_attn.k_conv1d.weight",
            f"model.layers.{layer_idx}.self_attn.v_conv1d.weight",
            f"model.layers.{layer_idx}.self_attn.A_log",
            f"model.layers.{layer_idx}.self_attn.f_a_proj.weight",
            f"model.layers.{layer_idx}.self_attn.f_b_proj.weight",
            f"model.layers.{layer_idx}.self_attn.dt_bias",
            f"model.layers.{layer_idx}.self_attn.b_proj.weight",
            f"model.layers.{layer_idx}.self_attn.g_a_proj.weight",
            f"model.layers.{layer_idx}.self_attn.g_b_proj.weight",
            f"model.layers.{layer_idx}.self_attn.o_norm.weight",
            f"model.layers.{layer_idx}.self_attn.o_proj.weight",
        ]
    if kind == "mla_attention":
        layer_idx = int(target["layer_idx"])
        return [
            f"model.layers.{layer_idx}.self_attn.q_proj.weight",
            f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight",
            f"model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight",
            f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight",
            f"model.layers.{layer_idx}.self_attn.o_proj.weight",
        ]
    if kind == "final_norm":
        return ["model.norm.weight"]
    if kind == "lm_head":
        return ["lm_head.weight"]
    raise ModuleProbeError(f"unsupported probe target kind '{kind}'")


def tensor_json(tensor: torch.Tensor) -> dict[str, Any]:
    array = tensor.detach().cpu().to(dtype=torch.float32).contiguous().numpy()
    return {"dims": list(array.shape), "values": array.reshape(-1).tolist()}


def tensor_from_json(payload: dict[str, Any]) -> torch.Tensor:
    dims = payload.get("dims")
    values = payload.get("values")
    if not isinstance(dims, list) or not isinstance(values, list):
        raise ModuleProbeError("module probe tensor payload must contain 'dims' and 'values'")
    return torch.tensor(values, dtype=torch.float32).reshape(dims)


def fnv64_tensor_payloads(tensor_names: list[str], payloads: dict[str, torch.Tensor]) -> str:
    hash_value = 0xCBF29CE484222325
    for tensor_name in tensor_names:
        hash_value = fnv64_bytes(hash_value, tensor_name.encode("utf-8"))
        array = payloads[tensor_name].detach().cpu().to(dtype=torch.float32).contiguous().numpy()
        for dim in array.shape:
            hash_value = fnv64_bytes(hash_value, struct.pack("<Q", int(dim)))
        hash_value = fnv64_bytes(hash_value, np.asarray(array, dtype=np.float32).tobytes(order="C"))
    return f"{hash_value:016x}"


def fnv64_bytes(hash_value: int, payload: bytes) -> int:
    for byte in payload:
        hash_value ^= int(byte)
        hash_value = (hash_value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return hash_value


def _validate_request_static(request: dict[str, Any]) -> None:
    expected_keys = {"kind", "version", "seed", "artifact", "tolerances", "probes"}
    actual_keys = set(request.keys())
    if actual_keys != expected_keys:
        raise ModuleProbeError(
            f"module probe request schema mismatch: expected keys {sorted(expected_keys)}, got {sorted(actual_keys)}"
        )
    if request["kind"] != KIMI_MODULE_PROBE_REQUEST_KIND:
        raise ModuleProbeError(
            f"expected module probe request kind '{KIMI_MODULE_PROBE_REQUEST_KIND}', got '{request['kind']}'"
        )
    if request["version"] != KIMI_MODULE_PROBE_VERSION:
        raise ModuleProbeError(
            f"expected module probe version {KIMI_MODULE_PROBE_VERSION}, got {request['version']}"
        )
    if not isinstance(request["seed"], int) or request["seed"] < 0:
        raise ModuleProbeError("module probe request field 'seed' must be a non-negative integer")
    _validate_artifact_spec(request["artifact"])
    tolerances = request["tolerances"]
    if tolerances.get("runtime_dtype") != KIMI_MODULE_PROBE_RUNTIME_DTYPE:
        raise ModuleProbeError(
            f"expected module probe runtime dtype '{KIMI_MODULE_PROBE_RUNTIME_DTYPE}', got '{tolerances.get('runtime_dtype')}'"
        )
    for field in ("output_max_abs_diff", "cache_max_abs_diff"):
        value = tolerances.get(field)
        if not isinstance(value, (int, float)) or value < 0:
            raise ModuleProbeError(f"module probe tolerance '{field}' must be finite and >= 0")
    probes = request["probes"]
    if not isinstance(probes, list) or not probes:
        raise ModuleProbeError("module probe request field 'probes' must be a non-empty list")


def _validate_artifact_spec(artifact: dict[str, Any]) -> None:
    expected_keys = {"model_type", "dtype", "num_hidden_layers", "hidden_size", "vocab_size"}
    if set(artifact.keys()) != expected_keys:
        raise ModuleProbeError("module probe request field 'artifact' has the wrong schema")
    if artifact["model_type"] != "kimi_linear":
        raise ModuleProbeError("module probe request field 'artifact.model_type' must be 'kimi_linear'")
    if artifact["dtype"] not in ("float32", "bfloat16"):
        raise ModuleProbeError("module probe request field 'artifact.dtype' must be 'float32' or 'bfloat16'")
    for field in ("num_hidden_layers", "hidden_size", "vocab_size"):
        if not isinstance(artifact[field], int) or artifact[field] <= 0:
            raise ModuleProbeError(f"module probe request field 'artifact.{field}' must be a positive integer")


def _validate_artifact_matches_request(artifact: dict[str, Any], config: dict[str, Any]) -> None:
    for field in ("model_type", "dtype", "num_hidden_layers", "hidden_size", "vocab_size"):
        if artifact[field] != config[field]:
            raise ModuleProbeError(
                f"module probe artifact field '{field}' expected {artifact[field]!r}, got {config[field]!r}"
            )


def validate_module_probe_artifact_config(config: dict[str, Any]) -> None:
    required_fields = [
        "model_type",
        "dtype",
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "kv_lora_rank",
        "q_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "mla_use_nope",
        "hidden_act",
        "tie_word_embeddings",
        "use_cache",
        "rms_norm_eps",
        "linear_attn_config",
    ]
    missing = [field for field in required_fields if field not in config]
    if missing:
        raise ModuleProbeError(f"artifact config is missing required fields: {missing}")
    if config["model_type"] != "kimi_linear":
        raise ModuleProbeError("artifact config field 'model_type' must be 'kimi_linear'")
    if config["dtype"] not in ("float32", "bfloat16"):
        raise ModuleProbeError("artifact config field 'dtype' must be 'float32' or 'bfloat16'")
    if config["hidden_act"] != "silu":
        raise ModuleProbeError("artifact config field 'hidden_act' must be 'silu'")
    if config["q_lora_rank"] is not None:
        raise ModuleProbeError("artifact config field 'q_lora_rank' must be null for the current module probe path")
    if bool(config["tie_word_embeddings"]):
        raise ModuleProbeError("artifact config field 'tie_word_embeddings' must be false")
    linear_attn = config["linear_attn_config"]
    if not isinstance(linear_attn, dict):
        raise ModuleProbeError("artifact config field 'linear_attn_config' must be an object")
    for field in ("full_attn_layers", "kda_layers", "num_heads", "head_dim", "short_conv_kernel_size"):
        if field not in linear_attn:
            raise ModuleProbeError(f"artifact config field 'linear_attn_config.{field}' is required")


def _validate_probe_request(probe: dict[str, Any], config: dict[str, Any]) -> None:
    expected_keys = {"name", "target", "input", "compare_decode"}
    if set(probe.keys()) != expected_keys:
        raise ModuleProbeError(f"module probe '{probe.get('name', '<unknown>')}' has the wrong schema")
    if not isinstance(probe["name"], str) or not probe["name"]:
        raise ModuleProbeError("module probe field 'name' must be a non-empty string")
    target = probe["target"]
    if not isinstance(target, dict) or "kind" not in target:
        raise ModuleProbeError(f"module probe '{probe['name']}' target must be an object with a 'kind' field")
    input_dims = probe["input"].get("dims")
    if not isinstance(input_dims, list) or len(input_dims) != 3 or input_dims[-1] != config["hidden_size"]:
        raise ModuleProbeError(
            f"module probe '{probe['name']}' expected input dims [batch, seq, {config['hidden_size']}], got {input_dims!r}"
        )

    kind = target["kind"]
    if kind in ("kda_attention", "mla_attention"):
        layer_idx = target.get("layer_idx")
        if not isinstance(layer_idx, int) or layer_idx < 0 or layer_idx >= config["num_hidden_layers"]:
            raise ModuleProbeError(
                f"module probe '{probe['name']}' layer_idx {layer_idx!r} is out of range for num_hidden_layers={config['num_hidden_layers']}"
            )
        if kind == "kda_attention" and (layer_idx + 1) not in config["linear_attn_config"]["kda_layers"]:
            raise ModuleProbeError(
                f"module probe '{probe['name']}' expected a KDA layer at index {layer_idx}"
            )
        if kind == "mla_attention" and (layer_idx + 1) not in config["linear_attn_config"]["full_attn_layers"]:
            raise ModuleProbeError(
                f"module probe '{probe['name']}' expected an MLA layer at index {layer_idx}"
            )
    elif kind in ("final_norm", "lm_head"):
        if probe["compare_decode"]:
            raise ModuleProbeError(
                f"module probe '{probe['name']}' requested decode comparison for non-attention target '{kind}'"
            )
    else:
        raise ModuleProbeError(f"module probe '{probe['name']}' uses unsupported target kind '{kind}'")
