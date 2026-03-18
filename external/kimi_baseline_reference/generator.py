from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file

BASELINE_SLICE_REQUEST_KIND = "attnres.kimi.baseline_slice_request"
BASELINE_SLICE_REQUEST_VERSION = 1
BASELINE_SLICE_PARITY_KIND = "attnres.kimi.baseline_slice_parity"
BASELINE_SLICE_PARITY_VERSION = 1
SEEDED_INIT_STATE_KIND = "attnres.kimi.seeded_init_state"
SEEDED_INIT_STATE_VERSION = 1

EXPECTED_MANIFEST: dict[str, Any] = {
    "kind": BASELINE_SLICE_REQUEST_KIND,
    "version": BASELINE_SLICE_REQUEST_VERSION,
    "seed": 20260318,
    "artifact": {
        "model_type": "kimi_linear",
        "dtype": "float32",
        "num_hidden_layers": 2,
        "hidden_size": 8,
        "vocab_size": 16,
    },
    "slice": {
        "import_selection": {
            "layer_indices": [0, 1],
            "include_embeddings": True,
            "include_final_norm": True,
            "include_lm_head": True,
        },
        "selected_hidden_layers": [0],
        "requested_modules": [
            "Embeddings",
            {"DecoderLayer": {"layer_idx": 0, "component": "InputNorm"}},
            {
                "DecoderLayer": {
                    "layer_idx": 0,
                    "component": {"Attention": {"kind": "LinearAttentionKda"}},
                }
            },
            {"DecoderLayer": {"layer_idx": 0, "component": "PostAttentionNorm"}},
            {
                "DecoderLayer": {
                    "layer_idx": 0,
                    "component": {"FeedForward": {"kind": "DenseMlp"}},
                }
            },
            {"DecoderLayer": {"layer_idx": 1, "component": "InputNorm"}},
            {
                "DecoderLayer": {
                    "layer_idx": 1,
                    "component": {"Attention": {"kind": "FullAttention"}},
                }
            },
            {"DecoderLayer": {"layer_idx": 1, "component": "PostAttentionNorm"}},
            {
                "DecoderLayer": {
                    "layer_idx": 1,
                    "component": {"FeedForward": {"kind": "SparseMoe"}},
                }
            },
            "FinalNorm",
            "LmHead",
        ],
        "required_tensors": [
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.1.input_layernorm.weight",
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.1.self_attn.o_proj.weight",
            "model.layers.1.post_attention_layernorm.weight",
            "model.layers.1.block_sparse_moe.gate.weight",
            "model.layers.1.block_sparse_moe.shared_experts.gate_proj.weight",
            "model.layers.1.block_sparse_moe.shared_experts.up_proj.weight",
            "model.layers.1.block_sparse_moe.shared_experts.down_proj.weight",
            "model.layers.1.block_sparse_moe.experts.0.w1.weight",
            "model.layers.1.block_sparse_moe.experts.0.w2.weight",
            "model.layers.1.block_sparse_moe.experts.0.w3.weight",
            "model.layers.1.block_sparse_moe.experts.1.w1.weight",
            "model.layers.1.block_sparse_moe.experts.1.w2.weight",
            "model.layers.1.block_sparse_moe.experts.1.w3.weight",
            "model.norm.weight",
            "lm_head.weight",
        ],
        "tolerances": {
            "metric": "max_abs_diff",
            "runtime_dtype": "float32",
            "logits_max_abs_diff": 0.5,
            "hidden_state_max_abs_diff": 1.0,
        },
    },
    "prompts": [
        {"name": "single_token_0", "input_ids": [0]},
        {"name": "single_token_5", "input_ids": [5]},
    ],
}


class GeneratorError(RuntimeError):
    pass


@dataclass
class LoadedFixtureInputs:
    manifest: dict[str, Any]
    config: dict[str, Any]
    tensors: dict[str, np.ndarray]


def generate_fixture(
    artifact_dir: Path,
    manifest_path: Path,
    seeded_init_path: Path,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    _validate_manifest(manifest)

    seeded_init = _load_json(seeded_init_path)
    _validate_seeded_init(seeded_init, manifest)

    config = _load_json(artifact_dir / "config.json")
    index = _load_json(artifact_dir / "model.safetensors.index.json")
    _validate_artifact_config(config)
    _validate_required_tensors(index, manifest["slice"]["required_tensors"])

    tensors = _load_seeded_init_tensors(seeded_init)
    tensors.update(_load_required_tensors(artifact_dir, index, manifest["slice"]["required_tensors"]))

    reference = KimiReferenceModel(config, tensors)
    prompt_results = []
    for prompt in manifest["prompts"]:
        prompt_results.append(
            reference.run_prompt(prompt["name"], prompt["input_ids"], manifest["slice"]["selected_hidden_layers"])
        )

    return {
        "kind": BASELINE_SLICE_PARITY_KIND,
        "version": BASELINE_SLICE_PARITY_VERSION,
        "seed": manifest["seed"],
        "artifact": manifest["artifact"],
        "slice": manifest["slice"],
        "prompts": manifest["prompts"],
        "prompt_results": prompt_results,
    }


class KimiReferenceModel:
    def __init__(self, config: dict[str, Any], tensors: dict[str, np.ndarray]) -> None:
        self.config = config
        self.tensors = tensors
        self.hidden_size = config["hidden_size"]
        self.num_hidden_layers = config["num_hidden_layers"]
        self.num_attention_heads = config["num_attention_heads"]
        self.num_key_value_heads = config["num_key_value_heads"]
        self.v_head_dim = config["v_head_dim"]
        self.qk_nope_head_dim = config["qk_nope_head_dim"]
        self.qk_rope_head_dim = config["qk_rope_head_dim"]
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.kv_repeat_factor = self.num_attention_heads // self.num_key_value_heads
        self.mla_use_nope = bool(config["mla_use_nope"])
        self.rms_norm_eps = np.float32(config["rms_norm_eps"])
        self.linear_attn = config["linear_attn_config"]

    def run_prompt(
        self,
        prompt_name: str,
        input_ids: list[int],
        selected_hidden_layers: list[int],
    ) -> dict[str, Any]:
        token_array = np.asarray(input_ids, dtype=np.int64)[None, :]
        hidden = self.embed_tokens(token_array)
        hidden_states = []
        for layer_idx in range(self.num_hidden_layers):
            hidden = self.forward_layer(layer_idx, hidden)
            if layer_idx in selected_hidden_layers:
                hidden_states.append(
                    {
                        "layer_idx": layer_idx,
                        "tensor": _tensor_json(hidden),
                    }
                )

        logits = self.linear(self.rms_norm(hidden, self.tensor("model.norm.weight")), "lm_head")
        return {
            "prompt_name": prompt_name,
            "input_ids": input_ids,
            "logits": _tensor_json(logits),
            "hidden_states": hidden_states,
        }

    def forward_layer(self, layer_idx: int, hidden: np.ndarray) -> np.ndarray:
        prefix = f"model.layers.{layer_idx}"

        residual = hidden
        normed = self.rms_norm(hidden, self.tensor(f"{prefix}.input_layernorm.weight"))
        if self.is_kda_layer(layer_idx):
            attention_out = self.forward_kda(layer_idx, normed)
        else:
            attention_out = self.forward_mla(layer_idx, normed)
        hidden = _f32(residual + attention_out)

        residual = hidden
        normed = self.rms_norm(hidden, self.tensor(f"{prefix}.post_attention_layernorm.weight"))
        if self.is_sparse_moe_layer(layer_idx):
            ff_out = self.forward_sparse_moe(layer_idx, normed)
        else:
            ff_out = self.forward_dense_mlp(layer_idx, normed)
        return _f32(residual + ff_out)

    def embed_tokens(self, input_ids: np.ndarray) -> np.ndarray:
        weights = self.tensor("model.embed_tokens.weight")
        return _f32(weights[input_ids])

    def rms_norm(self, x: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        variance = np.mean(np.square(x, dtype=np.float32), axis=-1, keepdims=True, dtype=np.float32)
        rms = np.sqrt(_f32(variance + self.rms_norm_eps))
        return _f32((x / rms) * gamma.reshape(1, 1, -1))

    def forward_kda(self, layer_idx: int, hidden: np.ndarray) -> np.ndarray:
        prefix = f"model.layers.{layer_idx}.self_attn"
        batch, seq_len, _ = hidden.shape
        num_heads = self.linear_attn["num_heads"]
        linear_head_dim = self.linear_attn["head_dim"]
        value_head_dim = self.v_head_dim

        q = _softplus(self.linear(hidden, f"{prefix}.q_proj")).reshape(batch, seq_len, num_heads, linear_head_dim)
        q = _f32(np.swapaxes(q, 1, 2) + np.float32(1e-6))
        k = _softplus(self.linear(hidden, f"{prefix}.k_proj")).reshape(batch, seq_len, num_heads, linear_head_dim)
        k = _f32(np.swapaxes(k, 1, 2) + np.float32(1e-6))
        v = self.linear(hidden, f"{prefix}.v_proj").reshape(batch, seq_len, num_heads, value_head_dim)
        v = _f32(np.swapaxes(v, 1, 2))

        kv = _f32(k[..., :, :, None] * v[..., :, None, :])
        prefix_kv = np.cumsum(kv, axis=2, dtype=np.float32)
        prefix_k = np.cumsum(k, axis=2, dtype=np.float32)
        numerator = np.sum(q[..., :, :, None] * prefix_kv, axis=3, dtype=np.float32)
        denominator = np.sum(q * prefix_k, axis=3, keepdims=True, dtype=np.float32) + np.float32(1e-6)
        linear_out = _f32(numerator / denominator)
        conv_out = self.short_conv_context(k, self.linear_attn["short_conv_kernel_size"])
        merged = _f32(linear_out + conv_out)
        merged = _f32(np.swapaxes(merged, 1, 2).reshape(batch, seq_len, num_heads * value_head_dim))
        return self.linear(merged, f"{prefix}.o_proj")

    def forward_mla(self, layer_idx: int, hidden: np.ndarray) -> np.ndarray:
        prefix = f"model.layers.{layer_idx}.self_attn"
        batch, seq_len, _ = hidden.shape

        q = self.linear(hidden, f"{prefix}.q_proj").reshape(
            batch, seq_len, self.num_attention_heads, self.qk_head_dim
        )
        q = _f32(np.swapaxes(self.apply_nope_policy(q), 1, 2))

        kv_latent = self.linear(hidden, f"{prefix}.kv_down")
        k = self.linear(kv_latent, f"{prefix}.k_up").reshape(
            batch, seq_len, self.num_key_value_heads, self.qk_head_dim
        )
        k = _f32(np.swapaxes(self.apply_nope_policy(k), 1, 2))

        v = self.linear(kv_latent, f"{prefix}.v_up").reshape(
            batch, seq_len, self.num_key_value_heads, self.v_head_dim
        )
        v = _f32(np.swapaxes(v, 1, 2))

        k = self.expand_kv_heads(k)
        v = self.expand_kv_heads(v)
        scores = np.matmul(q, np.swapaxes(k, 2, 3))
        scores = _f32(scores / np.float32(math.sqrt(self.qk_head_dim)))
        scores = _f32(scores + self.causal_mask(seq_len))
        weights = _softmax(scores, axis=3)
        attended = np.matmul(weights, v)
        attended = _f32(np.swapaxes(attended, 1, 2).reshape(batch, seq_len, self.num_attention_heads * self.v_head_dim))
        return self.linear(attended, f"{prefix}.o_proj")

    def forward_dense_mlp(self, layer_idx: int, hidden: np.ndarray) -> np.ndarray:
        prefix = f"model.layers.{layer_idx}.mlp"
        return self.forward_mlp_expert(prefix, hidden)

    def forward_sparse_moe(self, layer_idx: int, hidden: np.ndarray) -> np.ndarray:
        prefix = f"model.layers.{layer_idx}.block_sparse_moe"
        gate_logits = self.linear(hidden, f"{prefix}.gate")
        topk = np.argsort(-gate_logits, axis=2)[..., : self.config["num_experts_per_token"]]
        routing_mask = np.zeros_like(gate_logits, dtype=np.float32)
        np.put_along_axis(routing_mask, topk, np.float32(1.0), axis=2)
        sparse_logits = np.where(routing_mask == 0.0, np.float32(-1e9), gate_logits).astype(np.float32)
        routing_weights = _softmax(sparse_logits, axis=2)[..., None]

        expert_outputs = [
            self.forward_mlp_expert(f"{prefix}.experts.{expert_idx}", hidden)[:, :, None, :]
            for expert_idx in range(self.config["num_experts"])
        ]
        routed = np.concatenate(expert_outputs, axis=2)
        output = np.sum(routed * routing_weights, axis=2, dtype=np.float32)

        shared_expert_count = self.config["num_shared_experts"]
        if shared_expert_count:
            shared_outputs = [
                self.forward_mlp_expert(f"{prefix}.shared_experts", hidden)
                for _ in range(shared_expert_count)
            ]
            shared = np.sum(shared_outputs, axis=0, dtype=np.float32) / np.float32(shared_expert_count)
            output = _f32(output + shared)

        return _f32(output)

    def forward_mlp_expert(self, prefix: str, hidden: np.ndarray) -> np.ndarray:
        if prefix.endswith(".experts.0") or prefix.endswith(".experts.1"):
            gate = _silu(self.linear(hidden, f"{prefix}.w1"))
            up = self.linear(hidden, f"{prefix}.w3")
            return self.linear(_f32(gate * up), f"{prefix}.w2")

        gate = _silu(self.linear(hidden, f"{prefix}.gate_proj"))
        up = self.linear(hidden, f"{prefix}.up_proj")
        return self.linear(_f32(gate * up), f"{prefix}.down_proj")

    def apply_nope_policy(self, tensor: np.ndarray) -> np.ndarray:
        if self.mla_use_nope or self.qk_nope_head_dim == 0:
            return _f32(tensor)

        batch, seq_len, heads, _ = tensor.shape
        rope = tensor[:, :, :, self.qk_nope_head_dim : self.qk_head_dim]
        zeros = np.zeros((batch, seq_len, heads, self.qk_nope_head_dim), dtype=np.float32)
        return _f32(np.concatenate([zeros, rope], axis=3))

    def expand_kv_heads(self, tensor: np.ndarray) -> np.ndarray:
        if self.kv_repeat_factor == 1:
            return _f32(tensor)
        return _f32(np.repeat(tensor, self.kv_repeat_factor, axis=1))

    def causal_mask(self, seq_len: int) -> np.ndarray:
        positions = np.arange(seq_len, dtype=np.int64)
        allowed = positions[:, None] >= positions[None, :]
        mask = np.where(allowed, np.float32(0.0), np.float32(-1e9)).astype(np.float32)
        return mask.reshape(1, 1, seq_len, seq_len)

    def short_conv_context(self, projected_keys: np.ndarray, kernel_size: int) -> np.ndarray:
        batch, heads, seq_len, head_dim = projected_keys.shape
        if kernel_size <= 1:
            return np.zeros((batch, heads, seq_len, head_dim), dtype=np.float32)

        windows = []
        for token_offset in range(seq_len):
            end = token_offset + 1
            start = max(0, end - kernel_size)
            window = np.mean(projected_keys[:, :, start:end, :], axis=2, dtype=np.float32)
            windows.append(window[:, :, None, :])
        return _f32(np.concatenate(windows, axis=2))

    def linear(self, x: np.ndarray, prefix: str) -> np.ndarray:
        weight = self.tensor(f"{prefix}.weight")
        bias = self.tensor(f"{prefix}.bias")
        return _f32(np.matmul(x, weight) + bias)

    def tensor(self, name: str) -> np.ndarray:
        if name not in self.tensors:
            raise GeneratorError(f"missing tensor '{name}' in the seeded-init/reference state")
        return self.tensors[name]

    def is_kda_layer(self, layer_idx: int) -> bool:
        return (layer_idx + 1) in self.linear_attn["kda_layers"]

    def is_sparse_moe_layer(self, layer_idx: int) -> bool:
        return layer_idx >= self.config["first_k_dense_replace"] and layer_idx % self.config["moe_layer_freq"] == 0


def write_fixture(output_path: Path, fixture: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GeneratorError(f"missing json file '{path}'") from exc
    except json.JSONDecodeError as exc:
        raise GeneratorError(f"failed to parse json file '{path}': {exc}") from exc


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("kind") != BASELINE_SLICE_REQUEST_KIND:
        raise GeneratorError(
            f"expected baseline slice request manifest kind '{BASELINE_SLICE_REQUEST_KIND}', got '{manifest.get('kind')}'"
        )
    if manifest.get("version") != BASELINE_SLICE_REQUEST_VERSION:
        raise GeneratorError(
            f"expected baseline slice request manifest version {BASELINE_SLICE_REQUEST_VERSION}, got {manifest.get('version')}"
        )

    for field in ("seed", "artifact", "prompts"):
        _expect_equal(field, manifest.get(field), EXPECTED_MANIFEST[field], "baseline manifest")

    _expect_equal(
        "slice.import_selection",
        manifest["slice"].get("import_selection"),
        EXPECTED_MANIFEST["slice"]["import_selection"],
        "baseline manifest",
    )
    _expect_equal(
        "slice.selected_hidden_layers",
        manifest["slice"].get("selected_hidden_layers"),
        EXPECTED_MANIFEST["slice"]["selected_hidden_layers"],
        "baseline manifest",
    )
    _expect_equal(
        "slice.requested_modules",
        manifest["slice"].get("requested_modules"),
        EXPECTED_MANIFEST["slice"]["requested_modules"],
        "baseline manifest",
    )
    _expect_equal(
        "slice.required_tensors",
        manifest["slice"].get("required_tensors"),
        EXPECTED_MANIFEST["slice"]["required_tensors"],
        "baseline manifest",
    )
    _expect_equal(
        "slice.tolerances",
        manifest["slice"].get("tolerances"),
        EXPECTED_MANIFEST["slice"]["tolerances"],
        "baseline manifest",
    )


def _validate_seeded_init(seeded_init: dict[str, Any], manifest: dict[str, Any]) -> None:
    if seeded_init.get("kind") != SEEDED_INIT_STATE_KIND:
        raise GeneratorError(
            f"expected seeded init state kind '{SEEDED_INIT_STATE_KIND}', got '{seeded_init.get('kind')}'"
        )
    if seeded_init.get("version") != SEEDED_INIT_STATE_VERSION:
        raise GeneratorError(
            f"expected seeded init state version {SEEDED_INIT_STATE_VERSION}, got {seeded_init.get('version')}"
        )
    _expect_equal("seed", seeded_init.get("seed"), manifest["seed"], "seeded init state")
    _expect_equal("artifact", seeded_init.get("artifact"), manifest["artifact"], "seeded init state")


def _validate_artifact_config(config: dict[str, Any]) -> None:
    _expect_equal("model_type", config.get("model_type"), EXPECTED_MANIFEST["artifact"]["model_type"], "artifact config")
    _expect_equal("dtype", config.get("dtype"), EXPECTED_MANIFEST["artifact"]["dtype"], "artifact config")
    _expect_equal(
        "num_hidden_layers",
        config.get("num_hidden_layers"),
        EXPECTED_MANIFEST["artifact"]["num_hidden_layers"],
        "artifact config",
    )
    _expect_equal("hidden_size", config.get("hidden_size"), EXPECTED_MANIFEST["artifact"]["hidden_size"], "artifact config")
    _expect_equal("vocab_size", config.get("vocab_size"), EXPECTED_MANIFEST["artifact"]["vocab_size"], "artifact config")


def _validate_required_tensors(index: dict[str, Any], required_tensors: list[str]) -> None:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise GeneratorError("artifact shard index is missing 'weight_map'")
    actual = [tensor_name for tensor_name in required_tensors if tensor_name not in weight_map]
    if actual:
        raise GeneratorError(f"artifact shard index is missing required tensors: {actual}")


def _load_seeded_init_tensors(seeded_init: dict[str, Any]) -> dict[str, np.ndarray]:
    tensors = {}
    for name, payload in seeded_init["tensors"].items():
        dims = payload["dims"]
        values = np.asarray(payload["values"], dtype=np.float32)
        tensors[name] = values.reshape(dims)
    return tensors


def _load_required_tensors(
    artifact_dir: Path,
    index: dict[str, Any],
    required_tensors: list[str],
) -> dict[str, np.ndarray]:
    weight_map = index["weight_map"]
    tensors = {}
    shard_cache: dict[str, dict[str, np.ndarray]] = {}
    for tensor_name in required_tensors:
        shard_name = weight_map[tensor_name]
        if shard_name not in shard_cache:
            shard_cache[shard_name] = load_file(str(artifact_dir / shard_name))
        tensors[tensor_name] = np.asarray(shard_cache[shard_name][tensor_name], dtype=np.float32)
    return tensors


def _expect_equal(field: str, actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise GeneratorError(
            f"{label} field '{field}' expected {json.dumps(expected, sort_keys=True)}, got {json.dumps(actual, sort_keys=True)}"
        )


def _tensor_json(tensor: np.ndarray) -> dict[str, Any]:
    array = np.asarray(tensor, dtype=np.float32)
    return {
        "dims": list(array.shape),
        "values": array.reshape(-1).tolist(),
    }


def _f32(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _softplus(x: np.ndarray) -> np.ndarray:
    array = _f32(x)
    return _f32(np.log1p(np.exp(-np.abs(array))) + np.maximum(array, np.float32(0.0)))


def _silu(x: np.ndarray) -> np.ndarray:
    array = _f32(x)
    return _f32(array / (np.float32(1.0) + np.exp(-array)))


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    array = _f32(x)
    shifted = _f32(array - np.max(array, axis=axis, keepdims=True))
    exp = _f32(np.exp(shifted))
    return _f32(exp / np.sum(exp, axis=axis, keepdims=True, dtype=np.float32))
