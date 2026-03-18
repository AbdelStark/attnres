from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file

BASELINE_SLICE_REQUEST_KIND = "attnres.kimi.baseline_slice_request"
BASELINE_SLICE_REQUEST_VERSION = 1
BASELINE_SLICE_PARITY_KIND = "attnres.kimi.baseline_slice_parity"
BASELINE_SLICE_PARITY_VERSION = 1
LOCAL_INIT_CONTRACT_KIND = "attnres.kimi.local_init_contract"
LOCAL_INIT_CONTRACT_VERSION = 1
LOCAL_INIT_CONTRACT_STRATEGY = "burn.ndarray.lazy_linear_kaiming_uniform.v1"

EXPECTED_LOCAL_INIT_CONTRACT: dict[str, Any] = {
    "kind": LOCAL_INIT_CONTRACT_KIND,
    "version": LOCAL_INIT_CONTRACT_VERSION,
    "strategy": LOCAL_INIT_CONTRACT_STRATEGY,
}

# Stable float32-byte digest of the executed pilot local-init tensor set. This
# remains useful as a regression lock for the original fixed bundle, even
# though the manifest validator now accepts a broader set of attnres-emitted
# request manifests on the same supported surface.
PILOT_LOCAL_INIT_FLOAT32_DIGEST = "2791fc6202eb306e30b9ce2923c08d697300a8e87ebdd60ea679ce0f5bba248c"

_MASK32 = 0xFFFFFFFF
_MASK64 = 0xFFFFFFFFFFFFFFFF
_PCG32_MUL = 6364136223846793005
_PCG32_INC = 11634580027462260723


class GeneratorError(RuntimeError):
    pass


@dataclass(frozen=True)
class LinearModuleSpec:
    prefix: str
    d_input: int
    d_output: int


def generate_fixture(
    artifact_dir: Path,
    manifest_path: Path,
    local_init_contract_path: Path,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    compare_logits = _validate_manifest(manifest)

    local_init_contract = _load_json(local_init_contract_path)
    _validate_local_init_contract(local_init_contract)

    config = _load_json(artifact_dir / "config.json")
    index = _load_json(artifact_dir / "model.safetensors.index.json")
    _validate_artifact_config(config)
    _validate_manifest_artifact_matches_config(manifest["artifact"], config)
    _validate_required_tensors(index, manifest["slice"]["required_tensors"])

    tensors = _load_required_tensors(artifact_dir, index, manifest["slice"]["required_tensors"])
    tensors.update(
        reconstruct_local_init_tensors(
            config=config,
            seed=manifest["seed"],
            required_tensors=manifest["slice"]["required_tensors"],
            selected_hidden_layers=manifest["slice"]["selected_hidden_layers"],
            compare_logits=compare_logits,
        )
    )

    reference = KimiReferenceModel(config, tensors)
    prompt_results = []
    for prompt in manifest["prompts"]:
        prompt_results.append(
            reference.run_prompt(
                prompt["name"],
                prompt["input_ids"],
                manifest["slice"]["selected_hidden_layers"],
                compare_logits=compare_logits,
            )
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


def reconstruct_local_init_tensors(
    config: dict[str, Any],
    seed: int,
    required_tensors: list[str],
    selected_hidden_layers: list[int],
    compare_logits: bool,
) -> dict[str, np.ndarray]:
    required = set(required_tensors)
    rng = BurnNdArrayStdRng(seed)
    tensors: dict[str, np.ndarray] = {}

    for spec in _local_init_linear_specs(config, selected_hidden_layers, compare_logits):
        weight_name = f"{spec.prefix}.weight"
        if weight_name not in required:
            tensors[weight_name] = _kaiming_uniform_linear_tensor(
                rng=rng,
                shape=(spec.d_input, spec.d_output),
                fan_in=spec.d_input,
            )

        tensors[f"{spec.prefix}.bias"] = _kaiming_uniform_linear_tensor(
            rng=rng,
            shape=(spec.d_output,),
            fan_in=spec.d_input,
        )

    return tensors


def local_init_tensor_float32_digest(tensors: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(tensors[name].shape, dtype=np.uint64).tobytes())
        digest.update(np.asarray(tensors[name], dtype=np.float32).tobytes())
    return digest.hexdigest()


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
        compare_logits: bool,
    ) -> dict[str, Any]:
        token_array = np.asarray(input_ids, dtype=np.int64)[None, :]
        hidden = self.embed_tokens(token_array)
        hidden_states = []
        stop_after_layer = self.num_hidden_layers - 1 if compare_logits else max(selected_hidden_layers)
        for layer_idx in range(stop_after_layer + 1):
            hidden = self.forward_layer(layer_idx, hidden)
            if layer_idx in selected_hidden_layers:
                hidden_states.append({"layer_idx": layer_idx, "tensor": _tensor_json(hidden)})

        logits = None
        if compare_logits:
            logits = _tensor_json(self.linear(self.rms_norm(hidden, self.tensor("model.norm.weight")), "lm_head"))
        return {
            "prompt_name": prompt_name,
            "input_ids": input_ids,
            "logits": logits,
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

        q = self.linear(hidden, f"{prefix}.q_proj").reshape(batch, seq_len, self.num_attention_heads, self.qk_head_dim)
        q = _f32(np.swapaxes(self.apply_nope_policy(q), 1, 2))

        kv_latent = self.linear(hidden, f"{prefix}.kv_down")
        k = self.linear(kv_latent, f"{prefix}.k_up").reshape(batch, seq_len, self.num_key_value_heads, self.qk_head_dim)
        k = _f32(np.swapaxes(self.apply_nope_policy(k), 1, 2))

        v = self.linear(kv_latent, f"{prefix}.v_up").reshape(batch, seq_len, self.num_key_value_heads, self.v_head_dim)
        v = _f32(np.swapaxes(v, 1, 2))

        k = self.expand_kv_heads(k)
        v = self.expand_kv_heads(v)
        scores = np.matmul(q, np.swapaxes(k, 2, 3))
        scores = _f32(scores / np.float32(math.sqrt(self.qk_head_dim)))
        scores = _f32(scores + self.causal_mask(seq_len))
        weights = _softmax(scores, axis=3)
        attended = np.matmul(weights, v)
        attended = _f32(
            np.swapaxes(attended, 1, 2).reshape(batch, seq_len, self.num_attention_heads * self.v_head_dim)
        )
        return self.linear(attended, f"{prefix}.o_proj")

    def forward_dense_mlp(self, layer_idx: int, hidden: np.ndarray) -> np.ndarray:
        return self.forward_mlp_expert(f"model.layers.{layer_idx}.mlp", hidden)

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
        if ".experts." in prefix:
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
            raise GeneratorError(f"missing tensor '{name}' in the reconstructed/reference state")
        return self.tensors[name]

    def is_kda_layer(self, layer_idx: int) -> bool:
        return (layer_idx + 1) in self.linear_attn["kda_layers"]

    def is_sparse_moe_layer(self, layer_idx: int) -> bool:
        return layer_idx >= self.config["first_k_dense_replace"] and layer_idx % self.config["moe_layer_freq"] == 0


class BurnNdArrayStdRng:
    def __init__(self, seed: int) -> None:
        self._key_words = self._expand_seed(seed)
        self._block_counter = 0
        self._buffer: list[int] = []
        self._index = 64

    def next_u32(self) -> int:
        if self._index >= len(self._buffer):
            self._refill()
        value = self._buffer[self._index]
        self._index += 1
        return value

    def _refill(self) -> None:
        self._buffer = []
        for offset in range(4):
            self._buffer.extend(_chacha12_block(self._key_words, self._block_counter + offset))
        self._block_counter += 4
        self._index = 0

    @staticmethod
    def _expand_seed(seed: int) -> list[int]:
        state = seed & _MASK64
        seed_bytes = bytearray()

        for _ in range(8):
            state = (state * _PCG32_MUL + _PCG32_INC) & _MASK64
            xorshifted = (((state >> 18) ^ state) >> 27) & _MASK32
            rotation = (state >> 59) & 31
            word = ((xorshifted >> rotation) | ((xorshifted << ((-rotation) & 31)) & _MASK32)) & _MASK32
            seed_bytes.extend(word.to_bytes(4, "little"))

        return [int.from_bytes(seed_bytes[idx : idx + 4], "little") for idx in range(0, 32, 4)]


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


def _validate_manifest(manifest: dict[str, Any]) -> bool:
    _expect_exact_keys(manifest, ("kind", "version", "seed", "artifact", "slice", "prompts"), "baseline manifest")

    if manifest.get("kind") != BASELINE_SLICE_REQUEST_KIND:
        raise GeneratorError(
            f"expected baseline slice request manifest kind '{BASELINE_SLICE_REQUEST_KIND}', got '{manifest.get('kind')}'"
        )
    if manifest.get("version") != BASELINE_SLICE_REQUEST_VERSION:
        raise GeneratorError(
            f"expected baseline slice request manifest version {BASELINE_SLICE_REQUEST_VERSION}, got {manifest.get('version')}"
        )

    seed = manifest.get("seed")
    if not isinstance(seed, int) or seed < 0:
        raise GeneratorError(f"baseline manifest field 'seed' must be a non-negative integer, got {seed!r}")

    _validate_manifest_artifact(manifest.get("artifact"))
    compare_logits = _validate_manifest_slice(manifest.get("slice"), manifest["artifact"]["num_hidden_layers"])
    _validate_manifest_prompts(manifest.get("prompts"))
    return compare_logits


def _validate_local_init_contract(local_init_contract: dict[str, Any]) -> None:
    _expect_exact_keys(
        local_init_contract,
        EXPECTED_LOCAL_INIT_CONTRACT.keys(),
        "local init contract",
    )
    for field, expected in EXPECTED_LOCAL_INIT_CONTRACT.items():
        _expect_equal(field, local_init_contract.get(field), expected, "local init contract")


def _validate_artifact_config(config: dict[str, Any]) -> None:
    required_keys = (
        "model_type",
        "dtype",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "moe_intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "kv_lora_rank",
        "q_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "mla_use_nope",
        "hidden_act",
        "first_k_dense_replace",
        "moe_layer_freq",
        "num_experts",
        "num_experts_per_token",
        "num_shared_experts",
        "tie_word_embeddings",
        "use_cache",
        "rms_norm_eps",
        "linear_attn_config",
    )
    _expect_exact_keys(config, required_keys, "artifact config")

    if config.get("model_type") != "kimi_linear":
        raise GeneratorError("artifact config field 'model_type' must be 'kimi_linear'")
    if config.get("dtype") not in ("float32", "bfloat16"):
        raise GeneratorError("artifact config field 'dtype' must be 'float32' or 'bfloat16'")
    if config.get("hidden_act") != "silu":
        raise GeneratorError("artifact config field 'hidden_act' must be 'silu'")
    if config.get("q_lora_rank") is not None:
        raise GeneratorError("artifact config field 'q_lora_rank' must be null for the current external baseline generator")
    if bool(config.get("tie_word_embeddings")):
        raise GeneratorError("artifact config field 'tie_word_embeddings' must be false")
    if config.get("num_shared_experts") not in (0, 1):
        raise GeneratorError("artifact config field 'num_shared_experts' must be 0 or 1 for the current external baseline generator")

    for field in (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "moe_intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "first_k_dense_replace",
        "moe_layer_freq",
        "num_experts",
        "num_experts_per_token",
    ):
        _expect_positive_int(config.get(field), f"artifact config field '{field}'")

    linear_attn = config.get("linear_attn_config")
    if not isinstance(linear_attn, dict):
        raise GeneratorError("artifact config field 'linear_attn_config' must be an object")
    _expect_exact_keys(
        linear_attn,
        ("full_attn_layers", "kda_layers", "num_heads", "head_dim", "short_conv_kernel_size"),
        "artifact config field 'linear_attn_config'",
    )
    for field in ("num_heads", "head_dim", "short_conv_kernel_size"):
        _expect_positive_int(linear_attn.get(field), f"artifact config field 'linear_attn_config.{field}'")

    full_layers = _expect_int_list(linear_attn.get("full_attn_layers"), "artifact config field 'linear_attn_config.full_attn_layers'")
    kda_layers = _expect_int_list(linear_attn.get("kda_layers"), "artifact config field 'linear_attn_config.kda_layers'")
    num_hidden_layers = config["num_hidden_layers"]
    _expect_one_based_layers(full_layers, num_hidden_layers, "artifact config field 'linear_attn_config.full_attn_layers'")
    _expect_one_based_layers(kda_layers, num_hidden_layers, "artifact config field 'linear_attn_config.kda_layers'")

    covered = sorted(full_layers + kda_layers)
    expected = list(range(1, num_hidden_layers + 1))
    if covered != expected:
        raise GeneratorError(
            f"artifact config attention schedule must cover every layer exactly once, got {covered}, expected {expected}"
        )


def _validate_manifest_artifact_matches_config(artifact: dict[str, Any], config: dict[str, Any]) -> None:
    for field in ("model_type", "dtype", "num_hidden_layers", "hidden_size", "vocab_size"):
        _expect_equal(field, artifact.get(field), config.get(field), "baseline manifest artifact")


def _validate_manifest_artifact(artifact: Any) -> None:
    if not isinstance(artifact, dict):
        raise GeneratorError("baseline manifest field 'artifact' must be an object")
    _expect_exact_keys(
        artifact,
        ("model_type", "dtype", "num_hidden_layers", "hidden_size", "vocab_size"),
        "baseline manifest field 'artifact'",
    )
    if artifact.get("model_type") != "kimi_linear":
        raise GeneratorError("baseline manifest field 'artifact.model_type' must be 'kimi_linear'")
    if artifact.get("dtype") not in ("float32", "bfloat16"):
        raise GeneratorError("baseline manifest field 'artifact.dtype' must be 'float32' or 'bfloat16'")
    for field in ("num_hidden_layers", "hidden_size", "vocab_size"):
        _expect_positive_int(artifact.get(field), f"baseline manifest field 'artifact.{field}'")


def _validate_manifest_slice(slice_spec: Any, num_hidden_layers: int) -> bool:
    if not isinstance(slice_spec, dict):
        raise GeneratorError("baseline manifest field 'slice' must be an object")

    allowed_keys = {
        "import_selection",
        "selected_hidden_layers",
        "compare_logits",
        "requested_modules",
        "required_tensors",
        "tolerances",
    }
    actual_keys = set(slice_spec.keys())
    missing_keys = sorted({"import_selection", "selected_hidden_layers", "requested_modules", "required_tensors", "tolerances"} - actual_keys)
    extra_keys = sorted(actual_keys - allowed_keys)
    if missing_keys or extra_keys:
        detail = []
        if missing_keys:
            detail.append(f"missing keys {missing_keys}")
        if extra_keys:
            detail.append(f"unexpected keys {extra_keys}")
        raise GeneratorError(f"baseline manifest field 'slice' schema mismatch: {', '.join(detail)}")

    import_selection = slice_spec.get("import_selection")
    if not isinstance(import_selection, dict):
        raise GeneratorError("baseline manifest field 'slice.import_selection' must be an object")
    _expect_exact_keys(
        import_selection,
        ("layer_indices", "include_embeddings", "include_final_norm", "include_lm_head"),
        "baseline manifest field 'slice.import_selection'",
    )
    layer_indices = _expect_int_list(
        import_selection.get("layer_indices"),
        "baseline manifest field 'slice.import_selection.layer_indices'",
    )
    _expect_zero_based_layers(layer_indices, num_hidden_layers, "baseline manifest field 'slice.import_selection.layer_indices'")
    if not bool(import_selection.get("include_embeddings")):
        raise GeneratorError("baseline manifest field 'slice.import_selection.include_embeddings' must be true")

    selected_hidden_layers = _expect_int_list(
        slice_spec.get("selected_hidden_layers"),
        "baseline manifest field 'slice.selected_hidden_layers'",
    )
    _expect_zero_based_layers(selected_hidden_layers, num_hidden_layers, "baseline manifest field 'slice.selected_hidden_layers'")
    import_layer_set = set(layer_indices)
    for layer_idx in selected_hidden_layers:
        if layer_idx not in import_layer_set:
            raise GeneratorError(
                f"baseline manifest field 'slice.selected_hidden_layers' contains layer {layer_idx} outside import_selection.layer_indices"
            )

    compare_logits = slice_spec.get("compare_logits", True)
    if not isinstance(compare_logits, bool):
        raise GeneratorError("baseline manifest field 'slice.compare_logits' must be a boolean when present")
    if not compare_logits and not selected_hidden_layers:
        raise GeneratorError("baseline manifest field 'slice.selected_hidden_layers' must be non-empty when compare_logits = false")
    if compare_logits and not bool(import_selection.get("include_final_norm")):
        raise GeneratorError("baseline manifest field 'slice.import_selection.include_final_norm' must be true when compare_logits = true")
    if compare_logits and not bool(import_selection.get("include_lm_head")):
        raise GeneratorError("baseline manifest field 'slice.import_selection.include_lm_head' must be true when compare_logits = true")

    if not isinstance(slice_spec.get("requested_modules"), list) or not slice_spec["requested_modules"]:
        raise GeneratorError("baseline manifest field 'slice.requested_modules' must be a non-empty list")
    if not isinstance(slice_spec.get("required_tensors"), list) or not slice_spec["required_tensors"]:
        raise GeneratorError("baseline manifest field 'slice.required_tensors' must be a non-empty list")
    for tensor_name in slice_spec["required_tensors"]:
        if not isinstance(tensor_name, str) or not tensor_name:
            raise GeneratorError("baseline manifest field 'slice.required_tensors' must contain non-empty strings")

    tolerances = slice_spec.get("tolerances")
    if not isinstance(tolerances, dict):
        raise GeneratorError("baseline manifest field 'slice.tolerances' must be an object")
    _expect_exact_keys(
        tolerances,
        ("metric", "runtime_dtype", "logits_max_abs_diff", "hidden_state_max_abs_diff"),
        "baseline manifest field 'slice.tolerances'",
    )
    _expect_equal("metric", tolerances.get("metric"), "max_abs_diff", "baseline manifest field 'slice.tolerances'")
    _expect_equal("runtime_dtype", tolerances.get("runtime_dtype"), "float32", "baseline manifest field 'slice.tolerances'")
    for field in ("logits_max_abs_diff", "hidden_state_max_abs_diff"):
        value = tolerances.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise GeneratorError(
                f"baseline manifest field 'slice.tolerances.{field}' must be finite and >= 0, got {value!r}"
            )

    return compare_logits


def _validate_manifest_prompts(prompts: Any) -> None:
    if not isinstance(prompts, list) or not prompts:
        raise GeneratorError("baseline manifest field 'prompts' must be a non-empty list")

    seen_names: set[str] = set()
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, dict):
            raise GeneratorError(f"baseline manifest field 'prompts[{index}]' must be an object")
        _expect_exact_keys(prompt, ("name", "input_ids"), f"baseline manifest field 'prompts[{index}]'")
        name = prompt.get("name")
        if not isinstance(name, str) or not name:
            raise GeneratorError(f"baseline manifest field 'prompts[{index}].name' must be a non-empty string")
        if name in seen_names:
            raise GeneratorError(f"baseline manifest field 'prompts' contains duplicate prompt name '{name}'")
        seen_names.add(name)
        token_ids = _expect_int_list(
            prompt.get("input_ids"),
            f"baseline manifest field 'prompts[{index}].input_ids'",
            allow_duplicates=True,
        )
        if any(token_id < 0 for token_id in token_ids):
            raise GeneratorError(f"baseline manifest field 'prompts[{index}].input_ids' must contain non-negative integers")


def _expect_positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise GeneratorError(f"{label} must be a positive integer, got {value!r}")
    return value


def _expect_int_list(value: Any, label: str, allow_duplicates: bool = False) -> list[int]:
    if not isinstance(value, list) or any(not isinstance(item, int) for item in value):
        raise GeneratorError(f"{label} must be a list of integers")
    if not allow_duplicates and len(value) != len(set(value)):
        raise GeneratorError(f"{label} must not contain duplicate indices")
    return value


def _expect_zero_based_layers(layer_indices: list[int], num_hidden_layers: int, label: str) -> None:
    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= num_hidden_layers:
            raise GeneratorError(
                f"{label} contains out-of-range zero-based layer index {layer_idx} for num_hidden_layers={num_hidden_layers}"
            )


def _expect_one_based_layers(layer_indices: list[int], num_hidden_layers: int, label: str) -> None:
    for layer_idx in layer_indices:
        if layer_idx <= 0 or layer_idx > num_hidden_layers:
            raise GeneratorError(
                f"{label} contains out-of-range one-based layer index {layer_idx} for num_hidden_layers={num_hidden_layers}"
            )


def _validate_required_tensors(index: dict[str, Any], required_tensors: list[str]) -> None:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise GeneratorError("artifact shard index is missing 'weight_map'")
    missing = [tensor_name for tensor_name in required_tensors if tensor_name not in weight_map]
    if missing:
        raise GeneratorError(f"artifact shard index is missing required tensors: {missing}")


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


def _local_init_linear_specs(
    config: dict[str, Any],
    selected_hidden_layers: list[int],
    compare_logits: bool,
) -> list[LinearModuleSpec]:
    hidden = config["hidden_size"]
    linear_attn = config["linear_attn_config"]
    qk_head_dim = config["qk_nope_head_dim"] + config["qk_rope_head_dim"]
    specs: list[LinearModuleSpec] = []
    max_layer_exclusive = config["num_hidden_layers"] if compare_logits else (max(selected_hidden_layers) + 1)

    for layer_idx in range(max_layer_exclusive):
        prefix = f"model.layers.{layer_idx}"
        if (layer_idx + 1) in linear_attn["kda_layers"]:
            kda_qk_dim = linear_attn["num_heads"] * linear_attn["head_dim"]
            kda_v_dim = linear_attn["num_heads"] * config["v_head_dim"]
            specs.extend(
                [
                    LinearModuleSpec(f"{prefix}.self_attn.q_proj", hidden, kda_qk_dim),
                    LinearModuleSpec(f"{prefix}.self_attn.k_proj", hidden, kda_qk_dim),
                    LinearModuleSpec(f"{prefix}.self_attn.v_proj", hidden, kda_v_dim),
                    LinearModuleSpec(f"{prefix}.self_attn.o_proj", kda_v_dim, hidden),
                ]
            )
        else:
            if config["q_lora_rank"] is None:
                specs.append(
                    LinearModuleSpec(
                        f"{prefix}.self_attn.q_proj",
                        hidden,
                        config["num_attention_heads"] * qk_head_dim,
                    )
                )
            else:
                specs.extend(
                    [
                        LinearModuleSpec(f"{prefix}.self_attn.q_proj.down", hidden, config["q_lora_rank"]),
                        LinearModuleSpec(
                            f"{prefix}.self_attn.q_proj.up",
                            config["q_lora_rank"],
                            config["num_attention_heads"] * qk_head_dim,
                        ),
                    ]
                )
            specs.extend(
                [
                    LinearModuleSpec(f"{prefix}.self_attn.kv_down", hidden, config["kv_lora_rank"]),
                    LinearModuleSpec(
                        f"{prefix}.self_attn.k_up",
                        config["kv_lora_rank"],
                        config["num_key_value_heads"] * qk_head_dim,
                    ),
                    LinearModuleSpec(
                        f"{prefix}.self_attn.v_up",
                        config["kv_lora_rank"],
                        config["num_key_value_heads"] * config["v_head_dim"],
                    ),
                    LinearModuleSpec(
                        f"{prefix}.self_attn.o_proj",
                        config["num_attention_heads"] * config["v_head_dim"],
                        hidden,
                    ),
                ]
            )

        if layer_idx >= config["first_k_dense_replace"] and layer_idx % config["moe_layer_freq"] == 0:
            specs.append(LinearModuleSpec(f"{prefix}.block_sparse_moe.gate", hidden, config["num_experts"]))
            for shared_idx in range(config["num_shared_experts"]):
                if shared_idx != 0:
                    raise GeneratorError("local-init reconstruction supports exactly one shared expert for this pilot")
                shared_prefix = f"{prefix}.block_sparse_moe.shared_experts"
                specs.extend(
                    [
                        LinearModuleSpec(f"{shared_prefix}.gate_proj", hidden, config["moe_intermediate_size"]),
                        LinearModuleSpec(f"{shared_prefix}.up_proj", hidden, config["moe_intermediate_size"]),
                        LinearModuleSpec(f"{shared_prefix}.down_proj", config["moe_intermediate_size"], hidden),
                    ]
                )
            for expert_idx in range(config["num_experts"]):
                expert_prefix = f"{prefix}.block_sparse_moe.experts.{expert_idx}"
                specs.extend(
                    [
                        LinearModuleSpec(f"{expert_prefix}.w1", hidden, config["moe_intermediate_size"]),
                        LinearModuleSpec(f"{expert_prefix}.w3", hidden, config["moe_intermediate_size"]),
                        LinearModuleSpec(f"{expert_prefix}.w2", config["moe_intermediate_size"], hidden),
                    ]
                )
        else:
            specs.extend(
                [
                    LinearModuleSpec(f"{prefix}.mlp.gate_proj", hidden, config["intermediate_size"]),
                    LinearModuleSpec(f"{prefix}.mlp.up_proj", hidden, config["intermediate_size"]),
                    LinearModuleSpec(f"{prefix}.mlp.down_proj", config["intermediate_size"], hidden),
                ]
            )

    if compare_logits:
        specs.append(LinearModuleSpec("lm_head", hidden, config["vocab_size"]))
    return specs


def _kaiming_uniform_linear_tensor(
    rng: BurnNdArrayStdRng,
    shape: tuple[int, ...],
    fan_in: int,
) -> np.ndarray:
    bound = 1.0 / math.sqrt(fan_in)
    low, scale = _bounded_uniform_scale(np.float32(-bound), np.float32(bound))
    values = [_sample_uniform_float32(rng, low, scale) for _ in range(math.prod(shape))]
    return np.asarray(values, dtype=np.float32).reshape(shape)


def _bounded_uniform_scale(low: np.float32, high: np.float32) -> tuple[np.float32, np.float32]:
    scale = np.float32(high - low)
    max_rand = np.float32(1.0 - np.finfo(np.float32).eps)
    while np.float32(scale * max_rand + low) > high:
        scale = np.nextafter(scale, np.float32(-np.inf), dtype=np.float32)
    return low, scale


def _sample_uniform_float32(
    rng: BurnNdArrayStdRng,
    low: np.float32,
    scale: np.float32,
) -> np.float32:
    raw = rng.next_u32() >> (32 - 23)
    value_1_2 = struct.unpack("<f", struct.pack("<I", 0x3F800000 | raw))[0]
    value_0_1 = np.float32(value_1_2 - 1.0)
    return np.float32(value_0_1 * scale + low)


def _chacha12_block(key_words: list[int], block_counter: int, stream: int = 0) -> list[int]:
    state = [
        0x61707865,
        0x3320646E,
        0x79622D32,
        0x6B206574,
        *key_words,
        block_counter & _MASK32,
        (block_counter >> 32) & _MASK32,
        stream & _MASK32,
        (stream >> 32) & _MASK32,
    ]
    working = state.copy()

    for _ in range(6):
        _quarter_round(working, 0, 4, 8, 12)
        _quarter_round(working, 1, 5, 9, 13)
        _quarter_round(working, 2, 6, 10, 14)
        _quarter_round(working, 3, 7, 11, 15)
        _quarter_round(working, 0, 5, 10, 15)
        _quarter_round(working, 1, 6, 11, 12)
        _quarter_round(working, 2, 7, 8, 13)
        _quarter_round(working, 3, 4, 9, 14)

    return [(working[idx] + state[idx]) & _MASK32 for idx in range(16)]


def _quarter_round(state: list[int], a: int, b: int, c: int, d: int) -> None:
    state[a] = (state[a] + state[b]) & _MASK32
    state[d] = _rotate_left32(state[d] ^ state[a], 16)
    state[c] = (state[c] + state[d]) & _MASK32
    state[b] = _rotate_left32(state[b] ^ state[c], 12)
    state[a] = (state[a] + state[b]) & _MASK32
    state[d] = _rotate_left32(state[d] ^ state[a], 8)
    state[c] = (state[c] + state[d]) & _MASK32
    state[b] = _rotate_left32(state[b] ^ state[c], 7)


def _rotate_left32(value: int, shift: int) -> int:
    return ((value << shift) & _MASK32) | (value >> (32 - shift))


def _expect_equal(field: str, actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise GeneratorError(
            f"{label} field '{field}' expected {json.dumps(expected, sort_keys=True)}, got {json.dumps(actual, sort_keys=True)}"
        )


def _expect_exact_keys(payload: dict[str, Any], expected_keys, label: str) -> None:
    expected = set(expected_keys)
    actual = set(payload.keys())
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing keys {missing}")
        if extra:
            detail.append(f"unexpected keys {extra}")
        raise GeneratorError(f"{label} schema mismatch: {', '.join(detail)}")


def _tensor_json(tensor: np.ndarray) -> dict[str, Any]:
    array = np.asarray(tensor, dtype=np.float32)
    return {"dims": list(array.shape), "values": array.reshape(-1).tolist()}


def _f32(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _softplus(x: np.ndarray) -> np.ndarray:
    array = _f32(x)
    return _f32(np.log1p(np.exp(-np.abs(array))) + np.maximum(array, np.float32(0.0)))


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    shifted = _f32(x - np.max(x, axis=axis, keepdims=True))
    exp = np.exp(shifted, dtype=np.float32)
    return _f32(exp / np.sum(exp, axis=axis, keepdims=True, dtype=np.float32))


def _silu(x: np.ndarray) -> np.ndarray:
    array = _f32(x)
    return _f32(array / (np.float32(1.0) + np.exp(-array, dtype=np.float32)))
