from __future__ import annotations

import importlib.util
import math
import sys
import types
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import nn


class RemoteCodeError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    try:
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RemoteCodeError(f"missing json file '{path}'") from exc
    except Exception as exc:  # pragma: no cover - message quality only
        raise RemoteCodeError(f"failed to parse json file '{path}': {exc}") from exc


def ensure_hf_files(
    artifact_dir: Path,
    repo_id: str,
    filenames: list[str],
    revision: str | None = None,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        path = artifact_dir / filename
        if path.exists():
            continue
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(artifact_dir),
            revision=revision,
        )


def install_fla_backend() -> str:
    try:
        from fla.modules import FusedRMSNormGated as _  # noqa: F401
        from fla.modules import ShortConvolution as _  # noqa: F401
        from fla.ops.kda import chunk_kda as _  # noqa: F401
        from fla.ops.kda import fused_recurrent_kda as _  # noqa: F401
        from fla.ops.kda.gate import fused_kda_gate as _  # noqa: F401
        from fla.ops.utils.index import prepare_cu_seqlens_from_mask as _  # noqa: F401
        from fla.ops.utils.index import prepare_lens_from_mask as _  # noqa: F401
        from fla.utils import tensor_cache as _  # noqa: F401
        return "official"
    except Exception:
        return install_fla_fallback()


def install_fla_fallback() -> str:
    for module_name in list(sys.modules):
        if module_name == "fla" or module_name.startswith("fla."):
            del sys.modules[module_name]

    fla_module = types.ModuleType("fla")
    modules_module = types.ModuleType("fla.modules")
    ops_module = types.ModuleType("fla.ops")
    kda_module = types.ModuleType("fla.ops.kda")
    gate_module = types.ModuleType("fla.ops.kda.gate")
    utils_module = types.ModuleType("fla.ops.utils")
    index_module = types.ModuleType("fla.ops.utils.index")
    fla_utils_module = types.ModuleType("fla.utils")

    class ShortConvolution(nn.Module):
        def __init__(self, hidden_size: int, kernel_size: int, activation: str = "silu"):
            super().__init__()
            if activation != "silu":
                raise RemoteCodeError(
                    f"fallback ShortConvolution only supports activation='silu', got '{activation}'"
                )
            self.hidden_size = hidden_size
            self.kernel_size = kernel_size
            self.weight = nn.Parameter(torch.zeros(hidden_size, 1, kernel_size, dtype=torch.float32))

        def forward(
            self,
            x: torch.Tensor,
            cache: torch.Tensor | None = None,
            output_final_state: bool = False,
            cu_seqlens: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            if cu_seqlens is not None:
                raise RemoteCodeError("fallback ShortConvolution does not support cu_seqlens")
            if x.ndim != 3:
                raise RemoteCodeError(f"fallback ShortConvolution expected [batch, seq, hidden], got {tuple(x.shape)}")

            batch, seq_len, hidden = x.shape
            if hidden != self.hidden_size:
                raise RemoteCodeError(
                    f"fallback ShortConvolution expected hidden_size={self.hidden_size}, got {hidden}"
                )

            history_len = self.kernel_size - 1
            history = None
            if cache is not None:
                if cache.ndim != 3:
                    raise RemoteCodeError(f"fallback ShortConvolution cache expected [batch, hidden, history], got {tuple(cache.shape)}")
                history = cache.transpose(1, 2).to(dtype=torch.float32)
            combined = torch.cat([history, x.to(dtype=torch.float32)], dim=1) if history is not None else x.to(dtype=torch.float32)

            outputs = []
            for token_index in range(seq_len):
                end = (history.shape[1] if history is not None else 0) + token_index + 1
                start = max(0, end - self.kernel_size)
                window = combined[:, start:end, :]
                window_len = end - start
                weight_slice = self.weight[:, 0, self.kernel_size - window_len : self.kernel_size].transpose(0, 1)
                outputs.append(F.silu((window * weight_slice.unsqueeze(0)).sum(dim=1)))

            output = torch.stack(outputs, dim=1)
            if not output_final_state:
                return output, None

            if history_len == 0:
                next_cache = x.new_zeros((batch, hidden, 0), dtype=torch.float32)
            else:
                next_cache = combined[:, -history_len:, :].transpose(1, 2).contiguous()
            return output, next_cache

    class FusedRMSNormGated(nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1e-6, activation: str = "sigmoid"):
            super().__init__()
            if activation != "sigmoid":
                raise RemoteCodeError(
                    f"fallback FusedRMSNormGated only supports activation='sigmoid', got '{activation}'"
                )
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))

        def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
            hidden_states = hidden_states.to(dtype=torch.float32)
            variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
            normalized = hidden_states * torch.rsqrt(variance + self.eps)
            return normalized * self.weight * torch.sigmoid(gate.to(dtype=torch.float32))

    def fused_kda_gate(
        g: torch.Tensor,
        a_log: torch.Tensor,
        head_dim: int,
        g_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if g.ndim == 3:
            batch, seq_len, projection = g.shape
            num_heads = projection // head_dim
            g = g.view(batch, seq_len, num_heads, head_dim)
        if g_bias is not None:
            g_bias = g_bias.view(g.shape[2], head_dim).to(dtype=torch.float32)
        a = torch.exp(a_log.to(dtype=torch.float32)).view(1, 1, -1, 1)
        biased = g.to(dtype=torch.float32)
        if g_bias is not None:
            biased = biased + g_bias.view(1, 1, g.shape[2], head_dim)
        return -(a * F.softplus(biased))

    def _run_kda(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool,
        cu_seqlens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if cu_seqlens is not None:
            raise RemoteCodeError("fallback KDA kernels do not support cu_seqlens")

        q = q.to(dtype=torch.float32)
        k = k.to(dtype=torch.float32)
        v = v.to(dtype=torch.float32)
        g = g.to(dtype=torch.float32)
        beta = beta.to(dtype=torch.float32)
        batch, seq_len, num_heads, head_dim = q.shape

        if use_qk_l2norm_in_kernel:
            q = F.normalize(q, dim=-1, eps=1e-6) / math.sqrt(head_dim)
            k = F.normalize(k, dim=-1, eps=1e-6)

        state = initial_state.to(dtype=torch.float32) if initial_state is not None else q.new_zeros(
            (batch, num_heads, head_dim, head_dim)
        )
        outputs = []
        for token_index in range(seq_len):
            q_i = q[:, token_index]
            k_i = k[:, token_index]
            v_i = v[:, token_index]
            g_i = g[:, token_index]
            beta_i = beta[:, token_index]

            state = state * torch.exp(g_i).unsqueeze(-1)
            projected_value = (state * k_i.unsqueeze(-1)).sum(dim=2)
            delta_value = (v_i - projected_value) * beta_i.unsqueeze(-1)
            state = state + k_i.unsqueeze(-1) * delta_value.unsqueeze(-2)
            outputs.append((state * q_i.unsqueeze(-1)).sum(dim=2))

        output = torch.stack(outputs, dim=1)
        return output, state if output_final_state else None

    def chunk_kda(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = True,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return _run_kda(q, k, v, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel, cu_seqlens)

    def fused_recurrent_kda(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = True,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return _run_kda(q, k, v, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel, cu_seqlens)

    def prepare_lens_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        return attention_mask.sum(dim=-1, dtype=torch.int32)

    def prepare_cu_seqlens_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        lens = prepare_lens_from_mask(attention_mask)
        return torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=attention_mask.device),
                lens.cumsum(dim=0, dtype=torch.int32),
            ],
            dim=0,
        )

    def tensor_cache(fn):
        return fn

    modules_module.ShortConvolution = ShortConvolution
    modules_module.FusedRMSNormGated = FusedRMSNormGated
    kda_module.chunk_kda = chunk_kda
    kda_module.fused_recurrent_kda = fused_recurrent_kda
    gate_module.fused_kda_gate = fused_kda_gate
    index_module.prepare_cu_seqlens_from_mask = prepare_cu_seqlens_from_mask
    index_module.prepare_lens_from_mask = prepare_lens_from_mask
    fla_utils_module.tensor_cache = tensor_cache

    fla_module.modules = modules_module
    fla_module.ops = ops_module
    fla_module.utils = fla_utils_module
    ops_module.kda = kda_module
    ops_module.utils = utils_module
    utils_module.index = index_module
    kda_module.gate = gate_module

    sys.modules["fla"] = fla_module
    sys.modules["fla.modules"] = modules_module
    sys.modules["fla.ops"] = ops_module
    sys.modules["fla.ops.kda"] = kda_module
    sys.modules["fla.ops.kda.gate"] = gate_module
    sys.modules["fla.ops.utils"] = utils_module
    sys.modules["fla.ops.utils.index"] = index_module
    sys.modules["fla.utils"] = fla_utils_module
    return "cpu_fallback"


def load_remote_kimi_modules(remote_code_dir: Path) -> tuple[type, Any, str]:
    if not remote_code_dir.exists():
        raise RemoteCodeError(f"missing remote code directory '{remote_code_dir}'")

    backend = install_fla_backend()
    import importlib
    import transformers.utils as transformers_utils
    auto_docstring_module = importlib.import_module("transformers.utils.auto_docstring")

    original_auto_docstring_module = auto_docstring_module.auto_docstring
    original_transformers_auto_docstring = transformers_utils.auto_docstring
    no_op_auto_docstring = lambda *args, **kwargs: (lambda obj: obj)
    auto_docstring_module.auto_docstring = no_op_auto_docstring  # type: ignore[assignment]
    transformers_utils.auto_docstring = no_op_auto_docstring  # type: ignore[assignment]
    package_name = f"attnres_kimi_remote_{uuid.uuid4().hex}"
    package = types.ModuleType(package_name)
    package.__path__ = [str(remote_code_dir)]
    sys.modules[package_name] = package

    try:
        modules = {}
        for module_name in ("configuration_kimi", "modeling_kimi"):
            module_path = remote_code_dir / f"{module_name}.py"
            if not module_path.exists():
                raise RemoteCodeError(f"missing remote code file '{module_path}'")
            spec = importlib.util.spec_from_file_location(f"{package_name}.{module_name}", module_path)
            if spec is None or spec.loader is None:
                raise RemoteCodeError(f"failed to create import spec for '{module_path}'")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            modules[module_name] = module
    finally:
        auto_docstring_module.auto_docstring = original_auto_docstring_module  # type: ignore[assignment]
        transformers_utils.auto_docstring = original_transformers_auto_docstring  # type: ignore[assignment]

    return modules["configuration_kimi"].KimiLinearConfig, modules["modeling_kimi"], backend
