"""BWD-only FlyDSL override for ATen's fused RMSNorm backward operator."""

# mypy: allow-untyped-defs

from __future__ import annotations

import functools

import torch

from ... import flydsl_utils as fu


# This development manifest deliberately matches benchmark_bwd.py. It is a
# route-coverage gate, not the final performance crossover policy.
_SUPPORTED_M = frozenset(
    {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}
)
_SUPPORTED_N = frozenset({4096, 8192, 12288, 16384, 32768})
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@functools.cache
def _rmsnorm_bwd_entrypoint():
    """Import the FlyDSL runtime once, under the input device guard."""

    from .flydsl_rmsnorm_bwd_kernel import rmsnorm_bwd

    return rmsnorm_bwd


def _normalized_shape_1d(normalized_shape) -> int | None:
    try:
        shape = tuple(int(value) for value in normalized_shape)
    except TypeError:
        try:
            shape = (int(normalized_shape),)
        except (TypeError, ValueError):
            return None
    except ValueError:
        return None
    return shape[0] if len(shape) == 1 else None


def _fused_rms_norm_backward_cond(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    normalized_shape,
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    output_mask,
) -> bool:
    n = _normalized_shape_1d(normalized_shape)
    if n is None or n not in _SUPPORTED_N:
        return False
    if torch.version.hip is None or input.device.type != "cuda":
        return False
    if input.dtype not in _SUPPORTED_DTYPES:
        return False
    if input.ndim < 1 or input.shape[-1] != n or input.numel() == 0:
        return False
    if not input.is_contiguous():
        return False

    rows_m = input.numel() // n
    if rows_m not in _SUPPORTED_M or weight is None:
        return False
    if (
        weight.shape != (n,)
        or weight.dtype != input.dtype
        or weight.device != input.device
        or not weight.is_contiguous()
    ):
        return False
    if (
        grad_out.shape != input.shape
        or grad_out.dtype != input.dtype
        or grad_out.device != input.device
        or not grad_out.is_contiguous()
    ):
        return False
    if (
        rstd.device != input.device
        or rstd.dtype != torch.float32
        or rstd.shape != (*input.shape[:-1], 1)
        or not rstd.is_contiguous()
    ):
        return False
    if len(output_mask) != 2 or not any(bool(value) for value in output_mask):
        return False

    is_cow = torch._C._is_cow_tensor  # pyrefly: ignore[missing-attribute]
    return not any(is_cow(value) for value in (grad_out, input, rstd, weight))


def _fused_rms_norm_backward_impl(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    normalized_shape,
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    output_mask,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if weight is None:
        raise RuntimeError("FlyDSL RMSNorm backward requires an explicit weight")

    need_grad_input = bool(output_mask[0])
    need_grad_weight = bool(output_mask[1])
    with torch.cuda.device(input.device):
        grad_input, grad_weight = _rmsnorm_bwd_entrypoint()(
            grad_out,
            input,
            rstd,
            weight,
            need_grad_weight=need_grad_weight,
        )
    return grad_input if need_grad_input else None, grad_weight


def register_flydsl_rmsnorm_bwd_override() -> None:
    """Register BWD with transparent ATen fallback for unsupported calls."""

    fu.register_op_override(
        "aten",
        "_fused_rms_norm_backward",
        "CUDA",
        cond=_fused_rms_norm_backward_cond,
        impl=_fused_rms_norm_backward_impl,
        allow_multiple_override=True,
    )

