# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Self-contained FlyDSL RMSNorm backward kernel and PyTorch wrapper.

The device code is derived from ROCm/FlyDSL
kernels/norm/rmsnorm_bwd_kernel.py at commit
a85595136c647b2ac4532be43ad6e37beaedc085. Only the plain RMSNorm
backward path required by ATen is included.
"""

# mypy: allow-untyped-defs

import functools
import math

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import arith as _expr_arith
from flydsl.expr.typing import T
from flydsl.expr.vector import full
from flydsl.runtime.device import get_rocm_arch, is_rdna_arch

from torch._native.flydsl_cache import jit_cache


__all__ = [
    "clear_rmsnorm_bwd_caches",
    "rmsnorm_bwd",
    "rmsnorm_bwd_cache_info",
]

_BLOCK_THREADS = 256
_SUPPORTED_DTYPES: dict[torch.dtype, str] = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
}


def _get_llvm_ptr(ptr, offset, dtype_bytes, ptr_type=None):
    """Return a global-memory LLVM pointer at ptr + offset * dtype_bytes."""

    if ptr_type is None:
        ptr_type = ir.Type.parse("!llvm.ptr<1>")
    base_ptr = _fly.extract_aligned_pointer_as_index(ptr_type, ptr)
    base_ptr = _llvm.PtrToIntOp(T.i64, base_ptr).result
    byte_offset = _expr_arith.index_cast(
        T.i64, fx.Index(offset) * fx.Index(dtype_bytes)
    )
    llvm_ptr = _llvm.AddOp(
        base_ptr, byte_offset, _llvm.IntegerOverflowFlags(0)
    ).result
    llvm_ptr = _llvm.IntToPtrOp(ptr_type, llvm_ptr).result
    return llvm_ptr._value if const_expr(hasattr(llvm_ptr, "_value")) else llvm_ptr


def _atomic_add(
    dst,
    offset,
    value,
    *,
    dtype_bytes=4,
    syncscope="agent",
    ordering=None,
    alignment=None,
    ptr_type=None,
):
    """Atomically add value to dst[offset] in global memory."""

    ptr = _get_llvm_ptr(dst, offset, dtype_bytes, ptr_type=ptr_type)
    val = value.ir_value() if const_expr(hasattr(value, "ir_value")) else value
    elem_ty = val.type.element_type if isinstance(val.type, ir.VectorType) else val.type
    bin_op = (
        _llvm.AtomicBinOp.fadd
        if isinstance(elem_ty, ir.FloatType)
        else _llvm.AtomicBinOp.add
    )
    if ordering is None:
        ordering = _llvm.AtomicOrdering.monotonic
    if alignment is None:
        alignment = dtype_bytes
    return _llvm.AtomicRMWOp(
        bin_op,
        ptr,
        val,
        ordering,
        syncscope=syncscope,
        alignment=alignment,
    ).result


def _dtype_to_elem_type(dtype_str: str):
    if dtype_str == "f32":
        return fx.Float32
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    raise ValueError(
        f"unsupported dtype: {dtype_str!r} "
        "(expected 'f32', 'f16', or 'bf16')"
    )


def _dtype_str(dtype: torch.dtype) -> str:
    try:
        return _SUPPORTED_DTYPES[dtype]
    except KeyError as exc:
        raise TypeError(f"unsupported RMSNorm dtype for FlyDSL: {dtype}") from exc


def _warp_size_for_arch(arch: str) -> int:
    return 32 if is_rdna_arch(arch) else 64


def _make_single_reduction_storage(red_slots: int):
    @fx.struct
    class SharedStorage:
        s_red: fx.Array[fx.Float32, red_slots, 16]

    return SharedStorage


def _load_scalar(copy_atom, elem_dtype, divided_tensor, index):
    view = fx.slice(divided_tensor, (None, index))
    register = fx.make_rmem_tensor(1, elem_dtype)
    fx.copy_atom_call(copy_atom, view, register)
    return fx.memref_load_vec(register)[0]


def _store_scalar(
    copy_atom,
    elem_dtype,
    store_dtype,
    divided_tensor,
    index,
    value,
):
    register = fx.make_rmem_tensor(1, elem_dtype)
    typed_value = full(1, store_dtype(value), store_dtype)
    fx.memref_store_vec(typed_value, register)
    view = fx.slice(divided_tensor, (None, index))
    fx.copy_atom_call(copy_atom, register, view)


def _build_rmsnorm_bwd_module(n: int, dtype_str: str, arch: str):
    """Build one BWD specialization for N, dtype, and ROCm architecture."""

    warp_size = _warp_size_for_arch(arch)
    red_slots = max(1, (_BLOCK_THREADS + warp_size - 1) // warp_size)
    elem_bits = 32 if dtype_str == "f32" else 16
    SharedStorage = _make_single_reduction_storage(red_slots)

    @flyc.kernel
    def rmsnorm_bwd_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        DY: fx.Tensor,
        Rstd: fx.Tensor,
        DX: fx.Tensor,
        DWeight: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = _dtype_to_elem_type(dtype_str)
        fm_fast = arith.FastMathFlags.fast
        n_float = float(n)
        c_zero_f = fx.Float32(0.0)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = lds.s_red.view(fx.make_layout(red_slots, 1))

        def wave_reduce_add(value):
            reduced = value
            for shift_exp in range_constexpr(int(math.log2(warp_size))):
                offset = warp_size // (2 << shift_exp)
                peer = reduced.shuffle_xor(offset, warp_size)
                reduced = reduced.addf(peer, fastmath=fm_fast)
            return reduced

        def block_reduce_add(value):
            if const_expr(red_slots == 1):
                return wave_reduce_add(value)
            lane = tid % warp_size
            wave = tid // warp_size
            reduced = wave_reduce_add(value)
            if lane == 0:
                fx.memref_store(reduced, s_red, wave)
            gpu.barrier()
            if wave == 0:
                in_range = lane < red_slots
                lane_safe = in_range.select(lane, 0)
                partial = fx.memref_load(s_red, lane_safe)
                partial = in_range.select(partial, c_zero_f)
                partial = wave_reduce_add(partial)
                if lane == 0:
                    fx.memref_store(partial, s_red, 0)
            gpu.barrier()
            return fx.memref_load(s_red, 0)

        input_buf = fx.rocdl.make_buffer_tensor(Input)
        gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
        dy_buf = fx.rocdl.make_buffer_tensor(DY)
        rstd_buf = fx.rocdl.make_buffer_tensor(Rstd)
        dx_buf = fx.rocdl.make_buffer_tensor(DX)

        row_in = fx.slice(input_buf, (bid, None))
        row_dy = fx.slice(dy_buf, (bid, None))
        row_dx = fx.slice(dx_buf, (bid, None))

        copy_atom_s = fx.make_copy_atom(
            fx.rocdl.BufferCopy16b()
            if elem_bits <= 16
            else fx.rocdl.BufferCopy32b(),
            elem_bits,
        )
        copy_atom_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
        dy_div = fx.logical_divide(row_dy, fx.make_layout(1, 1))
        gamma_div = fx.logical_divide(gamma_buf, fx.make_layout(1, 1))
        dx_div = fx.logical_divide(row_dx, fx.make_layout(1, 1))
        rstd_div = fx.logical_divide(rstd_buf, fx.make_layout(1, 1))

        rstd = _load_scalar(copy_atom_f32, fx.Float32, rstd_div, bid)

        # Pass 1: c1 = mean((x * rstd) * (dy * gamma)).
        thread_acc = c_zero_f
        for base in range_constexpr(0, n, _BLOCK_THREADS):
            idx = tid + base
            is_valid = idx < n
            idx_safe = is_valid.select(idx, 0)
            x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx_safe)
            dy_e = _load_scalar(copy_atom_s, elem_dtype, dy_div, idx_safe)
            gamma_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx_safe)
            x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
            dy = dy_e if dtype_str == "f32" else dy_e.to(fx.Float32)
            gamma = gamma_e if dtype_str == "f32" else gamma_e.to(fx.Float32)
            x_hat = x * rstd
            weighted_dy = dy * gamma
            product = x_hat * weighted_dy
            thread_acc = thread_acc + is_valid.select(product, c_zero_f)

        c1 = block_reduce_add(thread_acc) / n_float

        # Pass 2: dx = (dy*gamma - x_hat*c1)*rstd; dw += dy*x_hat.
        for base in range_constexpr(0, n, _BLOCK_THREADS):
            idx = tid + base
            if idx < n:
                x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx)
                dy_e = _load_scalar(copy_atom_s, elem_dtype, dy_div, idx)
                gamma_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                dy = dy_e if dtype_str == "f32" else dy_e.to(fx.Float32)
                gamma = gamma_e if dtype_str == "f32" else gamma_e.to(fx.Float32)
                x_hat = x * rstd
                weighted_dy = dy * gamma
                dx = (weighted_dy - x_hat * c1) * rstd
                dx_e = dx if dtype_str == "f32" else dx.to(elem_dtype)
                _store_scalar(
                    copy_atom_s,
                    elem_dtype,
                    elem_dtype,
                    dx_div,
                    idx,
                    dx_e,
                )
                _atomic_add(DWeight, idx, dy * x_hat, dtype_bytes=4)

    @flyc.jit
    def launch_rmsnorm_bwd(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        DY: fx.Tensor,
        Rstd: fx.Tensor,
        DX: fx.Tensor,
        DWeight: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = rmsnorm_bwd_kernel(Input, Gamma, DY, Rstd, DX, DWeight)
        launcher.launch(
            grid=(m_in, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm_bwd


def _make_compile_arg(tensor: torch.Tensor):
    """Make only the row dimension dynamic across M values."""

    return flyc.from_torch_tensor(tensor).mark_shape_dynamic(0)


@functools.cache
def _compile_environment(device_index: int) -> tuple[str, str]:
    """Return target data while the caller holds the input device guard."""

    del device_index
    return str(get_rocm_arch()), flyc.compile_backend_name()


@jit_cache
def _compile_rmsnorm_bwd(
    n: int,
    dtype: str,
    arch: str,
    backend: str,
    device_index: int,
    *,
    compile_args,
) -> flyc.CompiledFunction:
    del backend, device_index
    input_2d, weight, grad_2d, rstd, grad_input, grad_weight, rows_m, stream = (
        compile_args
    )
    launch = _build_rmsnorm_bwd_module(n, dtype, arch)
    return flyc.compile(
        launch,
        _make_compile_arg(input_2d),
        flyc.from_torch_tensor(weight),
        _make_compile_arg(grad_2d),
        _make_compile_arg(rstd),
        _make_compile_arg(grad_input),
        flyc.from_torch_tensor(grad_weight),
        rows_m,
        stream,
    )


def rmsnorm_bwd(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor,
    *,
    need_grad_weight: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run FlyDSL BWD under the device guard held by the ATen override."""

    n = input.shape[-1]
    rows_m = input.numel() // n
    input_2d = input.view(rows_m, n)
    grad_2d = grad_out.view(rows_m, n)
    rstd_flat = rstd.view(rows_m)
    grad_input = torch.empty_like(input)
    grad_input_2d = grad_input.view(rows_m, n)

    # DWeight is accumulated atomically in FP32. Compilation may execute the
    # kernel while tracing, so clear the scratch exactly once after compilation.
    grad_weight_fp32 = torch.empty(
        n, device=input.device, dtype=torch.float32
    )
    stream = torch.cuda.current_stream(input.device)
    device_index = input.get_device()
    arch, backend = _compile_environment(device_index)

    compiled = _compile_rmsnorm_bwd(
        n,
        _dtype_str(input.dtype),
        arch,
        backend,
        device_index,
        compile_args=(
            input_2d,
            weight,
            grad_2d,
            rstd_flat,
            grad_input_2d,
            grad_weight_fp32,
            rows_m,
            stream,
        ),
    )
    grad_weight_fp32.zero_()
    compiled(
        input_2d,
        weight,
        grad_2d,
        rstd_flat,
        grad_input_2d,
        grad_weight_fp32,
        rows_m,
        stream,
    )

    grad_weight = (
        grad_weight_fp32.to(weight.dtype) if need_grad_weight else None
    )
    return grad_input, grad_weight


def clear_rmsnorm_bwd_caches() -> None:
    """Clear only the BWD compile and target-environment caches."""

    _compile_rmsnorm_bwd.cache_clear()
    _compile_environment.cache_clear()


def rmsnorm_bwd_cache_info():
    """Return BWD compile-cache statistics for tests and benchmarks."""

    return _compile_rmsnorm_bwd.cache_info()

