# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""V2 FlyDSL RMSNorm backward implementation.

V2 replaces the M*N global FP32 atomics used for dweight with three kernels:

* K1: one block per row computes k[row] = rstd**2/N * sum(x*dy*gamma).
* K2: a (column tile, split-M) grid writes dx and one FP32 dweight partial.
* K3: one column tile reduces the 64 partials and writes weight.dtype.

The public ``rmsnorm_bwd`` signature intentionally matches the production
implementation so this file can replace it for an isolated benchmark.
"""

# mypy: allow-untyped-defs

import functools
import math

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.vector import ReductionOp, full
from flydsl.runtime.device import get_rocm_arch, is_rdna_arch

from torch._native.flydsl_cache import jit_cache


__all__ = [
    "clear_rmsnorm_bwd_caches",
    "rmsnorm_bwd",
    "rmsnorm_bwd_cache_info",
]

_K1_BLOCK_THREADS = 256
_K2_BLOCK_THREADS_LOWP = 256
_K2_BLOCK_THREADS_F32 = 128
_K3_BLOCK_THREADS = 256
_DWEIGHT_SPLITS = 64
_PARTIAL_VEC_WIDTH = 4  # Four FP32 values == one 128-bit transaction.
_SUPPORTED_DTYPES: dict[torch.dtype, str] = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
}


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


def _store_scalar(copy_atom, elem_dtype, divided_tensor, index, value):
    register = fx.make_rmem_tensor(1, elem_dtype)
    fx.memref_store_vec(full(1, value, elem_dtype), register)
    view = fx.slice(divided_tensor, (None, index))
    fx.copy_atom_call(copy_atom, register, view)


def _load_vec(copy_atom, vec_width, elem_dtype, divided_tensor, index):
    register = fx.make_rmem_tensor(vec_width, elem_dtype)
    view = fx.slice(divided_tensor, (None, index))
    fx.copy_atom_call(copy_atom, view, register)
    return fx.memref_load_vec(register)


def _store_vec(copy_atom, vec_width, elem_dtype, divided_tensor, index, value):
    register = fx.make_rmem_tensor(vec_width, elem_dtype)
    fx.memref_store_vec(value, register)
    view = fx.slice(divided_tensor, (None, index))
    fx.copy_atom_call(copy_atom, register, view)


def _load_f32_vec4_scalar(copy_atom_f32, divided_tensor, vec_index):
    """Load one logical FP32 vec4 as four BufferCopy32b transactions."""

    base = vec_index * 4
    return Vec.from_elements(
        [
            _load_scalar(copy_atom_f32, fx.Float32, divided_tensor, base),
            _load_scalar(copy_atom_f32, fx.Float32, divided_tensor, base + 1),
            _load_scalar(copy_atom_f32, fx.Float32, divided_tensor, base + 2),
            _load_scalar(copy_atom_f32, fx.Float32, divided_tensor, base + 3),
        ],
        fx.Float32,
    )


def _store_f32_vec4_scalar(copy_atom_f32, divided_tensor, vec_index, value):
    """Store one logical FP32 vec4 as four BufferCopy32b transactions."""

    base = vec_index * 4
    value = Vec(value)
    _store_scalar(copy_atom_f32, fx.Float32, divided_tensor, base, value[0])
    _store_scalar(copy_atom_f32, fx.Float32, divided_tensor, base + 1, value[1])
    _store_scalar(copy_atom_f32, fx.Float32, divided_tensor, base + 2, value[2])
    _store_scalar(copy_atom_f32, fx.Float32, divided_tensor, base + 3, value[3])


def _to_elem_vec(dtype_str: str, elem_dtype, use_hw_cvt_bf16: bool, value):
    """Convert an FP32 vector to the output dtype."""

    if const_expr(dtype_str == "bf16"):
        if const_expr(use_hw_cvt_bf16):
            return value.to(elem_dtype)
        bits = value.bitcast(fx.Uint32)
        upper = bits >> 16
        lsb = upper & 1
        rounded = value.bitcast(fx.Uint32) + lsb + 0x7FFF
        bf16_bits = rounded >> 16
        even = bf16_bits.shuffle(bf16_bits, [0, 2, 4, 6])
        odd = bf16_bits.shuffle(bf16_bits, [1, 3, 5, 7])
        packed = even | (odd << 16)
        return packed.bitcast(elem_dtype)
    if const_expr(dtype_str == "f32"):
        return value
    return value.to(elem_dtype)


def _build_rmsnorm_bwd_module(n: int, dtype_str: str, arch: str):
    """Build V2's K1/K2/K3 specialization."""

    warp_size = _warp_size_for_arch(arch)
    red_slots = max(1, (_K1_BLOCK_THREADS + warp_size - 1) // warp_size)
    elem_bits = 32 if dtype_str == "f32" else 16
    elem_vec_width = 128 // elem_bits
    k2_copy_width = 1 if dtype_str == "f32" else elem_vec_width
    k2_block_threads = (
        _K2_BLOCK_THREADS_F32
        if dtype_str == "f32"
        else _K2_BLOCK_THREADS_LOWP
    )
    k1_tile_cols = _K1_BLOCK_THREADS * elem_vec_width
    k2_tile_cols = k2_block_threads * elem_vec_width
    k3_tile_cols = _K3_BLOCK_THREADS * elem_vec_width
    if n % k1_tile_cols != 0:
        raise ValueError(
            f"K1 requires N divisible by {k1_tile_cols}, got N={n}"
        )
    if n % k2_tile_cols != 0:
        raise ValueError(
            f"K2 requires N divisible by {k2_tile_cols}, got N={n}"
        )
    if n % k3_tile_cols != 0:
        raise ValueError(
            f"K3 requires N divisible by {k3_tile_cols}, got N={n}"
        )
    k1_col_tiles = n // k1_tile_cols
    k2_col_tiles = n // k2_tile_cols
    k3_col_tiles = n // k3_tile_cols
    use_hw_cvt_bf16 = arch == "gfx950" or str(arch).startswith("gfx95")
    SharedStorage = _make_single_reduction_storage(red_slots)

    @flyc.kernel
    def rmsnorm_bwd_row_reduce_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        DY: fx.Tensor,
        Rstd: fx.Tensor,
        RowK: fx.Tensor,
    ):
        """K1: one block owns one row and writes its scalar projection k."""

        row = fx.block_idx.x
        tid = fx.thread_idx.x
        elem_dtype = _dtype_to_elem_type(dtype_str)
        fm_fast = arith.FastMathFlags.fast
        c_zero_f = fx.Float32(0.0)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = lds.s_red.view(fx.make_layout(red_slots, 1))

        def wave_reduce_add(value):
            reduced = value
            for shift_exp in range_constexpr(int(math.log2(warp_size))):
                offset = warp_size // (2 << shift_exp)
                reduced = reduced.addf(
                    reduced.shuffle_xor(offset, warp_size), fastmath=fm_fast
                )
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
                if const_expr(red_slots == 4):
                    partial = partial.addf(
                        partial.shuffle_xor(2, warp_size), fastmath=fm_fast
                    )
                    partial = partial.addf(
                        partial.shuffle_xor(1, warp_size), fastmath=fm_fast
                    )
                else:
                    partial = wave_reduce_add(partial)
                if lane == 0:
                    fx.memref_store(partial, s_red, 0)
            gpu.barrier()
            return fx.memref_load(s_red, 0)

        input_buf = fx.rocdl.make_buffer_tensor(Input)
        gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
        dy_buf = fx.rocdl.make_buffer_tensor(DY)
        rstd_buf = fx.rocdl.make_buffer_tensor(Rstd)
        row_k_buf = fx.rocdl.make_buffer_tensor(RowK)

        row_in = fx.slice(input_buf, (row, None))
        row_dy = fx.slice(dy_buf, (row, None))
        row_div = fx.logical_divide(row_in, fx.make_layout(elem_vec_width, 1))
        dy_div = fx.logical_divide(row_dy, fx.make_layout(elem_vec_width, 1))
        gamma_div = fx.logical_divide(
            gamma_buf, fx.make_layout(elem_vec_width, 1)
        )
        rstd_div = fx.logical_divide(rstd_buf, fx.make_layout(1, 1))
        row_k_div = fx.logical_divide(row_k_buf, fx.make_layout(1, 1))
        copy_atom_v = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
        copy_atom_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        acc0 = c_zero_f
        acc1 = c_zero_f
        for tile_i in range_constexpr(k1_col_tiles):
            idx = tid + tile_i * _K1_BLOCK_THREADS
            x_e = _load_vec(
                copy_atom_v, elem_vec_width, elem_dtype, row_div, idx
            )
            dy_e = _load_vec(
                copy_atom_v, elem_vec_width, elem_dtype, dy_div, idx
            )
            gamma_e = _load_vec(
                copy_atom_v, elem_vec_width, elem_dtype, gamma_div, idx
            )
            x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
            dy = dy_e if dtype_str == "f32" else dy_e.to(fx.Float32)
            gamma = (
                gamma_e if dtype_str == "f32" else gamma_e.to(fx.Float32)
            )
            tile_sum = (x * (dy * gamma)).reduce(
                ReductionOp.ADD, fastmath=fm_fast
            )
            if const_expr(tile_i % 2 == 0):
                acc0 = acc0 + tile_sum
            else:
                acc1 = acc1 + tile_sum

        t = block_reduce_add(acc0 + acc1)
        rstd = _load_scalar(copy_atom_f32, fx.Float32, rstd_div, row)
        k = t * ((rstd * rstd) / float(n))
        if tid == 0:
            _store_scalar(copy_atom_f32, fx.Float32, row_k_div, row, k)

    @flyc.kernel
    def rmsnorm_bwd_dx_partial_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        DY: fx.Tensor,
        Rstd: fx.Tensor,
        RowK: fx.Tensor,
        DX: fx.Tensor,
        DWeightPartial: fx.Tensor,
        m_in: fx.Int32,
    ):
        """K2: logical column tiles compute dx and split-M dweight partials."""

        col_tile = fx.block_idx.x
        split = fx.block_idx.y
        tid = fx.thread_idx.x
        elem_dtype = _dtype_to_elem_type(dtype_str)
        fm_fast = arith.FastMathFlags.fast
        c_zero_f = fx.Float32(0.0)

        input_buf = fx.rocdl.make_buffer_tensor(Input)
        gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
        dy_buf = fx.rocdl.make_buffer_tensor(DY)
        rstd_buf = fx.rocdl.make_buffer_tensor(Rstd)
        row_k_buf = fx.rocdl.make_buffer_tensor(RowK)
        dx_buf = fx.rocdl.make_buffer_tensor(DX)
        partial_buf = fx.rocdl.make_buffer_tensor(DWeightPartial)

        copy_atom_v = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
        copy_atom_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)
        copy_atom_f32_v = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 32)
        gamma_div = fx.logical_divide(
            gamma_buf, fx.make_layout(k2_copy_width, 1)
        )
        rstd_div = fx.logical_divide(rstd_buf, fx.make_layout(1, 1))
        row_k_div = fx.logical_divide(row_k_buf, fx.make_layout(1, 1))

        vec_idx = col_tile * k2_block_threads + tid
        if const_expr(dtype_str == "f32"):
            gamma_e = _load_f32_vec4_scalar(
                copy_atom_f32, gamma_div, vec_idx
            )
        else:
            gamma_e = _load_vec(
                copy_atom_v, elem_vec_width, elem_dtype, gamma_div, vec_idx
            )
        gamma = gamma_e if dtype_str == "f32" else gamma_e.to(fx.Float32)
        zero_vec = full(elem_vec_width, c_zero_f, fx.Float32)

        for row, state in fx.range(
            fx.Index(split),
            fx.Index(m_in),
            _DWEIGHT_SPLITS,
            init=[zero_vec],
        ):
            dw_acc = Vec(state[0])
            row_in = fx.slice(input_buf, (row, None))
            row_dy = fx.slice(dy_buf, (row, None))
            row_dx = fx.slice(dx_buf, (row, None))
            row_div = fx.logical_divide(
                row_in, fx.make_layout(k2_copy_width, 1)
            )
            dy_div = fx.logical_divide(
                row_dy, fx.make_layout(k2_copy_width, 1)
            )
            dx_div = fx.logical_divide(
                row_dx, fx.make_layout(k2_copy_width, 1)
            )
            if const_expr(dtype_str == "f32"):
                x_e = _load_f32_vec4_scalar(
                    copy_atom_f32, row_div, vec_idx
                )
                dy_e = _load_f32_vec4_scalar(
                    copy_atom_f32, dy_div, vec_idx
                )
            else:
                x_e = _load_vec(
                    copy_atom_v, elem_vec_width, elem_dtype, row_div, vec_idx
                )
                dy_e = _load_vec(
                    copy_atom_v, elem_vec_width, elem_dtype, dy_div, vec_idx
                )
            x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
            dy = dy_e if dtype_str == "f32" else dy_e.to(fx.Float32)
            rstd = _load_scalar(copy_atom_f32, fx.Float32, rstd_div, row)
            k = _load_scalar(copy_atom_f32, fx.Float32, row_k_div, row)
            projected = fmath.fma(
                x,
                full(elem_vec_width, c_zero_f - k, fx.Float32),
                dy * gamma,
                fastmath=fm_fast,
            )
            dx = projected * rstd
            dx_e = _to_elem_vec(
                dtype_str, elem_dtype, use_hw_cvt_bf16, dx
            )
            if const_expr(dtype_str == "f32"):
                _store_f32_vec4_scalar(
                    copy_atom_f32, dx_div, vec_idx, dx_e
                )
            else:
                _store_vec(
                    copy_atom_v,
                    elem_vec_width,
                    elem_dtype,
                    dx_div,
                    vec_idx,
                    dx_e,
                )
            next_dw_acc = dw_acc + ((dy * x) * rstd)
            loop_results = yield [next_dw_acc]

        dw_acc = Vec(loop_results)
        partial_row = fx.slice(partial_buf, (split, None))
        partial_div = fx.logical_divide(
            partial_row, fx.make_layout(_PARTIAL_VEC_WIDTH, 1)
        )
        if const_expr(elem_vec_width == _PARTIAL_VEC_WIDTH):
            _store_vec(
                copy_atom_f32_v,
                _PARTIAL_VEC_WIDTH,
                fx.Float32,
                partial_div,
                vec_idx,
                dw_acc,
            )
        else:
            partial_idx = vec_idx * 2
            lo = dw_acc.shuffle(dw_acc, [0, 1, 2, 3])
            hi = dw_acc.shuffle(dw_acc, [4, 5, 6, 7])
            _store_vec(
                copy_atom_f32_v,
                _PARTIAL_VEC_WIDTH,
                fx.Float32,
                partial_div,
                partial_idx,
                lo,
            )
            _store_vec(
                copy_atom_f32_v,
                _PARTIAL_VEC_WIDTH,
                fx.Float32,
                partial_div,
                partial_idx + 1,
                hi,
            )

    @flyc.kernel
    def rmsnorm_bwd_dweight_reduce_kernel(
        DWeightPartial: fx.Tensor,
        DWeight: fx.Tensor,
    ):
        """K3: reduce 64 FP32 partial rows and directly write weight dtype."""

        col_tile = fx.block_idx.x
        tid = fx.thread_idx.x
        elem_dtype = _dtype_to_elem_type(dtype_str)
        c_zero_f = fx.Float32(0.0)
        partial_buf = fx.rocdl.make_buffer_tensor(DWeightPartial)
        dweight_buf = fx.rocdl.make_buffer_tensor(DWeight)
        copy_atom_f32_v = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 32)
        copy_atom_out = fx.make_copy_atom(
            fx.rocdl.BufferCopy128b(), elem_bits
        )

        vec_idx = col_tile * _K3_BLOCK_THREADS + tid
        acc_lo = full(_PARTIAL_VEC_WIDTH, c_zero_f, fx.Float32)
        if const_expr(elem_vec_width == 8):
            acc_hi = full(_PARTIAL_VEC_WIDTH, c_zero_f, fx.Float32)

        for split_i in range_constexpr(_DWEIGHT_SPLITS):
            partial_row = fx.slice(partial_buf, (split_i, None))
            partial_div = fx.logical_divide(
                partial_row, fx.make_layout(_PARTIAL_VEC_WIDTH, 1)
            )
            if const_expr(elem_vec_width == _PARTIAL_VEC_WIDTH):
                acc_lo = acc_lo + _load_vec(
                    copy_atom_f32_v,
                    _PARTIAL_VEC_WIDTH,
                    fx.Float32,
                    partial_div,
                    vec_idx,
                )
            else:
                partial_idx = vec_idx * 2
                acc_lo = acc_lo + _load_vec(
                    copy_atom_f32_v,
                    _PARTIAL_VEC_WIDTH,
                    fx.Float32,
                    partial_div,
                    partial_idx,
                )
                acc_hi = acc_hi + _load_vec(
                    copy_atom_f32_v,
                    _PARTIAL_VEC_WIDTH,
                    fx.Float32,
                    partial_div,
                    partial_idx + 1,
                )

        dweight_div = fx.logical_divide(
            dweight_buf, fx.make_layout(elem_vec_width, 1)
        )
        if const_expr(elem_vec_width == _PARTIAL_VEC_WIDTH):
            dw_f32 = acc_lo
        else:
            dw_f32 = acc_lo.shuffle(acc_hi, [0, 1, 2, 3, 4, 5, 6, 7])
        dw_e = _to_elem_vec(
            dtype_str, elem_dtype, use_hw_cvt_bf16, dw_f32
        )
        _store_vec(
            copy_atom_out,
            elem_vec_width,
            elem_dtype,
            dweight_div,
            vec_idx,
            dw_e,
        )

    @flyc.jit
    def launch_rmsnorm_bwd(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        DY: fx.Tensor,
        Rstd: fx.Tensor,
        RowK: fx.Tensor,
        DX: fx.Tensor,
        DWeightPartial: fx.Tensor,
        DWeight: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        rmsnorm_bwd_row_reduce_kernel(
            Input, Gamma, DY, Rstd, RowK
        ).launch(
            grid=(m_in, 1, 1),
            block=(_K1_BLOCK_THREADS, 1, 1),
            stream=stream,
        )
        rmsnorm_bwd_dx_partial_kernel(
            Input,
            Gamma,
            DY,
            Rstd,
            RowK,
            DX,
            DWeightPartial,
            m_in,
        ).launch(
            grid=(k2_col_tiles, _DWEIGHT_SPLITS, 1),
            block=(k2_block_threads, 1, 1),
            stream=stream,
        )
        rmsnorm_bwd_dweight_reduce_kernel(
            DWeightPartial, DWeight
        ).launch(
            grid=(k3_col_tiles, 1, 1),
            block=(_K3_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm_bwd


def _make_compile_arg(tensor: torch.Tensor):
    """Make only the row dimension dynamic across M values."""

    return flyc.from_torch_tensor(tensor).mark_shape_dynamic(0)


@functools.cache
def _compile_environment(device_index: int) -> tuple[str, str]:
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
    (
        input_2d,
        weight,
        grad_2d,
        rstd,
        row_k,
        grad_input,
        dweight_partial,
        grad_weight,
        rows_m,
        stream,
    ) = compile_args
    launch = _build_rmsnorm_bwd_module(n, dtype, arch)
    return flyc.compile(
        launch,
        _make_compile_arg(input_2d),
        flyc.from_torch_tensor(weight),
        _make_compile_arg(grad_2d),
        _make_compile_arg(rstd),
        _make_compile_arg(row_k),
        _make_compile_arg(grad_input),
        flyc.from_torch_tensor(dweight_partial),
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
    """Run the V2 three-kernel backward pipeline."""

    n = input.shape[-1]
    rows_m = input.numel() // n
    input_2d = input.view(rows_m, n)
    grad_2d = grad_out.view(rows_m, n)
    rstd_flat = rstd.view(rows_m)
    grad_input = torch.empty_like(input)
    grad_input_2d = grad_input.view(rows_m, n)
    row_k = torch.empty(rows_m, device=input.device, dtype=torch.float32)
    dweight_partial = torch.empty(
        (_DWEIGHT_SPLITS, n), device=input.device, dtype=torch.float32
    )
    # K3 overwrites every element, so neither zero_ nor a post-kernel cast is
    # required. The buffer is still allocated when output_mask omits dweight;
    # DX is computed unconditionally; the PyTorch impl honors output_mask at return.
    grad_weight_out = torch.empty_like(weight)

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
            row_k,
            grad_input_2d,
            dweight_partial,
            grad_weight_out,
            rows_m,
            stream,
        ),
    )
    compiled(
        input_2d,
        weight,
        grad_2d,
        rstd_flat,
        row_k,
        grad_input_2d,
        dweight_partial,
        grad_weight_out,
        rows_m,
        stream,
    )
    return grad_input, grad_weight_out if need_grad_weight else None


def clear_rmsnorm_bwd_caches() -> None:
    _compile_rmsnorm_bwd.cache_clear()
    _compile_environment.cache_clear()


def rmsnorm_bwd_cache_info():
    return _compile_rmsnorm_bwd.cache_info()
