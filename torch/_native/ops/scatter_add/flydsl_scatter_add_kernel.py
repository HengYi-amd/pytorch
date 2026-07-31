"""FlyDSL scatter_add kernels for ROCm native overrides.

Three paths, chosen per call:

- Global-atomic (``_build_scatter_add_kernel``): one wavefront per source row,
  16B gather of ``src`` + atomic-add into ``out[index[i], :]``. fp32 adds one
  element per atomic; fp16/bf16 use packed x2 atomics. Works for any shape but
  is global-atomic-throughput bound.

- LDS owner-binning (``_build_scatter_add_lds_kernel``): each block owns a
  disjoint range of output rows. It scans the index, bins matching entries via
  one LDS atomic on a counter, then all threads cooperate over the N columns.
  Each thread owns a fixed column partition and keeps that partition's fp32
  accumulator in register memory, so the accumulator needs neither LDS traffic
  nor per-column atomics. Finally the block writes its owned rows back with plain
  (non-atomic) global stores. This removes global atomics entirely and
  accumulates in fp32 (better fp16/bf16 accuracy). It re-scans the index once per
  block, so it is only profitable when the number of owner blocks stays bounded
  (see ``_lds_plan``); otherwise we fall back to global-atomic.

- Sort/segmented owner reduction (``_build_scatter_add_segmented_kernel``):
  for high-reuse gfx950 calls, sort the 1-D index on device, build
  row-segment offsets with ``searchsorted``, and launch one output-row owner
  block per segment. This removes both floating-point atomics and the
  owner-binning path's repeated full-index scans. The sort and workspaces are
  included in end-to-end timing. One shape-derived resource model is used for
  every dtype and size; low reuse, graph capture, and non-gfx950 devices retain
  the owner-binning/global fallbacks.
"""

# mypy: allow-untyped-defs

from collections import namedtuple

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as _fly, llvm, vector as _vector
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.vector import Vector as Vec, full
from flydsl.runtime.device import get_rocm_arch, is_rdna_arch

import torch
from torch._native.flydsl_cache import jit_cache


_WARPS_PER_BLOCK = 8
_VEC_BYTES = 16

# The two optimization choices below are intentionally global: every eligible
# dtype/shape uses the same policy. Alignment failures still use the scalar
# safety fallback, and graph capture/low-reuse calls retain the owner-binning
# path because sort/searchsorted need dynamic workspaces.
_USE_VECTOR_IO = True
_USE_SEGMENTED = True
_SEGMENTED_MIN_REUSE = 4

# Shape-derived owner-binning resource model. block_rows keeps roughly this many
# columns under one owner block, capped to bound match-list/control work. The
# thread model gives each owned row at least one wave and aims to keep no more
# than 32 fp32 accumulator elements per thread. 64 is the hard VGPR-pressure
# safety limit used by _lds_plan.
_LDS_TARGET_COLUMNS_PER_BLOCK = 8192
_LDS_MIN_BLOCK_ROWS = 2
_LDS_MAX_BLOCK_ROWS = 8
_LDS_TARGET_ACC_ELEMS_PER_THREAD = 32
_LDS_MAX_ACC_ELEMS_PER_THREAD = 64
# The redundant re-scan (num_blocks * num_entries i64 reads) versus src traffic
# (num_entries * n * elem_bytes) has ratio num_blocks/n -- independent of the
# entry count -- so gating on num_blocks alone bounds it. Calibrated on gfx950:
# num_blocks==16384 still wins at n=1024/f32, 65536 loses.
_LDS_NUM_BLOCKS_FACTOR = 4

_TORCH_TO_STR: dict[torch.dtype, str] = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
}


def _warp_size() -> int:
    return 32 if is_rdna_arch(get_rocm_arch()) else 64


WARP_SIZE = _warp_size()

_IS_GFX950 = get_rocm_arch().startswith("gfx950")
_BLOCK_THREADS = WARP_SIZE * _WARPS_PER_BLOCK


def _dtype_meta(dtype_str: str):
    """(elements per 16B chunk, element bytes). Safe to call outside a kernel."""
    if dtype_str == "f32":
        return 4, 4
    if dtype_str in ("f16", "bf16"):
        return 8, 2
    raise ValueError(f"unsupported scatter_add dtype: {dtype_str!r}")


def _elem_ir(dtype_str: str):
    """IR element type. Must be called inside a kernel (needs MLIR context)."""
    if dtype_str == "f32":
        return T.f32
    if dtype_str == "f16":
        return T.f16
    return T.bf16


def _llvm_ptr(base_tensor, elem_offset, elem_bytes):
    """Global ``!llvm.ptr<1>`` at ``base_tensor + elem_offset*elem_bytes``."""
    ptr_type = ir.Type.parse("!llvm.ptr<1>")
    base = _fly.extract_aligned_pointer_as_index(ptr_type, base_tensor)
    base = llvm.PtrToIntOp(T.i64, base).result
    byte_off = arith.index_cast(T.i64, fx.Index(elem_offset) * fx.Index(elem_bytes))
    p = llvm.AddOp(base, byte_off, llvm.IntegerOverflowFlags(0)).result
    p = llvm.IntToPtrOp(ptr_type, p).result
    return p._value if hasattr(p, "_value") else p


def _atomic_fadd(ptr, value):
    llvm.AtomicRMWOp(
        llvm.AtomicBinOp.fadd,
        ptr,
        value,
        llvm.AtomicOrdering.monotonic,
        syncscope="agent",
        alignment=4,
    )


def _build_scatter_add_kernel(dtype_str: str):
    vec_elems, elem_bytes = _dtype_meta(dtype_str)
    is_half = dtype_str != "f32"
    step = 2 if is_half else 1

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def scatter_add_kernel(
        out: fx.Tensor,
        index: fx.Tensor,
        src: fx.Tensor,
        num_entries: fx.Int32,
        cols_n: fx.Int32,
        out_rows_m: fx.Int32,
        src_row_stride: fx.Int32,
        out_row_stride: fx.Int32,
    ):
        elem_ty = _elem_ir(dtype_str)
        index_rsrc = buffer_ops.create_buffer_resource(index, max_size=True)
        src_rsrc = buffer_ops.create_buffer_resource(src, max_size=True)

        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        gdim = fx.grid_dim.x

        wave_in_block = tid // fx.Int32(WARP_SIZE)
        lane = tid % fx.Int32(WARP_SIZE)
        lane_offset = lane * fx.Int32(vec_elems)
        stride_elems = fx.Int32(WARP_SIZE * vec_elems)
        entries_per_block = fx.Int32(_WARPS_PER_BLOCK)

        base = bid * entries_per_block
        while base < num_entries:
            entry_id = base + wave_in_block
            if entry_id < num_entries:
                dst_row = buffer_ops.buffer_load(
                    index_rsrc, entry_id, vec_width=1, dtype=T.i64
                )
                if dst_row >= fx.Int64(0) and dst_row < fx.Int64(out_rows_m):
                    off = lane_offset
                    while off < cols_n:
                        # gather one 16B chunk, atomic-add into out
                        src_off = entry_id * src_row_stride + off
                        dst_off = dst_row * fx.Int64(out_row_stride) + fx.Int64(off)
                        vec = buffer_ops.buffer_load(
                            src_rsrc, src_off, vec_width=vec_elems, dtype=elem_ty
                        )
                        for j in range_constexpr(vec_elems // step):
                            if const_expr(is_half):
                                pair_ty = ir.VectorType.get([2], elem_ty)
                                a = _vector.extract(
                                    vec, static_position=[2 * j], dynamic_position=[]
                                )
                                b = _vector.extract(
                                    vec,
                                    static_position=[2 * j + 1],
                                    dynamic_position=[],
                                )
                                vdata = _vector.from_elements(pair_ty, [a, b])
                            else:
                                vdata = _vector.extract(
                                    vec, static_position=[j], dynamic_position=[]
                                )
                            ptr = _llvm_ptr(
                                out, dst_off + fx.Int64(j * step), elem_bytes
                            )
                            _atomic_fadd(ptr, vdata)
                        off = off + stride_elems
            base = base + gdim * entries_per_block

    @flyc.jit
    def launch(
        out: fx.Tensor,
        index: fx.Tensor,
        src: fx.Tensor,
        num_entries: fx.Int32,
        cols_n: fx.Int32,
        out_rows_m: fx.Int32,
        src_row_stride: fx.Int32,
        out_row_stride: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        scatter_add_kernel(
            out,
            index,
            src,
            num_entries,
            cols_n,
            out_rows_m,
            src_row_stride,
            out_row_stride,
        ).launch(grid=(grid_x, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


_FLOAT_CLS = {"f32": fx.Float32, "f16": fx.Float16, "bf16": fx.BFloat16}


def _lds_atomic_add_i32(memref, val):
    """LDS (addrspace 3) atomic add on a single-element i32 counter -> old value."""
    base = buffer_ops.create_llvm_ptr(
        buffer_ops.extract_base_index(memref, address_space=3), address_space=3
    )
    ptr = buffer_ops.get_element_ptr(base, byte_offset=arith.unwrap(fx.Int32(0)))
    return llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add,
        ptr,
        arith.unwrap(val),
        llvm.AtomicOrdering.monotonic,
        syncscope="workgroup",
        alignment=4,
    ).res


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _next_power_of_two(value: int) -> int:
    return 1 << (max(1, value) - 1).bit_length()


def _adaptive_lds_tuning(n: int, out_rows_m: int):
    """Derive owner rows and threads from one dtype-independent resource model."""
    proportional_rows = max(
        _LDS_MIN_BLOCK_ROWS, _LDS_TARGET_COLUMNS_PER_BLOCK // max(1, n)
    )
    block_rows = max(
        1, min(out_rows_m, _LDS_MAX_BLOCK_ROWS, proportional_rows)
    )

    min_threads = 4 * WARP_SIZE
    row_parallelism = WARP_SIZE * block_rows
    register_parallelism = _next_power_of_two(
        _ceil_div(
            block_rows * n, _LDS_TARGET_ACC_ELEMS_PER_THREAD
        )
    )
    block_threads = min(
        _BLOCK_THREADS,
        _next_power_of_two(
            max(min_threads, row_parallelism, register_parallelism)
        ),
    )
    return block_rows, block_threads


def _adaptive_segmented_threads(
    n: int, elem_bytes: int, use_vector: bool
) -> int:
    """Keep 2-4 scalar/vector work items per thread, without a shape table."""
    elements_per_item = _VEC_BYTES // elem_bytes if use_vector else 1
    work_items = _ceil_div(n, elements_per_item)
    items_per_thread = max(2, min(4, _ceil_div(n, 2048)))
    wanted = _next_power_of_two(_ceil_div(work_items, items_per_thread))
    return max(2 * WARP_SIZE, min(_BLOCK_THREADS, wanted))


def _vector_io_safe(
    out: torch.Tensor,
    src: torch.Tensor,
    cols_n: int,
    dtype_str: str,
) -> bool:
    if not _USE_VECTOR_IO:
        return False
    vec_elems, elem_bytes = _dtype_meta(dtype_str)
    return (
        cols_n % vec_elems == 0
        and out.data_ptr() % _VEC_BYTES == 0
        and src.data_ptr() % _VEC_BYTES == 0
        and out.stride(0) * elem_bytes % _VEC_BYTES == 0
        and src.stride(0) * elem_bytes % _VEC_BYTES == 0
    )


def _segmented_eligible(rows_m: int, out_rows_m: int) -> bool:
    """One architecture/reuse gate shared by every dtype and column size."""
    return (
        _USE_SEGMENTED
        and _IS_GFX950
        and out_rows_m > 0
        and rows_m >= _SEGMENTED_MIN_REUSE * out_rows_m
    )


def _build_scatter_add_lds_kernel(
    dtype_str: str, n: int, block_rows: int, block_threads: int, use_vector: bool
):
    vec_elems, _ = _dtype_meta(dtype_str)
    vec_chunks = n // vec_elems
    vecs_per_thread = (vec_chunks + block_threads - 1) // block_threads
    reg_vec_slots = block_rows * vecs_per_thread
    cols_per_thread = (n + block_threads - 1) // block_threads
    reg_elems = block_rows * cols_per_thread
    cap = block_threads  # one match per thread per chunk -> never overflows
    fcls = _FLOAT_CLS[dtype_str]

    @fx.struct
    class Shared:
        match_lr: fx.Array[fx.Int32, cap, 16]
        match_entry: fx.Array[fx.Int32, cap, 16]
        count: fx.Array[fx.Int32, 1, 16]

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def scatter_add_lds_kernel(
        out: fx.Tensor,
        index: fx.Tensor,
        src: fx.Tensor,
        num_entries: fx.Int32,
        out_rows_m: fx.Int32,
        src_row_stride: fx.Int32,
        out_row_stride: fx.Int32,
    ):
        elem_ty = _elem_ir(dtype_str)
        index_rsrc = buffer_ops.create_buffer_resource(index, max_size=True)
        src_rsrc = buffer_ops.create_buffer_resource(src, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out, max_size=True)

        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        BT = fx.Int32(block_threads)
        N = fx.Int32(n)
        row_base = bid * fx.Int32(block_rows)

        s = fx.SharedAllocator().allocate(Shared).peek()
        match_lr = s.match_lr.view(fx.make_layout(cap, 1))
        match_entry = s.match_entry.view(fx.make_layout(cap, 1))
        count = s.count.view(fx.make_layout(1, 1))
        if const_expr(use_vector):
            reg_acc_vec = []
            for _ in range_constexpr(reg_vec_slots):
                reg_acc_vec.append(fx.make_rmem_tensor(vec_elems, fx.Float32))
            zero_vec = full(vec_elems, fx.Float32(0.0), fx.Float32)
        else:
            reg_acc = fx.make_rmem_tensor(reg_elems, fx.Float32)

        # Seed the accumulator with the block's owned output rows (scatter_add is
        # += onto an existing out; blocks own disjoint rows so this is race-free).
        # The row and column-slot loops are compile-time unrolled: every register
        # access therefore has a constant offset and stays promotable to VGPRs.
        if const_expr(use_vector):
            for lr_c in range_constexpr(block_rows):
                grow = row_base + fx.Int32(lr_c)
                for vec_step in range_constexpr(vecs_per_thread):
                    vec_idx = tid + fx.Int32(vec_step * block_threads)
                    c = vec_idx * fx.Int32(vec_elems)
                    slot = lr_c * vecs_per_thread + vec_step
                    a_vec = zero_vec
                    if vec_idx < fx.Int32(vec_chunks) and grow < out_rows_m:
                        ov = buffer_ops.buffer_load(
                            out_rsrc,
                            grow * out_row_stride + c,
                            vec_width=vec_elems,
                            dtype=elem_ty,
                        )
                        a_vec = Vec(ov, vec_elems, fcls).to(fx.Float32)
                    fx.memref_store_vec(a_vec, reg_acc_vec[slot])
        else:
            for lr_c in range_constexpr(block_rows):
                grow = row_base + fx.Int32(lr_c)
                for col_step in range_constexpr(cols_per_thread):
                    c = tid + fx.Int32(col_step * block_threads)
                    a = fx.Float32(0.0)
                    if c < N and grow < out_rows_m:
                        ov = buffer_ops.buffer_load(
                            out_rsrc,
                            grow * out_row_stride + c,
                            vec_width=1,
                            dtype=elem_ty,
                        )
                        a = fcls(ov).to(fx.Float32)
                    fx.memref_store(
                        a, reg_acc, lr_c * cols_per_thread + col_step
                    )
        # Kept to preserve V1's synchronization schedule and isolate vector I/O.
        gpu.barrier()

        chunk = fx.Int32(0)
        while chunk < num_entries:
            if tid == fx.Int32(0):
                fx.memref_store(fx.Int32(0), count, 0)
            gpu.barrier()

            entry = chunk + tid
            if entry < num_entries:
                dst = buffer_ops.buffer_load(
                    index_rsrc, entry, vec_width=1, dtype=T.i64
                )
                lr = fx.Int32(dst) - row_base
                if lr >= fx.Int32(0) and lr < fx.Int32(block_rows):
                    pos = _lds_atomic_add_i32(count, fx.Int32(1))
                    fx.memref_store(lr, match_lr, fx.Int32(pos))
                    fx.memref_store(entry, match_entry, fx.Int32(pos))
            gpu.barrier()

            cnt = fx.memref_load(count, 0)
            m = fx.Int32(0)
            while m < cnt:
                lr = fx.memref_load(match_lr, m)
                e = fx.memref_load(match_entry, m)
                ebase = e * src_row_stride
                # Expand the runtime local-row choice into compile-time branches.
                # This keeps every reg_acc index constant instead of turning the
                # register tensor into dynamically indexed scratch memory.
                for lr_c in range_constexpr(block_rows):
                    if lr == fx.Int32(lr_c):
                        if const_expr(use_vector):
                            for vec_step in range_constexpr(vecs_per_thread):
                                vec_idx = tid + fx.Int32(vec_step * block_threads)
                                if vec_idx < fx.Int32(vec_chunks):
                                    c = vec_idx * fx.Int32(vec_elems)
                                    v = buffer_ops.buffer_load(
                                        src_rsrc,
                                        ebase + c,
                                        vec_width=vec_elems,
                                        dtype=elem_ty,
                                    )
                                    sv = Vec(v, vec_elems, fcls).to(fx.Float32)
                                    slot = lr_c * vecs_per_thread + vec_step
                                    cur = fx.memref_load_vec(reg_acc_vec[slot])
                                    fx.memref_store_vec(
                                        cur + sv, reg_acc_vec[slot]
                                    )
                        else:
                            for col_step in range_constexpr(cols_per_thread):
                                c = tid + fx.Int32(col_step * block_threads)
                                if c < N:
                                    v = buffer_ops.buffer_load(
                                        src_rsrc,
                                        ebase + c,
                                        vec_width=1,
                                        dtype=elem_ty,
                                    )
                                    slot = lr_c * cols_per_thread + col_step
                                    cur = fx.memref_load(reg_acc, slot)
                                    fx.memref_store(
                                        cur + fcls(v).to(fx.Float32), reg_acc, slot
                                    )
                m = m + fx.Int32(1)
            gpu.barrier()
            chunk = chunk + BT

        # Write owned rows back with plain stores (disjoint per block, no atomics).
        if const_expr(use_vector):
            for lr_c in range_constexpr(block_rows):
                grow = row_base + fx.Int32(lr_c)
                for vec_step in range_constexpr(vecs_per_thread):
                    vec_idx = tid + fx.Int32(vec_step * block_threads)
                    if vec_idx < fx.Int32(vec_chunks) and grow < out_rows_m:
                        c = vec_idx * fx.Int32(vec_elems)
                        a_vec = fx.memref_load_vec(
                            reg_acc_vec[lr_c * vecs_per_thread + vec_step]
                        )
                        out_v = a_vec if dtype_str == "f32" else a_vec.to(fcls)
                        buffer_ops.buffer_store(
                            out_v, out_rsrc, grow * out_row_stride + c
                        )
        else:
            for lr_c in range_constexpr(block_rows):
                grow = row_base + fx.Int32(lr_c)
                for col_step in range_constexpr(cols_per_thread):
                    c = tid + fx.Int32(col_step * block_threads)
                    if c < N and grow < out_rows_m:
                        a = fx.memref_load(
                            reg_acc, lr_c * cols_per_thread + col_step
                        )
                        out_v = a if dtype_str == "f32" else a.to(fcls)
                        buffer_ops.buffer_store(
                            out_v, out_rsrc, grow * out_row_stride + c
                        )

    @flyc.jit
    def launch(
        out: fx.Tensor,
        index: fx.Tensor,
        src: fx.Tensor,
        num_entries: fx.Int32,
        out_rows_m: fx.Int32,
        src_row_stride: fx.Int32,
        out_row_stride: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        scatter_add_lds_kernel(
            out,
            index,
            src,
            num_entries,
            out_rows_m,
            src_row_stride,
            out_row_stride,
        ).launch(grid=(grid_x, 1, 1), block=(block_threads, 1, 1), stream=stream)

    return launch


def _build_scatter_add_segmented_kernel(
    dtype_str: str,
    n: int,
    block_threads: int,
    use_vector: bool,
):
    """One pre-binned source segment per output-row block, without atomics."""
    vec_elems, _ = _dtype_meta(dtype_str)
    vec_chunks = n // vec_elems
    vecs_per_thread = (vec_chunks + block_threads - 1) // block_threads
    cols_per_thread = (n + block_threads - 1) // block_threads
    fcls = _FLOAT_CLS[dtype_str]

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def scatter_add_segmented_kernel(
        out: fx.Tensor,
        offsets: fx.Tensor,
        permutation: fx.Tensor,
        src: fx.Tensor,
        src_row_stride: fx.Int32,
        out_row_stride: fx.Int32,
    ):
        elem_ty = _elem_ir(dtype_str)
        offsets_rsrc = buffer_ops.create_buffer_resource(
            offsets, max_size=True
        )
        perm_rsrc = buffer_ops.create_buffer_resource(
            permutation, max_size=True
        )
        src_rsrc = buffer_ops.create_buffer_resource(src, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out, max_size=True)

        tid = fx.thread_idx.x
        row = fx.block_idx.x
        N = fx.Int32(n)
        begin = fx.Int32(
            buffer_ops.buffer_load(
                offsets_rsrc, row, vec_width=1, dtype=T.i64
            )
        )
        finish = fx.Int32(
            buffer_ops.buffer_load(
                offsets_rsrc, row + fx.Int32(1), vec_width=1, dtype=T.i64
            )
        )

        if const_expr(use_vector):
            reg_acc_vec = []
            for _ in range_constexpr(vecs_per_thread):
                reg_acc_vec.append(
                    fx.make_rmem_tensor(vec_elems, fx.Float32)
                )

            for vec_step in range_constexpr(vecs_per_thread):
                vec_idx = tid + fx.Int32(vec_step * block_threads)
                if vec_idx < fx.Int32(vec_chunks):
                    c = vec_idx * fx.Int32(vec_elems)
                    ov = buffer_ops.buffer_load(
                        out_rsrc,
                        row * out_row_stride + c,
                        vec_width=vec_elems,
                        dtype=elem_ty,
                    )
                    acc = Vec(ov, vec_elems, fcls).to(fx.Float32)
                    fx.memref_store_vec(acc, reg_acc_vec[vec_step])

            pos = begin
            while pos < finish:
                src_row_i64 = buffer_ops.buffer_load(
                    perm_rsrc, pos, vec_width=1, dtype=T.i64
                )
                src_base = fx.Int32(src_row_i64) * src_row_stride
                for vec_step in range_constexpr(vecs_per_thread):
                    vec_idx = tid + fx.Int32(vec_step * block_threads)
                    if vec_idx < fx.Int32(vec_chunks):
                        c = vec_idx * fx.Int32(vec_elems)
                        sv_raw = buffer_ops.buffer_load(
                            src_rsrc,
                            src_base + c,
                            vec_width=vec_elems,
                            dtype=elem_ty,
                        )
                        sv = Vec(sv_raw, vec_elems, fcls).to(fx.Float32)
                        cur = fx.memref_load_vec(reg_acc_vec[vec_step])
                        fx.memref_store_vec(
                            cur + sv, reg_acc_vec[vec_step]
                        )
                pos = pos + fx.Int32(1)

            for vec_step in range_constexpr(vecs_per_thread):
                vec_idx = tid + fx.Int32(vec_step * block_threads)
                if vec_idx < fx.Int32(vec_chunks):
                    c = vec_idx * fx.Int32(vec_elems)
                    acc = fx.memref_load_vec(reg_acc_vec[vec_step])
                    out_value = acc if dtype_str == "f32" else acc.to(fcls)
                    buffer_ops.buffer_store(
                        out_value,
                        out_rsrc,
                        row * out_row_stride + c,
                    )
        else:
            reg_acc = fx.make_rmem_tensor(cols_per_thread, fx.Float32)

            for col_step in range_constexpr(cols_per_thread):
                c = tid + fx.Int32(col_step * block_threads)
                if c < N:
                    ov = buffer_ops.buffer_load(
                        out_rsrc,
                        row * out_row_stride + c,
                        vec_width=1,
                        dtype=elem_ty,
                    )
                    fx.memref_store(
                        fcls(ov).to(fx.Float32), reg_acc, col_step
                    )

            pos = begin
            while pos < finish:
                src_row_i64 = buffer_ops.buffer_load(
                    perm_rsrc, pos, vec_width=1, dtype=T.i64
                )
                src_base = fx.Int32(src_row_i64) * src_row_stride
                for col_step in range_constexpr(cols_per_thread):
                    c = tid + fx.Int32(col_step * block_threads)
                    if c < N:
                        value = buffer_ops.buffer_load(
                            src_rsrc,
                            src_base + c,
                            vec_width=1,
                            dtype=elem_ty,
                        )
                        cur = fx.memref_load(reg_acc, col_step)
                        fx.memref_store(
                            cur + fcls(value).to(fx.Float32),
                            reg_acc,
                            col_step,
                        )
                pos = pos + fx.Int32(1)

            for col_step in range_constexpr(cols_per_thread):
                c = tid + fx.Int32(col_step * block_threads)
                if c < N:
                    acc = fx.memref_load(reg_acc, col_step)
                    out_value = acc if dtype_str == "f32" else acc.to(fcls)
                    buffer_ops.buffer_store(
                        out_value,
                        out_rsrc,
                        row * out_row_stride + c,
                    )

    @flyc.jit
    def launch(
        out: fx.Tensor,
        offsets: fx.Tensor,
        permutation: fx.Tensor,
        src: fx.Tensor,
        src_row_stride: fx.Int32,
        out_row_stride: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        scatter_add_segmented_kernel(
            out,
            offsets,
            permutation,
            src,
            src_row_stride,
            out_row_stride,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch


# Specializations already compiled. flyc.compile runs the kernel once while
# tracing (real tensors, no fake-tensor path as CuteDSL has), double-adding into
# out; we snapshot/restore out on each first compile (see scatter_add_into).
_COMPILED_KEYS: set = set()


@jit_cache
def _compile_scatter_add(
    dtype_str: str,
    device_index: int,
    *,
    compile_args,
) -> flyc.CompiledFunction:
    del device_index
    launch = _build_scatter_add_kernel(dtype_str)
    return flyc.compile(launch, *compile_args)


@jit_cache
def _compile_scatter_add_lds(
    dtype_str: str,
    cols_n: int,
    block_rows: int,
    block_threads: int,
    use_vector: bool,
    device_index: int,
    *,
    compile_args,
) -> flyc.CompiledFunction:
    del device_index
    launch = _build_scatter_add_lds_kernel(
        dtype_str, cols_n, block_rows, block_threads, use_vector
    )
    return flyc.compile(launch, *compile_args)


@jit_cache
def _compile_scatter_add_segmented(
    dtype_str: str,
    cols_n: int,
    block_threads: int,
    use_vector: bool,
    device_index: int,
    *,
    compile_args,
) -> flyc.CompiledFunction:
    del device_index
    launch = _build_scatter_add_segmented_kernel(
        dtype_str, cols_n, block_threads, use_vector
    )
    return flyc.compile(launch, *compile_args)


def _scatter_add_segmented_into(
    out: torch.Tensor,
    index_1d: torch.Tensor,
    src: torch.Tensor,
    rows_m: int,
    cols_n: int,
    out_rows_m: int,
    dtype_str: str,
    block_threads: int,
    use_vector: bool,
    device_index: int,
    stream: torch.cuda.Stream,
) -> bool:
    """Sort/pre-bin entries and reduce one contiguous segment per output row."""
    if rows_m >= 2**31:
        return False

    sorted_index, permutation = torch.sort(index_1d)
    boundaries = torch.arange(
        out_rows_m + 1,
        device=index_1d.device,
        dtype=torch.int64,
    )
    offsets = torch.searchsorted(sorted_index, boundaries)
    permutation.record_stream(stream)
    offsets.record_stream(stream)

    args = (
        out,
        offsets,
        permutation,
        src,
        src.stride(0),
        out.stride(0),
        out_rows_m,
        stream,
    )
    key = (
        "segmented",
        dtype_str,
        cols_n,
        block_threads,
        use_vector,
        device_index,
    )
    first_compile = key not in _COMPILED_KEYS
    if first_compile:
        out_before_compile = out.clone()
    compiled = _compile_scatter_add_segmented(
        dtype_str,
        cols_n,
        block_threads,
        use_vector,
        device_index,
        compile_args=args,
    )
    if first_compile:
        out.copy_(out_before_compile)
        _COMPILED_KEYS.add(key)
    compiled(*args)
    return True


def _lds_plan(
    rows_m: int,
    out_rows_m: int,
    cols_n: int,
    elem_bytes: int,
    block_rows: int | None = None,
    block_threads: int | None = None,
    vector_width: int = 1,
):
    """Return (block_rows, num_blocks) if owner-binning is profitable.

    V1 keeps the fp32 accumulator in registers. Bound per-thread register state
    instead of applying V0's obsolete LDS-accumulator byte limit. The redundant
    index re-scan
    (num_blocks * num_entries i64 reads) must stay cheap versus the src traffic
    (num_entries * cols_n * elem_bytes); we bound num_blocks by cols_n.
    """
    if block_rows is None or block_threads is None:
        block_rows, block_threads = _adaptive_lds_tuning(cols_n, out_rows_m)
    vector_items = _ceil_div(cols_n, vector_width)
    acc_elems_per_thread = (
        block_rows
        * _ceil_div(vector_items, block_threads)
        * vector_width
    )
    if acc_elems_per_thread > _LDS_MAX_ACC_ELEMS_PER_THREAD:
        return None
    num_blocks = (out_rows_m + block_rows - 1) // block_rows
    if num_blocks > _LDS_NUM_BLOCKS_FACTOR * cols_n * elem_bytes:
        return None
    return block_rows, num_blocks


def scatter_add_into(
    out: torch.Tensor,
    index_1d: torch.Tensor,
    src: torch.Tensor,
) -> None:
    """In-place ``out[index_1d[i], :] += src[i, :]``. 2D, inner stride 1."""
    rows_m, cols_n = src.shape
    out_rows_m = out.shape[0]
    dtype_str = _TORCH_TO_STR[src.dtype]
    elem_bytes = src.element_size()
    use_vector = _vector_io_safe(out, src, cols_n, dtype_str)
    vector_width = _VEC_BYTES // elem_bytes if use_vector else 1
    block_rows, block_threads = _adaptive_lds_tuning(cols_n, out_rows_m)
    plan = _lds_plan(
        rows_m,
        out_rows_m,
        cols_n,
        elem_bytes,
        block_rows,
        block_threads,
        vector_width,
    )

    with torch.cuda.device(out.device):
        stream = torch.cuda.current_stream()
        device_index = out.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()

        if (
            not torch.cuda.is_current_stream_capturing()
            and _segmented_eligible(rows_m, out_rows_m)
        ):
            segmented_threads = _adaptive_segmented_threads(
                cols_n, elem_bytes, use_vector
            )
            if _scatter_add_segmented_into(
                out,
                index_1d,
                src,
                rows_m,
                cols_n,
                out_rows_m,
                dtype_str,
                segmented_threads,
                use_vector,
                device_index,
                stream,
            ):
                return

        if plan is not None:
            _, num_blocks = plan
            args = (
                out,
                index_1d,
                src,
                rows_m,
                out_rows_m,
                src.stride(0),
                out.stride(0),
                num_blocks,
                stream,
            )
            key = (
                "lds",
                dtype_str,
                cols_n,
                block_rows,
                block_threads,
                use_vector,
                device_index,
            )
            first_compile = key not in _COMPILED_KEYS
            if first_compile:
                out_before_compile = out.clone()
            compiled = _compile_scatter_add_lds(
                dtype_str,
                cols_n,
                block_rows,
                block_threads,
                use_vector,
                device_index,
                compile_args=args,
            )
            if first_compile:
                out.copy_(out_before_compile)
                _COMPILED_KEYS.add(key)
            compiled(*args)
            return

        # one block per WARPS_PER_BLOCK rows (grid-stride loop covers any grid)
        grid_x = max(1, (rows_m + _WARPS_PER_BLOCK - 1) // _WARPS_PER_BLOCK)
        args = (
            out,
            index_1d,
            src,
            rows_m,
            cols_n,
            out_rows_m,
            src.stride(0),
            out.stride(0),
            grid_x,
            stream,
        )
        # restore out around each first compile's trace-time run (see _COMPILED_KEYS)
        key = (dtype_str, device_index)
        first_compile = key not in _COMPILED_KEYS
        if first_compile:
            out_before_compile = out.clone()
        compiled = _compile_scatter_add(dtype_str, device_index, compile_args=args)
        if first_compile:
            out.copy_(out_before_compile)
            _COMPILED_KEYS.add(key)
        compiled(*args)


def clear_scatter_add_cache() -> None:
    _compile_scatter_add.cache_clear()
    _compile_scatter_add_lds.cache_clear()
    _compile_scatter_add_segmented.cache_clear()
    _COMPILED_KEYS.clear()


_CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])


def scatter_add_cache_info():
    """Combined hits/misses across all scatter_add compile caches."""
    a = _compile_scatter_add.cache_info()
    b = _compile_scatter_add_lds.cache_info()
    c = _compile_scatter_add_segmented.cache_info()
    return _CacheInfo(
        a.hits + b.hits + c.hits,
        a.misses + b.misses + c.misses,
        None,
        a.currsize + b.currsize + c.currsize,
    )
