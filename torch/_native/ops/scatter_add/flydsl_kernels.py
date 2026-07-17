"""FlyDSL scatter_add kernels for ROCm native overrides.

Two paths, chosen per call:

- Global-atomic (``_build_scatter_add_kernel``): one wavefront per source row,
  16B gather of ``src`` + atomic-add into ``out[index[i], :]``. fp32 adds one
  element per atomic; fp16/bf16 use packed x2 atomics. Works for any shape but
  is global-atomic-throughput bound.

- LDS owner-binning (``_build_scatter_add_lds_kernel``): each block owns a
  disjoint range of output rows held in an LDS f32 accumulator. It scans the
  index, bins matching entries via one LDS atomic on a counter, then all threads
  cooperate over the N columns (each thread owns a fixed column partition, so the
  accumulator needs no per-column atomics), and finally writes its owned rows
  back with plain (non-atomic) global stores. This removes global atomics
  entirely and accumulates in fp32 (better fp16/bf16 accuracy). It re-scans the
  index once per block, so it is only profitable when the number of owner blocks
  stays bounded (see ``_use_lds_path``); otherwise we fall back to global-atomic.
"""

# mypy: allow-untyped-defs

from collections import namedtuple

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as _fly, llvm, vector as _vector
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch, is_rdna_arch

import torch
from torch._native.flydsl_cache import jit_cache


_WARPS_PER_BLOCK = 8
_VEC_BYTES = 16

# LDS owner-binning path tunables. Each thread bins at most one match per chunk
# (chunk size == block threads), so the match list never overflows.
_LDS_BYTES = 58 * 1024  # hard per-block LDS ceiling for the fp32 accumulator
_LDS_BLOCK_THREADS = 256
# block_rows tuning (gfx950, empirical). Large n wants block_rows==1 (huge
# per-row work amortizes the redundant re-scan, and more owner blocks help);
# small n wants a few rows per block so the re-scan is not repeated too often.
# block_rows ~= min(_LDS_SMALL_N_ROWS, _LDS_ACC_ELEMS // n) captures both, then
# capped by what fits in LDS and by out_rows.
_LDS_SMALL_N_ROWS = 4
_LDS_ACC_ELEMS = 4096
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


def _lds_block_rows(n: int, out_rows_m: int) -> int:
    """Output rows a block owns (see tuning constants above)."""
    lds_max = max(1, _LDS_BYTES // (n * 4))
    want = max(1, min(_LDS_SMALL_N_ROWS, _LDS_ACC_ELEMS // n))
    return max(1, min(want, lds_max, out_rows_m))


def _build_scatter_add_lds_kernel(dtype_str: str, n: int, block_rows: int):
    block_threads = _LDS_BLOCK_THREADS
    total = block_rows * n
    cap = block_threads  # one match per thread per chunk -> never overflows
    fcls = _FLOAT_CLS[dtype_str]

    @fx.struct
    class Shared:
        acc: fx.Array[fx.Float32, total, 16]
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
        acc = s.acc.view(fx.make_layout(total, 1))
        match_lr = s.match_lr.view(fx.make_layout(cap, 1))
        match_entry = s.match_entry.view(fx.make_layout(cap, 1))
        count = s.count.view(fx.make_layout(1, 1))

        # Seed the accumulator with the block's owned output rows (scatter_add is
        # += onto an existing out; blocks own disjoint rows so this is race-free).
        i = tid
        total_c = fx.Int32(total)
        while i < total_c:
            r = i // N
            c = i - r * N
            grow = row_base + r
            if grow < out_rows_m:
                ov = buffer_ops.buffer_load(
                    out_rsrc, grow * out_row_stride + c, vec_width=1, dtype=elem_ty
                )
                fx.memref_store(fcls(ov).to(fx.Float32), acc, i)
            i = i + BT
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
                lbase = lr * N
                ebase = e * src_row_stride
                c = tid
                while c < N:
                    v = buffer_ops.buffer_load(
                        src_rsrc, ebase + c, vec_width=1, dtype=elem_ty
                    )
                    cur = fx.memref_load(acc, lbase + c)
                    fx.memref_store(cur + fcls(v).to(fx.Float32), acc, lbase + c)
                    c = c + BT
                m = m + fx.Int32(1)
            gpu.barrier()
            chunk = chunk + BT

        # Write owned rows back with plain stores (disjoint per block, no atomics).
        i = tid
        while i < total_c:
            r = i // N
            c = i - r * N
            grow = row_base + r
            if grow < out_rows_m:
                a = fx.memref_load(acc, i)
                out_v = a if dtype_str == "f32" else a.to(fcls)
                buffer_ops.buffer_store(out_v, out_rsrc, grow * out_row_stride + c)
            i = i + BT

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
    device_index: int,
    *,
    compile_args,
) -> flyc.CompiledFunction:
    del device_index
    launch = _build_scatter_add_lds_kernel(dtype_str, cols_n, block_rows)
    return flyc.compile(launch, *compile_args)


def _lds_plan(rows_m: int, out_rows_m: int, cols_n: int, elem_bytes: int):
    """Return (block_rows, num_blocks) if the LDS path is profitable, else None.

    The fp32 accumulator must fit LDS, and the redundant index re-scan
    (num_blocks * num_entries i64 reads) must stay cheap versus the src traffic
    (num_entries * cols_n * elem_bytes); we bound num_blocks by cols_n.
    """
    if cols_n * 4 > _LDS_BYTES:
        return None
    block_rows = _lds_block_rows(cols_n, out_rows_m)
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
    plan = _lds_plan(rows_m, out_rows_m, cols_n, src.element_size())

    with torch.cuda.device(out.device):
        stream = torch.cuda.current_stream()
        device_index = out.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()

        if plan is not None:
            block_rows, num_blocks = plan
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
            key = ("lds", dtype_str, cols_n, block_rows, device_index)
            first_compile = key not in _COMPILED_KEYS
            if first_compile:
                out_before_compile = out.clone()
            compiled = _compile_scatter_add_lds(
                dtype_str, cols_n, block_rows, device_index, compile_args=args
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
    _COMPILED_KEYS.clear()


_CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])


def scatter_add_cache_info():
    """Combined hits/misses across both the LDS and global-atomic compile caches."""
    a = _compile_scatter_add.cache_info()
    b = _compile_scatter_add_lds.cache_info()
    return _CacheInfo(
        a.hits + b.hits, a.misses + b.misses, None, a.currsize + b.currsize
    )
