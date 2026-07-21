"""FlyDSL override registrations for ``aten::scatter_add`` on ROCm."""

# mypy: allow-untyped-defs

import functools

import torch
from torch._tensor_iterator import TensorIterator

from ... import flydsl_utils as fu
from ...registry import _OpCondFn, _OpImplFn


_SUPPORTED_DTYPES = {torch.float32, torch.float16, torch.bfloat16}
# 16B vectorized gather of src (LDG.128-equivalent buffer load) needs src base +
# row stride 16B-aligned. Paired x2 / scalar atomics into out need only the 4B
# atomic operand naturally aligned on base + row stride.
_SRC_ALIGN_BYTES = 16
_DST_ALIGN_BYTES = 4
_VEC_BYTES = 16


def _vec_elems(dtype: torch.dtype) -> int:
    return _VEC_BYTES // dtype.itemsize


@functools.cache
def _runtime_available() -> bool:
    return fu.runtime_available()


def _deterministic() -> bool:
    return torch.are_deterministic_algorithms_enabled()


def _any_cow(*tensors: torch.Tensor) -> bool:
    return any(
        torch._C._is_cow_tensor(t)  # pyrefly: ignore[missing-attribute]
        for t in tensors
    )


def _base_cond_ok(*tensors: torch.Tensor) -> bool:
    if not _runtime_available() or torch.version.hip is None:
        return False
    if _deterministic():
        return False
    if not all(t.is_cuda for t in tensors):
        return False
    if _any_cow(*tensors):
        return False
    return True


def _normalize_dim(dim: int, ndim: int) -> int:
    return dim + ndim if dim < 0 else dim


def _scatter_add_eligibility(
    self: torch.Tensor, d: int, index: torch.Tensor, src: torch.Tensor
) -> TensorIterator | None:
    if self.dtype not in _SUPPORTED_DTYPES or self.dtype != src.dtype:
        return None
    if index.dtype not in (torch.int32, torch.int64):
        return None
    if self.ndim != src.ndim or self.ndim != index.ndim or self.ndim == 0:
        return None
    if not 0 <= d < self.ndim:
        return None

    self_strides = list(self.stride())
    self_strides[d] = 0
    try:
        self_r = self.as_strided(index.shape, self_strides)
        src_r = src.as_strided(index.shape, src.stride())
        it = TensorIterator(
            outputs=[self_r],
            const_inputs=[src_r, index],
            check_mem_overlap=False,
            check_all_same_dtype=False,
            resize_outputs=False,
        )
    except RuntimeError:
        return None

    if it.ndim != 2:
        return None
    elem = self.element_size()
    idx_elem = index.element_size()
    s_self, s_src, s_idx = it.strides(0), it.strides(1), it.strides(2)
    if (
        s_idx[0] == 0
        and s_idx[1] == idx_elem
        and s_self[0] == elem
        and s_src[0] == elem
        and s_self[1] == 0
    ):
        return it
    return None


def _alignment_ok(dst: torch.Tensor, src: torch.Tensor, dim: int) -> bool:
    elem = dst.element_size()
    return (
        src.data_ptr() % _SRC_ALIGN_BYTES == 0
        and (src.stride(dim) * elem) % _SRC_ALIGN_BYTES == 0
        and dst.data_ptr() % _DST_ALIGN_BYTES == 0
        and (dst.stride(dim) * elem) % _DST_ALIGN_BYTES == 0
    )


def _is_supported(
    self: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor
) -> bool:
    d = _normalize_dim(dim, self.ndim)
    it = _scatter_add_eligibility(self, d, index, src)
    if it is None:
        return False
    # Vectorized 16B gather requires the coalesced inner dim (N == it.shape[0])
    # to be a whole number of vec_elems chunks (fp32: %4, fp16/bf16: %8).
    if it.shape[0] % _vec_elems(self.dtype) != 0:
        return False
    return _alignment_ok(self, src, d)


def _prepare_kernel_inputs(
    self: torch.Tensor,
    d: int,
    index: torch.Tensor,
    src: torch.Tensor,
    it: TensorIterator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, m_src = it.shape[0], it.shape[1]
    m_self = self.shape[d]
    self_2d = self.as_strided((m_self, n), (self.stride(d), 1))
    src_2d = src.as_strided((m_src, n), (src.stride(d), 1))
    index_1d = index.as_strided((m_src,), (1,))
    if index_1d.dtype != torch.int64:
        index_1d = index_1d.to(torch.int64)
    return self_2d, index_1d, src_2d


def _cond(self, dim, index, src, *args, **kwargs):
    return _base_cond_ok(self, index, src) and _is_supported(self, int(dim), index, src)


def _out_cond(self, dim, index, src, *, out):
    if not _base_cond_ok(self, index, src, out):
        return False
    if out.dtype != self.dtype or out.shape != self.shape or out.ndim == 0:
        return False
    return _is_supported(out, int(dim), index, src)


def _copy_if_distinct(out: torch.Tensor, self: torch.Tensor) -> None:
    if out.data_ptr() != self.data_ptr():
        out.copy_(self)


def _kernel():
    from .flydsl_scatter_add_kernel import scatter_add_into

    return scatter_add_into


def _run(dst: torch.Tensor, dim: int, index, src) -> torch.Tensor:
    d = _normalize_dim(dim, dst.ndim)
    it = _scatter_add_eligibility(dst, d, index, src)
    if it is None:
        raise RuntimeError("scatter_add flydsl: cond approved but iter rebuild failed")
    if it.numel == 0:
        return dst
    dst_2d, index_1d, src_2d = _prepare_kernel_inputs(dst, d, index, src, it)
    _kernel()(dst_2d, index_1d, src_2d)
    return dst


def _impl(self, dim, index, src, *args, **kwargs):
    return _run(self.clone(), int(dim), index, src)


def _out_impl(self, dim, index, src, *, out):
    _copy_if_distinct(out, self)
    return _run(out, int(dim), index, src)


def _inplace_impl(self, dim, index, src, *args, **kwargs):
    return _run(self, int(dim), index, src)


def _register_one(op_symbol: str, cond: _OpCondFn, impl: _OpImplFn) -> None:
    fu.register_op_override(
        "aten",
        op_symbol,
        "CUDA",
        cond=cond,
        impl=impl,
        allow_multiple_override=True,
    )


def register_to_dispatch() -> None:
    for op_symbol, cond, impl in (
        ("scatter_add", _cond, _impl),
        ("scatter_add.out", _out_cond, _out_impl),
        ("scatter_add_", _cond, _inplace_impl),
    ):
        _register_one(op_symbol, cond, impl)
