# Owner(s): ["module: dsl-native-ops"]

import unittest

import torch
import torch.backends.python_native as pn
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def _flydsl_scatter_add_registered() -> bool:
    try:
        ops = set(pn.get_dsl_operations("flydsl"))
        return {"scatter_add", "scatter_add.out", "scatter_add_"}.issubset(ops)
    except Exception:
        return False


_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@unittest.skipUnless(TEST_CUDA and torch.version.hip is not None, "ROCm required")
@unittest.skipUnless(
    _flydsl_scatter_add_registered(), "FlyDSL scatter_add override not registered"
)
class TestFlyDSLScatterAdd(TestCase):
    def setUp(self):
        super().setUp()
        from torch._native.ops.scatter_add.flydsl_scatter_add_kernel import (
            clear_scatter_add_cache,
        )

        clear_scatter_add_cache()
        torch.manual_seed(0)

    def _make_inputs(self, dtype):
        dst = torch.randn((128, 64), device="cuda", dtype=dtype)
        index = torch.randint(
            0, dst.shape[0], (512, 1), device="cuda", dtype=torch.int64
        )
        index = index.expand(512, dst.shape[1])
        src = torch.randn((512, dst.shape[1]), device="cuda", dtype=dtype)
        return dst, index, src

    def _assert_no_worse_than_aten(self, got, aten, dst, index, src):
        # scatter_add accumulates in the working dtype, so fp16/bf16 results
        # legitimately differ from aten by rounding/ordering. Compare both to an
        # fp32 ground truth and require the FlyDSL kernel to be no less accurate.
        with pn.flydsl.disabled():
            truth = torch.scatter_add(dst.float(), 0, index, src.float())
        got_err = (got.float() - truth).abs().max().item()
        aten_err = (aten.float() - truth).abs().max().item()
        self.assertLessEqual(got_err, aten_err * 1.5 + 1e-4)

    @parametrize("dtype", list(_DTYPES))
    def test_scatter_add_matches_aten_and_uses_cache(self, dtype):
        from torch._native.ops.scatter_add.flydsl_scatter_add_kernel import (
            scatter_add_cache_info,
        )

        torch_dtype = _DTYPES[dtype]
        dst, index, src = self._make_inputs(torch_dtype)
        with pn.flydsl.disabled():
            ref = torch.scatter_add(dst, 0, index, src)

        got = torch.scatter_add(dst, 0, index, src)
        self._assert_no_worse_than_aten(got, ref, dst, index, src)

        info = scatter_add_cache_info()
        self.assertEqual(info.misses, 1)
        self.assertGreaterEqual(info.hits, 0)

    @parametrize("dtype", list(_DTYPES))
    def test_scatter_add_out_and_inplace_match_aten(self, dtype):
        torch_dtype = _DTYPES[dtype]
        dst, index, src = self._make_inputs(torch_dtype)

        out = torch.empty_like(dst)
        with pn.flydsl.disabled():
            ref = torch.scatter_add(dst, 0, index, src)
        torch.scatter_add(dst, 0, index, src, out=out)
        self._assert_no_worse_than_aten(out, ref, dst, index, src)

        inplace = dst.clone()
        with pn.flydsl.disabled():
            ref_inplace = dst.clone()
            ref_inplace.scatter_add_(0, index, src)
        inplace.scatter_add_(0, index, src)
        self._assert_no_worse_than_aten(inplace, ref_inplace, dst, index, src)

    def test_unsupported_dtype_falls_back_without_compiling(self):
        from torch._native.ops.scatter_add.flydsl_scatter_add_kernel import (
            scatter_add_cache_info,
        )

        # float64 is not in the FlyDSL supported set -> must fall back to aten.
        dst = torch.randn((128, 64), device="cuda", dtype=torch.float64)
        index = torch.randint(
            0, dst.shape[0], (512, 1), device="cuda", dtype=torch.int64
        )
        index = index.expand(512, dst.shape[1])
        src = torch.randn((512, dst.shape[1]), device="cuda", dtype=torch.float64)

        with pn.flydsl.disabled():
            ref = torch.scatter_add(dst, 0, index, src)
        got = torch.scatter_add(dst, 0, index, src)
        torch.testing.assert_close(got, ref)
        self.assertEqual(scatter_add_cache_info().misses, 0)

    def test_non_vectorizable_n_falls_back_without_compiling(self):
        from torch._native.ops.scatter_add.flydsl_scatter_add_kernel import (
            scatter_add_cache_info,
        )

        # N=60 is not a multiple of vec_elems (fp16 needs %8) -> fall back.
        dst = torch.randn((128, 60), device="cuda", dtype=torch.float16)
        index = torch.randint(
            0, dst.shape[0], (512, 1), device="cuda", dtype=torch.int64
        )
        index = index.expand(512, dst.shape[1])
        src = torch.randn((512, dst.shape[1]), device="cuda", dtype=torch.float16)

        with pn.flydsl.disabled():
            ref = torch.scatter_add(dst, 0, index, src)
        got = torch.scatter_add(dst, 0, index, src)
        torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)
        self.assertEqual(scatter_add_cache_info().misses, 0)


instantiate_parametrized_tests(TestFlyDSLScatterAdd)


if __name__ == "__main__":
    run_tests()
