# Owner(s): ["module: dsl-native-ops"]

import os
import unittest

import torch
import torch.backends.python_native as pn
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


EPS = 1e-5
M_VALUES = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
N_VALUES = (4096, 8192, 12288, 16384, 32768)
FULL_CASES = tuple((m, n) for n in N_VALUES for m in M_VALUES)
SMOKE_CASES = tuple((128, n) for n in N_VALUES)
RUN_FULL_MATRIX = os.environ.get("PYTORCH_FLYDSL_RMSNORM_BWD_FULL_MATRIX") == "1"
TEST_CASES = FULL_CASES if RUN_FULL_MATRIX else SMOKE_CASES


def _flydsl_bwd_registered() -> bool:
    try:
        return "_fused_rms_norm_backward" in pn.get_dsl_operations("flydsl")
    except Exception:
        return False


def _cache_counts() -> dict[str, int]:
    from torch._native.ops.norm.flydsl_rmsnorm_bwd_kernel import (
        rmsnorm_bwd_cache_info,
    )

    info = rmsnorm_bwd_cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "currsize": info.currsize,
    }


def _route_delta(before, after) -> int:
    return (
        after["hits"]
        + after["misses"]
        - before["hits"]
        - before["misses"]
    )


def _shared_rstd(x: torch.Tensor) -> torch.Tensor:
    """Create one FP32 rstd fixture without invoking any RMSNorm FWD path."""

    return torch.rsqrt(
        x.float().square().mean(dim=-1, keepdim=True) + EPS
    ).contiguous()


def _tolerance(dtype: torch.dtype, result: str) -> tuple[float, float]:
    if result == "dweight":
        return {
            torch.float32: (1e-4, 1e-2),
            torch.float16: (3e-2, 2e-1),
            torch.bfloat16: (1e-1, 1.0),
        }[dtype]
    return {
        torch.float32: (1e-4, 1e-3),
        torch.float16: (3e-2, 3e-2),
        torch.bfloat16: (1e-1, 2e-1),
    }[dtype]


def _case_name(case: tuple[int, int]) -> str:
    return f"M{case[0]}_N{case[1]}"


@unittest.skipUnless(TEST_CUDA and torch.version.hip is not None, "ROCm required")
@unittest.skipUnless(
    _flydsl_bwd_registered(), "FlyDSL RMSNorm backward is not registered"
)
class TestFlyDSLRMSNormBwd(TestCase):
    def _make_inputs(self, device, m, n, dtype):
        torch.manual_seed(m + n)
        x = torch.randn((m, n), device=device, dtype=dtype)
        weight = torch.randn((n,), device=device, dtype=dtype)
        grad_out = torch.randn((m, n), device=device, dtype=dtype)
        return x, weight, grad_out

    def _assert_close(self, actual, expected, dtype, result):
        rtol, atol = _tolerance(dtype, result)
        self.assertEqual(actual, expected, rtol=rtol, atol=atol)

    def _call_bwd(self, grad_out, x, rstd, weight, output_mask):
        return torch.ops.aten._fused_rms_norm_backward.default(
            grad_out,
            x,
            [x.shape[-1]],
            rstd,
            weight,
            output_mask,
        )

    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize("case", TEST_CASES, name_fn=_case_name)
    def test_backward_matches_aten_with_shared_rstd(self, device, dtype, case):
        m, n = case
        x, weight, grad_out = self._make_inputs(device, m, n, dtype)
        with torch.inference_mode():
            rstd = _shared_rstd(x)

        before = _cache_counts()
        with torch.inference_mode(), pn.flydsl.disabled():
            ref_dx, ref_dw = self._call_bwd(
                grad_out, x, rstd, weight, [True, True]
            )
        after_native = _cache_counts()
        self.assertEqual(_route_delta(before, after_native), 0)

        with torch.inference_mode():
            got_dx, got_dw = self._call_bwd(
                grad_out, x, rstd, weight, [True, True]
            )
        after_flydsl = _cache_counts()
        self.assertEqual(_route_delta(after_native, after_flydsl), 1)
        self.assertEqual(got_dx.dtype, dtype)
        self.assertEqual(got_dw.dtype, dtype)
        self._assert_close(got_dx, ref_dx, dtype, "dx")
        self._assert_close(got_dw, ref_dw, dtype, "dweight")

    def test_backward_rezeros_atomic_buffer_and_honors_masks(self, device):
        m, n, dtype = 128, 12288, torch.float16
        x, weight, grad_out = self._make_inputs(device, m, n, dtype)
        with torch.inference_mode():
            rstd = _shared_rstd(x)
        with torch.inference_mode(), pn.flydsl.disabled():
            ref_dx, ref_dw = self._call_bwd(
                grad_out, x, rstd, weight, [True, True]
            )

        for _ in range(2):
            before = _cache_counts()
            with torch.inference_mode():
                got_dx, got_dw = self._call_bwd(
                    grad_out, x, rstd, weight, [True, True]
                )
            self.assertEqual(_route_delta(before, _cache_counts()), 1)
            self._assert_close(got_dx, ref_dx, dtype, "dx")
            self._assert_close(got_dw, ref_dw, dtype, "dweight")

        before = _cache_counts()
        with torch.inference_mode():
            dx_only, missing_dw = self._call_bwd(
                grad_out, x, rstd, weight, [True, False]
            )
            missing_dx, dw_only = self._call_bwd(
                grad_out, x, rstd, weight, [False, True]
            )
        self.assertEqual(_route_delta(before, _cache_counts()), 2)
        self.assertIsNone(missing_dw)
        self.assertIsNone(missing_dx)
        self._assert_close(dx_only, ref_dx, dtype, "dx")
        self._assert_close(dw_only, ref_dw, dtype, "dweight")

    def test_unsupported_m_falls_back_to_aten(self, device):
        m, n, dtype = 127, 4096, torch.float16
        x, weight, grad_out = self._make_inputs(device, m, n, dtype)
        with torch.inference_mode():
            rstd = _shared_rstd(x)
        with torch.inference_mode(), pn.flydsl.disabled():
            ref_dx, ref_dw = self._call_bwd(
                grad_out, x, rstd, weight, [True, True]
            )
        before = _cache_counts()
        with torch.inference_mode():
            got_dx, got_dw = self._call_bwd(
                grad_out, x, rstd, weight, [True, True]
            )
        self.assertEqual(_route_delta(before, _cache_counts()), 0)
        self._assert_close(got_dx, ref_dx, dtype, "dx")
        self._assert_close(got_dw, ref_dw, dtype, "dweight")


instantiate_device_type_tests(
    TestFlyDSLRMSNormBwd,
    globals(),
    only_for="cuda",
)


if __name__ == "__main__":
    run_tests()

