"""Lazy exports for optional FlyDSL vendored kernels."""

import importlib
from typing import Any


__all__ = [
    "MXFP4GemmParams",
    "make_mxfp4_param_and_validate",
    "make_mxfp4_scaled_mm_gfx950",
    "mxfp4_gemm_derived",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(f"{__name__}.gemm_mxfp4_gfx950")
    value = getattr(module, name)
    globals()[name] = value
    return value
