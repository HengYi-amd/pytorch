# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr


GFX950_DMA_BYTES = 16
GFX950_WAVE_SIZE = 64


def make_gfx950_wave_layout(m_waves, n_waves):
    return fx.make_layout((m_waves, n_waves, 1), (n_waves, 1, 0))


def make_row_major_swizzled_lds_layout(rows, inner_extent, swizzle):
    return fx.make_composed_layout(
        swizzle,
        fx.make_ordered_layout((rows, inner_extent), (1, 0)),
    )


def make_transposed_swizzled_lds_layout(rows, inner_extent, granule_bits):
    base_layout = fx.make_ordered_layout((rows, inner_extent), (0, 1))
    if const_expr(rows == 64):
        return fx.make_composed_layout(
            fx.static(fx.SwizzleType.get(2, granule_bits, 2)), base_layout
        )
    if const_expr(rows == 128):
        return fx.make_composed_layout(
            fx.static(fx.SwizzleType.get(2, granule_bits, 3)), base_layout
        )
    if const_expr(rows == 256):
        return fx.make_composed_layout(
            fx.static(fx.SwizzleType.get(2, granule_bits, 4)), base_layout
        )
    return base_layout


def waitcnt_barrier(vmcnt=0):
    llvm.InlineAsmOp(
        None,
        [],
        f"s_waitcnt vmcnt({vmcnt})\n\ts_barrier",
        "",
        has_side_effects=True,
    )


def waitcnt(vmcnt=0):
    llvm.InlineAsmOp(None, [], f"s_waitcnt vmcnt({vmcnt})", "", has_side_effects=True)


def waitcnt_lgkm(lgkmcnt=0):
    llvm.InlineAsmOp(
        None,
        [],
        f"s_waitcnt lgkmcnt({lgkmcnt})",
        "",
        has_side_effects=True,
    )
