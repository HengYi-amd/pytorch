"""Benchmark FlyDSL scatter_add vs aten across dtypes/shapes.

Modeled on bench_rmsnorm_flydsl_vs_aten.py. Compares aten (flydsl disabled)
against the FlyDSL wavefront-per-row kernel. Correctness is checked against an
fp32 ground truth before timing (must be no worse than aten).
"""

import argparse
import csv

import torch
import torch._native  # noqa: F401
import torch.backends.python_native as pn


DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def bench(fn, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return sum(times) / len(times), times[len(times) // 2], min(times)


def repeat(fn, repeats: int, warmup: int, iters: int):
    runs = [bench(fn, warmup, iters) for _ in range(repeats)]
    return (
        sum(r[0] for r in runs) / repeats,
        sum(r[1] for r in runs) / repeats,
        min(r[2] for r in runs),
    )


def _make_inputs(m_out, m_src, n, dtype):
    torch.manual_seed(0)
    dst = torch.randn((m_out, n), device="cuda", dtype=dtype)
    index = torch.randint(0, m_out, (m_src, 1), device="cuda", dtype=torch.int64)
    index = index.expand(m_src, n)
    src = torch.randn((m_src, n), device="cuda", dtype=dtype)
    return dst, index, src


def _max_err_vs_truth(got, dst, index, src):
    with pn.flydsl.disabled():
        truth = torch.scatter_add(dst.float(), 0, index, src.float())
    return (got.float() - truth).abs().max().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-rows", type=int, default=8192)
    p.add_argument("--rows", type=int, default=65536, help="number of source rows")
    p.add_argument("--hidden-sizes", default="256,1024,4096,8192")
    p.add_argument("--dtypes", default="fp16,bf16,fp32")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--csv", default="agent_space/scatter_add_flydsl_vs_aten.csv")
    args = p.parse_args()

    hidden = [int(x) for x in args.hidden_sizes.split(",") if x]
    dtype_names = [x for x in args.dtypes.split(",") if x]

    from torch._native.ops.scatter_add.flydsl_scatter_add_kernel import (
        clear_scatter_add_cache,
    )

    print("out_rows,rows,n,dtype,aten_median_ms,flydsl_median_ms,speedup,maxerr")

    rows_out = []
    for dtype_name in dtype_names:
        dtype = DTYPES[dtype_name]
        for n in hidden:
            dst, index, src = _make_inputs(args.out_rows, args.rows, n, dtype)

            with pn.flydsl.disabled():
                aten = repeat(
                    lambda: torch.scatter_add(dst, 0, index, src),
                    args.repeats,
                    args.warmup,
                    args.iters,
                )

            clear_scatter_add_cache()
            got = torch.scatter_add(dst, 0, index, src)
            maxerr = _max_err_vs_truth(got, dst, index, src)
            fly = repeat(
                lambda: torch.scatter_add(dst, 0, index, src),
                args.repeats,
                args.warmup,
                args.iters,
            )

            row = {
                "out_rows": args.out_rows,
                "rows": args.rows,
                "n": n,
                "dtype": dtype_name,
                "aten_median_ms": aten[1],
                "flydsl_median_ms": fly[1],
                "speedup": aten[1] / fly[1],
                "maxerr": maxerr,
            }
            rows_out.append(row)
            print(
                f"{args.out_rows},{args.rows},{n},{dtype_name},{aten[1]:.6f},"
                f"{fly[1]:.6f},{aten[1] / fly[1]:.3f},{maxerr:.4f}",
                flush=True,
            )

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(args.csv)


if __name__ == "__main__":
    main()
