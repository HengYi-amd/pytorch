#!/usr/bin/env python3
"""BWD-only correctness and wall-to-sync benchmark for FlyDSL RMSNorm."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import torch
import torch.backends.python_native as pn

DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}
M_VALUES = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
N_VALUES = (4096, 8192, 12288, 16384)
SOURCE_PATHS = (
    "benchmarks/flydsl_rmsnorm/benchmark_bwd.py",
    "test/python_native/test_flydsl_registry.py",
    "test/python_native/test_rmsnorm_flydsl_bwd.py",
    "torch/_native/__init__.py",
    "torch/_native/flydsl_cache.py",
    "torch/_native/flydsl_utils.py",
    "torch/_native/ops/norm/__init__.py",
    "torch/_native/ops/norm/flydsl_rmsnorm_bwd_kernel.py",
    "torch/_native/ops/norm/flydsl_rmsnorm_bwd_impl.py",
)
CSV_COLUMNS = (
    "m",
    "n",
    "dtype",
    "flydsl_wall_to_sync_ms",
    "aten_wall_to_sync_ms",
    "speedup",
    "flydsl_p10_ms",
    "flydsl_p90_ms",
    "flydsl_cv_percent",
    "aten_p10_ms",
    "aten_p90_ms",
    "aten_cv_percent",
    "correctness",
    "rounds",
    "samples_per_block",
    "inner_iters",
    "effective_samples_per_implementation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--m-values",
        default=",".join(str(value) for value in M_VALUES),
        help="Comma-separated subset of the fixed M manifest.",
    )
    parser.add_argument(
        "--n-values",
        default=",".join(str(value) for value in N_VALUES),
        help="Comma-separated subset of the fixed N manifest.",
    )
    parser.add_argument(
        "--dtypes",
        default=",".join(DTYPES),
        help="Comma-separated subset of fp16,bf16,fp32.",
    )
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument(
        "--inner-iters",
        type=int,
        default=1,
        help=(
            "Calls per synchronized sample. Keep the default 1 for literal "
            "single-call wall-to-sync latency."
        ),
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=4,
        help="Positive even number of ABBA/BAAB rounds.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        default="artifacts/rmsnorm_bwd_benchmark",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed case. Progress is still checkpointed.",
    )
    return parser.parse_args()


def parse_int_values(value: str, allowed: tuple[int, ...], label: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"--{label} must select at least one value")
    if len(values) != len(set(values)):
        raise ValueError(f"--{label} contains duplicate values: {values}")
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(
            f"--{label} contains values outside the fixed manifest: {sorted(unknown)}"
        )
    return values


def parse_dtypes(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError("--dtypes must select at least one dtype")
    if len(names) != len(set(names)):
        raise ValueError(f"--dtypes contains duplicates: {names}")
    unknown = set(names) - set(DTYPES)
    if unknown:
        raise ValueError(f"unknown dtypes: {sorted(unknown)}")
    return names


def make_manifest(
    m_values: list[int], n_values: list[int], dtype_names: list[str]
) -> list[dict[str, object]]:
    return [
        {"m": m, "n": n, "dtype": dtype_name}
        for dtype_name in dtype_names
        for m in m_values
        for n in n_values
    ]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize_ms(samples: list[float]) -> dict[str, object]:
    mean = statistics.fmean(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": mean,
        "p10_ms": percentile(samples, 0.10),
        "p90_ms": percentile(samples, 0.90),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stdev_ms": stdev,
        "cv_percent": 100.0 * stdev / mean if mean else 0.0,
        "samples_ms": samples,
    }


def measurement_schedule(round_index: int) -> tuple[str, ...]:
    if round_index % 2 == 0:
        return ("aten", "flydsl", "flydsl", "aten")
    return ("flydsl", "aten", "aten", "flydsl")


def implementation_context(name: str):
    return pn.flydsl.disabled() if name == "aten" else nullcontext()


def make_shared_rstd(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute the shared FP32 rstd fixture outside the timed region."""

    return torch.rsqrt(
        x.float().square().mean(dim=-1, keepdim=True) + eps
    ).contiguous()


def fused_bwd(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor,
):
    return torch.ops.aten._fused_rms_norm_backward.default(
        grad_out,
        x,
        [x.shape[-1]],
        rstd,
        weight,
        [True, True],
    )


def cache_counters() -> dict[str, int]:
    from torch._native.ops.norm.flydsl_rmsnorm_bwd_kernel import (
        rmsnorm_bwd_cache_info,
    )

    info = rmsnorm_bwd_cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "currsize": info.currsize,
    }


def route_delta(before, after) -> int:
    return (
        after["hits"]
        + after["misses"]
        - before["hits"]
        - before["misses"]
    )


def tolerances(dtype: torch.dtype, result: str) -> tuple[float, float]:
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


def assert_result_chunked(
    actual: torch.Tensor,
    expected: torch.Tensor,
    dtype: torch.dtype,
    result: str,
) -> dict[str, object]:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{result} shape mismatch: {actual.shape} != {expected.shape}"
        )
    if actual.dtype != expected.dtype:
        raise AssertionError(
            f"{result} dtype mismatch: {actual.dtype} != {expected.dtype}"
        )

    rtol, atol = tolerances(dtype, result)
    actual_flat = actual.detach().reshape(-1)
    expected_flat = expected.detach().reshape(-1)
    chunk_elements = 8 * 1024 * 1024
    max_abs_error = 0.0
    max_rel_error = 0.0
    violations = 0
    for start in range(0, actual_flat.numel(), chunk_elements):
        stop = min(start + chunk_elements, actual_flat.numel())
        actual_f = actual_flat[start:stop].float()
        expected_f = expected_flat[start:stop].float()
        abs_error = (actual_f - expected_f).abs()
        close = torch.isclose(
            actual_f,
            expected_f,
            rtol=rtol,
            atol=atol,
            equal_nan=False,
        )
        violations += torch.count_nonzero(~close).item()
        max_abs_error = max(max_abs_error, abs_error.max().item())
        denominator = expected_f.abs().clamp_min(1e-12)
        max_rel_error = max(
            max_rel_error,
            (abs_error / denominator).max().item(),
        )
    if violations:
        raise AssertionError(
            f"{result} has {violations} values outside rtol={rtol}, atol={atol}; "
            f"max_abs_error={max_abs_error}, max_rel_error={max_rel_error}"
        )
    return {
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "rtol": rtol,
        "atol": atol,
        "checked_elements": actual_flat.numel(),
    }


def correctness_case(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_out: torch.Tensor,
    eps: float,
) -> tuple[dict[str, object], torch.Tensor]:
    with torch.inference_mode():
        shared_rstd = make_shared_rstd(x, eps)

    before_native = cache_counters()
    with torch.inference_mode(), pn.flydsl.disabled():
        ref_dx, ref_dw = fused_bwd(grad_out, x, shared_rstd, weight)
    after_native = cache_counters()
    if route_delta(before_native, after_native) != 0:
        raise RuntimeError("ATen reference unexpectedly used FlyDSL BWD")

    with torch.inference_mode():
        got_dx, got_dw = fused_bwd(grad_out, x, shared_rstd, weight)
    after_flydsl = cache_counters()
    bwd_routes = route_delta(after_native, after_flydsl)
    if bwd_routes != 1:
        raise RuntimeError(
            f"FlyDSL BWD correctness route count was {bwd_routes}, expected 1"
        )

    details: dict[str, object] = {
        "passed": True,
        "shared_rstd_source": (
            "torch.rsqrt(mean(x.float() ** 2) + eps), outside timing"
        ),
        "flydsl_bwd_route_count": bwd_routes,
    }
    try:
        details["grad_input"] = assert_result_chunked(
            got_dx,
            ref_dx,
            x.dtype,
            "dx",
        )
        details["grad_weight"] = assert_result_chunked(
            got_dw,
            ref_dw,
            x.dtype,
            "dweight",
        )
    except AssertionError as exc:
        details.update(
            passed=False,
            error=repr(exc),
            traceback=traceback.format_exc(),
        )
    return details, shared_rstd


def measure_wall_to_sync_ms(
    fn: Callable[[], object],
    *,
    samples: int,
    inner_iters: int,
    device: torch.device,
) -> list[float]:
    values = []
    for _ in range(samples):
        torch.cuda.synchronize(device)
        start_ns = time.perf_counter_ns()
        for _ in range(inner_iters):
            fn()
        torch.cuda.synchronize(device)
        end_ns = time.perf_counter_ns()
        values.append((end_ns - start_ns) / 1_000_000.0 / inner_iters)
    return values


def benchmark_bwd_case(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_out: torch.Tensor,
    shared_rstd: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, object]:
    fn = lambda: fused_bwd(grad_out, x, shared_rstd, weight)
    cache_before = cache_counters()
    warmup_validation = []
    for implementation in ("aten", "flydsl"):
        before = cache_counters()
        with torch.inference_mode(), implementation_context(implementation):
            for _ in range(args.warmup):
                fn()
            torch.cuda.synchronize(x.device)
        after = cache_counters()
        observed_bwd = route_delta(before, after)
        expected_bwd = args.warmup if implementation == "flydsl" else 0
        if observed_bwd != expected_bwd:
            raise RuntimeError(
                f"{implementation} warmup routes: bwd={observed_bwd} "
                f"(expected {expected_bwd})"
            )
        warmup_validation.append(
            {
                "implementation": implementation,
                "bwd_expected": expected_bwd,
                "bwd_observed": observed_bwd,
            }
        )

    aggregate = {"aten": [], "flydsl": []}
    schedules = []
    blocks = []
    flydsl_timed_calls = 0
    for round_index in range(args.rounds):
        schedule = measurement_schedule(round_index)
        schedules.append(schedule)
        for block_index, implementation in enumerate(schedule):
            before = cache_counters()
            with torch.inference_mode(), implementation_context(implementation):
                samples_ms = measure_wall_to_sync_ms(
                    fn,
                    samples=args.samples,
                    inner_iters=args.inner_iters,
                    device=x.device,
                )
            after = cache_counters()
            observed_bwd = route_delta(before, after)
            expected_bwd = (
                args.samples * args.inner_iters
                if implementation == "flydsl"
                else 0
            )
            if observed_bwd != expected_bwd:
                raise RuntimeError(
                    f"{implementation} timed routes: bwd={observed_bwd} "
                    f"(expected {expected_bwd})"
                )
            if implementation == "flydsl":
                flydsl_timed_calls += args.samples * args.inner_iters
            aggregate[implementation].extend(samples_ms)
            blocks.append(
                {
                    "round": round_index,
                    "block": block_index,
                    "implementation": implementation,
                    "bwd_expected": expected_bwd,
                    "bwd_observed": observed_bwd,
                    "wall_to_sync": summarize_ms(samples_ms),
                }
            )

    cache_after = cache_counters()
    expected_total = args.warmup + flydsl_timed_calls
    observed_total = route_delta(cache_before, cache_after)
    if observed_total != expected_total:
        raise RuntimeError(
            f"overall routes: bwd={observed_total} (expected {expected_total})"
        )

    aten_stats = summarize_ms(aggregate["aten"])
    flydsl_stats = summarize_ms(aggregate["flydsl"])
    speedup = aten_stats["median_ms"] / flydsl_stats["median_ms"]
    return {
        "measurement": "wall_to_sync",
        "unit": "ms",
        "schedule": schedules,
        "blocks": blocks,
        "warmup_route_validation": warmup_validation,
        "route_validation": {
            "bwd_expected": expected_total,
            "bwd_observed": observed_total,
        },
        "aten": aten_stats,
        "flydsl": flydsl_stats,
        "aten_wall_to_sync_ms": aten_stats["median_ms"],
        "flydsl_wall_to_sync_ms": flydsl_stats["median_ms"],
        "speedup": speedup,
    }


def git_text(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=30,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unknown"


def environment(device: torch.device, output_dir: Path) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    props = torch.cuda.get_device_properties(device)
    try:
        flydsl_version = importlib.metadata.version("flydsl")
    except importlib.metadata.PackageNotFoundError:
        flydsl_version = "not-installed"
    source_sha = git_text(repo_root, "rev-parse", "HEAD")
    built_sha = getattr(torch.version, "git_version", None)
    build_matches = (
        None
        if not built_sha or source_sha == "unknown"
        else built_sha.startswith(source_sha) or source_sha.startswith(built_sha)
    )
    expected_sources = {
        "torch": (repo_root / "torch/__init__.py").resolve(),
        "bwd_impl": (
            repo_root
            / "torch/_native/ops/norm/flydsl_rmsnorm_bwd_impl.py"
        ).resolve(),
        "bwd_kernel": (
            repo_root
            / "torch/_native/ops/norm/flydsl_rmsnorm_bwd_kernel.py"
        ).resolve(),
    }
    loaded_sources = {"torch": Path(torch.__file__).resolve()}
    for label, module_name in (
        ("bwd_impl", "torch._native.ops.norm.flydsl_rmsnorm_bwd_impl"),
        ("bwd_kernel", "torch._native.ops.norm.flydsl_rmsnorm_bwd_kernel"),
    ):
        spec = importlib.util.find_spec(module_name)
        loaded_sources[label] = (
            Path(spec.origin).resolve()
            if spec is not None and spec.origin is not None
            else None
        )
    sources_match = all(
        loaded_sources[label] is not None
        and os.path.normcase(str(loaded_sources[label]))
        == os.path.normcase(str(expected_path))
        for label, expected_path in expected_sources.items()
    )
    disk = shutil.disk_usage(output_dir.parent)
    visibility = {
        name: os.environ.get(name)
        for name in (
            "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL",
            "FLYDSL_RUNTIME_CACHE_DIR",
        )
    }
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "command": [sys.executable, *sys.argv],
        "benchmark_script": str(Path(__file__).resolve()),
        "torch_file": torch.__file__,
        "torch_version": torch.__version__,
        "torch_git_version": built_sha,
        "torch_source_sha": source_sha,
        "build_matches_checkout": build_matches,
        "loaded_source_paths": {
            label: str(path) if path is not None else None
            for label, path in loaded_sources.items()
        },
        "expected_source_paths": {
            label: str(path) for label, path in expected_sources.items()
        },
        "loaded_sources_match_checkout": sources_match,
        "torch_branch": git_text(repo_root, "branch", "--show-current"),
        "flydsl_version": flydsl_version,
        "hip_version": torch.version.hip,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_arch": getattr(props, "gcnArchName", "unknown"),
        "gpu_total_memory_bytes": props.total_memory,
        "visibility_environment": visibility,
        "flydsl_operations": pn.get_dsl_operations("flydsl"),
        "output_filesystem_free_bytes_before": disk.free,
    }


def capture_gpu_state() -> dict[str, object]:
    commands = (
        ("amd_smi_metric", ["amd-smi", "metric", "--gpu", "all"]),
        ("amd_smi_process", ["amd-smi", "process", "--gpu", "all", "--general"]),
        ("rocm_smi", ["rocm-smi"]),
        ("rocm_smi_pids", ["rocm-smi", "--showpids"]),
    )
    results: dict[str, object] = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "commands": {},
    }
    command_results = results["commands"]
    assert isinstance(command_results, dict)
    for name, command in commands:
        if shutil.which(command[0]) is None:
            continue
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=15,
                check=False,
            )
            command_results[name] = {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        except (OSError, subprocess.TimeoutExpired) as exc:
            command_results[name] = {"command": command, "error": repr(exc)}
    return results


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def snapshot_sources(output_dir: Path) -> list[dict[str, str]]:
    repo_root = Path(__file__).resolve().parents[2]
    snapshot_root = output_dir / "source_snapshot"
    entries = []
    for relative in SOURCE_PATHS:
        source = repo_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"required source file is missing: {source}")
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        entries.append(
            {
                "relative_path": relative,
                "source_path": str(source),
                "snapshot_path": str(destination),
                "sha256": digest,
            }
        )

    atomic_write_text(
        output_dir / "source_manifest.json",
        json.dumps(entries, indent=2, sort_keys=True),
    )
    checksum_lines = [
        f"{entry['sha256']}  {entry['relative_path']}" for entry in entries
    ]
    atomic_write_text(
        output_dir / "source_SHA256SUMS.txt",
        "\n".join(checksum_lines) + "\n",
    )
    atomic_write_text(
        output_dir / "git_head.txt",
        git_text(repo_root, "rev-parse", "HEAD") + "\n",
    )
    atomic_write_text(
        output_dir / "git_status_relevant.txt",
        git_text(repo_root, "status", "--short", "--", *SOURCE_PATHS) + "\n",
    )
    atomic_write_text(
        output_dir / "git_diff_relevant.patch",
        git_text(repo_root, "diff", "--binary", "--", *SOURCE_PATHS) + "\n",
    )
    return entries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    writer.writerows({key: row.get(key) for key in CSV_COLUMNS} for row in rows)
    atomic_write_text(path, buffer.getvalue())


def write_report(
    path: Path,
    payload: dict[str, object],
    rows: list[dict[str, object]],
) -> None:
    config = payload["config"]
    environment_data = payload["environment"]
    assert isinstance(config, dict)
    assert isinstance(environment_data, dict)
    lines = [
        "# FlyDSL RMSNorm BWD Wall-to-Sync Benchmark",
        "",
        "## Method",
        "",
        (
            f"BWD only; warmup={config['warmup']}; samples per block="
            f"{config['samples']}; inner iterations={config['inner_iters']}; "
            f"rounds={config['rounds']}; fixed ABBA/BAAB counterbalancing."
        ),
        "",
        "Both implementations call aten::_fused_rms_norm_backward with the same "
        "inputs and the same FP32 rstd fixture computed outside timing.",
        "",
        "wall_to_sync starts after a device synchronize, includes Python dispatch, "
        "allocation, zero/cast, launch, and GPU completion, and ends after a device "
        "synchronize. Each sample is divided by inner_iters.",
        "",
        "The primary speedup is aten_wall_to_sync_ms / flydsl_wall_to_sync_ms. "
        "Values above 1.0 favor FlyDSL.",
        "",
        "Rows marked correctness=FAIL retain timing for diagnostics and matrix "
        "completeness, but their speedup must not be used as a performance claim.",
        "",
        "## Environment",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    for key, value in environment_data.items():
        rendered = str(value).replace("|", "/").replace("\n", " ")
        lines.append(f"| {key} | {rendered} |")
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| M | N | dtype | FlyDSL wall ms | ATen wall ms | speedup | "
            "FlyDSL CV | ATen CV | correctness |",
            "|---:|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['m']} | {row['n']} | {row['dtype']} | "
            f"{row['flydsl_wall_to_sync_ms']:.6f} | "
            f"{row['aten_wall_to_sync_ms']:.6f} | {row['speedup']:.4f}x | "
            f"{row['flydsl_cv_percent']:.2f}% | "
            f"{row['aten_cv_percent']:.2f}% | {row['correctness']} |"
        )
    failures = [
        case
        for case in payload["cases"]
        if not case.get("correctness", {}).get("passed", False)
    ]
    if failures:
        lines.extend(["", "## Failed cases", ""])
        for case in failures:
            error = str(case["correctness"].get("error", "unknown")).replace(
                "\n", " "
            )
            lines.append(
                f"- M={case['m']} N={case['n']} {case['dtype']}: {error}"
            )
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            "The source_snapshot directory contains exactly the nine BWD-only target "
            "files. source_SHA256SUMS.txt is authoritative for this run.",
            "",
        ]
    )
    atomic_write_text(path, "\n".join(lines))


def write_progress(
    output_dir: Path,
    payload: dict[str, object],
    rows: list[dict[str, object]],
    *,
    final: bool,
) -> None:
    completed_cases = payload["cases"]
    failed_cases = sum(
        not case.get("correctness", {}).get("passed", False)
        for case in completed_cases
    )
    payload["progress"] = {
        "completed_cases": len(completed_cases),
        "total_cases": len(payload["config"]["case_manifest"]),
        "successful_cases": len(completed_cases) - failed_cases,
        "failed_cases": failed_cases,
        "timed_cases": len(rows),
        "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    atomic_write_text(output_dir / "checkpoint.json", serialized)
    write_csv(output_dir / "results.csv", rows)
    write_report(output_dir / "report.md", payload, rows)
    if final:
        atomic_write_text(output_dir / "results.json", serialized)


def prepare_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def run_case(
    spec: dict[str, object],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, object]]:
    m = int(spec["m"])
    n = int(spec["n"])
    dtype_name = str(spec["dtype"])
    dtype = DTYPES[dtype_name]
    x = torch.randn((m, n), device=device, dtype=dtype)
    weight = torch.randn((n,), device=device, dtype=dtype)
    grad_out = torch.randn((m, n), device=device, dtype=dtype)
    correctness, shared_rstd = correctness_case(x, weight, grad_out, args.eps)
    timing = benchmark_bwd_case(
        x,
        weight,
        grad_out,
        shared_rstd,
        args,
    )
    aten = timing["aten"]
    flydsl = timing["flydsl"]
    detail = {
        "m": m,
        "n": n,
        "dtype": dtype_name,
        "correctness": correctness,
        "timing": timing,
    }
    row = {
        "m": m,
        "n": n,
        "dtype": dtype_name,
        "flydsl_wall_to_sync_ms": timing["flydsl_wall_to_sync_ms"],
        "aten_wall_to_sync_ms": timing["aten_wall_to_sync_ms"],
        "speedup": timing["speedup"],
        "flydsl_p10_ms": flydsl["p10_ms"],
        "flydsl_p90_ms": flydsl["p90_ms"],
        "flydsl_cv_percent": flydsl["cv_percent"],
        "aten_p10_ms": aten["p10_ms"],
        "aten_p90_ms": aten["p90_ms"],
        "aten_cv_percent": aten["cv_percent"],
        "correctness": "PASS" if correctness["passed"] else "FAIL",
        "rounds": args.rounds,
        "samples_per_block": args.samples,
        "inner_iters": args.inner_iters,
        "effective_samples_per_implementation": len(aten["samples_ms"]),
    }
    return detail, row


def main() -> int:
    args = parse_args()
    if args.warmup <= 0:
        raise ValueError("--warmup must be positive so JIT is excluded")
    if args.samples <= 0 or args.inner_iters <= 0:
        raise ValueError("--samples and --inner-iters must be positive")
    if args.rounds <= 0 or args.rounds % 2:
        raise ValueError("--rounds must be a positive even number")
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("This benchmark requires a ROCm PyTorch build")

    m_values = parse_int_values(args.m_values, M_VALUES, "m-values")
    n_values = parse_int_values(args.n_values, N_VALUES, "n-values")
    dtype_names = parse_dtypes(args.dtypes)
    case_manifest = make_manifest(m_values, n_values, dtype_names)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    if "flydsl" not in pn.all_dsls:
        raise RuntimeError("FlyDSL is not registered in python_native")
    flydsl_operations = pn.get_dsl_operations("flydsl")
    if "_fused_rms_norm_backward" not in flydsl_operations:
        raise RuntimeError("FlyDSL RMSNorm backward is not registered")
    if "_fused_rms_norm" in flydsl_operations:
        raise RuntimeError("BWD-only benchmark found a FlyDSL RMSNorm FWD route")

    output_dir = Path(args.output_dir).resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    run_environment = environment(device, output_dir)
    if run_environment["build_matches_checkout"] is not True:
        raise RuntimeError(
            "Cannot prove that the loaded PyTorch build matches this checkout: "
            f"built={run_environment['torch_git_version']}, "
            f"source={run_environment['torch_source_sha']}"
        )
    if run_environment["loaded_sources_match_checkout"] is not True:
        raise RuntimeError(
            "Loaded torch/BWD module paths do not match this checkout: "
            f"loaded={run_environment['loaded_source_paths']}, "
            f"expected={run_environment['expected_source_paths']}"
        )
    prepare_output_dir(output_dir)
    source_manifest = snapshot_sources(output_dir)

    from torch._native.ops.norm.flydsl_rmsnorm_bwd_kernel import (
        clear_rmsnorm_bwd_caches,
    )

    clear_rmsnorm_bwd_caches()
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "environment": run_environment,
        "gpu_state_before": capture_gpu_state(),
        "gpu_state_after": None,
        "source_manifest": source_manifest,
        "measurement_semantics": {
            "primary_metric": "wall_to_sync",
            "unit": "ms",
            "start": "after torch.cuda.synchronize(device)",
            "end": "after timed torch.cuda.synchronize(device) returns",
            "amortization": "elapsed batch wall time divided by inner_iters",
            "speedup": "aten_wall_to_sync_ms / flydsl_wall_to_sync_ms",
            "jit_included": False,
            "rstd": "one shared FP32 formula rstd generated outside timing",
            "correctness_failure_timing": (
                "retained for diagnostics; invalid for speedup claims"
            ),
        },
        "config": {
            "fixed_m_manifest": list(M_VALUES),
            "fixed_n_manifest": list(N_VALUES),
            "fixed_dtype_manifest": list(DTYPES),
            "selected_m_values": m_values,
            "selected_n_values": n_values,
            "selected_dtypes": dtype_names,
            "case_manifest": case_manifest,
            "eps": args.eps,
            "warmup": args.warmup,
            "samples": args.samples,
            "inner_iters": args.inner_iters,
            "rounds": args.rounds,
            "measurement_order": "ABBA/BAAB",
            "seed": args.seed,
            "device": str(device),
        },
        "cases": [],
        "cache": None,
    }
    rows: list[dict[str, object]] = []
    write_progress(output_dir, payload, rows, final=False)

    failed = False
    for index, spec in enumerate(case_manifest, 1):
        print(
            f"[{index}/{len(case_manifest)}] BWD M={spec['m']} N={spec['n']} "
            f"dtype={spec['dtype']}",
            flush=True,
        )
        try:
            detail, row = run_case(spec, device, args)
            payload["cases"].append(detail)
            rows.append(row)
            if not detail["correctness"]["passed"]:
                failed = True
            print(
                f"  correctness={row['correctness']}, flydsl_wall_to_sync_ms="
                f"{row['flydsl_wall_to_sync_ms']:.6f}, "
                f"aten_wall_to_sync_ms={row['aten_wall_to_sync_ms']:.6f}, "
                f"speedup={row['speedup']:.4f}x",
                flush=True,
            )
        except Exception as exc:
            failed = True
            payload["cases"].append(
                {
                    **spec,
                    "correctness": {
                        "passed": False,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                }
            )
            print(f"  FAILED: {exc!r}", flush=True)
        finally:
            try:
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
            except Exception:
                pass
            payload["cache"] = cache_counters()
            write_progress(output_dir, payload, rows, final=False)
        if failed and args.fail_fast:
            break

    payload["gpu_state_after"] = capture_gpu_state()
    payload["status"] = "complete_with_failures" if failed else "complete"
    payload["cache"] = cache_counters()
    write_progress(output_dir, payload, rows, final=True)
    print(f"Results: {output_dir / 'results.csv'}", flush=True)
    print(f"Report: {output_dir / 'report.md'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

