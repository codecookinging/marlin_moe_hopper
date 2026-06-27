"""Microbenchmark: cluster_streamk_reduce vs atomic global reduce.

Runs an isolated CUDA benchmark (not full Marlin GEMM) that compares:
  - cluster DSMEM 2-CTA reduce via marlin_hopper::cluster_streamk_reduce
  - atomic global reduce mirroring Marlin MoE Stream-K tail (lock + atomicAdd)

Usage:
  PYTHONPATH=$PWD/python ./.venv/bin/pytest tests/test_cluster_reduce_benchmark.py -s -rs
  PYTHONPATH=$PWD/python ./.venv/bin/python tests/test_cluster_reduce_benchmark.py
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from marlin_v100 import ops


def _benchmark_schema_text() -> str:
    op = torch.ops._moe_C.benchmark_streamk_reduce
    schema = getattr(op, "_schema", None)
    return str(schema) if schema is not None else ""


def _benchmark_has_device_guard() -> bool:
    op = torch.ops._moe_C.benchmark_streamk_reduce
    schema = getattr(op, "_schema", None)
    if schema is not None:
        return any(arg.name == "device_guard" for arg in schema.arguments)
    return "device_guard" in _benchmark_schema_text()


def _moe_extension_path() -> str:
    try:
        import marlin_v100._moe_C as moe_ext

        return str(getattr(moe_ext, "__file__", "<unknown>"))
    except Exception as exc:  # pragma: no cover
        return f"<unavailable: {exc}>"


def _benchmark_environment() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA is required"
    cap = torch.cuda.get_device_capability()
    if cap[0] < 9:
        return False, f"SM90+ required (current capability={cap})"
    try:
        ops._load_moe()
    except Exception as exc:
        return False, f"marlin moe extension is not available: {exc}"
    if not hasattr(torch.ops, "_moe_C") or not hasattr(
        torch.ops._moe_C, "benchmark_streamk_reduce"
    ):
        return False, "benchmark_streamk_reduce is not registered in _moe_C"
    if not _benchmark_has_device_guard():
        return (
            False,
            "stale _moe_C schema (missing device_guard Tensor argument); "
            f"loaded from {_moe_extension_path()}; schema={_benchmark_schema_text()!r}; "
            "rebuild with: PYTHONPATH=$PWD/python ./.venv/bin/python setup.py build_ext --inplace",
        )
    return True, ""


def _require_sm90_benchmark(*, fail_instead_of_skip: bool = False) -> None:
    ok, message = _benchmark_environment()
    if ok:
        return
    print(f"\n[cluster reduce benchmark skipped]\n{message}\n")
    if fail_instead_of_skip:
        pytest.fail(message)
    pytest.skip(message)


def _parse_result(raw: torch.Tensor) -> dict[str, float]:
    """Decode benchmark_streamk_reduce output vector."""
    values = raw.detach().cpu().tolist()
    num_floats = int(values[0])
    num_threads = int(values[1])
    atomic_ns_per_pair = float(values[2])
    cluster_ns_per_pair = float(values[3])
    max_abs_diff = float(values[4])
    num_pairs = int(values[5])
    atomic_ns_per_launch = float(values[6])
    cluster_ns_per_launch = cluster_ns_per_pair * num_pairs
    speedup = atomic_ns_per_pair / cluster_ns_per_pair if cluster_ns_per_pair > 0 else 0.0
    return {
        "num_floats": num_floats,
        "num_threads": num_threads,
        "num_pairs": num_pairs,
        "atomic_ns_per_pair": atomic_ns_per_pair,
        "cluster_ns_per_pair": cluster_ns_per_pair,
        "atomic_ns_per_launch": atomic_ns_per_launch,
        "cluster_ns_per_launch": cluster_ns_per_launch,
        "speedup": speedup,
        "max_abs_diff": max_abs_diff,
    }


def run_benchmark(
    *,
    num_floats: int = 32,
    num_threads: int = 256,
    num_pairs: int = 4096,
    warmup_iters: int = 20,
    bench_iters: int = 200,
    run_verify: bool = True,
) -> dict[str, float]:
    raw = ops.benchmark_streamk_reduce(
        num_floats,
        num_threads,
        num_pairs,
        warmup_iters,
        bench_iters,
        run_verify,
    )
    return _parse_result(raw)


def _print_row(row: dict[str, float]) -> None:
    print(
        f"num_floats={row['num_floats']:>2} "
        f"threads={row['num_threads']:>3} "
        f"pairs={row['num_pairs']:>5} | "
        f"atomic {row['atomic_ns_per_pair']:8.1f} ns/pair | "
        f"cluster {row['cluster_ns_per_pair']:8.1f} ns/pair | "
        f"speedup {row['speedup']:5.2f}x | "
        f"max_diff {row['max_abs_diff']:.2e}"
    )


def print_cluster_reduce_benchmark_table() -> None:
    print("\ncluster_streamk_reduce vs atomic global reduce (isolated microbench)")
    print(f"_moe_C: {_moe_extension_path()}")
    print(f"schema: {_benchmark_schema_text()}")
    print("-" * 88)
    for num_floats in (16, 32, 64):
        for num_threads in (128, 256):
            row = run_benchmark(
                num_floats=num_floats,
                num_threads=num_threads,
                num_pairs=4096,
                warmup_iters=20,
                bench_iters=200,
                run_verify=True,
            )
            _print_row(row)
    print("-" * 88)


@pytest.mark.parametrize("num_floats", [16, 32, 64])
@pytest.mark.parametrize("num_threads", [128, 256])
def test_cluster_streamk_reduce_matches_atomic(num_floats: int, num_threads: int) -> None:
    _require_sm90_benchmark()
    row = run_benchmark(
        num_floats=num_floats,
        num_threads=num_threads,
        num_pairs=512,
        warmup_iters=5,
        bench_iters=10,
        run_verify=True,
    )
    assert row["max_abs_diff"] < 1e-3, (
        f"cluster vs atomic mismatch: max_abs_diff={row['max_abs_diff']}"
    )


def test_cluster_streamk_reduce_faster_than_atomic() -> None:
    """Soft performance check; skip when explicitly disabled."""
    if os.environ.get("MARLIN_SKIP_REDUCE_PERF_ASSERT") == "1":
        pytest.skip("performance assertion disabled")

    _require_sm90_benchmark()
    row = run_benchmark(
        num_floats=32,
        num_threads=256,
        num_pairs=4096,
        warmup_iters=30,
        bench_iters=300,
        run_verify=True,
    )
    _print_row(row)
    assert row["speedup"] > 1.2, (
        "expected cluster DSMEM reduce to beat atomic global reduce "
        f"(speedup={row['speedup']:.2f}x)"
    )


def test_print_cluster_reduce_benchmark_table() -> None:
    """Print a small sweep table (use pytest -s to see stdout)."""
    _require_sm90_benchmark(fail_instead_of_skip=True)
    print_cluster_reduce_benchmark_table()


if __name__ == "__main__":
    ok, message = _benchmark_environment()
    if not ok:
        raise SystemExit(message)
    print_cluster_reduce_benchmark_table()
