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
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from marlin_v100 import ops


def _benchmark_schema_text() -> str:
    return ops._benchmark_schema_text()


def _benchmark_has_device_guard() -> bool:
    return ops._benchmark_schema_has_device_guard()


def _moe_extension_path() -> str:
    try:
        import marlin_v100._moe_C as moe_ext

        return str(getattr(moe_ext, "__file__", "<unknown>"))
    except Exception as exc:  # pragma: no cover
        return f"<unavailable: {exc}>"


def _extension_project_root() -> str:
    ext_path = _moe_extension_path()
    if ext_path.startswith("<"):
        return "<unknown>"
    # .../python/marlin_v100/_moe_C*.so -> repo root is three levels up
    return str(Path(ext_path).resolve().parents[2])


def _bindings_source_has_device_guard() -> bool | None:
    root = _extension_project_root()
    if root == "<unknown>":
        return None
    bindings = Path(root) / "csrc/moe/torch_bindings_marlin.cpp"
    if not bindings.is_file():
        return None
    return "device_guard" in bindings.read_text(encoding="utf-8")


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

    ext_path = _moe_extension_path()
    project_root = _extension_project_root()
    schema = _benchmark_schema_text()
    if _benchmark_has_device_guard():
        return True, ""

    src_ok = _bindings_source_has_device_guard()
    rebuild_cmd = (
        f"cd {project_root} && "
        "rm -rf build python/marlin_v100/_moe_C*.so && "
        "PYTHONPATH=$PWD/python ./.venv/bin/python setup.py build_ext --inplace"
    )
    if src_ok is True:
        reason = (
            "loaded _moe_C binary is stale even though source already contains "
            "device_guard; rebuild in the directory that owns the loaded .so"
        )
    elif src_ok is False:
        reason = (
            "loaded project copy is stale: csrc/moe/torch_bindings_marlin.cpp "
            "in the loaded tree still lacks device_guard (sync/copy issue)"
        )
    else:
        reason = "loaded _moe_C binary is stale relative to benchmark source"

    return (
        False,
        f"{reason}\n"
        f"  loaded _moe_C: {ext_path}\n"
        f"  project root:  {project_root}\n"
        f"  schema:        {schema!r}\n"
        f"  rebuild:\n"
        f"    {rebuild_cmd}",
    )


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


def _print_benchmark_env() -> tuple[int, int, int, int]:
    """Return (num_pairs, warmup_iters, bench_iters, chunk_pairs) for print sweep."""
    return (
        int(os.environ.get("MARLIN_REDUCE_BENCH_PAIRS", "256")),
        int(os.environ.get("MARLIN_REDUCE_BENCH_WARMUP", "5")),
        int(os.environ.get("MARLIN_REDUCE_BENCH_ITERS", "30")),
        int(os.environ.get("MARLIN_REDUCE_BENCH_CHUNK", "32")),
    )


def print_cluster_reduce_benchmark_table() -> None:
    num_pairs, warmup_iters, bench_iters, chunk_pairs = _print_benchmark_env()
    print("\ncluster_streamk_reduce vs atomic global reduce (isolated microbench)")
    print(f"_moe_C: {_moe_extension_path()}")
    print(f"schema: {_benchmark_schema_text()}")
    print(
        f"sweep: pairs={num_pairs} chunk={chunk_pairs} "
        f"warmup={warmup_iters} bench={bench_iters} "
        "(MARLIN_REDUCE_BENCH_* env vars; set MARLIN_REDUCE_BENCH_CLUSTER=0 "
        "to skip cluster path)"
    )
    print("-" * 88)
    # Phase 1: atomic-only (set MARLIN_REDUCE_BENCH_CLUSTER=0 in rebuilt _moe_C).
    print("  smoke atomic-only num_floats=16 threads=128 pairs=4 ...", flush=True)
    prev_cluster = os.environ.get("MARLIN_REDUCE_BENCH_CLUSTER")
    os.environ["MARLIN_REDUCE_BENCH_CLUSTER"] = "0"
    os.environ.setdefault("MARLIN_REDUCE_BENCH_CHUNK", str(chunk_pairs))
    try:
        smoke_atomic = run_benchmark(
            num_floats=16,
            num_threads=128,
            num_pairs=4,
            warmup_iters=1,
            bench_iters=2,
            run_verify=False,
        )
        _print_row(smoke_atomic)
    finally:
        if prev_cluster is None:
            os.environ.pop("MARLIN_REDUCE_BENCH_CLUSTER", None)
        else:
            os.environ["MARLIN_REDUCE_BENCH_CLUSTER"] = prev_cluster

    # Phase 2: cluster + atomic verify on a tiny grid.
    print("  smoke cluster num_floats=16 threads=128 pairs=4 ...", flush=True)
    smoke = run_benchmark(
        num_floats=16,
        num_threads=128,
        num_pairs=4,
        warmup_iters=1,
        bench_iters=2,
        run_verify=True,
    )
    _print_row(smoke)
    print("-" * 88)
    first = True
    for num_floats in (16, 32, 64):
        for num_threads in (128, 256):
            print(
                f"  running num_floats={num_floats} threads={num_threads} ...",
                flush=True,
            )
            row = run_benchmark(
                num_floats=num_floats,
                num_threads=num_threads,
                num_pairs=num_pairs,
                warmup_iters=warmup_iters,
                bench_iters=bench_iters,
                run_verify=first,
            )
            first = False
            _print_row(row)
    print("-" * 88)


@pytest.mark.parametrize("num_floats", [16, 32, 64])
@pytest.mark.parametrize("num_threads", [128, 256])
def test_cluster_streamk_reduce_matches_atomic(num_floats: int, num_threads: int) -> None:
    _require_sm90_benchmark()
    row = run_benchmark(
        num_floats=num_floats,
        num_threads=num_threads,
        num_pairs=128,
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
        num_pairs=1024,
        warmup_iters=10,
        bench_iters=100,
        run_verify=True,
    )
    _print_row(row)
    assert row["speedup"] > 1.2, (
        "expected cluster DSMEM reduce to beat atomic global reduce "
        f"(speedup={row['speedup']:.2f}x)"
    )


def test_print_cluster_reduce_benchmark_table() -> None:
    """Print a small sweep table (use pytest -s to see stdout)."""
    print("\n[cluster reduce benchmark] starting...", flush=True)
    _require_sm90_benchmark(fail_instead_of_skip=True)
    print_cluster_reduce_benchmark_table()


if __name__ == "__main__":
    ok, message = _benchmark_environment()
    if not ok:
        raise SystemExit(message)
    print_cluster_reduce_benchmark_table()
