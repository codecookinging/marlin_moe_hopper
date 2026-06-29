"""Microbenchmark: cluster_streamk_reduce vs atomic global reduce.

Runs an isolated CUDA benchmark (not full Marlin GEMM) that compares:
  - cluster DSMEM 2-CTA reduce via marlin_hopper::cluster_streamk_reduce
  - atomic global reduce mirroring Marlin MoE Stream-K tail (lock + atomicAdd)

Realistic simulation knobs (slices_per_tile, in_kernel_iters, large num_floats):
  - slices_per_tile=2: cluster vs atomic on independent tiles
  - slices_per_tile>2: many CTAs hammer the same output tile (Stream-K contention)
  - in_kernel_iters=True: inner loop inside kernel (closer to Marlin, no per-iter launch)

Usage:
  PYTHONPATH=$PWD/python ./.venv/bin/pytest tests/test_cluster_reduce_benchmark.py -s -rs
  PYTHONPATH=$PWD/python ./.venv/bin/pytest tests/test_cluster_reduce_benchmark.py -s -k realistic
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
    if _benchmark_has_device_guard() and "slices_per_tile" in schema:
        return True, ""

    src_ok = _bindings_source_has_device_guard()
    rebuild_cmd = (
        f"cd {project_root} && "
        "rm -rf build python/marlin_v100/_moe_C*.so && "
        "PYTHONPATH=$PWD/python ./.venv/bin/python setup.py build_ext --inplace"
    )
    if src_ok is True:
        reason = (
            "loaded _moe_C binary is stale; rebuild in the directory that owns the loaded .so"
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
    """Decode benchmark_streamk_reduce output vector (9 floats)."""
    values = raw.detach().cpu().tolist()
    num_floats = int(values[0])
    num_threads = int(values[1])
    atomic_ns_per_tile = float(values[2])
    cluster_ns_per_tile = float(values[3])
    max_abs_diff = float(values[4])
    num_tiles = int(values[5])
    atomic_ns_per_launch = float(values[6])
    slices_per_tile = int(values[7]) if len(values) > 7 else 2
    in_kernel_iters = bool(values[8]) if len(values) > 8 else False
    cluster_ns_per_launch = cluster_ns_per_tile * num_tiles
    speedup = (
        atomic_ns_per_tile / cluster_ns_per_tile if cluster_ns_per_tile > 0 else 0.0
    )
    return {
        "num_floats": num_floats,
        "num_threads": num_threads,
        "num_tiles": num_tiles,
        "num_pairs": num_tiles,
        "slices_per_tile": slices_per_tile,
        "in_kernel_iters": in_kernel_iters,
        "atomic_ns_per_pair": atomic_ns_per_tile,
        "cluster_ns_per_pair": cluster_ns_per_tile,
        "atomic_ns_per_tile": atomic_ns_per_tile,
        "cluster_ns_per_tile": cluster_ns_per_tile,
        "atomic_ns_per_launch": atomic_ns_per_launch,
        "cluster_ns_per_launch": cluster_ns_per_launch,
        "speedup": speedup,
        "max_abs_diff": max_abs_diff,
    }


def run_benchmark(
    *,
    num_floats: int = 32,
    num_threads: int = 256,
    num_tiles: int | None = None,
    num_pairs: int | None = None,
    warmup_iters: int = 20,
    bench_iters: int = 200,
    run_verify: bool = True,
    slices_per_tile: int = 2,
    in_kernel_iters: bool = False,
) -> dict[str, float]:
    tiles = num_tiles if num_tiles is not None else (num_pairs if num_pairs is not None else 64)
    raw = ops.benchmark_streamk_reduce(
        num_floats,
        num_threads,
        tiles,
        warmup_iters,
        bench_iters,
        run_verify,
        slices_per_tile,
        in_kernel_iters,
    )
    return _parse_result(raw)


def _print_row(row: dict[str, float]) -> None:
    mode = "inkernel" if row.get("in_kernel_iters") else "launch"
    cluster_ns = row["cluster_ns_per_tile"]
    cluster_str = f"{cluster_ns:8.1f}" if cluster_ns > 0 else "     n/a"
    speedup = row["speedup"]
    speedup_str = f"{speedup:5.2f}x" if cluster_ns > 0 else "   n/a"
    print(
        f"floats={row['num_floats']:>4} thr={row['num_threads']:>3} "
        f"tiles={row['num_tiles']:>4} slices={row['slices_per_tile']:>2} "
        f"{mode:>7} | "
        f"atomic {row['atomic_ns_per_tile']:8.1f} ns/tile | "
        f"cluster {cluster_str} ns/tile | "
        f"speedup {speedup_str} | "
        f"max_diff {row['max_abs_diff']:.2e}"
    )


def print_realistic_streamk_simulation() -> None:
    """Sweep large NumFloats, in-kernel iters, and multi-CTA tile contention."""
    num_tiles = int(os.environ.get("MARLIN_REDUCE_BENCH_TILES", "64"))
    chunk = int(os.environ.get("MARLIN_REDUCE_BENCH_CHUNK", "8"))
    warmup = int(os.environ.get("MARLIN_REDUCE_BENCH_WARMUP", "3"))
    bench = int(os.environ.get("MARLIN_REDUCE_BENCH_ITERS", "20"))

    print("\n=== Realistic Stream-K simulation ===")
    print(f"_moe_C: {_moe_extension_path()}")
    print(
        f"tiles={num_tiles} chunk={chunk} warmup={warmup} bench={bench}\n"
        "Sections:\n"
        "  A) large NumFloats, slices=2, in-kernel (Marlin-like 2-CTA reduce)\n"
        "  B) slices=2, in-kernel vs per-launch (launch overhead)\n"
        "  C) multi-slice same tile (atomic contention; cluster n/a when slices>2)\n"
    )
    os.environ.setdefault("MARLIN_REDUCE_BENCH_CHUNK", str(chunk))

    print("--- A) large NumFloats, slices=2, in_kernel ---")
    for num_floats in (64, 128, 256, 1024):
        print(f"  running floats={num_floats} ...", flush=True)
        row = run_benchmark(
            num_floats=num_floats,
            num_threads=256,
            num_tiles=num_tiles,
            warmup_iters=warmup,
            bench_iters=bench,
            run_verify=num_floats <= 256,
            slices_per_tile=2,
            in_kernel_iters=True,
        )
        _print_row(row)

    print("--- B) launch mode vs in-kernel (floats=256, slices=2) ---")
    for in_kernel in (False, True):
        label = "in-kernel" if in_kernel else "per-launch"
        print(f"  running {label} ...", flush=True)
        row = run_benchmark(
            num_floats=256,
            num_threads=256,
            num_tiles=num_tiles,
            warmup_iters=warmup,
            bench_iters=bench,
            run_verify=in_kernel,
            slices_per_tile=2,
            in_kernel_iters=in_kernel,
        )
        _print_row(row)

    print("--- C) multi-CTA same output tile (atomic contention) ---")
    for slices in (2, 4, 8, 16):
        print(f"  running slices={slices} ...", flush=True)
        row = run_benchmark(
            num_floats=256,
            num_threads=256,
            num_tiles=num_tiles,
            warmup_iters=warmup,
            bench_iters=bench,
            run_verify=slices == 2,
            slices_per_tile=slices,
            in_kernel_iters=True,
        )
        _print_row(row)
    print("=" * 88)


def _print_benchmark_env() -> tuple[int, int, int, int]:
    return (
        int(os.environ.get("MARLIN_REDUCE_BENCH_PAIRS", "64")),
        int(os.environ.get("MARLIN_REDUCE_BENCH_WARMUP", "3")),
        int(os.environ.get("MARLIN_REDUCE_BENCH_ITERS", "10")),
        int(os.environ.get("MARLIN_REDUCE_BENCH_CHUNK", "8")),
    )


def print_cluster_reduce_benchmark_table() -> None:
    num_tiles, warmup_iters, bench_iters, chunk_pairs = _print_benchmark_env()
    print("\ncluster_streamk_reduce vs atomic global reduce (basic sweep)")
    print(f"_moe_C: {_moe_extension_path()}")
    print(f"schema: {_benchmark_schema_text()}")
    print(
        f"sweep: tiles={num_tiles} chunk={chunk_pairs} "
        f"warmup={warmup_iters} bench={bench_iters}"
    )
    print("-" * 88)

    print("  smoke cluster verify tiles=4 slices=2 ...", flush=True)
    smoke = run_benchmark(
        num_floats=16,
        num_threads=128,
        num_tiles=4,
        warmup_iters=1,
        bench_iters=2,
        run_verify=True,
        slices_per_tile=2,
        in_kernel_iters=False,
    )
    _print_row(smoke)
    print("-" * 88)

    for num_floats in (16, 32, 64):
        for num_threads in (128, 256):
            print(
                f"  running num_floats={num_floats} threads={num_threads} ...",
                flush=True,
            )
            row = run_benchmark(
                num_floats=num_floats,
                num_threads=num_threads,
                num_tiles=num_tiles,
                warmup_iters=warmup_iters,
                bench_iters=bench_iters,
                run_verify=False,
                slices_per_tile=2,
                in_kernel_iters=False,
            )
            _print_row(row)
    print("-" * 88)
    print_realistic_streamk_simulation()


@pytest.mark.parametrize("num_floats", [16, 32, 64])
@pytest.mark.parametrize("num_threads", [128, 256])
def test_cluster_streamk_reduce_matches_atomic(num_floats: int, num_threads: int) -> None:
    _require_sm90_benchmark()
    row = run_benchmark(
        num_floats=num_floats,
        num_threads=num_threads,
        num_tiles=64,
        warmup_iters=3,
        bench_iters=5,
        run_verify=True,
        slices_per_tile=2,
        in_kernel_iters=False,
    )
    assert row["max_abs_diff"] < 1e-3, (
        f"cluster vs atomic mismatch: max_abs_diff={row['max_abs_diff']}"
    )


def test_large_num_floats_in_kernel_matches_atomic() -> None:
    _require_sm90_benchmark()
    for num_floats in (128, 256, 1024):
        row = run_benchmark(
            num_floats=num_floats,
            num_threads=256,
            num_tiles=32,
            warmup_iters=2,
            bench_iters=5,
            run_verify=True,
            slices_per_tile=2,
            in_kernel_iters=True,
        )
        assert row["max_abs_diff"] < 1e-2, (
            f"num_floats={num_floats} max_abs_diff={row['max_abs_diff']}"
        )


def test_multi_slice_atomic_contention_runs() -> None:
    """Higher slices_per_tile stresses same-output atomic path (cluster n/a)."""
    _require_sm90_benchmark()
    for slices in (4, 8, 16):
        row = run_benchmark(
            num_floats=256,
            num_threads=256,
            num_tiles=32,
            warmup_iters=2,
            bench_iters=5,
            run_verify=False,
            slices_per_tile=slices,
            in_kernel_iters=True,
        )
        assert row["cluster_ns_per_tile"] == 0.0
        assert row["atomic_ns_per_tile"] > 0.0


def test_in_kernel_cluster_can_beat_launch_mode() -> None:
    """Soft check: in-kernel cluster should not lose badly to in-kernel atomic."""
    if os.environ.get("MARLIN_SKIP_REDUCE_PERF_ASSERT") == "1":
        pytest.skip("performance assertion disabled")

    _require_sm90_benchmark()
    common = dict(
        num_floats=256,
        num_threads=256,
        num_tiles=64,
        warmup_iters=5,
        bench_iters=30,
        run_verify=True,
        slices_per_tile=2,
    )
    launch_row = run_benchmark(**common, in_kernel_iters=False)
    inkernel_row = run_benchmark(**common, in_kernel_iters=True)
    _print_row(launch_row)
    _print_row(inkernel_row)
    # In-kernel should improve both; cluster may exceed atomic on large tiles.
    assert inkernel_row["speedup"] >= launch_row["speedup"] * 0.8, (
        f"in-kernel speedup regressed: launch={launch_row['speedup']:.2f}x "
        f"inkernel={inkernel_row['speedup']:.2f}x"
    )


def test_cluster_streamk_reduce_faster_than_atomic() -> None:
    if os.environ.get("MARLIN_SKIP_REDUCE_PERF_ASSERT") == "1":
        pytest.skip("performance assertion disabled")

    _require_sm90_benchmark()
    row = run_benchmark(
        num_floats=256,
        num_threads=256,
        num_tiles=64,
        warmup_iters=5,
        bench_iters=30,
        run_verify=True,
        slices_per_tile=2,
        in_kernel_iters=True,
    )
    _print_row(row)
    assert row["speedup"] > 1.0, (
        "expected in-kernel cluster DSMEM reduce to beat atomic on large tiles "
        f"(speedup={row['speedup']:.2f}x)"
    )


def test_print_cluster_reduce_benchmark_table() -> None:
    print("\n[cluster reduce benchmark] starting...", flush=True)
    _require_sm90_benchmark(fail_instead_of_skip=True)
    print_cluster_reduce_benchmark_table()


def test_print_realistic_streamk_simulation() -> None:
    print("\n[realistic stream-k simulation] starting...", flush=True)
    _require_sm90_benchmark(fail_instead_of_skip=True)
    print_realistic_streamk_simulation()


if __name__ == "__main__":
    ok, message = _benchmark_environment()
    if not ok:
        raise SystemExit(message)
    print_realistic_streamk_simulation()
