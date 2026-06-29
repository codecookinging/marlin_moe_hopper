from __future__ import annotations

import importlib

import torch

_dense_loaded = False
_moe_loaded = False


def _load_dense() -> None:
    global _dense_loaded
    if not _dense_loaded:
        importlib.import_module("marlin_v100._C")
        _dense_loaded = True


def _load_moe() -> None:
    global _moe_loaded
    if not _moe_loaded:
        importlib.import_module("marlin_v100._moe_C")
        _moe_loaded = True


def marlin_gemm(*args, **kwargs) -> torch.Tensor:
    _load_dense()
    return torch.ops._C.marlin_gemm(*args, **kwargs)


def gptq_marlin_repack(*args, **kwargs) -> torch.Tensor:
    _load_dense()
    return torch.ops._C.gptq_marlin_repack(*args, **kwargs)


def awq_marlin_repack(*args, **kwargs) -> torch.Tensor:
    _load_dense()
    return torch.ops._C.awq_marlin_repack(*args, **kwargs)


def marlin_int4_fp8_preprocess(*args, **kwargs) -> torch.Tensor:
    _load_dense()
    return torch.ops._C.marlin_int4_fp8_preprocess(*args, **kwargs)


def topk_softmax(*args, **kwargs) -> None:
    _load_moe()
    return torch.ops._moe_C.topk_softmax(*args, **kwargs)


def topk_sigmoid(*args, **kwargs) -> None:
    _load_moe()
    return torch.ops._moe_C.topk_sigmoid(*args, **kwargs)


def grouped_topk(*args, **kwargs):
    _load_moe()
    return torch.ops._moe_C.grouped_topk(*args, **kwargs)


def moe_align_block_size(*args, **kwargs) -> None:
    _load_moe()
    return torch.ops._moe_C.moe_align_block_size(*args, **kwargs)


def batched_moe_align_block_size(*args, **kwargs) -> None:
    _load_moe()
    return torch.ops._moe_C.batched_moe_align_block_size(*args, **kwargs)


def moe_wna16_marlin_gemm(*args, **kwargs) -> torch.Tensor:
    _load_moe()
    return torch.ops._moe_C.moe_wna16_marlin_gemm(*args, **kwargs)


def _get_benchmark_op_schema():
    """Resolve FunctionSchema for benchmark_streamk_reduce (OpOverload or OpOverloadPacket)."""
    op = torch.ops._moe_C.benchmark_streamk_reduce
    schemas = getattr(op, "_schemas", None)
    if schemas:
        return next(iter(schemas.values()))
    schema = getattr(op, "_schema", None)
    if schema is not None:
        return schema
    if hasattr(op, "default"):
        return op.default._schema
    for name in op.overloads() if hasattr(op, "overloads") else []:
        overload = getattr(op, name) if name else op.default
        return overload._schema
    return None


def _benchmark_schema_text() -> str:
    schema = _get_benchmark_op_schema()
    return str(schema) if schema is not None else ""


def _benchmark_schema_has_device_guard() -> bool:
    schema = _get_benchmark_op_schema()
    if schema is not None:
        return any(arg.name == "device_guard" for arg in schema.arguments)
    return "device_guard" in _benchmark_schema_text()


def benchmark_streamk_reduce(
    num_floats: int,
    num_threads: int,
    num_tiles: int,
    warmup_iters: int,
    bench_iters: int,
    run_verify: bool,
    slices_per_tile: int = 2,
    in_kernel_iters: bool = False,
    *,
    num_pairs: int | None = None,
    device_guard: torch.Tensor | None = None,
) -> torch.Tensor:
    """Benchmark cluster vs atomic Stream-K reduce.

    num_tiles: independent output tiles (legacy alias: num_pairs).
    slices_per_tile: CTAs sharing one output tile (2=cluster comparable, >2=contention).
    in_kernel_iters: loop reduces inside the kernel (closer to Marlin).
    """
    _load_moe()
    op = torch.ops._moe_C.benchmark_streamk_reduce
    tiles = num_tiles if num_pairs is None else num_pairs
    has_device_guard = _benchmark_schema_has_device_guard()

    args = (
        num_floats,
        num_threads,
        tiles,
        warmup_iters,
        bench_iters,
        run_verify,
        slices_per_tile,
        in_kernel_iters,
    )
    if has_device_guard:
        if device_guard is not None:
            return op(device_guard, *args)
        guard = torch.empty((), device="cuda")
        return op(guard, *args)

    schema = _benchmark_schema_text()
    raise RuntimeError(
        "Loaded _moe_C is stale: benchmark_streamk_reduce schema is missing "
        f"device_guard or new args (schema={schema!r}). Rebuild the extension:\n"
        "  rm -f python/marlin_v100/_moe_C*.so && "
        "PYTHONPATH=$PWD/python ./.venv/bin/python setup.py build_ext --inplace"
    )
