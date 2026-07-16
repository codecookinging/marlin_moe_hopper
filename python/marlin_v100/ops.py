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


def cutlass69_pack_only(q_weight_int8: torch.Tensor) -> torch.Tensor:
    _load_moe()
    return torch.ops._moe_C.cutlass69_pack_only(q_weight_int8)


def cutlass69_pack_only(q_weight_int8: torch.Tensor) -> torch.Tensor:
    _load_moe()
    return torch.ops._moe_C.cutlass69_pack_only(q_weight_int8)


def cutlass69_pack_and_reorder(q_weight_int8: torch.Tensor) -> torch.Tensor:
    _load_moe()
    return torch.ops._moe_C.cutlass69_pack_and_reorder(q_weight_int8)


def cutlass69_dequant_packed(
    packed: torch.Tensor, scales: torch.Tensor, group_size: int
) -> torch.Tensor:
    _load_moe()
    return torch.ops._moe_C.cutlass69_dequant_packed(packed, scales, group_size)


def cutlass69_dequant_reordered(
    reordered: torch.Tensor, scales: torch.Tensor, group_size: int
) -> torch.Tensor:
    _load_moe()
    return torch.ops._moe_C.cutlass69_dequant_reordered(
        reordered, scales, group_size
    )


def cutlass69_dequant_packed(
    packed: torch.Tensor, scales: torch.Tensor, group_size: int
) -> torch.Tensor:
    _load_moe()
    return torch.ops._moe_C.cutlass69_dequant_packed(packed, scales, group_size)


def moe_sum(input: torch.Tensor, output: torch.Tensor) -> None:
    _load_moe()
    torch.ops._moe_C.moe_sum(input, output)


def moe_cutlass69_fused_moe(*args, **kwargs) -> torch.Tensor:
    _load_moe()
    return torch.ops._moe_C.moe_cutlass69_fused_moe(*args, **kwargs)
