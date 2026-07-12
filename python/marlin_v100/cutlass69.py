from __future__ import annotations

import torch

from . import dense, ops, quant_utils

_BRIDGE_CACHE: dict[tuple[int, int, int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _quantize_unsigned_with_bias(
    weight: torch.Tensor, group_size: int, bias: int
) -> tuple[torch.Tensor, torch.Tensor]:
    size_k, size_n = weight.shape
    if group_size == -1:
        group_size = size_k
    if size_k % group_size != 0:
        raise ValueError(f"group_size={group_size} must divide size_k={size_k}")

    groups = size_k // group_size
    reshaped = weight.reshape(groups, group_size, size_n)
    max_abs = reshaped.abs().amax(dim=1, keepdim=False).clamp_min(1e-6)
    scales = max_abs / float(bias - 1)
    scales = scales.to(weight.dtype)

    q = torch.round(reshaped / scales.unsqueeze(1)).clamp(-bias, bias - 1).to(torch.int32)
    q = (q + bias).reshape(size_k, size_n)
    return q, scales


def quantize_experts(
    weights: torch.Tensor,
    group_size: int,
    bias: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize expert weights for CUTLASS69 grouped GEMM.

    weights: [num_experts, size_k, size_n]
    Returns:
      q_weights_uint8: [num_experts, size_n, size_k / 2] packed/reordered for CUTLASS
      scales: [num_experts, size_n, scale_k] row-major for CUTLASS
    """
    num_experts, size_k, size_n = weights.shape
    if group_size == -1:
        group_size = size_k
    groups = size_k // group_size

    reshaped = weights.reshape(num_experts, groups, group_size, size_n)
    max_abs = reshaped.abs().amax(dim=2).clamp_min(1e-6)
    scales = (max_abs / float(bias - 1)).to(weights.dtype)
    q = torch.round(reshaped / scales.unsqueeze(2)).clamp(-bias, bias - 1)
    q = (q + bias).reshape(num_experts, size_k, size_n)

    q_t = q.transpose(1, 2).contiguous().to(torch.int8)
    scales_t = scales.transpose(1, 2).contiguous()
    packed = ops.cutlass69_pack_and_reorder(q_t)
    return packed, scales_t


def _marlin_unpack_batched(
    q_weights: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int = 4,
) -> torch.Tensor:
    """Unpack Marlin expert weights. Input [E, ...] -> [E, size_k, size_n]."""
    perm = quant_utils.get_weight_perm(num_bits, is_a_8bit=False).to(
        q_weights.device, dtype=torch.long
    )
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), device=q_weights.device, dtype=torch.long)

    packed = q_weights.to(torch.int64) & 0xFFFFFFFF
    pack_factor = quant_utils.get_pack_factor(num_bits)
    unpacked = torch.stack(
        [
            (packed >> (num_bits * i)) & ((1 << num_bits) - 1)
            for i in range(pack_factor)
        ],
        dim=-1,
    )
    unpacked = unpacked.to(torch.int32).reshape(
        q_weights.size(0), q_weights.shape[1], q_weights.shape[2] * pack_factor
    )
    unpacked = (
        unpacked.reshape(q_weights.size(0), -1, perm.numel())[:, :, inv_perm]
        .reshape(q_weights.size(0), size_k // 16, size_n * 16)
        .reshape(q_weights.size(0), size_k // 16, size_n // 16, 16, 16)
    )
    return unpacked.permute(0, 1, 3, 2, 4).reshape(q_weights.size(0), size_k, size_n)


def _marlin_unpermute_scales_batched(
    scales: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
) -> torch.Tensor:
    scale_perm, scale_perm_single = dense.get_scale_perms()
    perm = scale_perm if group_size < size_k and group_size != -1 else scale_perm_single
    perm = torch.tensor(perm, device=scales.device, dtype=torch.long)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), device=scales.device, dtype=torch.long)
    unperm = (
        scales.reshape(scales.size(0), -1, perm.numel())[:, :, inv_perm]
        .reshape(scales.size(0), -1, size_n)
        .transpose(1, 2)
        .contiguous()
    )
    return unperm


def _is_cutlass69_packed(q_weight: torch.Tensor, size_k: int, size_n: int) -> bool:
    return (
        q_weight.dtype == torch.uint8
        and q_weight.dim() == 3
        and q_weight.size(1) == size_n
        and q_weight.size(2) * 2 == size_k
    )


def marlin_experts_to_cutlass69(
    q_weight: torch.Tensor,
    scales: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert Marlin-packed expert weights/scales to CUTLASS69 layout."""
    if group_size is None:
        if scales.dim() != 3:
            raise ValueError("CUTLASS69 bridge expects rank-3 Marlin scales.")
        group_size = size_k // scales.size(1)

    unpacked = _marlin_unpack_batched(q_weight, size_k, size_n)
    q_t = unpacked.transpose(1, 2).contiguous().to(torch.int8)
    packed = ops.cutlass69_pack_and_reorder(q_t)
    cutlass_scales = _marlin_unpermute_scales_batched(
        scales, size_k, size_n, group_size
    )
    return packed, cutlass_scales


def ensure_cutlass69_expert_weights(
    q_weight: torch.Tensor,
    scales: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CUTLASS69-ready weights, converting from Marlin layout if needed."""
    if _is_cutlass69_packed(q_weight, size_k, size_n):
        scale_k = (size_k + 127) // 128
        if (
            scales.dim() == 3
            and scales.size(1) == size_n
            and scales.size(2) == scale_k
        ):
            return q_weight, scales

    cache_key = (
        q_weight.data_ptr(),
        scales.data_ptr(),
        size_k,
        size_n,
        0 if group_size is None else group_size,
    )
    cached = _BRIDGE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    converted = marlin_experts_to_cutlass69(
        q_weight, scales, size_k, size_n, group_size=group_size
    )
    _BRIDGE_CACHE[cache_key] = converted
    return converted
