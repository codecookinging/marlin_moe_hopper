from __future__ import annotations

import torch

from . import ops


_SHORT_BATCH_M_THRESHOLD = 512
_SHORT_BATCH_MOE_BLOCK_SIZE = 16
_MOE_BLOCK_SIZE_CANDIDATES = [64, 32, 16, 8]

def get_adaptive_moe_block_size(
    m: int, topk: int, num_experts: int, input_dtype=None
) -> int:
    """Pick MoE align block size.

    Short batch (m < 1024) always uses 16 tokens/block so the Marlin kernel
    stays on thread_m_blocks == 1 (small-batch tile table). Long batch may use
    up to 64 when expert load is dense enough.
    """
    if m <= _SHORT_BATCH_M_THRESHOLD:
        block_size_m = _SHORT_BATCH_MOE_BLOCK_SIZE
    else:
        block_size_m = 64
        for candidate in _MOE_BLOCK_SIZE_CANDIDATES:
            # If the average tokens per expert is significantly less than the
            # candidate block size, shrink the block to reduce padding overhead.
            if m * topk / num_experts / candidate < 0.9:
                block_size_m = candidate
            else:
                break

    if input_dtype is not None and input_dtype.itemsize == 1:
        block_size_m = max(block_size_m, _SHORT_BATCH_MOE_BLOCK_SIZE)
    return block_size_m


def normalize_moe_block_size(moe_block_size: int, m: int) -> int:
    """Force short-batch launches onto thread_m_blocks == 1."""
    if m < _SHORT_BATCH_M_THRESHOLD and moe_block_size > _SHORT_BATCH_MOE_BLOCK_SIZE:
        return _SHORT_BATCH_MOE_BLOCK_SIZE
    return moe_block_size

def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    max_num_m_blocks = max_num_tokens_padded // block_size + 1
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    ops.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_map,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type_id: int,
    moe_block_size: int | None = None,
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    global_scale1: torch.Tensor | None = None,
    global_scale2: torch.Tensor | None = None,
    g_idx1: torch.Tensor | None = None,
    g_idx2: torch.Tensor | None = None,
    sort_indices1: torch.Tensor | None = None,
    sort_indices2: torch.Tensor | None = None,
    w1_zeros: torch.Tensor | None = None,
    w2_zeros: torch.Tensor | None = None,
    is_k_full: bool = True,
    use_tma: bool = False,
) -> torch.Tensor:
    m, k = hidden_states.shape
    topk = topk_ids.shape[1]
    intermediate_size = w1_scale.shape[2]
    n = intermediate_size // 2
    output_size = w2_scale.shape[2]
    if intermediate_size % 2 != 0:
        raise ValueError(
            f"Expected first-layer MoE scale width to be even, got {intermediate_size}."
        )

    if moe_block_size is None:
        if m <= _SHORT_BATCH_M_THRESHOLD:
            moe_block_size = _SHORT_BATCH_MOE_BLOCK_SIZE
            do_split = False
        else:
            moe_block_size = _SHORT_BATCH_MOE_BLOCK_SIZE
            do_split = False
    else:
        moe_block_size = normalize_moe_block_size(moe_block_size, m)
        do_split = False

    sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
        topk_ids, moe_block_size, w1.shape[0]
    )
    
    if do_split:
        num_experts = w1.shape[0]
        total_blocks = num_tokens_post_pad.item() // 16
        
        expert_ids = expert_ids[:total_blocks]
        sorted_ids = sorted_ids[:total_blocks * 16]
        
        valid_topk_ids = topk_ids[topk_ids >= 0]
        expert_counts = torch.bincount(valid_topk_ids.flatten(), minlength=num_experts)
        
        block_indices = torch.arange(total_blocks, device=expert_ids.device)
        expert_block_counts = torch.bincount(expert_ids, minlength=num_experts)
        expert_block_offsets = torch.cumsum(expert_block_counts, dim=0) - expert_block_counts
        block_expert_idx = block_indices - expert_block_offsets[expert_ids]
        
        is_full_block = block_expert_idx < (expert_counts[expert_ids] // 64) * 4
        
        full_block_mask = is_full_block
        partial_block_mask = ~is_full_block
        
        sorted_ids_2d = sorted_ids.view(-1, 16)
        
        full_sorted_ids = sorted_ids_2d[full_block_mask].view(-1)
        full_expert_ids = expert_ids[full_block_mask][::4]
        full_num_tokens = torch.tensor([full_sorted_ids.numel()], dtype=torch.int32, device=expert_ids.device)
        
        partial_sorted_ids = sorted_ids_2d[partial_block_mask].view(-1)
        partial_expert_ids = expert_ids[partial_block_mask]
        partial_num_tokens = torch.tensor([partial_sorted_ids.numel()], dtype=torch.int32, device=expert_ids.device)
    if workspace is None:
        props = torch.cuda.get_device_properties(hidden_states.device)
        max_blocks_per_sm = 6 if props.major >= 9 else 4
        workspace = torch.zeros(
            props.multi_processor_count * max_blocks_per_sm,
            dtype=torch.int,
            device=hidden_states.device,
        )

    intermediate = torch.empty(
        (m * topk, intermediate_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    intermediate = ops.moe_wna16_marlin_gemm(
        hidden_states,
        intermediate,
        w1,
        bias1,
        w1_scale,
        None,
        global_scale1,
        w1_zeros,
        g_idx1,
        sort_indices1,
        workspace,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        topk_weights,
        moe_block_size,
        topk,
        False,
        quant_type_id,
        m,
        intermediate_size,
        k,
        is_k_full,
        False,
        True,
        False,
        -1,
        -1,
        -1,
        use_tma,
    )
    gate, up = intermediate.view(m * topk, intermediate_size).chunk(2, dim=-1)
    activated = torch.nn.functional.silu(gate) * up
    output = torch.empty(
        (m * topk, output_size), dtype=hidden_states.dtype, device=hidden_states.device
    )
    output = ops.moe_wna16_marlin_gemm(
        activated,
        output,
        w2,
        bias2,
        w2_scale,
        None,
        global_scale2,
        w2_zeros,
        g_idx2,
        sort_indices2,
        workspace,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        topk_weights,
        moe_block_size,
        1,
        True,
        quant_type_id,
        m * topk,
        output_size,
        n,
        is_k_full,
        False,
        True,
        False,
        -1,
        -1,
        -1,
        use_tma,
    )
    return output.view(m, topk, output_size).sum(dim=1)
