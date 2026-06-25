from __future__ import annotations

import torch

from . import ops

MAX_BLOCK_SEGMENTS = 16


class MoeAlignResult:
    __slots__ = (
        "sorted_ids",
        "expert_ids",
        "num_tokens_post_pad",
        "block_token_offsets",
        "block_num_segments",
        "block_segment_experts",
        "block_segment_row_starts",
        "block_segment_counts",
        "use_packed",
    )

    def __init__(
        self,
        sorted_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_pad: torch.Tensor,
        *,
        block_token_offsets: torch.Tensor | None = None,
        block_num_segments: torch.Tensor | None = None,
        block_segment_experts: torch.Tensor | None = None,
        block_segment_row_starts: torch.Tensor | None = None,
        block_segment_counts: torch.Tensor | None = None,
        use_packed: bool = False,
    ) -> None:
        self.sorted_ids = sorted_ids
        self.expert_ids = expert_ids
        self.num_tokens_post_pad = num_tokens_post_pad
        self.block_token_offsets = block_token_offsets
        self.block_num_segments = block_num_segments
        self.block_segment_experts = block_segment_experts
        self.block_segment_row_starts = block_segment_row_starts
        self.block_segment_counts = block_segment_counts
        self.use_packed = use_packed


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    *,
    use_packed: bool = True,
) -> MoeAlignResult:
    if use_packed:
        return moe_align_block_size_packed(
            topk_ids, block_size, num_experts, expert_map=expert_map
        )

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
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
    return MoeAlignResult(sorted_ids, expert_ids, num_tokens_post_pad, use_packed=False)


def moe_align_block_size_packed(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
) -> MoeAlignResult:
    numel = topk_ids.numel()
    max_num_m_blocks = numel + 1
    sorted_ids = torch.empty((numel,), dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    block_token_offsets = torch.empty(
        (max_num_m_blocks + 1,), dtype=torch.int32, device=topk_ids.device
    )
    block_num_segments = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
    flat_segments = max_num_m_blocks * MAX_BLOCK_SEGMENTS
    block_segment_experts = torch.empty(
        (flat_segments,), dtype=torch.int32, device=topk_ids.device
    )
    block_segment_row_starts = torch.empty(
        (flat_segments,), dtype=torch.int32, device=topk_ids.device
    )
    block_segment_counts = torch.empty(
        (flat_segments,), dtype=torch.int32, device=topk_ids.device
    )
    ops.moe_align_block_size_packed(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        block_token_offsets,
        block_num_segments,
        block_segment_experts,
        block_segment_row_starts,
        block_segment_counts,
        expert_map,
    )
    num_moe_blocks = int(num_tokens_post_pad.item()) // block_size
    return MoeAlignResult(
        sorted_ids,
        expert_ids[:num_moe_blocks],
        num_tokens_post_pad,
        block_token_offsets=block_token_offsets[: num_moe_blocks + 1],
        block_num_segments=block_num_segments[:num_moe_blocks],
        block_segment_experts=block_segment_experts[: num_moe_blocks * MAX_BLOCK_SEGMENTS],
        block_segment_row_starts=block_segment_row_starts[
            : num_moe_blocks * MAX_BLOCK_SEGMENTS
        ],
        block_segment_counts=block_segment_counts[: num_moe_blocks * MAX_BLOCK_SEGMENTS],
        use_packed=True,
    )


def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type_id: int,
    moe_block_size: int = 16,
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
    use_packed_align: bool = True,
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
    align = moe_align_block_size(
        topk_ids, moe_block_size, w1.shape[0], use_packed=use_packed_align
    )
    if workspace is None:
        props = torch.cuda.get_device_properties(hidden_states.device)
        max_blocks_per_sm = 6 if props.major >= 9 else 4
        workspace = torch.zeros(
            props.multi_processor_count * max_blocks_per_sm,
            dtype=torch.int,
            device=hidden_states.device,
        )

    packed_kwargs = {}
    if align.use_packed:
        packed_kwargs = dict(
            block_token_offsets_or_none=align.block_token_offsets,
            block_num_segments_or_none=align.block_num_segments,
            block_segment_experts_or_none=align.block_segment_experts,
            block_segment_row_starts_or_none=align.block_segment_row_starts,
            block_segment_counts_or_none=align.block_segment_counts,
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
        align.sorted_ids,
        align.expert_ids,
        align.num_tokens_post_pad,
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
        **packed_kwargs,
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
        align.sorted_ids,
        align.expert_ids,
        align.num_tokens_post_pad,
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
        **packed_kwargs,
    )
    return output.view(m, topk, output_size).sum(dim=1)
