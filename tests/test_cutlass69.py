import os
import torch
import pytest
from marlin_v100 import moe, ops, dense
from .helpers import scalar_types, marlin_quantize_experts, marlin_make_workspace_new, marlin_make_empty_g_idx, marlin_moe_reference
from .cutlass_helpers import cutlass69_quantize_experts

@pytest.mark.parametrize("m", [16, 64])
@pytest.mark.parametrize("fused", [False, True])
def test_cutlass69(m, fused):
    if fused:
        os.environ["MARLIN_MOE_USE_CUTLASS69_FUSED_GEMM1"] = "1"
        os.environ["MARLIN_MOE_USE_CUTLASS69"] = "0"
    else:
        os.environ["MARLIN_MOE_USE_CUTLASS69_FUSED_GEMM1"] = "0"
        os.environ["MARLIN_MOE_USE_CUTLASS69"] = "1"
        
    num_experts = 8
    topk = 2
    k = 6144
    n = 256
    if fused:
        n = 512 # intermediate size for GEMM1 is 2N
    
    hidden_states = torch.randn((m, k), dtype=torch.float16, device="cuda") / 10
    w1 = torch.randn((num_experts, k, n), dtype=torch.float16, device="cuda") / 10
    
    topk_weights = torch.rand((m, topk), dtype=torch.float32, device="cuda")
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = torch.randint(0, num_experts, (m, topk), dtype=torch.int32, device="cuda")
    
    quant_type = scalar_types.uint4b8
    group_size = 128
    
    # Quantize to INT8
    q_weights_int8, scales = cutlass69_quantize_experts(w1, quant_type, group_size)
    
    # Pack and reorder using our new C++ function
    q_weights_cutlass = ops.cutlass69_pack_and_reorder(q_weights_int8.to("cuda"))
    scales = scales.to("cuda")
    
    # We also need Marlin packed weights for reference
    q_weights_marlin, scales_marlin, _ = marlin_quantize_experts(
        w1, quant_type, group_size, False
    )
    q_weights_marlin = q_weights_marlin.to("cuda")
    scales_marlin = scales_marlin.to("cuda")
    
    workspace = marlin_make_workspace_new(hidden_states.device)
    g_idx = marlin_make_empty_g_idx(hidden_states.device)
    sort_indices = torch.empty(0, dtype=torch.int, device=hidden_states.device)
    
    moe_block_size = 16
    sorted_ids, expert_ids, num_tokens_post_pad = moe.moe_align_block_size(
        topk_ids, moe_block_size, num_experts
    )
    
    # Run Marlin
    os.environ["MARLIN_MOE_USE_CUTLASS69"] = "0"
    os.environ["MARLIN_MOE_USE_CUTLASS69_FUSED_GEMM1"] = "0"
    
    out_marlin = torch.empty((m * topk, n), dtype=torch.float16, device="cuda")
    ops.moe_wna16_marlin_gemm(
        hidden_states,
        out_marlin,
        q_weights_marlin,
        None,
        scales_marlin,
        None, None, None, g_idx, sort_indices, workspace,
        sorted_ids, expert_ids, num_tokens_post_pad, topk_weights,
        moe_block_size, topk, False, quant_type.id,
        m, n, k, True, False, True, False, -1, -1, -1, False
    )
    
    # Run CUTLASS
    if fused:
        os.environ["MARLIN_MOE_USE_CUTLASS69_FUSED_GEMM1"] = "1"
        out_cutlass = torch.empty((m * topk, n // 2), dtype=torch.float16, device="cuda")
    else:
        os.environ["MARLIN_MOE_USE_CUTLASS69"] = "1"
        out_cutlass = torch.empty((m * topk, n), dtype=torch.float16, device="cuda")
        
    ops.moe_wna16_marlin_gemm(
        hidden_states,
        out_cutlass,
        q_weights_cutlass,
        None,
        scales,
        None, None, None, g_idx, sort_indices, workspace,
        sorted_ids, expert_ids, num_tokens_post_pad, topk_weights,
        moe_block_size, topk, False, quant_type.id,
        m, n, k, True, False, True, False, -1, -1, -1, False
    )
    
    if fused:
        gate, up = out_marlin.chunk(2, dim=-1)
        out_marlin = torch.nn.functional.silu(gate) * up
        
    torch.testing.assert_close(out_cutlass, out_marlin, rtol=1e-2, atol=1e-2)

