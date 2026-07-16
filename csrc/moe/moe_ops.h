#pragma once

#include <torch/all.h>

void topk_softmax(torch::Tensor& topk_weights, torch::Tensor& topk_indices,
                  torch::Tensor& token_expert_indices,
                  torch::Tensor& gating_output, bool renormalize,
                  std::optional<torch::Tensor> bias);

void topk_sigmoid(torch::Tensor& topk_weights, torch::Tensor& topk_indices,
                  torch::Tensor& token_expert_indices,
                  torch::Tensor& gating_output, bool renormalize,
                  std::optional<torch::Tensor> bias);

void moe_sum(torch::Tensor& input, torch::Tensor& output);

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad,
                          std::optional<torch::Tensor> maybe_expert_map);

void batched_moe_align_block_size(int64_t max_tokens_per_batch,
                                  int64_t block_size,
                                  torch::Tensor const& expert_num_tokens,
                                  torch::Tensor sorted_ids,
                                  torch::Tensor expert_ids,
                                  torch::Tensor num_tokens_post_pad);

void moe_lora_align_block_size(
    torch::Tensor topk_ids, torch::Tensor token_lora_mapping,
    int64_t num_experts, int64_t block_size, int64_t max_loras,
    int64_t max_num_tokens_padded, int64_t max_num_m_blocks,
    torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad, torch::Tensor adapter_enabled,
    torch::Tensor lora_ids, std::optional<torch::Tensor> maybe_expert_map);
#ifndef USE_ROCM
torch::Tensor moe_wna16_gemm(torch::Tensor input, torch::Tensor output,
                             torch::Tensor b_qweight, torch::Tensor b_scales,
                             std::optional<torch::Tensor> b_qzeros,
                             std::optional<torch::Tensor> topk_weights,
                             torch::Tensor sorted_token_ids,
                             torch::Tensor expert_ids,
                             torch::Tensor num_tokens_post_pad, int64_t top_k,
                             int64_t BLOCK_SIZE_M, int64_t BLOCK_SIZE_N,
                             int64_t BLOCK_SIZE_K, int64_t bit);

std::tuple<torch::Tensor, torch::Tensor> grouped_topk(
    torch::Tensor const& scores, int64_t n_group, int64_t topk_group,
    int64_t topk, bool renormalize, double routed_scaling_factor,
    torch::Tensor const& bias, int64_t scoring_func);
#endif

bool moe_permute_unpermute_supported();

void shuffle_rows(const torch::Tensor& input_tensor,
                  const torch::Tensor& dst2src_map,
                  torch::Tensor& output_tensor);

#ifndef USE_ROCM
// cuBLAS bf16 x bf16 -> fp32 router GEMM (fallback for non-SM90 / batch > 16)
torch::Tensor router_gemm_bf16_fp32(torch::Tensor const& input,
                                    torch::Tensor const& weight);

// DeepSeek V3 optimized router GEMM kernel for SM90+
// Computes output = mat_a @ mat_b.T where:
//   mat_a: [num_tokens, hidden_dim] in bf16
//   mat_b: [num_experts, hidden_dim] in bf16
//   output: [num_tokens, num_experts] in bf16 or fp32
// Supports num_tokens in [1, 16], num_experts in {256, 384}, hidden_dim = 7168
void dsv3_router_gemm(torch::Tensor& output, const torch::Tensor& mat_a,
                      const torch::Tensor& mat_b);

torch::Tensor cutlass69_pack_only(torch::Tensor q_weight_int8);
torch::Tensor cutlass69_reorder_packed(torch::Tensor packed);
torch::Tensor cutlass69_dequant_packed(torch::Tensor packed, torch::Tensor scales,
                                       int64_t group_size);
torch::Tensor cutlass69_pack_only(torch::Tensor q_weight_int8);
torch::Tensor cutlass69_pack_only(torch::Tensor q_weight_int8);
torch::Tensor cutlass69_pack_and_reorder(torch::Tensor q_weight_int8);
torch::Tensor cutlass69_dequant_packed(torch::Tensor packed, torch::Tensor scales,
                                       int64_t group_size);
torch::Tensor cutlass69_dequant_reordered(torch::Tensor reordered,
                                          torch::Tensor scales,
                                          int64_t group_size);
torch::Tensor cutlass69_dequant_packed(torch::Tensor packed,
                                       torch::Tensor scales,
                                       int64_t group_size);
torch::Tensor moe_cutlass69_fused_moe(
    torch::Tensor hidden, torch::Tensor w1, torch::Tensor w1_scales,
    torch::Tensor w2, torch::Tensor w2_scales, torch::Tensor topk_weights,
    torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
    torch::Tensor num_tokens_past_padded, int64_t moe_block_size, int64_t top_k,
    int64_t prob_m, int64_t prob_n1, int64_t prob_k1, int64_t prob_n2,
    int64_t prob_k2, int64_t group_size);
#endif
