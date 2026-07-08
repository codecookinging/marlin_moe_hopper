#pragma once

#include <cuda_runtime.h>

#include "core/scalar_type.hpp"

namespace marlin_moe_cutlass69_host {

struct HostSupport {
  bool supported = false;
  const char* reason = nullptr;
};

bool cutlass69_compiled();
bool cutlass69_env_enabled();

HostSupport select_host_path(int major_capability, int a_bits, int b_bits,
                             int prob_m, int prob_n, int prob_k,
                             bool has_act_order, bool has_zp, int moe_block_size,
                             int group_size);

void dispatch_marlin_moe_cutlass69(
    const void* A, const void* B, void* C, const void* b_scales,
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    const int32_t* num_tokens_past_padded, const float* topk_weights,
    int moe_block_size, int num_experts, int top_k, bool mul_topk_weights,
    int prob_m, int prob_n, int prob_k, vllm::ScalarType const& a_type,
    vllm::ScalarType const& b_type, vllm::ScalarType const& c_type,
    int group_size, int dev, cudaStream_t stream);

}  // namespace marlin_moe_cutlass69_host
