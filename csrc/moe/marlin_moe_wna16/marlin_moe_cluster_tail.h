#pragma once

#include <cuda_runtime.h>

#include "core/scalar_type.hpp"

namespace marlin_moe_host {

void dispatch_marlin_moe_cluster_tail(
    vllm::ScalarTypeId c_type_id, int thread_m_blocks, int thread_n_blocks,
    int num_threads, bool m_block_size_8, bool is_a_8bit, int num_floats,
    float* partials, const int* tail_meta, int4* C,
    const int32_t* sorted_token_ids, const float* topk_weights, int prob_m,
    int prob_n, int top_k, int moe_block_size, bool mul_topk_weights,
    int num_pairs, cudaStream_t stream);

}  // namespace marlin_moe_host
