#ifndef MARLIN_NAMESPACE_NAME
#define MARLIN_NAMESPACE_NAME marlin_moe_wna16
#endif

#include "marlin_moe_cluster_tail.cuh"

#include <algorithm>
#include <cuda_runtime.h>

#include "core/scalar_type.hpp"
#include "marlin_moe_cluster_cache.h"

namespace marlin_moe_host {

inline int tail_smem_bytes(int thread_n_blocks, int num_floats) {
  int sh_red = (2 * thread_n_blocks + 1) * 16 * static_cast<int>(sizeof(int4));
  int sh_pack = num_floats * static_cast<int>(sizeof(float));
  return std::max(sh_red, sh_pack);
}

template <const vllm::ScalarTypeId c_type_id, int thread_n_blocks, int num_threads,
          int num_floats>
void launch_tail_kernel(float* partials, const int* tail_meta, int4* C,
                        const int32_t* sorted_token_ids,
                        const float* topk_weights, int prob_m, int prob_n,
                        int top_k, int moe_block_size, bool mul_topk_weights,
                        int num_pairs, cudaStream_t stream) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
  if (num_pairs <= 0) {
    return;
  }
  auto* kernel =
      marlin_moe_tail::marlin_moe_cluster_tail_kernel<c_type_id, thread_n_blocks,
                                                    num_threads, num_floats>;
  const void* kernel_ptr = reinterpret_cast<const void*>(kernel);
  const int smem = tail_smem_bytes(thread_n_blocks, num_floats);
  const int max_cluster =
      query_max_cluster_size_cached(kernel_ptr, num_threads, smem);
  int cluster_size = 2;
  if (max_cluster > 0) {
    cluster_size = std::min(max_cluster, cluster_size);
  }
  if (cluster_size < 2) {
    return;
  }
  ensure_non_portable_cluster_attr_cached(kernel_ptr);
  const int blocks = num_pairs * cluster_size;
  cudaLaunchConfig_t config{};
  config.gridDim = blocks;
  config.blockDim = num_threads;
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = cluster_size;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;
  config.attrs = &attr;
  config.numAttrs = 1;
  cudaLaunchKernelEx(
      &config, kernel, partials, tail_meta, C, sorted_token_ids, topk_weights,
      prob_m, prob_n, top_k, moe_block_size, mul_topk_weights, num_pairs);
#else
  (void)partials;
  (void)tail_meta;
  (void)C;
  (void)sorted_token_ids;
  (void)topk_weights;
  (void)prob_m;
  (void)prob_n;
  (void)top_k;
  (void)moe_block_size;
  (void)mul_topk_weights;
  (void)num_pairs;
  (void)stream;
#endif
}

void dispatch_marlin_moe_cluster_tail(
    vllm::ScalarTypeId c_type_id, int thread_m_blocks, int thread_n_blocks,
    int num_threads, bool m_block_size_8, bool is_a_8bit, int num_floats,
    float* partials, const int* tail_meta, int4* C,
    const int32_t* sorted_token_ids, const float* topk_weights, int prob_m,
    int prob_n, int top_k, int moe_block_size, bool mul_topk_weights,
    int num_pairs, cudaStream_t stream) {
  if (num_pairs <= 0) {
    return;
  }
  if (thread_m_blocks != 1 || m_block_size_8 || is_a_8bit) {
    return;
  }
  if (c_type_id == vllm::kBFloat16.id() && thread_n_blocks == 8 &&
      num_threads == 256 && num_floats == 32) {
    launch_tail_kernel<vllm::kBFloat16.id(), 8, 256, 32>(
        partials, tail_meta, C, sorted_token_ids, topk_weights, prob_m, prob_n,
        top_k, moe_block_size, mul_topk_weights, num_pairs, stream);
    return;
  }
  if (c_type_id == vllm::kFloat16.id() && thread_n_blocks == 8 &&
      num_threads == 256 && num_floats == 32) {
    launch_tail_kernel<vllm::kFloat16.id(), 8, 256, 32>(
        partials, tail_meta, C, sorted_token_ids, topk_weights, prob_m, prob_n,
        top_k, moe_block_size, mul_topk_weights, num_pairs, stream);
    return;
  }
  if (c_type_id == vllm::kBFloat16.id() && thread_n_blocks == 4 &&
      num_threads == 256 && num_floats == 32) {
    launch_tail_kernel<vllm::kBFloat16.id(), 4, 256, 32>(
        partials, tail_meta, C, sorted_token_ids, topk_weights, prob_m, prob_n,
        top_k, moe_block_size, mul_topk_weights, num_pairs, stream);
  }
}

}  // namespace marlin_moe_host
