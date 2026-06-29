#pragma once

#include <algorithm>
#include <cstdlib>

#include <cuda_runtime.h>

#include "kernel.h"

namespace marlin_moe_host {

inline int div_ceil(int a, int b) { return (a + b - 1) / b; }

struct StreamKHostParams {
  int k_tiles;
  int part2_mn_tiles;
  int iters;
};

inline StreamKHostParams compute_streamk_host_params(int parallel_padded,
                                                     int prob_k, int prob_n,
                                                     int thread_k_blocks,
                                                     int thread_n_blocks,
                                                     int logical_blocks) {
  int k_tiles = prob_k / 16 / thread_k_blocks;
  int n_tiles = prob_n / 16 / thread_n_blocks;
  int global_mn_tiles = parallel_padded * n_tiles;
  int part2_mn_tiles = global_mn_tiles;
  if (global_mn_tiles > logical_blocks) {
    part2_mn_tiles = global_mn_tiles % logical_blocks;
    if (part2_mn_tiles * 3 <= logical_blocks) {
      part2_mn_tiles += logical_blocks;
    }
  }
  int iters = div_ceil(k_tiles * part2_mn_tiles, logical_blocks);
  return {k_tiles, part2_mn_tiles, iters};
}

inline bool needs_part2_streamk_tail(int parallel_padded, int prob_k,
                                     int prob_n, int thread_k_blocks,
                                     int thread_n_blocks, int logical_blocks) {
  int n_tiles = prob_n / 16 / thread_n_blocks;
  int global_mn_tiles = parallel_padded * n_tiles;
  return global_mn_tiles > logical_blocks;
}

inline bool can_use_identity_pair_cluster(int logical_blocks, int cluster_size,
                                        int k_tiles, int iters,
                                        int part2_mn_tiles) {
  if (cluster_size != 2 || part2_mn_tiles >= logical_blocks) {
    return false;
  }
  int b = 0;
  while (b < logical_blocks) {
    int group_key = (iters * b) / k_tiles;
    int end = b + 1;
    while (end < logical_blocks && (iters * end) / k_tiles == group_key) {
      ++end;
    }
    if (b % cluster_size != 0 || end - b != cluster_size) {
      return false;
    }
    b = end;
  }
  return true;
}

struct ClusterLaunchPlan {
  bool use_cluster = false;
  int cluster_size = 1;
  int launch_blocks = 0;
  int logical_blocks = 0;
  bool use_atomic_add = true;
  bool use_fp32_reduce = false;
};

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

inline ClusterLaunchPlan compute_cluster_launch_plan(
    int major_capability, int parallel_padded, int prob_k, int prob_n,
    int thread_k_blocks, int thread_n_blocks, int logical_blocks,
    int launch_blocks, bool use_atomic_add_in, bool use_fp32_reduce_in,
    MarlinFuncPtr kernel, int num_threads, int max_shared_mem) {
  ClusterLaunchPlan plan{};
  plan.launch_blocks = launch_blocks;
  plan.logical_blocks = logical_blocks;
  plan.use_atomic_add = use_atomic_add_in;
  plan.use_fp32_reduce = use_fp32_reduce_in;

  const char* disable_cluster_env =
      std::getenv("MARLIN_MOE_DISABLE_CLUSTER_REDUCE");
  if (disable_cluster_env != nullptr && disable_cluster_env[0] == '1') {
    return plan;
  }
  if (major_capability < 9) {
    return plan;
  }
  if (!needs_part2_streamk_tail(parallel_padded, prob_k, prob_n,
                                thread_k_blocks, thread_n_blocks,
                                logical_blocks)) {
    return plan;
  }

  int cluster_size = 2;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
  cudaLaunchConfig_t occ_cfg{};
  occ_cfg.blockDim = num_threads;
  occ_cfg.dynamicSmemBytes = max_shared_mem;
  int max_cluster = 0;
  if (cudaOccupancyMaxPotentialClusterSize(&max_cluster, kernel, &occ_cfg) ==
          cudaSuccess &&
      max_cluster > 0) {
    cluster_size = std::min(max_cluster, cluster_size);
  }
#endif
  if (cluster_size <= 1 || logical_blocks % cluster_size != 0) {
    return plan;
  }

  StreamKHostParams sk = compute_streamk_host_params(
      parallel_padded, prob_k, prob_n, thread_k_blocks, thread_n_blocks,
      logical_blocks);
  if (!can_use_identity_pair_cluster(logical_blocks, cluster_size, sk.k_tiles,
                                     sk.iters, sk.part2_mn_tiles)) {
    return plan;
  }

  plan.use_cluster = true;
  plan.cluster_size = cluster_size;
  plan.launch_blocks = logical_blocks;
  plan.use_atomic_add = false;
  plan.use_fp32_reduce = false;
  return plan;
}

// Host-side compile-time specialization: only the cluster launch path is
// compiled into the UseClusterReduce=true instantiation.
template <bool UseClusterReduce>
inline void launch_marlin_moe_kernel(
    MarlinFuncPtr kernel, const ClusterLaunchPlan& plan, int num_threads,
    int max_shared_mem, cudaStream_t stream, const int4* A, const int4* B,
    int4* C, int4* C_tmp, const int4* bias_ptr, const float* a_s_ptr,
    const int4* b_s_ptr, const float* g_s_ptr, const int4* zp_ptr,
    const int* g_idx_ptr, const int32_t* sorted_token_ids_ptr,
    const int32_t* expert_ids_ptr, const int32_t* num_tokens_past_padded_ptr,
    const float* topk_weights_ptr, int top_k, bool mul_topk_weights,
    int num_groups, int prob_m, int prob_n, int prob_k, int* locks,
    bool has_bias, bool use_atomic_add, bool use_fp32_reduce) {
  static_assert(UseClusterReduce == true || UseClusterReduce == false,
                "UseClusterReduce must be a compile-time boolean");

  constexpr bool kUseClusterReduce = UseClusterReduce;

  if constexpr (UseClusterReduce) {
    cudaFuncSetAttribute(kernel,
                         cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaLaunchConfig_t config{};
    config.gridDim = plan.launch_blocks;
    config.blockDim = num_threads;
    config.dynamicSmemBytes = max_shared_mem;
    config.stream = stream;
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = plan.cluster_size;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;
    config.attrs = &attr;
    config.numAttrs = 1;
    cudaLaunchKernelEx(
        &config, kernel, A, B, C, C_tmp, bias_ptr, a_s_ptr, b_s_ptr, g_s_ptr,
        zp_ptr, g_idx_ptr, sorted_token_ids_ptr, expert_ids_ptr,
        num_tokens_past_padded_ptr, topk_weights_ptr, top_k, mul_topk_weights,
        num_groups, prob_m, prob_n, prob_k, plan.logical_blocks, nullptr,
        locks, has_bias, use_atomic_add, use_fp32_reduce, kUseClusterReduce);
  } else {
    kernel<<<plan.launch_blocks, num_threads, max_shared_mem, stream>>>(
        A, B, C, C_tmp, bias_ptr, a_s_ptr, b_s_ptr, g_s_ptr, zp_ptr, g_idx_ptr,
        sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,
        topk_weights_ptr, top_k, mul_topk_weights, num_groups, prob_m, prob_n,
        prob_k, plan.logical_blocks, nullptr, locks, has_bias, use_atomic_add,
        use_fp32_reduce, kUseClusterReduce);
  }
}

template <typename MarlinFuncPtrT>
inline void dispatch_marlin_moe_launch(
    MarlinFuncPtrT kernel, const ClusterLaunchPlan& plan, int num_threads,
    int max_shared_mem, cudaStream_t stream, const int4* A, const int4* B,
    int4* C, int4* C_tmp, const int4* bias_ptr, const float* a_s_ptr,
    const int4* b_s_ptr, const float* g_s_ptr, const int4* zp_ptr,
    const int* g_idx_ptr, const int32_t* sorted_token_ids_ptr,
    const int32_t* expert_ids_ptr, const int32_t* num_tokens_past_padded_ptr,
    const float* topk_weights_ptr, int top_k, bool mul_topk_weights,
    int num_groups, int prob_m, int prob_n, int prob_k, int* locks,
    bool has_bias, bool use_atomic_add, bool use_fp32_reduce) {
  if (plan.use_cluster) {
    launch_marlin_moe_kernel<true>(
        kernel, plan, num_threads, max_shared_mem, stream, A, B, C, C_tmp,
        bias_ptr, a_s_ptr, b_s_ptr, g_s_ptr, zp_ptr, g_idx_ptr,
        sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,
        topk_weights_ptr, top_k, mul_topk_weights, num_groups, prob_m, prob_n,
        prob_k, locks, has_bias, use_atomic_add, use_fp32_reduce);
  } else {
    launch_marlin_moe_kernel<false>(
        kernel, plan, num_threads, max_shared_mem, stream, A, B, C, C_tmp,
        bias_ptr, a_s_ptr, b_s_ptr, g_s_ptr, zp_ptr, g_idx_ptr,
        sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,
        topk_weights_ptr, top_k, mul_topk_weights, num_groups, prob_m, prob_n,
        prob_k, locks, has_bias, use_atomic_add, use_fp32_reduce);
  }
}

}  // namespace marlin_moe_host
