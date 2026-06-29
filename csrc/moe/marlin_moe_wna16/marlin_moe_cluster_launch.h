#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "kernel.h"
#include "marlin_moe_cluster_cache.h"
#include "marlin_moe_cluster_tail.h"
#include "core/scalar_type.hpp"

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

// Mirror device init_part2_slice() slice_count for the first Part2 matmul slice.
inline int part2_first_slice_count(int cta_block, int k_tiles, int iters,
                                     int part2_mn_tiles) {
  int slice_col_par = (iters * cta_block) / k_tiles;
  int slice_row = (iters * cta_block) % k_tiles;
  int slice_iters =
      iters * (cta_block + 1) - (k_tiles * slice_col_par + slice_row);
  if (slice_iters < 0 || slice_col_par >= part2_mn_tiles) {
    return 0;
  }
  if (slice_row + slice_iters > k_tiles) {
    slice_iters = k_tiles - slice_row;
  }
  if (slice_iters == 0) {
    return 0;
  }

  int slice_count = 1;
  int col_first = iters * div_ceil(k_tiles * slice_col_par, iters);
  if (col_first <= k_tiles * (slice_col_par + 1)) {
    int col_off = col_first - k_tiles * slice_col_par;
    slice_count = div_ceil(k_tiles - col_off, iters);
    if (col_off > 0) {
      slice_count++;
    }
  }
  return slice_count;
}

struct Part2SliceStats {
  int active_blocks = 0;
  int slice_eq_2 = 0;
  int slice_gt_2 = 0;
};

inline Part2SliceStats simulate_part2_slice_stats(int logical_blocks,
                                                    int k_tiles, int iters,
                                                    int part2_mn_tiles) {
  Part2SliceStats stats{};
  for (int cta_block = 0; cta_block < logical_blocks; ++cta_block) {
    int slice_count =
        part2_first_slice_count(cta_block, k_tiles, iters, part2_mn_tiles);
    if (slice_count <= 0) {
      continue;
    }
    stats.active_blocks++;
    if (slice_count == 2) {
      stats.slice_eq_2++;
    } else if (slice_count > 2) {
      stats.slice_gt_2++;
    }
  }
  return stats;
}

struct TailClusterPlan {
  bool use_tail_cluster = false;
  int num_pairs = 0;
  int num_floats = 0;
  std::vector<int32_t> cta_pair_id;
};

struct ClusterLaunchPlan {
  bool use_cluster = false;
  int cluster_size = 1;
  int launch_blocks = 0;
  int logical_blocks = 0;
  bool use_atomic_add = true;
  bool use_fp32_reduce = false;
};

inline bool env_flag_enabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] == '1';
}

// Whole-grid clusterDim=2 launch taxes Part1 GEMM and every CTA. Microbench
// shows isolated DSMEM reduce is not faster than atomic. Only opt in when
// explicitly requested for experiments (not production).
inline bool allow_whole_kernel_cluster_launch() {
  return env_flag_enabled("MARLIN_MOE_WHOLE_KERNEL_CLUSTER");
}

// Minimum fraction of Part2-active CTAs that hit slice_count==2 before we even
// consider whole-kernel cluster launch (still gated by allow_* above).
constexpr float kMinSliceEq2Fraction = 0.80f;

inline bool cluster_roi_allows_whole_kernel_launch(
    const Part2SliceStats& stats) {
  if (stats.active_blocks <= 0) {
    return false;
  }
  float eq2_frac =
      static_cast<float>(stats.slice_eq_2) /
      static_cast<float>(stats.active_blocks);
  return eq2_frac >= kMinSliceEq2Fraction && stats.slice_gt_2 == 0;
}

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

inline int query_max_cluster_size(MarlinFuncPtr kernel, int num_threads,
                                  int max_shared_mem) {
  return query_max_cluster_size_cached(reinterpret_cast<const void*>(kernel),
                                       num_threads, max_shared_mem);
}

inline void ensure_non_portable_cluster_attr(MarlinFuncPtr kernel) {
  ensure_non_portable_cluster_attr_cached(reinterpret_cast<const void*>(kernel));
}

inline TailClusterPlan compute_tail_cluster_plan(
    int major_capability, int parallel_padded, int prob_k, int prob_n,
    int thread_k_blocks, int thread_n_blocks, int logical_blocks,
    int thread_m_blocks, bool m_block_size_8, bool is_a_8bit,
    vllm::ScalarTypeId c_type_id, int num_threads) {
  TailClusterPlan plan{};
  if (env_flag_enabled("MARLIN_MOE_DISABLE_CLUSTER_REDUCE") ||
      env_flag_enabled("MARLIN_MOE_DISABLE_TAIL_CLUSTER_REDUCE")) {
    return plan;
  }
  if (major_capability < 9) {
    return plan;
  }
  if (thread_m_blocks != 1 || m_block_size_8 || is_a_8bit) {
    return plan;
  }
  if (!needs_part2_streamk_tail(parallel_padded, prob_k, prob_n,
                                thread_k_blocks, thread_n_blocks,
                                logical_blocks)) {
    return plan;
  }

  StreamKHostParams sk = compute_streamk_host_params(
      parallel_padded, prob_k, prob_n, thread_k_blocks, thread_n_blocks,
      logical_blocks);
  if (!can_use_identity_pair_cluster(logical_blocks, 2, sk.k_tiles, sk.iters,
                                     sk.part2_mn_tiles)) {
    return plan;
  }

  plan.num_floats = thread_m_blocks * (is_a_8bit ? 2 : 4) * 2 * 4;
  plan.cta_pair_id.assign(logical_blocks, -1);

  int b = 0;
  while (b < logical_blocks) {
    int group_key = (sk.iters * b) / sk.k_tiles;
    int end = b + 1;
    while (end < logical_blocks &&
           (sk.iters * end) / sk.k_tiles == group_key) {
      ++end;
    }
    if (end - b == 2) {
      int sc0 =
          part2_first_slice_count(b, sk.k_tiles, sk.iters, sk.part2_mn_tiles);
      int sc1 = part2_first_slice_count(b + 1, sk.k_tiles, sk.iters,
                                        sk.part2_mn_tiles);
      if (sc0 == 2 && sc1 == 2) {
        int pair_id = plan.num_pairs++;
        plan.cta_pair_id[b] = pair_id;
        plan.cta_pair_id[b + 1] = pair_id;
      }
    }
    b = end;
  }

  if (plan.num_pairs <= 0) {
    return plan;
  }

  const bool tail_dispatch_ok =
      (c_type_id == vllm::kBFloat16.id() || c_type_id == vllm::kFloat16.id()) &&
      plan.num_floats == 32 && num_threads == 256 &&
      (thread_n_blocks == 8 || thread_n_blocks == 4);
  if (!tail_dispatch_ok) {
    plan.num_pairs = 0;
    plan.cta_pair_id.assign(logical_blocks, -1);
    return plan;
  }

  plan.use_tail_cluster = true;
  if (env_flag_enabled("MARLIN_MOE_CLUSTER_DEBUG")) {
    fprintf(stderr,
            "[Marlin MoE tail cluster] enable pairs=%d floats=%d blocks=%d\n",
            plan.num_pairs, plan.num_floats, logical_blocks);
  }
  return plan;
}

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

  if (env_flag_enabled("MARLIN_MOE_DISABLE_CLUSTER_REDUCE")) {
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

  // Production default: normal launch + atomic reduce (same as non-cluster).
  // Whole-kernel cluster launch is opt-in only and still ROI-gated.
  if (!allow_whole_kernel_cluster_launch()) {
    return plan;
  }

  int cluster_size = 2;
  int max_cluster =
      query_max_cluster_size(kernel, num_threads, max_shared_mem);
  if (max_cluster > 0) {
    cluster_size = std::min(max_cluster, cluster_size);
  }
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

  Part2SliceStats stats = simulate_part2_slice_stats(
      logical_blocks, sk.k_tiles, sk.iters, sk.part2_mn_tiles);
  if (!cluster_roi_allows_whole_kernel_launch(stats)) {
    if (env_flag_enabled("MARLIN_MOE_CLUSTER_DEBUG")) {
      fprintf(stderr,
              "[Marlin MoE cluster] skip whole-kernel cluster: "
              "active=%d eq2=%d gt2=%d (need eq2>=%.0f%%, gt2=0)\n",
              stats.active_blocks, stats.slice_eq_2, stats.slice_gt_2,
              kMinSliceEq2Fraction * 100.0f);
    }
    return plan;
  }

  plan.use_cluster = true;
  plan.cluster_size = cluster_size;
  plan.launch_blocks = logical_blocks;
  // Keep atomic enabled: slice_count!=2 and cluster failure fall back to atomic
  // without forcing lock+global reduce.
  plan.use_atomic_add = use_atomic_add_in;
  plan.use_fp32_reduce = use_fp32_reduce_in;
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
    bool has_bias, bool use_atomic_add, bool use_fp32_reduce,
    float* cluster_partials, const int* cluster_cta_pair_id,
    int* cluster_tail_meta, bool use_tail_cluster_reduce) {
  static_assert(UseClusterReduce == true || UseClusterReduce == false,
                "UseClusterReduce must be a compile-time boolean");

  constexpr bool kUseClusterReduce = UseClusterReduce;

  if constexpr (UseClusterReduce) {
    ensure_non_portable_cluster_attr(kernel);
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
        locks, has_bias, use_atomic_add, use_fp32_reduce, kUseClusterReduce,
        cluster_partials, cluster_cta_pair_id, cluster_tail_meta,
        use_tail_cluster_reduce);
  } else {
    kernel<<<plan.launch_blocks, num_threads, max_shared_mem, stream>>>(
        A, B, C, C_tmp, bias_ptr, a_s_ptr, b_s_ptr, g_s_ptr, zp_ptr, g_idx_ptr,
        sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,
        topk_weights_ptr, top_k, mul_topk_weights, num_groups, prob_m, prob_n,
        prob_k, plan.logical_blocks, nullptr, locks, has_bias, use_atomic_add,
        use_fp32_reduce, kUseClusterReduce, cluster_partials,
        cluster_cta_pair_id, cluster_tail_meta, use_tail_cluster_reduce);
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
    bool has_bias, bool use_atomic_add, bool use_fp32_reduce,
    float* cluster_partials, const int* cluster_cta_pair_id,
    int* cluster_tail_meta, bool use_tail_cluster_reduce) {
  if (plan.use_cluster) {
    launch_marlin_moe_kernel<true>(
        kernel, plan, num_threads, max_shared_mem, stream, A, B, C, C_tmp,
        bias_ptr, a_s_ptr, b_s_ptr, g_s_ptr, zp_ptr, g_idx_ptr,
        sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,
        topk_weights_ptr, top_k, mul_topk_weights, num_groups, prob_m, prob_n,
        prob_k, locks, has_bias, use_atomic_add, use_fp32_reduce,
        cluster_partials, cluster_cta_pair_id, cluster_tail_meta,
        use_tail_cluster_reduce);
  } else {
    launch_marlin_moe_kernel<false>(
        kernel, plan, num_threads, max_shared_mem, stream, A, B, C, C_tmp,
        bias_ptr, a_s_ptr, b_s_ptr, g_s_ptr, zp_ptr, g_idx_ptr,
        sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,
        topk_weights_ptr, top_k, mul_topk_weights, num_groups, prob_m, prob_n,
        prob_k, locks, has_bias, use_atomic_add, use_fp32_reduce,
        cluster_partials, cluster_cta_pair_id, cluster_tail_meta,
        use_tail_cluster_reduce);
  }
}

template <typename MarlinFuncPtrT>
inline void dispatch_marlin_moe_launch_and_tail(
    MarlinFuncPtrT kernel, const ClusterLaunchPlan& plan,
    const TailClusterPlan& tail_plan, int num_threads, int max_shared_mem,
    cudaStream_t stream, const int4* A, const int4* B, int4* C, int4* C_tmp,
    const int4* bias_ptr, const float* a_s_ptr, const int4* b_s_ptr,
    const float* g_s_ptr, const int4* zp_ptr, const int* g_idx_ptr,
    const int32_t* sorted_token_ids_ptr, const int32_t* expert_ids_ptr,
    const int32_t* num_tokens_past_padded_ptr, const float* topk_weights_ptr,
    int top_k, bool mul_topk_weights, int num_groups, int prob_m, int prob_n,
    int prob_k, int* locks, bool has_bias, bool use_atomic_add,
    bool use_fp32_reduce, float* cluster_partials,
    const int* cluster_cta_pair_id, int* cluster_tail_meta,
    vllm::ScalarTypeId c_type_id, int thread_m_blocks, int thread_n_blocks,
    bool m_block_size_8, bool is_a_8bit, int moe_block_size, int top_k) {
  const bool use_tail = tail_plan.use_tail_cluster && cluster_partials != nullptr;
  dispatch_marlin_moe_launch(
      kernel, plan, num_threads, max_shared_mem, stream, A, B, C, C_tmp,
      bias_ptr, a_s_ptr, b_s_ptr, g_s_ptr, zp_ptr, g_idx_ptr,
      sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,
      topk_weights_ptr, top_k, mul_topk_weights, num_groups, prob_m, prob_n,
      prob_k, locks, has_bias, use_atomic_add, use_fp32_reduce,
      cluster_partials, cluster_cta_pair_id, cluster_tail_meta, use_tail);
  if (use_tail) {
    dispatch_marlin_moe_cluster_tail(
        c_type_id, thread_m_blocks, thread_n_blocks, num_threads,
        m_block_size_8, is_a_8bit, tail_plan.num_floats, cluster_partials,
        cluster_tail_meta, C, sorted_token_ids_ptr, topk_weights_ptr, prob_m,
        prob_n, top_k, moe_block_size, mul_topk_weights, tail_plan.num_pairs,
        stream);
  }
}

}  // namespace marlin_moe_host
