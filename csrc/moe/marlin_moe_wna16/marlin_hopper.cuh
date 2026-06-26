// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// H100 (SM90/Hopper) specific PTX helpers for marlin_template.h.
//
// All helpers are guarded by __CUDA_ARCH__ >= 900 so they compile safely on
// any target but only emit SM90 PTX when the device code path requires it.

#pragma once
#include <cuda.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

#include <cooperative_groups.h>

namespace marlin_hopper {

namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------
// cp.async variants with L2-friendly cache policy
// ---------------------------------------------------------------------------
//
// NOTE: cp.async supports .L2::64B/.L2::128B/.L2::256B prefetch-size hints,
// which are useful for contiguous reusable data such as weights and scales.
// Eviction-priority hints such as .L2::evict_first are not legal on cp.async
// and ptxas rejects them with:
//   Illegal modifier '.level::eviction_priority' for instruction 'cp.async'
//
// For scattered activation gathers we use plain .cg.  This bypasses L1 and
// avoids polluting it with one-shot MoE token reads, while leaving L2 eviction
// to hardware policy.

// For weight / scale tiles: request a 128-byte L2 prefetch around each 16-byte
// async copy.  The copy size remains 16 bytes; .L2::128B is only a prefetch
// hint for nearby data that subsequent lanes / iterations are likely to use.
__device__ __forceinline__ void cp_async4_l2_128(void* smem_ptr,
                                                 const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
               :
               : "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

// For activation tiles: use .cg so scattered reads don't thrash L1.

// Predicated variant matches the original cp_async4_pred() signature.
__device__ __forceinline__ void cp_async4_pred_cg(void* smem_ptr,
                                                  const void* glob_ptr,
                                                  bool pred = true) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  setp.ne.b32 p, %0, 0;\n"
      "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
      "}\n"
      :
      : "r"((int)pred), "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

// ---------------------------------------------------------------------------
// Cluster DSMEM Stream-K global reduce
// ---------------------------------------------------------------------------
//
// Reduces the common Stream-K two-CTA conflict through DSMEM. This is a narrow
// fast path: the host only enables cluster launch when every conflict group is
// exactly one CTA pair, so the device side can avoid metadata validation and
// multi-batch global merging.

struct ClusterReduceStatus {
  bool ok = false;
};

__device__ inline ClusterReduceStatus cluster_streamk_reduce(
    float* frag_c, int num_floats, int4* sh_red, int slice_idx,
    int slice_count) {
  ClusterReduceStatus status;
  cg::cluster_group cluster = cg::this_cluster();
  const int cluster_size = cluster.num_blocks();

  if (slice_count != 2 || cluster_size != 2 ||
      (slice_idx != 0 && slice_idx != 1)) {
    return status;
  }

  float* pack = reinterpret_cast<float*>(sh_red);
  for (int i = threadIdx.x; i < num_floats; i += blockDim.x) {
    pack[i] = frag_c[i];
  }
  __syncthreads();
  cluster.sync();

  if (slice_idx == 0) {
    float* peer_pack = cluster.map_shared_rank(pack, 1);
    for (int i = threadIdx.x; i < num_floats; i += blockDim.x) {
      frag_c[i] += peer_pack[i];
    }
  }
  __syncthreads();
  cluster.sync();

  status.ok = true;
  return status;
}

}  // namespace marlin_hopper

#endif  // __CUDA_ARCH__ >= 900
