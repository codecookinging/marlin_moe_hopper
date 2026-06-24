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
// Reduces register-resident partial sums across CTAs in the same thread-block
// cluster via distributed shared memory (DSMEM).  Stream-K slice peers must be
// packed into the same cluster by the host launch plan.  When slice_count is
// larger than the cluster size, only the current batch is reduced in DSMEM and
// batch leaders must merge through the caller's global path.

struct ClusterReduceStatus {
  bool ok = false;
  // True when slice_count > cluster_size and only the intra-cluster batch was
  // reduced; batch leaders still need a global merge before write-out.
  bool merge_batches = false;
};

template <int MaxClusterBlocks = 8>
__device__ inline ClusterReduceStatus cluster_streamk_reduce(
    float* frag_c, int num_floats, int4* sh_red, int reduce_group_id,
    int slice_idx, int slice_count) {
  ClusterReduceStatus status;
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = cluster.block_rank();
  const int cluster_size = cluster.num_blocks();

  if (slice_count <= 1 || cluster_size <= 1 || cluster_size > MaxClusterBlocks) {
    return status;
  }

  const int batch_id = slice_idx / cluster_size;
  const int num_batches = (slice_count + cluster_size - 1) / cluster_size;
  const int batch_base = batch_id * cluster_size;
  const int local_slice_idx = slice_idx - batch_base;
  const int local_slice_count =
      slice_count - batch_base > cluster_size ? cluster_size
                                              : slice_count - batch_base;

  if (local_slice_count <= 1 || local_slice_idx >= local_slice_count) {
    return status;
  }

  int* meta = reinterpret_cast<int*>(sh_red);
  if (threadIdx.x == 0) {
    meta[0] = reduce_group_id;
    meta[1] = local_slice_idx;
    meta[2] = local_slice_count;
    meta[3] = batch_id;
    meta[4] = 0;
  }
  __syncthreads();
  cluster.sync();

  __shared__ int cluster_valid;
  if (rank == 0 && threadIdx.x == 0) {
    cluster_valid = 1;
    int seen[MaxClusterBlocks];
#pragma unroll
    for (int i = 0; i < MaxClusterBlocks; ++i) {
      seen[i] = 0;
    }

    for (int r = 0; r < local_slice_count; ++r) {
      int* peer_meta =
          reinterpret_cast<int*>(cluster.map_shared_rank(sh_red, r));
      if (peer_meta[0] != reduce_group_id || peer_meta[2] != local_slice_count ||
          peer_meta[3] != batch_id) {
        cluster_valid = 0;
        break;
      }
      const int idx = peer_meta[1];
      if (idx < 0 || idx >= local_slice_count || seen[idx]) {
        cluster_valid = 0;
        break;
      }
      seen[idx] = 1;
    }

    if (cluster_valid) {
      for (int i = 0; i < local_slice_count; ++i) {
        if (!seen[i]) {
          cluster_valid = 0;
          break;
        }
      }
    }
  }
  if (threadIdx.x == 0) {
    __syncthreads();
    if (rank == 0) {
      meta[4] = cluster_valid;
    }
    __syncthreads();
  } else {
    __syncthreads();
  }
  cluster.sync();

  if (threadIdx.x == 0) {
    const int* leader_meta =
        reinterpret_cast<const int*>(cluster.map_shared_rank(sh_red, 0));
    cluster_valid = leader_meta[4];
  }
  __syncthreads();
  if (cluster_valid != 1) {
    return status;
  }

  float* pack = reinterpret_cast<float*>(sh_red) + 8;
  for (int i = threadIdx.x; i < num_floats; i += blockDim.x) {
    pack[i] = frag_c[i];
  }
  __syncthreads();
  cluster.sync();

  if (local_slice_idx == 0) {
    for (int r = 1; r < local_slice_count; ++r) {
      float* peer_pack = cluster.map_shared_rank(pack, r);
      for (int i = threadIdx.x; i < num_floats; i += blockDim.x) {
        frag_c[i] += peer_pack[i];
      }
    }
  }
  __syncthreads();
  cluster.sync();

  status.ok = true;
  status.merge_batches = num_batches > 1;
  return status;
}

}  // namespace marlin_hopper

#endif  // __CUDA_ARCH__ >= 900
