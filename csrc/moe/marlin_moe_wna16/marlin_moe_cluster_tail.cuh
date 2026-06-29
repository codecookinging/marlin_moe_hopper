#pragma once

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

#include <cooperative_groups.h>

#include "marlin_hopper.cuh"
#include "quantization/marlin/marlin_dtypes.cuh"

namespace marlin_moe_tail {

__device__ inline int div_ceil(int a, int b) { return (a + b - 1) / b; }

// Tail write for thread_m_blocks=1, !m_block_size_8, !is_a_8bit (GLM-5 path).
template <vllm::ScalarTypeId c_type_id, int thread_n_blocks, int NumThreads>
__device__ void tail_write_m1_n4(float* frag_c, int4* sh_red, int4* C,
                                 const int32_t* sorted_token_ids_ptr,
                                 int block_id, int slice_col, int prob_n,
                                 int prob_m, int top_k, int moe_block_size,
                                 bool mul_topk_weights,
                                 const float* topk_weights_ptr) {
  using Cdtype = MarlinScalarType<c_type_id>;
  using c_scalar_t = typename Cdtype::scalar_t;
  using c_scalar_t2 = typename Cdtype::scalar_t2;

  int32_t sorted_ids[16];
  int block_num_valid_tokens = moe_block_size;
  int4* sorted_ids_int4 = reinterpret_cast<int4*>(sorted_ids);
  for (int i = 0; i < moe_block_size / 4; i++) {
    sorted_ids_int4[i] = reinterpret_cast<const int4*>(sorted_token_ids_ptr)
                               [block_id * moe_block_size / 4 + i];
  }
  for (int i = 0; i < moe_block_size; i++) {
    if (sorted_ids[i] >= prob_m * top_k) {
      block_num_valid_tokens = i;
      break;
    }
  }

  int c_gl_stride = prob_n / 8;
  constexpr int c_sh_stride = 2 * thread_n_blocks + 1;
  constexpr int tb_n_warps = thread_n_blocks / 4;
  int c_gl_wr_delta = c_gl_stride * (NumThreads / (2 * thread_n_blocks));
  int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                (threadIdx.x % (2 * thread_n_blocks));
  c_gl_wr += (2 * thread_n_blocks) * slice_col;
  int c_sh_wr =
      (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
  c_sh_wr += 32 * (threadIdx.x / 32);
  int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                (threadIdx.x % (2 * thread_n_blocks));

  auto getf = [&](int j, int m, int g) -> float {
    return frag_c[(j * 2 + m) * 4 + g];
  };

  if (threadIdx.x / 32 < tb_n_warps) {
#pragma unroll
    for (int j = 0; j < 4; j++) {
      int wr = c_sh_wr + 8 * j;
      c_scalar_t2 r0 = Cdtype::nums2num2(Cdtype::float2num(getf(j, 0, 0)),
                                          Cdtype::float2num(getf(j, 0, 1)));
      c_scalar_t2 r1 = Cdtype::nums2num2(Cdtype::float2num(getf(j, 0, 2)),
                                          Cdtype::float2num(getf(j, 0, 3)));
      c_scalar_t2 r2 = Cdtype::nums2num2(Cdtype::float2num(getf(j, 1, 0)),
                                          Cdtype::float2num(getf(j, 1, 1)));
      c_scalar_t2 r3 = Cdtype::nums2num2(Cdtype::float2num(getf(j, 1, 2)),
                                          Cdtype::float2num(getf(j, 1, 3)));
      reinterpret_cast<c_scalar_t2*>(&sh_red[wr + (4 * c_sh_stride) * 0 + 0])[0] =
          r0;
      reinterpret_cast<c_scalar_t2*>(&sh_red[wr + (4 * c_sh_stride) * 8 + 0])[0] =
          r1;
      reinterpret_cast<c_scalar_t2*>(&sh_red[wr + (4 * c_sh_stride) * 0 + 4])[0] =
          r2;
      reinterpret_cast<c_scalar_t2*>(&sh_red[wr + (4 * c_sh_stride) * 8 + 4])[0] =
          r3;
    }
  }
  __syncthreads();

#pragma unroll
  for (int i = 0;
       i < div_ceil(16, NumThreads / (2 * thread_n_blocks)); i++) {
    int row = c_gl_wr / c_gl_stride;
    if (row < block_num_valid_tokens) {
      int64_t sorted_row = sorted_ids[row];
      int64_t true_idx = sorted_row * c_gl_stride + c_gl_wr % c_gl_stride;
      c_scalar_t2* sh_red_half2 =
          reinterpret_cast<c_scalar_t2*>(&sh_red[c_sh_rd]);
      if (mul_topk_weights) {
        c_scalar_t2 topk_weight_score = Cdtype::nums2num2(
            Cdtype::float2num(topk_weights_ptr[sorted_row]),
            Cdtype::float2num(topk_weights_ptr[sorted_row]));
#pragma unroll
        for (int a = 0; a < 4; a++) {
          sh_red_half2[a] = __hmul2(sh_red_half2[a], topk_weight_score);
        }
      }
      C[true_idx] = *reinterpret_cast<int4*>(sh_red_half2);
      c_gl_wr += c_gl_wr_delta;
      c_sh_rd += c_sh_stride * (NumThreads / (2 * thread_n_blocks));
    }
  }
}

template <vllm::ScalarTypeId c_type_id, int thread_n_blocks, int NumThreads,
          int NumFloats>
__global__ void marlin_moe_cluster_tail_kernel(
    float* __restrict__ cluster_partials,
    const int* __restrict__ cluster_tail_meta, int4* __restrict__ C,
    const int32_t* __restrict__ sorted_token_ids_ptr,
    const float* __restrict__ topk_weights_ptr, int prob_m, int prob_n,
    int top_k, int moe_block_size, bool mul_topk_weights, int num_pairs) {
  cooperative_groups::cluster_group cluster =
      cooperative_groups::this_cluster();
  if (cluster.num_blocks() != 2) {
    return;
  }

  const int pair_id = blockIdx.x / 2;
  const int slice_idx = cluster.block_rank();
  const bool active = pair_id < num_pairs;
  if (!active) {
    return;
  }

  float frag_c[NumFloats];
  const float* src =
      cluster_partials + static_cast<int64_t>(pair_id) * 2 * NumFloats +
      static_cast<int64_t>(slice_idx) * NumFloats;
  for (int i = threadIdx.x; i < NumFloats; i += blockDim.x) {
    frag_c[i] = src[i];
  }
  __syncthreads();

  extern __shared__ int4 sh_mem[];
  int4* sh_red = sh_mem;
  int4* sh_pack = sh_mem + (2 * thread_n_blocks + 1) * 16;

  marlin_hopper::ClusterReduceStatus status =
      marlin_hopper::cluster_streamk_reduce<NumFloats>(
          frag_c, sh_pack, slice_idx, /*slice_count=*/2);
  if (!status.ok) {
    return;
  }

  if (slice_idx != 0) {
    return;
  }

  const int block_id = cluster_tail_meta[pair_id * 2 + 0];
  const int slice_col = cluster_tail_meta[pair_id * 2 + 1];
  tail_write_m1_n4<c_type_id, thread_n_blocks, NumThreads>(
      frag_c, sh_red, C, sorted_token_ids_ptr, block_id, slice_col, prob_n,
      prob_m, top_k, moe_block_size, mul_topk_weights, topk_weights_ptr);
}

}  // namespace marlin_moe_tail

#endif  // __CUDA_ARCH__ >= 900
