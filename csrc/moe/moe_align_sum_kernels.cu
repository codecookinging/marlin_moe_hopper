#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>

#include <ATen/ATen.h>
#include <ATen/cuda/Atomic.cuh>

#include "../cuda_compat.h"
#include "../dispatch_utils.h"
#include "core/math.hpp"

#include <algorithm>
#include <vector>

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

namespace {

constexpr int32_t kMaxSegmentsPerBlock = 16;

struct PackedMoeBlock {
  int32_t num_segments = 0;
  int32_t first_expert = -1;
  int32_t token_start = 0;
  int32_t token_count = 0;
  int32_t segment_experts[kMaxSegmentsPerBlock];
  int32_t segment_row_starts[kMaxSegmentsPerBlock];
  int32_t segment_counts[kMaxSegmentsPerBlock];
};

// Greedy bin-pack consecutive experts into MoE blocks.  Multiple small experts
// can share one block when their token counts sum to <= block_size.
void pack_moe_blocks_cpu(const int32_t* expert_counts_in, int32_t num_experts,
                         int32_t block_size,
                         std::vector<PackedMoeBlock>& blocks,
                         std::vector<int32_t>& dense_cumsum) {
  blocks.clear();
  dense_cumsum.assign(num_experts + 1, 0);
  std::vector<int32_t> remaining(num_experts);
  for (int32_t e = 0; e < num_experts; ++e) {
    remaining[e] = expert_counts_in[e];
    dense_cumsum[e + 1] = dense_cumsum[e] + expert_counts_in[e];
  }

  int32_t e = 0;
  while (e < num_experts) {
    if (remaining[e] == 0) {
      ++e;
      continue;
    }
    PackedMoeBlock block;
    int32_t block_fill = 0;
    while (e < num_experts) {
      if (remaining[e] == 0) {
        ++e;
        continue;
      }
      const int32_t take = std::min(remaining[e], block_size - block_fill);
      if (take <= 0) {
        break;
      }
      block.segment_experts[block.num_segments] = e;
      block.segment_row_starts[block.num_segments] = block_fill;
      block.segment_counts[block.num_segments] = take;
      ++block.num_segments;
      block_fill += take;
      remaining[e] -= take;
      if (remaining[e] > 0) {
        break;
      }
      ++e;
      if (block_fill == block_size) {
        break;
      }
    }
    block.token_count = block_fill;
    block.first_expert = block.segment_experts[0];
    blocks.push_back(block);
  }
}

template <typename scalar_t>
__global__ void count_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids, int32_t* __restrict__ expert_counts,
    int32_t* __restrict__ expert_map, int32_t num_experts, size_t numel,
    bool has_expert_map) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = static_cast<int32_t>(topk_ids[i]);
    if (expert_id >= num_experts) {
      continue;
    }
    if (has_expert_map) {
      expert_id = expert_map[expert_id];
      if (expert_id == -1) {
        continue;
      }
    }
    atomicAdd(&expert_counts[expert_id], 1);
  }
}

template <typename scalar_t>
__global__ void sort_expert_tokens_dense_kernel(
    const scalar_t* __restrict__ topk_ids, int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ write_cursor, const int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ expert_map, int32_t num_experts, size_t numel,
    bool has_expert_map) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = static_cast<int32_t>(topk_ids[i]);
    if (expert_id >= num_experts) {
      continue;
    }
    if (has_expert_map) {
      expert_id = expert_map[expert_id];
      if (expert_id == -1) {
        continue;
      }
    }
    int32_t rank = atomicAdd(&write_cursor[expert_id], 1);
    sorted_token_ids[expert_offsets[expert_id] + rank] =
        static_cast<int32_t>(i);
  }
}

}  // namespace

namespace vllm {
namespace moe {
namespace batched_moe_align_block_size {

// Note num_threads needs to be 1024 for BlockScan Reduction in the kernel.
static constexpr int32_t num_threads = 1024;
static constexpr int32_t num_blocks = 1;
__global__ void batched_moe_align_block_size_kernel(
    int32_t const num_batches, int32_t const max_tokens_per_batch,
    int32_t const block_size, int32_t const* __restrict__ batch_num_tokens,
    int32_t* __restrict__ sorted_ids, int32_t* __restrict__ block_ids,
    int32_t* __restrict__ num_tokens_post_pad) {
  // TODO(varun): This is a naive implementation. Could be optimized.

  size_t const batch_id = threadIdx.x;
  size_t const stride = blockDim.x * gridDim.x;
  int32_t const num_blocks_per_batch =
      CEILDIV(max_tokens_per_batch, block_size);
  int32_t const sorted_ids_size =
      num_blocks_per_batch * num_batches * block_size;
  int32_t const block_ids_size = sorted_ids_size / block_size;
  int32_t const SENTINEL =
      num_batches * max_tokens_per_batch;  // To denote invalid entries.
  // Intialize sorted_ids
  for (size_t i = threadIdx.x; i < sorted_ids_size; i += stride) {
    sorted_ids[i] = SENTINEL;
  }
  // Intialize expert_ids with -1
  for (size_t i = threadIdx.x; i < block_ids_size; i += stride) {
    block_ids[i] = -1;
  }

  int32_t b_num_tokens = 0;
  if (batch_id < num_batches) {
    b_num_tokens = batch_num_tokens[batch_id];
  }
  int32_t const ceil_b_num_tokens =
      CEILDIV(b_num_tokens, block_size) * block_size;

  // Compute prefix sum over token counts per expert
  using BlockScan = cub::BlockScan<int32_t, 1024>;
  __shared__ typename BlockScan::TempStorage temp_storage;
  int cumsum_val;
  BlockScan(temp_storage).ExclusiveSum(ceil_b_num_tokens, cumsum_val);
  __syncthreads();

  bool const is_last_batch = batch_id == (num_batches - 1);
  if (is_last_batch) {
    *num_tokens_post_pad = cumsum_val + ceil_b_num_tokens;
  }

  if (batch_id < num_batches) {
    int32_t const batch_offset = batch_id * max_tokens_per_batch;
    for (size_t i = 0; i < b_num_tokens; ++i) {
      sorted_ids[cumsum_val + i] = batch_offset + i;
    }

    int32_t const block_start = cumsum_val / block_size;
    int32_t const num_blocks = ceil_b_num_tokens / block_size;
    for (size_t i = 0; i < num_blocks; ++i) {
      block_ids[block_start + i] = batch_id;
    }
  }
}
}  // namespace batched_moe_align_block_size

template <typename scalar_t>
__device__ void _moe_align_block_size(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t* __restrict__ expert_map, int32_t num_experts,
    int32_t padded_num_experts, int32_t experts_per_warp, int32_t block_size,
    size_t numel, int32_t* __restrict__ cumsum, int32_t max_num_tokens_padded,
    int32_t max_num_m_blocks, int32_t model_offset, int32_t inactive_expert_id,
    int32_t topk_num, int32_t* token_mask, bool has_expert_map) {
  extern __shared__ int32_t shared_counts[];

  // Compute input buffer offsets. Typically these will all be 0, except when
  // using Multi LoRA.
  int sorted_token_ids_offset = max_num_tokens_padded * model_offset;
  int expert_ids_offset = max_num_m_blocks * model_offset;
  int cumsum_offset = (num_experts + 1) * model_offset;

  // Use separate threadblocks to fill sorted_token_ids.
  // This is safe since the current kernel does not use sorted_token_ids.
  if (blockIdx.x % 2) {
    // Initialize sorted_token_ids with numel
    for (size_t it = threadIdx.x; it < max_num_tokens_padded;
         it += blockDim.x) {
      sorted_token_ids[sorted_token_ids_offset + it] = numel;
    }
    return;
  }

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int my_expert_start = warp_id * experts_per_warp;

  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < padded_num_experts) {
      shared_counts[warp_id * experts_per_warp + i] = 0;
    }
  }

  __syncthreads();

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    if (expert_id >= num_experts) {
      continue;
    }
    if (has_expert_map) {
      expert_id = expert_map[expert_id];
      // filter invalid experts
      if (expert_id == -1) continue;
    }
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    int mask = token_mask == nullptr ? 1 : token_mask[i / topk_num];
    atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset],
              mask);
  }

  __syncthreads();

  // Compute prefix sum over token counts per expert
  using BlockScan = cub::BlockScan<int32_t, 1024>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  int expert_count = 0;
  int expert_id = threadIdx.x;
  if (expert_id < num_experts) {
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];
    expert_count = CEILDIV(expert_count, block_size) * block_size;
  }

  int cumsum_val;
  BlockScan(temp_storage).ExclusiveSum(expert_count, cumsum_val);
  if (expert_id <= num_experts) {
    cumsum[cumsum_offset + expert_id] = cumsum_val;
  }

  if (expert_id == num_experts) {
    total_tokens_post_pad[model_offset] = cumsum_val;
  }

  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = cumsum[cumsum_offset + threadIdx.x];
         i < cumsum[cumsum_offset + threadIdx.x + 1]; i += block_size) {
      expert_ids[expert_ids_offset + i / block_size] = threadIdx.x;
    }
  }

  // Fill remaining expert_ids with -1
  const size_t fill_start_idx =
      cumsum[cumsum_offset + num_experts] / block_size + threadIdx.x;
  for (size_t i = fill_start_idx; i < max_num_m_blocks; i += blockDim.x) {
    expert_ids[expert_ids_offset + i] = inactive_expert_id;
  }
}

template <typename scalar_t, int32_t fill_threads>
__device__ void _moe_align_block_size_small_batch_expert(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t* __restrict__ expert_map, int32_t num_experts, int32_t block_size,
    size_t numel, int32_t max_num_tokens_padded, int32_t max_num_m_blocks,
    int32_t inactive_expert_id, int32_t model_offset, int32_t topk_num,
    int32_t* token_mask, bool has_expert_map) {
  // Compute input buffer offsets. Typically these will all be 0, except when
  // using Multi LoRA.
  int sorted_token_ids_offset = max_num_tokens_padded * model_offset;
  int expert_ids_offset = max_num_m_blocks * model_offset;

  // Use an additional group of threads to fill sorted_token_ids.
  // Since the current kernel will use sorted_token_ids afterward,
  // we fill sorted_token_ids within the same threadblock to make
  // synchronization easier.
  if (threadIdx.x < fill_threads) {
    // Initialize sorted_token_ids with numel
    for (size_t it = threadIdx.x; it < max_num_tokens_padded;
         it += fill_threads) {
      sorted_token_ids[sorted_token_ids_offset + it] = numel;
    }
    // Three __syncthreads() corresponding to the other threads
    __syncthreads();
    __syncthreads();
    __syncthreads();
    return;
  }

  const size_t tid = threadIdx.x - fill_threads;
  const size_t stride = blockDim.x - fill_threads;

  extern __shared__ int32_t shared_mem[];
  int32_t* cumsum = shared_mem;
  int32_t* tokens_cnts = (int32_t*)(shared_mem + num_experts + 1);

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[(tid + 1) * num_experts + i] = 0;
  }

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    if (has_expert_map) {
      expert_id = expert_map[expert_id];
      // filter invalid expert
      if (expert_id == -1) continue;
    }
    int mask = token_mask == nullptr ? 1 : token_mask[i / topk_num];
    tokens_cnts[(tid + 1) * num_experts + expert_id] += mask;
  }

  __syncthreads();

  if (tid < num_experts) {
    tokens_cnts[tid] = 0;
    for (int i = 1; i <= stride; ++i) {
      tokens_cnts[i * num_experts + tid] +=
          tokens_cnts[(i - 1) * num_experts + tid];
    }
  }

  __syncthreads();

  if (tid == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] =
          cumsum[i - 1] +
          CEILDIV(tokens_cnts[stride * num_experts + i - 1], block_size) *
              block_size;
    }
    total_tokens_post_pad[model_offset] =
        static_cast<int32_t>(cumsum[num_experts]);
  }

  __syncthreads();

  if (tid < num_experts) {
    for (int i = cumsum[tid]; i < cumsum[tid + 1]; i += block_size) {
      expert_ids[expert_ids_offset + i / block_size] = tid;
    }
  }

  // Fill remaining expert_ids with -1
  const size_t fill_start_idx = cumsum[num_experts] / block_size + tid;
  for (size_t i = fill_start_idx; i < max_num_m_blocks; i += stride) {
    expert_ids[expert_ids_offset + i] = inactive_expert_id;
  }

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    if (has_expert_map) {
      expert_id = expert_map[expert_id];
      // filter invalid expert
      if (expert_id == -1) continue;
    }
    int32_t rank_post_pad =
        tokens_cnts[tid * num_experts + expert_id] + cumsum[expert_id];

    if (token_mask == nullptr || token_mask[i / topk_num]) {
      sorted_token_ids[sorted_token_ids_offset + rank_post_pad] = i;
      ++tokens_cnts[tid * num_experts + expert_id];
    }
  }
}

template <typename scalar_t>
__device__ void _count_and_sort_expert_tokens(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ cumsum_buffer,
    int32_t* __restrict__ expert_map, size_t numel, int32_t num_experts,
    int32_t max_num_tokens_padded, int32_t* __restrict__ token_mask,
    int32_t model_offset, int32_t topk_num, bool has_expert_map) {
  const size_t tid = blockIdx.y * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.y;

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    if (expert_id >= num_experts) {
      continue;
    }

    if (has_expert_map) {
      expert_id = expert_map[expert_id];
      // filter invalid experts
      if (expert_id == -1) continue;
    }

    if (token_mask == nullptr || token_mask[i / topk_num]) {
      int32_t rank_post_pad = atomicAdd(
          &cumsum_buffer[(model_offset * (num_experts + 1)) + expert_id], 1);
      sorted_token_ids[max_num_tokens_padded * model_offset + rank_post_pad] =
          i;
    }
  }
}

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t* __restrict__ expert_map, int32_t num_experts,
    int32_t padded_num_experts, int32_t experts_per_warp, int32_t block_size,
    size_t numel, int32_t* __restrict__ cumsum, int32_t max_num_tokens_padded,
    int32_t topk_num, bool has_expert_map) {
  _moe_align_block_size(
      topk_ids, sorted_token_ids, expert_ids, total_tokens_post_pad, expert_map,
      num_experts, padded_num_experts, experts_per_warp, block_size, numel,
      cumsum, max_num_tokens_padded, CEILDIV(max_num_tokens_padded, block_size),
      0, -1, topk_num, nullptr, has_expert_map);
}

template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ cumsum_buffer,
    int32_t* __restrict__ expert_map, size_t numel, int32_t num_experts,
    int32_t max_num_tokens_padded, int32_t topk_num, bool has_expert_map) {
  _count_and_sort_expert_tokens(
      topk_ids, sorted_token_ids, cumsum_buffer, expert_map, numel, num_experts,
      max_num_tokens_padded, nullptr, 0, topk_num, has_expert_map);
}

template <typename scalar_t, int TOPK>
__global__ void moe_sum_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., topk, d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    scalar_t x = 0.0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      x += VLLM_LDG(&input[token_idx * TOPK * d + k * d + idx]);
    }
    out[token_idx * d + idx] = x;
  }
}

template <typename scalar_t, int32_t fill_threads>
__global__ void moe_align_block_size_small_batch_expert_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t* __restrict__ expert_map, int32_t num_experts, int32_t block_size,
    size_t numel, int32_t max_num_tokens_padded, int32_t topk_num,
    bool has_expert_map) {
  _moe_align_block_size_small_batch_expert<scalar_t, fill_threads>(
      topk_ids, sorted_token_ids, expert_ids, total_tokens_post_pad, expert_map,
      num_experts, block_size, numel, max_num_tokens_padded,
      CEILDIV(max_num_tokens_padded, block_size), -1, 0, topk_num, nullptr,
      has_expert_map);
}

template <typename scalar_t>
__global__ void moe_lora_align_block_size_kernel(
    scalar_t* __restrict__ topk_ids, int32_t* __restrict__ token_lora_mapping,
    int64_t block_size, int32_t* __restrict__ expert_map, int num_experts,
    int max_loras, size_t numel, int max_num_tokens_padded,
    int max_num_m_blocks, int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids, int32_t topk_num,
    int32_t* total_tokens_post_pad, int32_t* adapter_enabled,
    int32_t* __restrict__ cumsum, int32_t experts_per_warp,
    int32_t padded_num_experts, int32_t* lora_ids,
    int32_t* __restrict__ token_mask, bool has_expert_map) {
  int lora_idx = blockIdx.x / 2;
  int lora_id = lora_ids[lora_idx];
  if (lora_id == -1 || adapter_enabled[lora_id] == 0) {
    return;
  }

  // Populate the token_mask based on the token-LoRA mapping
  int num_tokens = numel / topk_num;
  if (threadIdx.x == 0) {
    total_tokens_post_pad[lora_id] = 0;

    for (int i = 0; i < num_tokens; i++) {
      token_mask[(lora_id * num_tokens) + i] =
          (int)token_lora_mapping[i] == lora_id;
    }
  }

  __syncthreads();

  _moe_align_block_size(
      topk_ids, sorted_token_ids, expert_ids, total_tokens_post_pad, expert_map,
      num_experts, padded_num_experts, experts_per_warp, block_size, numel,
      cumsum, max_num_tokens_padded, max_num_m_blocks, lora_id, -1, topk_num,
      &token_mask[(lora_id * num_tokens)], has_expert_map);
}

template <typename scalar_t>
__global__ void lora_count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ cumsum_buffer,
    int32_t* __restrict__ expert_map, size_t numel, int32_t num_experts,
    int32_t max_num_tokens_padded, int32_t topk_num, int32_t* token_mask,
    int32_t* lora_ids, bool has_expert_map) {
  int lora_idx = blockIdx.x;
  int lora_id = lora_ids[lora_idx];
  if (lora_id == -1) {
    return;
  }

  int num_tokens = numel / topk_num;

  _count_and_sort_expert_tokens(
      topk_ids, sorted_token_ids, cumsum_buffer, expert_map, numel, num_experts,
      max_num_tokens_padded, &token_mask[(lora_id * num_tokens)], lora_id,
      topk_num, has_expert_map);
}

template <typename scalar_t, int32_t fill_threads>
__global__ void moe_lora_align_block_size_small_batch_expert_kernel(
    scalar_t* __restrict__ topk_ids, int32_t* token_lora_mapping,
    int64_t block_size, int32_t* __restrict__ expert_map, int num_experts,
    int max_loras, size_t numel, int max_num_tokens_padded,
    int max_num_m_blocks, int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids, int topk_num,
    int32_t* total_tokens_post_pad, int32_t* adapter_enabled, int32_t* lora_ids,
    int32_t* token_mask, bool has_expert_map) {
  int lora_idx = blockIdx.x;
  int lora_id = lora_ids[lora_idx];
  if (lora_id == -1 || adapter_enabled[lora_id] == 0) {
    return;
  }

  int num_tokens = numel / topk_num;
  if (threadIdx.x == 0) {
    total_tokens_post_pad[lora_id] = 0;

    for (int i = 0; i < num_tokens; i++) {
      token_mask[(lora_id * num_tokens) + i] =
          (int)token_lora_mapping[i] == lora_id;
    }
  }

  __syncthreads();

  _moe_align_block_size_small_batch_expert<scalar_t, fill_threads>(
      topk_ids, sorted_token_ids, expert_ids, total_tokens_post_pad, expert_map,
      num_experts, block_size, numel, max_num_tokens_padded, max_num_m_blocks,
      -1, lora_id, topk_num, &token_mask[(lora_id * num_tokens)],
      has_expert_map);
}

}  // namespace moe
}  // namespace vllm

// taken from
// https://github.com/sgl-project/sglang/blob/8b5f83ed3b7d2a49ad5c5cd5aa61c5d502f47dbc
void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad,
                          std::optional<torch::Tensor> maybe_expert_map) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t padded_num_experts =
      ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  int experts_per_warp = WARP_SIZE;
  int threads = 1024;
  threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

  // BlockScan uses 1024 threads and assigns one thread per expert.
  TORCH_CHECK(padded_num_experts < 1024,
              "padded_num_experts must be less than 1024");
  auto options_int =
      torch::TensorOptions().dtype(torch::kInt).device(topk_ids.device());
  bool has_expert_map = maybe_expert_map.has_value();
  torch::Tensor expert_map;
  if (has_expert_map) {
    expert_map = maybe_expert_map.value();
  } else {
    expert_map = torch::empty({0}, options_int);
  }

  VLLM_DISPATCH_INTEGRAL_AND_UNSIGNED_TYPES(
      topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
        // calc needed amount of shared mem for `cumsum` tensors
        bool small_batch_expert_mode =
            (topk_ids.numel() < 1024) && (num_experts <= 64);

        if (small_batch_expert_mode) {
          const int32_t threads = max((int32_t)num_experts, WARP_SIZE);
          const int32_t shared_mem_size =
              ((threads + 1) * num_experts + (num_experts + 1)) *
              sizeof(int32_t);

          // threadIdx.x >= fill_threads: counting experts and aligning
          // threadIdx.x < fill_threads: filling sorted_token_ids
          constexpr int32_t fill_threads = 256;
          auto small_batch_expert_kernel =
              vllm::moe::moe_align_block_size_small_batch_expert_kernel<
                  scalar_t, fill_threads>;
          small_batch_expert_kernel<<<1, fill_threads + threads,
                                      shared_mem_size, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              experts_ids.data_ptr<int32_t>(),
              num_tokens_post_pad.data_ptr<int32_t>(),
              expert_map.data_ptr<int32_t>(), num_experts, block_size,
              topk_ids.numel(), sorted_token_ids.size(0), topk_ids.size(1),
              has_expert_map);
        } else {
          torch::Tensor cumsum_buffer =
              torch::empty({num_experts + 1}, options_int);
          auto align_kernel = vllm::moe::moe_align_block_size_kernel<scalar_t>;

          size_t num_warps = CEILDIV(padded_num_experts, experts_per_warp);
          size_t shared_mem_size =
              num_warps * experts_per_warp * sizeof(int32_t);

          // launch two threadblocks
          // blockIdx.x == 0: counting experts and aligning
          // blockIdx.x == 1: filling sorted_token_ids
          align_kernel<<<2, threads, shared_mem_size, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              experts_ids.data_ptr<int32_t>(),
              num_tokens_post_pad.data_ptr<int32_t>(),
              expert_map.data_ptr<int32_t>(), num_experts, padded_num_experts,
              experts_per_warp, block_size, topk_ids.numel(),
              cumsum_buffer.data_ptr<int32_t>(), sorted_token_ids.size(0),
              topk_ids.size(1), has_expert_map);

          const int block_threads = std::min(256, (int)threads);
          const int num_blocks =
              (topk_ids.numel() + block_threads - 1) / block_threads;
          const int max_blocks = 65535;
          const int actual_blocks = std::min(num_blocks, max_blocks);
          dim3 gridDims(1, actual_blocks);

          auto sort_kernel =
              vllm::moe::count_and_sort_expert_tokens_kernel<scalar_t>;
          sort_kernel<<<gridDims, block_threads, 0, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              cumsum_buffer.data_ptr<int32_t>(), expert_map.data_ptr<int32_t>(),
              topk_ids.numel(), num_experts, sorted_token_ids.size(0),
              topk_ids.size(1), has_expert_map);
        }
      });
}

void batched_moe_align_block_size(int64_t max_tokens_per_batch,
                                  int64_t block_size,
                                  torch::Tensor const& batch_num_tokens,
                                  torch::Tensor sorted_ids,
                                  torch::Tensor batch_ids,
                                  torch::Tensor num_tokens_post_pad) {
  namespace batched_kernel = vllm::moe::batched_moe_align_block_size;

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int32_t const B = batch_num_tokens.size(0);
  int32_t const num_blocks_per_batch =
      round_to_next_multiple_of(max_tokens_per_batch, block_size) / block_size;
  int32_t const num_blocks = num_blocks_per_batch * B;
  int64_t const sorted_ids_size = num_blocks * block_size;

  TORCH_CHECK(sorted_ids.size(0) == sorted_ids_size);
  TORCH_CHECK(batch_ids.size(0) == sorted_ids_size / block_size);
  TORCH_CHECK(num_tokens_post_pad.size(0) == 1);
  TORCH_CHECK(B <= batched_kernel::num_threads);

  batched_kernel::batched_moe_align_block_size_kernel<<<
      batched_kernel::num_blocks, batched_kernel::num_threads, 0, stream>>>(
      B, max_tokens_per_batch, block_size, batch_num_tokens.data_ptr<int32_t>(),
      sorted_ids.data_ptr<int32_t>(), batch_ids.data_ptr<int32_t>(),
      num_tokens_post_pad.data_ptr<int32_t>());
}

void moe_sum(torch::Tensor& input,   // [num_tokens, topk, hidden_size]
             torch::Tensor& output)  // [num_tokens, hidden_size]
{
  const int hidden_size = input.size(-1);
  const auto num_tokens = output.numel() / hidden_size;
  const int topk = input.size(1);

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (topk) {
    case 2:
      VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        vllm::moe::moe_sum_kernel<scalar_t, 2><<<grid, block, 0, stream>>>(
            output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
            hidden_size);
      });
      break;

    case 3:
      VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        vllm::moe::moe_sum_kernel<scalar_t, 3><<<grid, block, 0, stream>>>(
            output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
            hidden_size);
      });
      break;

    case 4:
      VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        vllm::moe::moe_sum_kernel<scalar_t, 4><<<grid, block, 0, stream>>>(
            output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
            hidden_size);
      });
      break;

    default:
      at::sum_out(output, input, 1);
      break;
  }
}

void moe_lora_align_block_size(
    torch::Tensor topk_ids, torch::Tensor token_lora_mapping,
    int64_t num_experts, int64_t block_size, int64_t max_loras,
    int64_t max_num_tokens_padded, int64_t max_num_m_blocks,
    torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad, torch::Tensor adapter_enabled,
    torch::Tensor lora_ids, std::optional<torch::Tensor> maybe_expert_map) {
  const int topk_num = topk_ids.size(1);

  TORCH_CHECK(block_size > 0, "block_size should be greater than 0. ");

  int device_max_shared_mem;
  auto dev = topk_ids.get_device();
  cudaDeviceGetAttribute(&device_max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t padded_num_experts =
      ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

  // BlockScan uses 1024 threads and assigns one thread per expert.
  TORCH_CHECK(padded_num_experts < 1024,
              "padded_num_experts must be less than 1024");

  auto options_int =
      torch::TensorOptions().dtype(torch::kInt).device(topk_ids.device());
  torch::Tensor token_mask =
      torch::empty({max_loras * topk_ids.size(0)}, options_int);
  bool has_expert_map = maybe_expert_map.has_value();
  torch::Tensor expert_map;
  if (has_expert_map) {
    expert_map = maybe_expert_map.value();
  } else {
    expert_map = torch::empty({0}, options_int);
  }

  VLLM_DISPATCH_INTEGRAL_TYPES(
      topk_ids.scalar_type(), "moe_lora_align_sum_kernel", [&] {
        bool small_batch_expert_mode =
            (topk_ids.numel() < 1024) && (num_experts <= 64);

        if (small_batch_expert_mode) {
          const int32_t num_thread = max((int32_t)num_experts, 128);
          const int32_t shared_mem =
              (num_thread + 1) * num_experts * sizeof(int32_t) +
              (num_experts + 1) * sizeof(int32_t);
          if (shared_mem > device_max_shared_mem) {
            TORCH_CHECK(false, "Shared memory usage exceeds device limit.");
          }

          // threadIdx.x >= fill_threads: counting experts and aligning
          // threadIdx.x < fill_threads: filling sorted_token_ids
          constexpr int32_t fill_threads = 256;

          dim3 blockDim(num_thread + fill_threads);
          auto kernel =
              vllm::moe::moe_lora_align_block_size_small_batch_expert_kernel<
                  scalar_t, fill_threads>;
          AT_CUDA_CHECK(VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(
              (void*)kernel, shared_mem));
          kernel<<<max_loras, blockDim, shared_mem, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              token_lora_mapping.data_ptr<int32_t>(), block_size,
              expert_map.data_ptr<int32_t>(), num_experts, max_loras,
              topk_ids.numel(), max_num_tokens_padded, max_num_m_blocks,
              sorted_token_ids.data_ptr<int32_t>(),
              expert_ids.data_ptr<int32_t>(), topk_num,
              num_tokens_post_pad.data_ptr<int32_t>(),
              adapter_enabled.data_ptr<int32_t>(), lora_ids.data_ptr<int32_t>(),
              token_mask.data_ptr<int32_t>(), has_expert_map);
        } else {
          int num_thread = 1024;
          dim3 blockDim(num_thread);
          size_t num_warps = CEILDIV(padded_num_experts, WARP_SIZE);

          size_t shared_mem_size = num_warps * WARP_SIZE * sizeof(int32_t);

          // cumsum buffer
          torch::Tensor cumsum =
              torch::zeros({max_loras * (num_experts + 1)}, options_int);

          auto align_kernel =
              vllm::moe::moe_lora_align_block_size_kernel<scalar_t>;

          // launch two threadblocks for each lora
          // blockIdx.x % 2 == 0: counting experts and aligning
          // blockIdx.x % 2 == 1: filling sorted_token_ids
          align_kernel<<<max_loras * 2, blockDim, shared_mem_size, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              token_lora_mapping.data_ptr<int32_t>(), block_size,
              expert_map.data_ptr<int32_t>(), num_experts, max_loras,
              topk_ids.numel(), max_num_tokens_padded, max_num_m_blocks,
              sorted_token_ids.data_ptr<int32_t>(),
              expert_ids.data_ptr<int32_t>(), topk_num,
              num_tokens_post_pad.data_ptr<int32_t>(),
              adapter_enabled.data_ptr<int32_t>(), cumsum.data_ptr<int32_t>(),
              WARP_SIZE, padded_num_experts, lora_ids.data_ptr<int32_t>(),
              token_mask.data_ptr<int32_t>(), has_expert_map);

          const int block_threads = std::min(256, (int)num_thread);
          const int num_blocks =
              (topk_ids.numel() + block_threads - 1) / block_threads;

          const int max_blocks = 65535;
          const int actual_blocks = std::min(num_blocks, max_blocks);

          dim3 gridDims(max_loras, actual_blocks);
          auto sort_kernel =
              vllm::moe::lora_count_and_sort_expert_tokens_kernel<scalar_t>;

          sort_kernel<<<gridDims, block_threads, 0, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(), cumsum.data_ptr<int32_t>(),
              expert_map.data_ptr<int32_t>(), topk_ids.numel(), num_experts,
              max_num_tokens_padded, topk_num, token_mask.data_ptr<int32_t>(),
              lora_ids.data_ptr<int32_t>(), has_expert_map);
        }
      });
}

void moe_align_block_size_packed(
    torch::Tensor topk_ids, int64_t num_experts, int64_t block_size,
    torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad, torch::Tensor block_token_offsets,
    torch::Tensor block_num_segments, torch::Tensor block_segment_experts,
    torch::Tensor block_segment_row_starts, torch::Tensor block_segment_counts,
    std::optional<torch::Tensor> maybe_expert_map) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int32_t num_experts_i = static_cast<int32_t>(num_experts);
  const int32_t block_size_i = static_cast<int32_t>(block_size);
  const size_t numel = topk_ids.numel();

  TORCH_CHECK(sorted_token_ids.scalar_type() == torch::kInt32);
  TORCH_CHECK(expert_ids.scalar_type() == torch::kInt32);
  TORCH_CHECK(num_tokens_post_pad.scalar_type() == torch::kInt32);
  TORCH_CHECK(block_token_offsets.scalar_type() == torch::kInt32);
  TORCH_CHECK(block_num_segments.scalar_type() == torch::kInt32);
  TORCH_CHECK(block_segment_experts.scalar_type() == torch::kInt32);
  TORCH_CHECK(block_segment_row_starts.scalar_type() == torch::kInt32);
  TORCH_CHECK(block_segment_counts.scalar_type() == torch::kInt32);
  TORCH_CHECK(sorted_token_ids.numel() >= static_cast<int64_t>(numel));

  bool has_expert_map = maybe_expert_map.has_value();
  torch::Tensor expert_map;
  if (has_expert_map) {
    expert_map = maybe_expert_map.value();
  } else {
    expert_map = torch::empty({0}, sorted_token_ids.options());
  }

  auto options_int = sorted_token_ids.options();
  torch::Tensor expert_counts = torch::zeros({num_experts_i}, options_int);
  torch::Tensor write_cursor = torch::zeros({num_experts_i}, options_int);
  torch::Tensor expert_offsets = torch::empty({num_experts_i + 1}, options_int);

  const int block_threads = 256;
  const int count_blocks =
      static_cast<int>((numel + block_threads - 1) / block_threads);

  VLLM_DISPATCH_INTEGRAL_AND_UNSIGNED_TYPES(
      topk_ids.scalar_type(), "moe_align_block_size_packed", [&] {
        count_expert_tokens_kernel<scalar_t>
            <<<count_blocks, block_threads, 0, stream>>>(
                topk_ids.data_ptr<scalar_t>(),
                expert_counts.data_ptr<int32_t>(),
                expert_map.data_ptr<int32_t>(), num_experts_i, numel,
                has_expert_map);
      });

  std::vector<int32_t> counts_host(num_experts_i);
  std::vector<int32_t> dense_cumsum;
  std::vector<PackedMoeBlock> blocks;
  cudaMemcpyAsync(counts_host.data(), expert_counts.data_ptr<int32_t>(),
                  num_experts_i * sizeof(int32_t), cudaMemcpyDeviceToHost,
                  stream);
  cudaStreamSynchronize(stream);

  pack_moe_blocks_cpu(counts_host.data(), num_experts_i, block_size_i, blocks,
                      dense_cumsum);

  const int32_t num_moe_blocks = static_cast<int32_t>(blocks.size());
  const int32_t num_tokens_post_pad_val = num_moe_blocks * block_size_i;

  TORCH_CHECK(block_token_offsets.numel() >= num_moe_blocks + 1);
  TORCH_CHECK(expert_ids.numel() >= num_moe_blocks);
  TORCH_CHECK(block_num_segments.numel() >= num_moe_blocks);
  TORCH_CHECK(block_segment_experts.numel() >=
              static_cast<int64_t>(num_moe_blocks) * kMaxSegmentsPerBlock);
  TORCH_CHECK(block_segment_row_starts.numel() >=
              static_cast<int64_t>(num_moe_blocks) * kMaxSegmentsPerBlock);
  TORCH_CHECK(block_segment_counts.numel() >=
              static_cast<int64_t>(num_moe_blocks) * kMaxSegmentsPerBlock);

  std::vector<int32_t> block_token_offsets_host(num_moe_blocks + 1);
  std::vector<int32_t> expert_ids_host(num_moe_blocks, -1);
  std::vector<int32_t> block_num_segments_host(num_moe_blocks, 0);
  std::vector<int32_t> block_segment_experts_host(
      num_moe_blocks * kMaxSegmentsPerBlock, -1);
  std::vector<int32_t> block_segment_row_starts_host(
      num_moe_blocks * kMaxSegmentsPerBlock, 0);
  std::vector<int32_t> block_segment_counts_host(
      num_moe_blocks * kMaxSegmentsPerBlock, 0);

  int32_t dense_offset = 0;
  for (int32_t b = 0; b < num_moe_blocks; ++b) {
    const PackedMoeBlock& block = blocks[b];
    block_token_offsets_host[b] = dense_offset;
    expert_ids_host[b] = block.first_expert;
    block_num_segments_host[b] = block.num_segments;
    for (int32_t s = 0; s < block.num_segments; ++s) {
      const int32_t flat = b * kMaxSegmentsPerBlock + s;
      block_segment_experts_host[flat] = block.segment_experts[s];
      block_segment_row_starts_host[flat] = block.segment_row_starts[s];
      block_segment_counts_host[flat] = block.segment_counts[s];
    }
    dense_offset += block.token_count;
  }
  block_token_offsets_host[num_moe_blocks] = dense_offset;

  cudaMemcpyAsync(expert_offsets.data_ptr<int32_t>(), dense_cumsum.data(),
                  dense_cumsum.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
                  stream);
  sorted_token_ids.fill_(static_cast<int32_t>(numel));

  VLLM_DISPATCH_INTEGRAL_AND_UNSIGNED_TYPES(
      topk_ids.scalar_type(), "sort_expert_tokens_dense_kernel", [&] {
        sort_expert_tokens_dense_kernel<scalar_t>
            <<<count_blocks, block_threads, 0, stream>>>(
                topk_ids.data_ptr<scalar_t>(),
                sorted_token_ids.data_ptr<int32_t>(),
                write_cursor.data_ptr<int32_t>(),
                expert_offsets.data_ptr<int32_t>(),
                expert_map.data_ptr<int32_t>(), num_experts_i, numel,
                has_expert_map);
      });

  const int32_t num_tokens_post_pad_host = num_tokens_post_pad_val;
  cudaMemcpyAsync(num_tokens_post_pad.data_ptr<int32_t>(),
                  &num_tokens_post_pad_host, sizeof(int32_t),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(block_token_offsets.data_ptr<int32_t>(),
                  block_token_offsets_host.data(),
                  (num_moe_blocks + 1) * sizeof(int32_t), cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(expert_ids.data_ptr<int32_t>(), expert_ids_host.data(),
                  num_moe_blocks * sizeof(int32_t), cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(block_num_segments.data_ptr<int32_t>(),
                  block_num_segments_host.data(),
                  num_moe_blocks * sizeof(int32_t), cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(block_segment_experts.data_ptr<int32_t>(),
                  block_segment_experts_host.data(),
                  num_moe_blocks * kMaxSegmentsPerBlock * sizeof(int32_t),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(block_segment_row_starts.data_ptr<int32_t>(),
                  block_segment_row_starts_host.data(),
                  num_moe_blocks * kMaxSegmentsPerBlock * sizeof(int32_t),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(block_segment_counts.data_ptr<int32_t>(),
                  block_segment_counts_host.data(),
                  num_moe_blocks * kMaxSegmentsPerBlock * sizeof(int32_t),
                  cudaMemcpyHostToDevice, stream);
}