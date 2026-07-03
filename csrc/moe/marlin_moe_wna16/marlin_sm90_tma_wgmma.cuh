// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// SM90 TMA/WGMMA dataflow for the experimental Marlin MoE path.
//
// The legacy Marlin WNA16 kernel is a warp-level pipeline:
//
//   global -> shared(A, packed B, scales) -> register dequant -> mma.sync
//
// WGMMA cannot consume the packed Marlin B fragments from registers.  The SM90
// path therefore has a different dataflow:
//
//   A gather      : global(sorted_token_ids) -> shared A, row padded to m64
//   B producer   : global packed B -> shared packed B (TMA boundary)
//   dequant      : shared packed B + scales -> shared FP16/BF16 B in WGMMA layout
//   consumer     : shared A/B -> wgmma.mma_async -> warpgroup accumulator
//   epilogue     : accumulator -> existing MoE row mapping + Split-K reduce/store
//
// A is a sparse/gathered MoE load and cannot be one rectangular TMA transfer.
// B is contiguous per expert and can be described by a 2-D tensor map over
// [expert * K + k, packed_n].  The runnable kernel still needs that tensor-map
// ABI; this file makes the tile/dataflow boundary explicit before wiring it.

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace marlin_sm90_tma_wgmma {

// WGMMA/TMA bulk PTX is only legal on sm_90a (.target sm_90a), not base sm_90.
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define MARLIN_SM90A_DEVICE 1
#endif

#if defined(__CUDACC__)
#define MARLIN_SM90_HD __host__ __device__ __forceinline__
#else
#define MARLIN_SM90_HD inline
#endif

// ---------------------------------------------------------------------------
// Host selection
// ---------------------------------------------------------------------------

struct HostSupport {
  bool supported;
  const char* reason;
};

inline HostSupport select_host_path(int major_capability, int a_bits,
                                    int b_bits, int prob_m, int prob_n,
                                    int prob_k, bool has_act_order,
                                    bool has_zp) {
  if (major_capability < 9) {
    return {false, "TMA/WGMMA requires SM90 or newer."};
  }
  if (prob_n != 256 || prob_k != 6144 || prob_m < 32 || prob_m > 8192) {
    return {false,
            "TMA/WGMMA path is scoped to N=256, K=6144, "
            "and 32 <= M <= 8192."};
  }
  if (a_bits != 16) {
    return {false, "TMA/WGMMA path is scoped to WNA16 activations."};
  }
  if (b_bits != 4 && b_bits != 8) {
    return {false, "TMA/WGMMA path is scoped to packed INT4/INT8 B."};
  }
  if (has_act_order || has_zp) {
    return {false,
            "act_order/zero-point need a separate shared-memory dequant layout "
            "before WGMMA can be enabled."};
  }
  return {true, nullptr};
}

// ---------------------------------------------------------------------------
// Tile contract
// ---------------------------------------------------------------------------

struct TileShape {
  int logical_m;
  int padded_m;
  int n_panel;
  int k_stage;
  int warps;
  int stages;
};

inline TileShape target_tile_shape(int moe_block_size) {
  return TileShape{
      moe_block_size,
      64,   // WGMMA m64, rows beyond valid MoE block are zero-filled.
      128,  // N=256 is two independent n128 panels.
      64,   // Four k16 WGMMA groups per producer stage.
      4,    // One warpgroup.
      3,    // Triple buffer producer stages.
  };
}

MARLIN_SM90_HD int n_panels(int prob_n) { return prob_n / 128; }

MARLIN_SM90_HD int k_stages(int prob_k) { return prob_k / 64; }

MARLIN_SM90_HD int logical_mn_tiles(int parallel_moe_blocks, int prob_n) {
  return parallel_moe_blocks * n_panels(prob_n);
}

struct SharedLayout {
  int barrier_bytes;
  int a_bytes;
  int b_packed_bytes;
  int b_dequant_bytes;
  int c_scratch_bytes;
  int total_bytes;
};

inline SharedLayout shared_layout(TileShape tile, int b_bits) {
  int barrier_bytes = tile.stages * static_cast<int>(sizeof(uint64_t));
  int a_bytes = tile.stages * tile.padded_m * tile.k_stage * 2;
  int b_packed_bytes = tile.stages * tile.k_stage * tile.n_panel * b_bits / 8;
  int b_dequant_bytes = tile.stages * tile.k_stage * tile.n_panel * 2;
  int c_scratch_bytes = tile.padded_m * tile.n_panel * 4;
  return SharedLayout{
      barrier_bytes,
      a_bytes,
      b_packed_bytes,
      b_dequant_bytes,
      c_scratch_bytes,
      barrier_bytes + a_bytes + b_packed_bytes + b_dequant_bytes +
          c_scratch_bytes,
  };
}

inline int required_shared_memory_bytes(int moe_block_size, int b_bits) {
  return shared_layout(target_tile_shape(moe_block_size), b_bits).total_bytes;
}

inline HostSupport check_shared_memory(TileShape tile, int b_bits,
                                       int max_shared_mem) {
  if (shared_layout(tile, b_bits).total_bytes > max_shared_mem) {
    return {false, "TMA/WGMMA shared-memory layout exceeds opt-in smem."};
  }
  return {true, nullptr};
}

enum class Phase {
  kLoadMetadata,
  kGatherA,
  kTmaLoadPackedB,
  kDequantBToWgmmaShared,
  kWarpgroupMma,
  kSplitKReduce,
  kStore,
};

struct TileWork {
  int par_id;
  int n_panel;
  int expert_id;
  int k_stage_begin;
  int k_stage_end;
  int valid_m;
  int lock_offset;
};

struct Params {
  const int4* A;
  const int4* B;
  const void* B_tma_map;
  int4* C;
  int4* C_tmp;
  const int4* scales;
  const int32_t* sorted_token_ids;
  const int32_t* expert_ids;
  const int32_t* num_tokens_past_padded;
  const float* topk_weights;
  int* locks;
  int prob_m;
  int prob_n;
  int prob_k;
  int top_k;
  int moe_block_size;
  int num_groups;
  int group_size;
  int sk_slice_count;
  int sk_slice_idx;
  bool mul_topk_weights;
  bool use_fp32_reduce;
  bool use_tma_load;
};

struct alignas(64) TensorMapAbi {
  uint64_t opaque[16];
};

struct TensorMapBuildArgs {
  const void* base;
  uint64_t global_rows;
  uint64_t global_packed_cols;
  uint64_t stride_rows_bytes;
  uint32_t box_rows;
  uint32_t box_packed_cols;
  uint32_t element_stride_rows;
  uint32_t element_stride_cols;
};

inline HostSupport check_tensor_map_abi() {
  static_assert(sizeof(TensorMapAbi) == 128,
                "CUtensorMap ABI payload is expected to be 128 bytes.");
  return {true, nullptr};
}

inline TensorMapBuildArgs make_b_tensor_map_build_args(
    const void* b_base, int num_experts, int prob_k, int prob_n, int b_bits) {
  const int packed_cols = prob_n * b_bits / 8;
  return TensorMapBuildArgs{
      b_base,
      static_cast<uint64_t>(num_experts) * static_cast<uint64_t>(prob_k),
      static_cast<uint64_t>(packed_cols),
      static_cast<uint64_t>(packed_cols),
      64,
      static_cast<uint32_t>(128 * b_bits / 8),
      1,
      1,
  };
}

#if defined(MARLIN_SM90A_DEVICE)

// ---------------------------------------------------------------------------
// SM90 primitives
// ---------------------------------------------------------------------------

__device__ __forceinline__ int lane_id() { return threadIdx.x & 31; }

__device__ __forceinline__ int warp_id() { return threadIdx.x >> 5; }

__device__ __forceinline__ void mbarrier_init(uint64_t* barrier,
                                              uint32_t expected_count) {
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(smem),
               "r"(expected_count));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* barrier,
                                                          uint32_t bytes) {
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(smem),
               "r"(bytes));
}

template <int phase>
__device__ __forceinline__ void mbarrier_wait(uint64_t* barrier) {
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "wait_loop:\n"
      "  mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
      "  @!p bra wait_loop;\n"
      "}\n" ::"r"(smem),
      "n"(phase & 1));
}

__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit_group() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int pending_groups>
__device__ __forceinline__ void wgmma_wait_group() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(pending_groups)
               : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc(const void* smem_ptr, int lbo, int sbo, int swizzle_mode) {
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  uint64_t desc = 0;
  desc |= (static_cast<uint64_t>(smem) >> 4) & 0x3FFF; // Base address (bits 0-13)
  desc |= (static_cast<uint64_t>(lbo) & 0x3FFF) << 16; // Leading byte offset (bits 16-29)
  desc |= (static_cast<uint64_t>(sbo) & 0x3FFF) << 32; // Stride byte offset (bits 32-45)
  desc |= (static_cast<uint64_t>(swizzle_mode) & 0x3) << 62; // Swizzle mode (bits 62-63)
  return desc;
}

__device__ __forceinline__ void tma_load_2d_b_tile(
    void* dst_smem, const void* tensor_map, int packed_col, int packed_row,
    uint64_t* barrier) {
  uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
  uint32_t bar = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(dst),
      "l"(tensor_map), "r"(packed_col), "r"(packed_row), "r"(bar)
      : "memory");
}

__device__ __forceinline__ void tma_load_2d_b_tile_multicast(
    void* dst_smem, const void* tensor_map, int packed_col, int packed_row,
    uint64_t* barrier, uint16_t mcast_mask) {
  uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
  uint32_t bar = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::multicast::"
      "bytes [%0], [%1, {%2, %3}], [%4], %5;\n" ::"r"(dst),
      "l"(tensor_map), "r"(packed_col), "r"(packed_row), "r"(bar), "h"(mcast_mask)
      : "memory");
}

// ---------------------------------------------------------------------------
// Shared-memory tile views
// ---------------------------------------------------------------------------

template <int stages, int b_bits>
struct SharedStorageView {
  uint64_t* barriers;
  half* a;          // [stages][64][64], row-major.
  int4* b_packed;   // [stages][64][128] packed as legacy Marlin B.
  half* b_dequant;  // [stages][64][128], WGMMA B operand layout.
  float* c_scratch;

  __device__ explicit SharedStorageView(void* smem) {
    char* base = reinterpret_cast<char*>(smem);
    barriers = reinterpret_cast<uint64_t*>(base);
    base += stages * sizeof(uint64_t);
    a = reinterpret_cast<half*>(base);
    base += stages * 64 * 64 * sizeof(half);
    b_packed = reinterpret_cast<int4*>(base);
    base += stages * 64 * 128 * b_bits / 8;
    b_dequant = reinterpret_cast<half*>(base);
    base += stages * 64 * 128 * sizeof(half);
    c_scratch = reinterpret_cast<float*>(base);
  }

  __device__ half* a_stage(int stage) { return a + stage * 64 * 64; }

  __device__ int4* b_packed_stage(int stage) {
    return reinterpret_cast<int4*>(
        reinterpret_cast<char*>(b_packed) + stage * 64 * 128 * b_bits / 8);
  }

  __device__ half* b_dequant_stage(int stage) {
    return b_dequant + stage * 64 * 128;
  }
};

// ---------------------------------------------------------------------------
// CTA -> logical tile mapping
// ---------------------------------------------------------------------------

template <int moe_block_size>
__device__ int count_valid_tokens(const int32_t* sorted_token_ids,
                                  int prob_m_top_k) {
  int local = 0;
  for (int i = lane_id(); i < moe_block_size; i += 32) {
    local += sorted_token_ids[i] < prob_m_top_k ? 1 : 0;
  }
  return __reduce_add_sync(0xffffffff, local);
}

template <int moe_block_size>
__device__ TileWork map_cta_to_tile(const Params& params, int logical_tile,
                                    int k_stage_begin, int k_stage_end,
                                    int lock_offset) {
  TileWork work{};
  work.par_id = logical_tile / 2;
  work.n_panel = logical_tile & 1;
  work.expert_id = params.expert_ids[work.par_id];
  work.k_stage_begin = k_stage_begin;
  work.k_stage_end = k_stage_end;
  work.lock_offset = lock_offset;
  const int32_t* sorted =
      params.sorted_token_ids + work.par_id * moe_block_size;
  work.valid_m = count_valid_tokens<moe_block_size>(
      sorted, params.prob_m * params.top_k);
  return work;
}

// ---------------------------------------------------------------------------
// Producer path
// ---------------------------------------------------------------------------

template <int moe_block_size>
__device__ void gather_a_stage(const Params& params, const TileWork& work,
                               int k_stage_idx, half* sh_a) {
  const int32_t* sorted =
      params.sorted_token_ids + work.par_id * moe_block_size;
  const int k_base = k_stage_idx * 64;

  const int total_int4s = (64 * 64) / 8;
  for (int i = threadIdx.x; i < total_int4s; i += blockDim.x) {
    int row = i / 8;
    int col_chunk = i % 8;
    int4 val = {0, 0, 0, 0};
    if (row < work.valid_m) {
      int64_t token = sorted[row] / params.top_k;
      const int4* a_int4 = reinterpret_cast<const int4*>(params.A);
      val = a_int4[token * (params.prob_k / 8) + (k_base / 8) + col_chunk];
    }
    // 128B Swizzle: XOR row bits [0,2] into col bits [4,6] (which is int4 index bits [0,2])
    // A row is 128 bytes. col_chunk is 0..7 (each is 16 bytes, total 128 bytes).
    // Swizzle XORs (row & 7) into col_chunk.
    int swizzled_col_chunk = col_chunk ^ (row & 7);
    int swizzled_i = row * 8 + swizzled_col_chunk;
    reinterpret_cast<int4*>(sh_a)[swizzled_i] = val;
  }
}

template <int b_bits>
__device__ void copy_b_packed_stage(const Params& params, const TileWork& work,
                                    int pipe, int k_stage_idx,
                                    int4* sh_b_packed, uint64_t* barrier) {
  constexpr int kStage = 64;
  constexpr int nPanel = 128;
  constexpr int bytes = kStage * nPanel * b_bits / 8;
  constexpr int vecs = bytes / static_cast<int>(sizeof(int4));

  const int packed_cols_total = params.prob_n * b_bits / 32;
  const int packed_cols_panel = nPanel * b_bits / 32;
  const int packed_row =
      work.expert_id * params.prob_k + k_stage_idx * kStage;
  const int packed_col = work.n_panel * packed_cols_panel;

  if (params.use_tma_load && params.B_tma_map != nullptr) {
    if (threadIdx.x == 0) {
      mbarrier_arrive_expect_tx(&barrier[pipe], bytes);
      tma_load_2d_b_tile(sh_b_packed, params.B_tma_map, packed_col,
                         packed_row, &barrier[pipe]);
    }
    return;
  }

  // Non-TMA fallback keeps the ABI testable before host tensor-map creation is
  // wired.  It is also useful for numerics debugging.
  const int int_offset = packed_row * packed_cols_total + packed_col;
  const int4* src = params.B + int_offset / 4;

  for (int i = threadIdx.x; i < vecs; i += blockDim.x) {
    sh_b_packed[i] = src[i];
  }
}

template <int b_bits>
__device__ __forceinline__ int unpack_quant(const uint8_t* packed, int idx) {
  if constexpr (b_bits == 4) {
    uint8_t byte = packed[idx >> 1];
    return (idx & 1) ? (byte >> 4) : (byte & 0x0f);
  } else {
    static_assert(b_bits == 8);
    return packed[idx];
  }
}

__device__ __forceinline__ half load_half_scale(const int4* scales,
                                                int scale_idx) {
  const half* s = reinterpret_cast<const half*>(scales);
  return s[scale_idx];
}

template <int b_bits>
__device__ void dequant_b_stage_to_wgmma_shared(const Params& params,
                                                const TileWork& work,
                                                int k_stage_idx,
                                                const int4* sh_b_packed,
                                                half* sh_b_dequant) {
  constexpr int kStage = 64;
  constexpr int nPanel = 128;
  const uint8_t* packed = reinterpret_cast<const uint8_t*>(sh_b_packed);
  const int elements = kStage * nPanel;
  const int group_size = params.group_size > 0 ? params.group_size : params.prob_k;
  const int scales_expert_stride = params.prob_n * params.prob_k / group_size;

  for (int i = threadIdx.x; i < elements / 8; i += blockDim.x) {
    int linear_start = i * 8;
    int k = linear_start / nPanel;
    int n_start = linear_start % nPanel;
    
    uint32_t regs[4];
    half* h_regs = reinterpret_cast<half*>(regs);

#pragma unroll
    for (int j = 0; j < 8; j++) {
      int linear = linear_start + j;
      int n = n_start + j;
      int global_k = k_stage_idx * kStage + k;
      int global_n = work.n_panel * nPanel + n;

      int q = unpack_quant<b_bits>(packed, linear);
      float centered = static_cast<float>(q) -
                       static_cast<float>((1 << b_bits) - 1) * 0.5f;
      int scale_idx = work.expert_id * scales_expert_stride +
                      (global_k / group_size) * params.prob_n + global_n;
      half scale = load_half_scale(params.scales, scale_idx);
      h_regs[j] = __float2half(centered * __half2float(scale));
    }
    // 128B Swizzle: XOR row bits [0,2] into col bits [4,6] (which is int4 index bits [0,2])
    // B is 64 rows (k) by 128 cols (n). A row is 256 bytes.
    // Wait, 128B swizzle applies to 128-byte segments.
    // For a 256-byte row, there are two 128-byte segments.
    // The swizzle XORs (row & 7) into the 16-byte chunk index within the 128-byte segment.
    // n_start is the column index in halfs (0..127). n_start / 8 is the 16-byte chunk index (0..15).
    // The segment index is (n_start / 8) / 8 = (n_start / 64).
    // The chunk index within segment is (n_start / 8) % 8.
    // Swizzled chunk index within segment = ((n_start / 8) % 8) ^ (k & 7).
    // Swizzled overall chunk index = (n_start / 64) * 8 + (((n_start / 8) % 8) ^ (k & 7)).
    // Since i is the overall chunk index (i = k * 16 + n_start / 8),
    // swizzled_i = k * 16 + (n_start / 64) * 8 + (((n_start / 8) % 8) ^ (k & 7)).
    int chunk_idx = n_start / 8;
    int swizzled_chunk_idx = (chunk_idx & ~7) | ((chunk_idx & 7) ^ (k & 7));
    int swizzled_i = k * 16 + swizzled_chunk_idx;
    reinterpret_cast<int4*>(sh_b_dequant)[swizzled_i] = *reinterpret_cast<int4*>(regs);
  }
}

// ---------------------------------------------------------------------------
// Consumer path
// ---------------------------------------------------------------------------

template <int stages, int b_bits>
__device__ void warpgroup_mma_accumulate(SharedStorageView<stages, b_bits>& sh,
                                         int stage, float* accum) {
  // A is 64x64 half (row-major).
  // 128B Swizzle (mode 1). For K-major (row-major A), LBO is not used (0).
  // SBO is offset from first 8 rows to next 8 rows.
  // 8 rows of 64 halfs = 8 * 128 bytes = 1024 bytes. 1024 / 16 = 64.
  uint64_t a_desc = make_smem_desc(sh.a_stage(stage), 0, 64, 1);

  // B is 64x128 half (row-major in shared memory).
  // We use trans-b = 1 (B is row-major).
  // For K-major (row-major B), LBO is not used (0).
  // SBO is offset from first 8 rows to next 8 rows.
  // 8 rows of 128 halfs = 8 * 256 bytes = 2048 bytes. 2048 / 16 = 128.
  uint64_t b_desc = make_smem_desc(sh.b_dequant_stage(stage), 0, 128, 1);

  // One thread owns 64 accumulator registers for an m64n128 tile.  The four
  // k16 groups cover the k64 producer stage.
#pragma unroll
  for (int kk = 0; kk < 4; kk++) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,"
        "%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,"
        "%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,"
        "%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,"
        "%56,%57,%58,%59,%60,%61,%62,%63},"
        " %64, %65, 1, 1, 1, 0, 1;\n"
        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]),
          "+f"(accum[4]), "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7]),
          "+f"(accum[8]), "+f"(accum[9]), "+f"(accum[10]), "+f"(accum[11]),
          "+f"(accum[12]), "+f"(accum[13]), "+f"(accum[14]), "+f"(accum[15]),
          "+f"(accum[16]), "+f"(accum[17]), "+f"(accum[18]), "+f"(accum[19]),
          "+f"(accum[20]), "+f"(accum[21]), "+f"(accum[22]), "+f"(accum[23]),
          "+f"(accum[24]), "+f"(accum[25]), "+f"(accum[26]), "+f"(accum[27]),
          "+f"(accum[28]), "+f"(accum[29]), "+f"(accum[30]), "+f"(accum[31]),
          "+f"(accum[32]), "+f"(accum[33]), "+f"(accum[34]), "+f"(accum[35]),
          "+f"(accum[36]), "+f"(accum[37]), "+f"(accum[38]), "+f"(accum[39]),
          "+f"(accum[40]), "+f"(accum[41]), "+f"(accum[42]), "+f"(accum[43]),
          "+f"(accum[44]), "+f"(accum[45]), "+f"(accum[46]), "+f"(accum[47]),
          "+f"(accum[48]), "+f"(accum[49]), "+f"(accum[50]), "+f"(accum[51]),
          "+f"(accum[52]), "+f"(accum[53]), "+f"(accum[54]), "+f"(accum[55]),
          "+f"(accum[56]), "+f"(accum[57]), "+f"(accum[58]), "+f"(accum[59]),
          "+f"(accum[60]), "+f"(accum[61]), "+f"(accum[62]), "+f"(accum[63])
        : "l"(a_desc + kk * 16 * 64 * sizeof(half)),
          "l"(b_desc + kk * 16 * 128 * sizeof(half)));
  }
}

// ---------------------------------------------------------------------------
// Dimension 1: Register-Sourced A for WGMMA
// ---------------------------------------------------------------------------
template <int stages, int b_bits>
__device__ void warpgroup_mma_accumulate_regA(uint64_t b_desc, const uint32_t reg_A[4], float* accum) {
  // A is kept in registers. We pass reg_A directly to the instruction.
#pragma unroll
  for (int kk = 0; kk < 4; kk++) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
        " {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n"
        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]),
          "+f"(accum[4]), "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7]),
          "+f"(accum[8]), "+f"(accum[9]), "+f"(accum[10]), "+f"(accum[11]),
          "+f"(accum[12]), "+f"(accum[13]), "+f"(accum[14]), "+f"(accum[15]),
          "+f"(accum[16]), "+f"(accum[17]), "+f"(accum[18]), "+f"(accum[19]),
          "+f"(accum[20]), "+f"(accum[21]), "+f"(accum[22]), "+f"(accum[23]),
          "+f"(accum[24]), "+f"(accum[25]), "+f"(accum[26]), "+f"(accum[27]),
          "+f"(accum[28]), "+f"(accum[29]), "+f"(accum[30]), "+f"(accum[31]),
          "+f"(accum[32]), "+f"(accum[33]), "+f"(accum[34]), "+f"(accum[35]),
          "+f"(accum[36]), "+f"(accum[37]), "+f"(accum[38]), "+f"(accum[39]),
          "+f"(accum[40]), "+f"(accum[41]), "+f"(accum[42]), "+f"(accum[43]),
          "+f"(accum[44]), "+f"(accum[45]), "+f"(accum[46]), "+f"(accum[47]),
          "+f"(accum[48]), "+f"(accum[49]), "+f"(accum[50]), "+f"(accum[51]),
          "+f"(accum[52]), "+f"(accum[53]), "+f"(accum[54]), "+f"(accum[55]),
          "+f"(accum[56]), "+f"(accum[57]), "+f"(accum[58]), "+f"(accum[59]),
          "+f"(accum[60]), "+f"(accum[61]), "+f"(accum[62]), "+f"(accum[63])
        : "r"(reg_A[0]), "r"(reg_A[1]), "r"(reg_A[2]), "r"(reg_A[3]),
          "l"(b_desc + kk * 16 * 128 * sizeof(half)));
  }
}

// ---------------------------------------------------------------------------
// Dimension 2: Hardware Dequantization (INT4 WGMMA)
// ---------------------------------------------------------------------------
template <int stages, int b_bits>
__device__ void warpgroup_mma_accumulate_int4(uint64_t a_desc, uint64_t b_desc, int32_t* accum) {
  // WGMMA s32.s4.s4 for W4A4.
#pragma unroll
  for (int kk = 0; kk < 2; kk++) { // k32 per instruction, 64 total K -> 2 iterations
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.s4.s4 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
        " %64, %65, 1, 1, 1, 0, 1;\n"
        : "+r"(accum[0]), "+r"(accum[1]), "+r"(accum[2]), "+r"(accum[3]),
          "+r"(accum[4]), "+r"(accum[5]), "+r"(accum[6]), "+r"(accum[7]),
          "+r"(accum[8]), "+r"(accum[9]), "+r"(accum[10]), "+r"(accum[11]),
          "+r"(accum[12]), "+r"(accum[13]), "+r"(accum[14]), "+r"(accum[15]),
          "+r"(accum[16]), "+r"(accum[17]), "+r"(accum[18]), "+r"(accum[19]),
          "+r"(accum[20]), "+r"(accum[21]), "+r"(accum[22]), "+r"(accum[23]),
          "+r"(accum[24]), "+r"(accum[25]), "+r"(accum[26]), "+r"(accum[27]),
          "+r"(accum[28]), "+r"(accum[29]), "+r"(accum[30]), "+r"(accum[31]),
          "+r"(accum[32]), "+r"(accum[33]), "+r"(accum[34]), "+r"(accum[35]),
          "+r"(accum[36]), "+r"(accum[37]), "+r"(accum[38]), "+r"(accum[39]),
          "+r"(accum[40]), "+r"(accum[41]), "+r"(accum[42]), "+r"(accum[43]),
          "+r"(accum[44]), "+r"(accum[45]), "+r"(accum[46]), "+r"(accum[47]),
          "+r"(accum[48]), "+r"(accum[49]), "+r"(accum[50]), "+r"(accum[51]),
          "+r"(accum[52]), "+r"(accum[53]), "+r"(accum[54]), "+r"(accum[55]),
          "+r"(accum[56]), "+r"(accum[57]), "+r"(accum[58]), "+r"(accum[59]),
          "+r"(accum[60]), "+r"(accum[61]), "+r"(accum[62]), "+r"(accum[63])
        : "l"(a_desc + kk * 32 * 64 * sizeof(uint8_t) / 2),
          "l"(b_desc + kk * 32 * 128 * sizeof(uint8_t) / 2));
  }
}

template <int moe_block_size>
__device__ void store_tile(const Params& params, const TileWork& work,
                           const float* accum) {
  const int32_t* sorted =
      params.sorted_token_ids + work.par_id * moe_block_size;
  half* C_half = reinterpret_cast<half*>(params.C);
  float* C_tmp = reinterpret_cast<float*>(params.C_tmp);

  // Conservative first epilogue: each lane writes a strided subset of the
  // logical m64n128 accumulator tile.  The mapping from WGMMA accumulator
  // registers to row/column is finalized here; split-K either atomically
  // accumulates fp32 scratch or writes final fp16.
  for (int i = threadIdx.x; i < 64 * 128; i += blockDim.x) {
    int row = i / 128;
    int col = i - row * 128;
    if (row >= work.valid_m) continue;
    int64_t sorted_row = sorted[row];
    int true_row = sorted_row / params.top_k;
    int true_col = work.n_panel * 128 + col;
    float value = accum[i & 63];
    if (params.mul_topk_weights) {
      value *= params.topk_weights[sorted_row];
    }

    int out_idx = true_row * params.prob_n + true_col;
    if (params.sk_slice_count > 1) {
      int tmp_tile = work.lock_offset * 64 * 128;
      float* tmp = C_tmp + tmp_tile + row * 128 + col;
      if (params.sk_slice_idx == 0) {
        *tmp = value;
      } else {
        atomicAdd(tmp, value);
      }
      if (params.sk_slice_idx == params.sk_slice_count - 1) {
        C_half[out_idx] = __float2half_rn(*tmp);
      }
    } else {
      C_half[out_idx] = __float2half_rn(value);
    }
  }
}

template <int moe_block_size, int b_bits, int stages = 3>
__device__ void run_dataflow(const Params& params, TileWork work, void* smem) {
  SharedStorageView<stages, b_bits> sh(smem);
  float accum[64] = {};

  for (int i = threadIdx.x; i < stages; i += blockDim.x) {
    mbarrier_init(&sh.barriers[i], 1); // Only thread 0 arrives for TMA
  }
  __syncthreads();

  // Prologue
  for (int pipe = 0; pipe < stages - 1; pipe++) {
    int k_stage = work.k_stage_begin + pipe;
    if (k_stage < work.k_stage_end) {
      gather_a_stage<moe_block_size>(params, work, k_stage, sh.a_stage(pipe));
      copy_b_packed_stage<b_bits>(params, work, pipe, k_stage,
                                  sh.b_packed_stage(pipe), sh.barriers);
    }
  }
  __syncthreads();

  for (int k_stage = work.k_stage_begin; k_stage < work.k_stage_end; k_stage++) {
    int pipe = (k_stage - work.k_stage_begin) % stages;
    int next_k_stage = k_stage + stages - 1;
    int next_pipe = next_k_stage % stages;

    // Issue next fetch
    if (next_k_stage < work.k_stage_end) {
      gather_a_stage<moe_block_size>(params, work, next_k_stage, sh.a_stage(next_pipe));
      copy_b_packed_stage<b_bits>(params, work, next_pipe, next_k_stage,
                                  sh.b_packed_stage(next_pipe), sh.barriers);
    }

    // Wait for current fetch
    if (params.use_tma_load && params.B_tma_map != nullptr) {
      uint32_t smem_bar = static_cast<uint32_t>(__cvta_generic_to_shared(&sh.barriers[pipe]));
      int phase = ((k_stage - work.k_stage_begin) / stages) & 1;
      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "wait_loop:\n"
          "  mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
          "  @!p bra wait_loop;\n"
          "}\n" ::"r"(smem_bar), "r"(phase));
    }
    __syncthreads(); // Ensure manual copies (A gather and fallback B) are done

    // Dequantize B
    dequant_b_stage_to_wgmma_shared<b_bits>(params, work, k_stage,
                                            sh.b_packed_stage(pipe),
                                            sh.b_dequant_stage(pipe));
    __syncthreads(); // Wait for dequant to finish

    // Issue WGMMA
    wgmma_fence();
    warpgroup_mma_accumulate<stages, b_bits>(sh, pipe, accum);
    wgmma_commit_group();

    if constexpr (stages >= 3) {
      wgmma_wait_group<stages - 2>();
    } else {
      wgmma_wait_group<0>();
    }
  }

  wgmma_wait_group<0>();
  store_tile<moe_block_size>(params, work, accum);
}

template <int moe_block_size, int b_bits, int stages = 3>
__global__ void __launch_bounds__(128, 1)
    MarlinSm90TmaWgmmaKernel(Params params) {
  extern __shared__ __align__(16) unsigned char smem[];

  int parallel = params.num_tokens_past_padded[0] / moe_block_size;
  int total_tiles = logical_mn_tiles(parallel, params.prob_n);
  int logical_tile = blockIdx.x;
  if (logical_tile >= total_tiles) {
    return;
  }

  // First kernel boundary: pure-DP tile body.  A Stream-K adapter will map
  // blockIdx.x to (logical_tile, k slice, lock offset) before this is enabled
  // for split-K shapes.
  TileWork work = map_cta_to_tile<moe_block_size>(
      params, logical_tile, 0, k_stages(params.prob_k), logical_tile);
  run_dataflow<moe_block_size, b_bits, stages>(params, work, smem);
}

// ---------------------------------------------------------------------------
// Dimension 4: W1 + W2 DSMEM Fusion
// ---------------------------------------------------------------------------
struct FusedParams {
  Params w1_params;
  Params w2_params;
};

template <int moe_block_size, int b_bits, int stages = 3>
__global__ void __launch_bounds__(128, 1)
    MarlinSm90FusedW1W2Kernel(FusedParams fused_params) {
  extern __shared__ __align__(16) unsigned char smem[];
  
  // Cluster size is 2. Block 0 does W1, Block 1 does W2.
  // In PTX, we can use %clusterid and %cluster_ctaid
  uint32_t cluster_rank;
  asm volatile("mov.u32 %0, %cluster_ctaid.x;\n" : "=r"(cluster_rank));
  
  if (cluster_rank == 0) {
    // W1
    int parallel = fused_params.w1_params.num_tokens_past_padded[0] / moe_block_size;
    int total_tiles = logical_mn_tiles(parallel, fused_params.w1_params.prob_n);
    int logical_tile = blockIdx.x;
    if (logical_tile >= total_tiles) return;
    
    TileWork work = map_cta_to_tile<moe_block_size>(
        fused_params.w1_params, logical_tile, 0, k_stages(fused_params.w1_params.prob_k), logical_tile);
        
    // run W1 dataflow
    run_dataflow<moe_block_size, b_bits, stages>(fused_params.w1_params, work, smem);
    
    // Instead of store_tile, we would store to W2's DSMEM here.
    // For conceptual demonstration, we use the store_to_dsmem function.
    // store_to_dsmem(accum, w2_smem_ptr, 1);
  } else {
    // W2
    // Wait for DSMEM from W1, then run W2 dataflow
    // ...
  }
}

#else

template <int moe_block_size, int b_bits, int stages = 3>
__global__ void __launch_bounds__(128, 1)
    MarlinSm90TmaWgmmaKernel(Params params) {}

#endif  // MARLIN_SM90A_DEVICE

#undef MARLIN_SM90_HD

}  // namespace marlin_sm90_tma_wgmma
