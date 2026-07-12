#include "marlin_moe_cutlass69.h"

#include <c10/util/Exception.h>

#include <cstdlib>
#include <vector>

namespace marlin_moe_cutlass69_host {

bool cutlass69_compiled() {
#ifdef MARLIN_MOE_HAS_CUTLASS
  return true;
#else
  return false;
#endif
}

bool cutlass69_env_enabled() {
  const char* env = std::getenv("MARLIN_MOE_USE_CUTLASS69");
  return env != nullptr && env[0] == '1';
}

bool cutlass69_fused_gemm1_env_enabled() {
  const char* env = std::getenv("MARLIN_MOE_USE_CUTLASS69_FUSED_GEMM1");
  return env != nullptr && env[0] == '1';
}

HostSupport select_host_path(int major_capability, int a_bits, int b_bits,
                             int prob_m, int prob_n, int prob_k,
                             bool has_act_order, bool has_zp, int moe_block_size,
                             int group_size) {
  if (!cutlass69_compiled()) {
    return {false,
            "CUTLASS was not found at build time. Rebuild with CUTLASS_DIR "
            "pointing to an NVIDIA/cutlass checkout."};
  }
  if (major_capability < 9) {
    return {false, "CUTLASS example-69 path requires SM90 or newer."};
  }
  if (prob_k != 6144 || prob_m < 16 || prob_m > 8192) {
    return {false,
            "CUTLASS example-69 path is scoped to K=6144 and "
            "16 <= M <= 8192."};
  }
  if (prob_n != 256 && prob_n != 512) {
    return {false,
            "CUTLASS example-69 path supports N=256 (GEMM2) or N=512 (GEMM1)."};
  }
  if (a_bits != 16) {
    return {false, "CUTLASS example-69 path is scoped to WNA16 activations."};
  }
  if (b_bits != 4) {
    return {false, "CUTLASS example-69 path currently supports INT4 B only."};
  }
  if (has_act_order || has_zp) {
    return {false,
            "act_order/zero-point are not supported on the CUTLASS69 path."};
  }
  if (moe_block_size != 16 && moe_block_size != 32 && moe_block_size != 64) {
    return {false,
            "CUTLASS example-69 path supports moe_block_size in {16, 32, 64}."};
  }
  if (group_size != 128 && group_size != -1) {
    return {false,
            "CUTLASS example-69 path expects group_size=128 or channelwise (-1)."};
  }
  return {true, nullptr};
}

} // namespace marlin_moe_cutlass69_host

#ifndef MARLIN_MOE_HAS_CUTLASS

namespace marlin_moe_cutlass69_host {

void dispatch_marlin_moe_cutlass69(
    const void* A, const void* B, void* C, const void* b_scales,
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    const int32_t* num_tokens_past_padded, const float* topk_weights,
    int moe_block_size, int num_experts, int top_k, bool mul_topk_weights,
    int prob_m, int prob_n, int prob_k, vllm::ScalarType const& a_type,
    vllm::ScalarType const& b_type, vllm::ScalarType const& c_type,
    int group_size, int dev, cudaStream_t stream) {
  (void)A;
  (void)B;
  (void)C;
  (void)b_scales;
  (void)sorted_token_ids;
  (void)expert_ids;
  (void)num_tokens_past_padded;
  (void)topk_weights;
  (void)moe_block_size;
  (void)num_experts;
  (void)top_k;
  (void)mul_topk_weights;
  (void)prob_m;
  (void)prob_n;
  (void)prob_k;
  (void)a_type;
  (void)b_type;
  (void)c_type;
  (void)group_size;
  (void)dev;
  (void)stream;
  TORCH_CHECK(false,
              "MARLIN_MOE_USE_CUTLASS69=1 but extension was built without "
              "CUTLASS. Set CUTLASS_DIR and rebuild.");
}

void dispatch_marlin_moe_cutlass69_fused_gemm1(
    const void* A, const void* B, void* C, const void* b_scales,
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    const int32_t* num_tokens_past_padded, int moe_block_size, int num_experts,
    int top_k, int prob_m, int prob_n, int prob_k,
    vllm::ScalarType const& a_type, vllm::ScalarType const& b_type,
    vllm::ScalarType const& c_type, int group_size, int dev,
    cudaStream_t stream) {
  (void)A;
  (void)B;
  (void)C;
  (void)b_scales;
  (void)sorted_token_ids;
  (void)expert_ids;
  (void)num_tokens_past_padded;
  (void)moe_block_size;
  (void)num_experts;
  (void)top_k;
  (void)prob_m;
  (void)prob_n;
  (void)prob_k;
  (void)a_type;
  (void)b_type;
  (void)c_type;
  (void)group_size;
  (void)dev;
  (void)stream;
  TORCH_CHECK(false,
              "MARLIN_MOE_USE_CUTLASS69_FUSED_GEMM1=1 requires CUTLASS. "
              "Set CUTLASS_DIR and rebuild.");
}

} // namespace marlin_moe_cutlass69_host

#else  // MARLIN_MOE_HAS_CUTLASS

#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/util/debug.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/mixed_dtype_utils.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#ifndef CUDA_CHECK
#define CUDA_CHECK(status)                                                   \
  {                                                                          \
    cudaError_t error = status;                                              \
    if (error != cudaSuccess) {                                              \
      throw std::runtime_error(std::string("CUDA error: ") +                 \
                               cudaGetErrorString(error));                   \
    }                                                                        \
  }
#endif

#ifndef CUTLASS_CHECK
#define CUTLASS_CHECK(status)                                                \
  {                                                                          \
    cutlass::Status error = status;                                          \
    if (error != cutlass::Status::kSuccess) {                                \
      throw std::runtime_error(                                              \
          std::string("CUTLASS error: ") +                                   \
          cutlass::cutlassGetStatusString(error));                           \
    }                                                                        \
  }
#endif

#include <chrono>
#include <cstdio>
#include <type_traits>

#include <cub/device/device_scan.cuh>

namespace marlin_moe_cutlass69_host {

using namespace cute;

namespace {

struct Cutlass69GemmProfile {
  double host_setup_ms = 0.0;
  double gemm_init_ms = 0.0;
  double gemm_run_ms = 0.0;
  int num_expert_problems = 0;
};

struct Cutlass69FusedProfile {
  double host_prep_ms = 0.0;
  double buffer_alloc_ms = 0.0;
  double gather_a_ms = 0.0;
  Cutlass69GemmProfile gemm{};
  double fused_silu_scatter_ms = 0.0;
  int groups = 0;
  int padded_m = 0;
  int packed_m = 0;
  bool bfull = false;
};

bool cutlass69_bfull_enabled(int prob_m) {
  const char* env = std::getenv("MARLIN_MOE_CUTLASS69_BFULL");
  if (env != nullptr) {
    return env[0] != '0';
  }
  return prob_m >= 1024;
}

bool cutlass69_profile_enabled() {
  const char* env = std::getenv("MARLIN_MOE_CUTLASS69_PROFILE");
  return env != nullptr && env[0] == '1';
}

double cutlass69_elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
  float ms = 0.f;
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  return static_cast<double>(ms);
}

void cutlass69_print_fused_profile(int call_id, const Cutlass69FusedProfile& profile,
                                   int prob_m, int prob_n, int prob_k) {
  const double gemm_total = profile.gemm.host_setup_ms + profile.gemm.gemm_init_ms +
                            profile.gemm.gemm_run_ms;
  const double total = profile.host_prep_ms + profile.buffer_alloc_ms +
                       profile.gather_a_ms + gemm_total +
                       profile.fused_silu_scatter_ms;
  const double denom = total > 0.0 ? total : 1.0;
  fprintf(stderr,
          "[CUTLASS69 fused profile #%d] M=%d N=%d K=%d groups=%d padded_M=%d "
          "packed_M=%d bfull=%d expert_problems=%d total=%.3f ms\n",
          call_id, prob_m, prob_n, prob_k, profile.groups, profile.padded_m,
          profile.packed_m, profile.bfull ? 1 : 0,
          profile.gemm.num_expert_problems, total);
  if (profile.bfull) {
    fprintf(stderr, "  expert_pack_a (replaces gather): %8.3f ms (%5.1f%%)\n",
            profile.gather_a_ms, 100.0 * profile.gather_a_ms / denom);
  } else {
    fprintf(stderr, "  gather_a:                        %8.3f ms (%5.1f%%)\n",
            profile.gather_a_ms, 100.0 * profile.gather_a_ms / denom);
  }
  fprintf(stderr, "  host_prep (num_tokens D2H+sync): %8.3f ms (%5.1f%%)\n",
          profile.host_prep_ms, 100.0 * profile.host_prep_ms / denom);
  fprintf(stderr, "  buffer_alloc (A/C temp):         %8.3f ms (%5.1f%%)\n",
          profile.buffer_alloc_ms, 100.0 * profile.buffer_alloc_ms / denom);
  fprintf(stderr, "  gemm_host_setup (expert_ids+H2D): %8.3f ms (%5.1f%%)\n",
          profile.gemm.host_setup_ms,
          100.0 * profile.gemm.host_setup_ms / denom);
  fprintf(stderr, "  gemm_init (workspace+initialize): %8.3f ms (%5.1f%%)\n",
          profile.gemm.gemm_init_ms, 100.0 * profile.gemm.gemm_init_ms / denom);
  fprintf(stderr, "  gemm_run (CUTLASS kernel):        %8.3f ms (%5.1f%%)\n",
          profile.gemm.gemm_run_ms, 100.0 * profile.gemm.gemm_run_ms / denom);
  fprintf(stderr, "  fused_silu_scatter:              %8.3f ms (%5.1f%%)\n",
          profile.fused_silu_scatter_ms,
          100.0 * profile.fused_silu_scatter_ms / denom);
  fprintf(stderr, "  --- gemm substages sum:          %8.3f ms\n", gemm_total);
}

using ProblemShape =
    cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using QuantType = cutlass::int4b_t;
constexpr int kScaleChunk = 128;

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

template <typename MmaType, typename ElementC_, int TileN_, int ClusterN_,
          bool RowMajorD = false>
struct Cutlass69GroupedGemmTypes {
  using ElementA = MmaType;
  using LayoutA = cutlass::layout::RowMajor;
  constexpr static int AlignmentA =
      128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = QuantType;
  using LayoutB = cutlass::layout::ColumnMajor;
  constexpr static int AlignmentB =
      128 / cutlass::sizeof_bits<ElementB>::value;

  using LayoutA_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  using LayoutB_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutB>::type;

  using StrideA =
      cute::remove_pointer_t<cutlass::detail::TagToStrideA_t<LayoutA*>>;
  using StrideB =
      cute::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutB*>>;

  using ValueShuffle = Layout<Shape<_2, _4>, Stride<_4, _1>>;
  constexpr static int NumShuffleAtoms = 1;
  using MmaAtomShape = Layout<Shape<_1, Int<NumShuffleAtoms>>>;
  using LayoutAtomQuant = decltype(
      cutlass::compute_memory_reordering_atom<MmaType, MmaAtomShape,
                                            ValueShuffle>());
  using LayoutB_Reordered = decltype(cute::tile_to_shape(
      LayoutAtomQuant{},
      Layout<Shape<int, int, Int<1>>, StrideB>{}));

  using ElementScale = MmaType;
  using LayoutScale = cutlass::layout::RowMajor;

  using ElementC = ElementC_;
  using LayoutC = cutlass::layout::RowMajor;
  using ElementD = ElementC;
  using LayoutD = LayoutC;
  constexpr static int AlignmentC =
      128 / cutlass::sizeof_bits<ElementC>::value;
  constexpr static int AlignmentD = AlignmentC;

  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  constexpr static int TileShapeK =
      128 * 8 / cutlass::sizeof_bits<MmaType>::value;
  using TileShape = Shape<_128, Int<TileN_>, Int<TileShapeK>>;
  using ClusterShape = Shape<_1, Int<ClusterN_>, _1>;
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;

  using LayoutCOut =
      std::conditional_t<RowMajorD, LayoutC,
                         typename cutlass::layout::LayoutTranspose<LayoutC>::type>;
  using LayoutDOut = LayoutCOut;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, TileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementAccumulator,
          ElementC, LayoutCOut*, AlignmentC,
          ElementD, LayoutDOut*, AlignmentD, EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass,
          cute::tuple<ElementB, ElementScale>, LayoutB_Reordered*, AlignmentB,
          ElementA, LayoutA_Transpose*, AlignmentA, ElementAccumulator,
          TileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::GroupScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideC = typename GemmKernel::InternalStrideC;
  using StrideD = StrideC;
  using StrideS = typename CollectiveMainloop::StrideScale;

  constexpr static bool kRowMajorD = RowMajorD;
};

using Cutlass69GemmTypesSmall =
    Cutlass69GroupedGemmTypes<cutlass::half_t, cutlass::half_t, 16, 1, false>;
using Cutlass69GemmTypesLarge =
    Cutlass69GroupedGemmTypes<cutlass::half_t, cutlass::half_t, 256, 2, true>;
using Cutlass69GemmTypesLargeBf16 =
    Cutlass69GroupedGemmTypes<cutlass::bfloat16_t, cutlass::bfloat16_t, 256, 2,
                              true>;
using Cutlass69GemmTypesSmallBf16 =
    Cutlass69GroupedGemmTypes<cutlass::bfloat16_t, cutlass::bfloat16_t, 16, 1,
                              false>;

// CUTLASS69 grouped GEMM stores D in column-major [rows, cols] (stride {N,M}).
template <typename MmaType>
__global__ void scatter_moe_c_kernel(
    const MmaType* __restrict__ C_grouped, MmaType* __restrict__ C_out,
    const int32_t* __restrict__ sorted_ids, int groups, int moe_block_size,
    int prob_n, int prob_m, int top_k) {
  const int64_t total_rows =
      static_cast<int64_t>(groups) * moe_block_size;
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = total_rows * prob_n;
  if (idx >= total) {
    return;
  }
  const int col = static_cast<int>(idx / total_rows);
  const int row = static_cast<int>(idx % total_rows);
  const int g = row / moe_block_size;
  const int r = row % moe_block_size;

  const int32_t sorted = sorted_ids[g * moe_block_size + r];
  if (sorted < 0 || sorted >= prob_m * top_k) {
    return;
  }

  const MmaType val = C_grouped[static_cast<int64_t>(col) * total_rows + row];
  C_out[static_cast<int64_t>(sorted) * prob_n + col] = val;
}

template <typename MmaType>
__global__ void scatter_moe_c_fused_silu_kernel(
    const MmaType* __restrict__ C_grouped, MmaType* __restrict__ C_out,
    const int32_t* __restrict__ sorted_ids, int groups, int moe_block_size,
    int prob_n, int prob_m, int top_k) {
  const int out_n = prob_n / 2;
  const int64_t total_rows =
      static_cast<int64_t>(groups) * moe_block_size;
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = total_rows * out_n;
  if (idx >= total) {
    return;
  }
  const int col = static_cast<int>(idx / total_rows);
  const int row = static_cast<int>(idx % total_rows);
  const int g = row / moe_block_size;
  const int r = row % moe_block_size;

  const int32_t sorted = sorted_ids[g * moe_block_size + r];
  if (sorted < 0 || sorted >= prob_m * top_k) {
    return;
  }

  const int64_t row_off = row;
  const float gate =
      static_cast<float>(C_grouped[row_off + static_cast<int64_t>(col) * total_rows]);
  const float up = static_cast<float>(
      C_grouped[row_off + static_cast<int64_t>(col + out_n) * total_rows]);
  const float silu_gate = gate / (1.0f + expf(-gate));
  C_out[static_cast<int64_t>(sorted) * out_n + col] = MmaType(silu_gate * up);
}

template <typename MmaType>
__global__ void gather_moe_a_kernel(const MmaType* __restrict__ A,
                                    MmaType* __restrict__ A_grouped,
                                    const int32_t* __restrict__ sorted_ids,
                                    int groups, int moe_block_size, int prob_m,
                                    int top_k, int prob_k) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total =
      static_cast<int64_t>(groups) * moe_block_size * prob_k;
  if (idx >= total) {
    return;
  }
  const int g = static_cast<int>(idx / (moe_block_size * prob_k));
  const int rem = static_cast<int>(idx % (moe_block_size * prob_k));
  const int row = rem / prob_k;
  const int col = rem % prob_k;

  MmaType* dst = A_grouped +
                 (static_cast<int64_t>(g) * moe_block_size + row) * prob_k +
                 col;
  const int32_t sorted = sorted_ids[g * moe_block_size + row];
  if (sorted < 0 || sorted >= prob_m * top_k) {
    *dst = MmaType(0);
    return;
  }
  const int64_t token = sorted / top_k;
  const MmaType* src = A + token * prob_k + col;
  *dst = *src;
}

template <typename MmaType>
__global__ void count_expert_tokens_kernel(int32_t* __restrict__ expert_counts,
                                           const int32_t* __restrict__ sorted_ids,
                                           const int32_t* __restrict__ expert_ids,
                                           int groups, int moe_block_size,
                                           int prob_m, int top_k, int num_experts) {
  const int64_t total = static_cast<int64_t>(groups) * moe_block_size;
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int g = static_cast<int>(idx / moe_block_size);
  const int expert = expert_ids[g];
  if (expert < 0 || expert >= num_experts) {
    return;
  }
  const int32_t sorted = sorted_ids[idx];
  if (sorted < 0 || sorted >= prob_m * top_k) {
    return;
  }
  atomicAdd(&expert_counts[expert], 1);
}

template <typename MmaType>
__global__ void pack_a_expert_rows_kernel(
    const MmaType* __restrict__ A, MmaType* __restrict__ A_packed,
    int32_t* __restrict__ expert_write_cursor, int32_t* __restrict__ packed_to_sorted,
    const int32_t* __restrict__ sorted_ids, const int32_t* __restrict__ expert_ids,
    int groups, int moe_block_size, int prob_m, int top_k, int prob_k,
    int num_experts) {
  const int64_t total_slots = static_cast<int64_t>(groups) * moe_block_size;
  const int64_t slot = static_cast<int64_t>(blockIdx.x);
  if (slot >= total_slots) {
    return;
  }
  const int g = static_cast<int>(slot / moe_block_size);
  const int expert = expert_ids[g];
  if (expert < 0 || expert >= num_experts) {
    return;
  }
  const int32_t sorted = sorted_ids[slot];
  if (sorted < 0 || sorted >= prob_m * top_k) {
    return;
  }
  const int dst_row = atomicAdd(&expert_write_cursor[expert], 1);
  packed_to_sorted[dst_row] = sorted;

  const int64_t token = static_cast<int64_t>(sorted) / top_k;
  const MmaType* src = A + token * prob_k;
  MmaType* dst = A_packed + static_cast<int64_t>(dst_row) * prob_k;
  for (int k = threadIdx.x; k < prob_k; k += blockDim.x) {
    dst[k] = src[k];
  }
}

template <typename MmaType>
__global__ void scatter_packed_fused_silu_rowmajor_kernel(
    const MmaType* __restrict__ C_packed, MmaType* __restrict__ C_out,
    const int32_t* __restrict__ packed_to_sorted, int packed_m, int prob_n) {
  const int out_n = prob_n / 2;
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(packed_m) * out_n;
  if (idx >= total) {
    return;
  }
  const int row = static_cast<int>(idx / out_n);
  const int col = static_cast<int>(idx % out_n);
  const int32_t sorted = packed_to_sorted[row];
  const int64_t row_base = static_cast<int64_t>(row) * prob_n;
  const float gate = static_cast<float>(C_packed[row_base + col]);
  const float up = static_cast<float>(C_packed[row_base + col + out_n]);
  const float silu_gate = gate / (1.0f + expf(-gate));
  C_out[static_cast<int64_t>(sorted) * out_n + col] = MmaType(silu_gate * up);
}

template <typename MmaType>
__global__ void scatter_moe_c_rowmajor_kernel(
    const MmaType* __restrict__ C_grouped, MmaType* __restrict__ C_out,
    const int32_t* __restrict__ sorted_ids, int groups, int moe_block_size,
    int prob_n, int prob_m, int top_k) {
  const int64_t total_rows =
      static_cast<int64_t>(groups) * moe_block_size;
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = total_rows * prob_n;
  if (idx >= total) {
    return;
  }
  const int col = static_cast<int>(idx % prob_n);
  const int row = static_cast<int>(idx / prob_n);
  const int g = row / moe_block_size;
  const int r = row % moe_block_size;

  const int32_t sorted = sorted_ids[g * moe_block_size + r];
  if (sorted < 0 || sorted >= prob_m * top_k) {
    return;
  }

  const MmaType val = C_grouped[static_cast<int64_t>(row) * prob_n + col];
  C_out[static_cast<int64_t>(sorted) * prob_n + col] = val;
}

template <typename MmaType>
__global__ void scatter_moe_c_fused_silu_rowmajor_kernel(
    const MmaType* __restrict__ C_grouped, MmaType* __restrict__ C_out,
    const int32_t* __restrict__ sorted_ids, int groups, int moe_block_size,
    int prob_n, int prob_m, int top_k) {
  const int out_n = prob_n / 2;
  const int64_t total_rows =
      static_cast<int64_t>(groups) * moe_block_size;
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = total_rows * out_n;
  if (idx >= total) {
    return;
  }
  const int col = static_cast<int>(idx % out_n);
  const int row = static_cast<int>(idx / out_n);
  const int g = row / moe_block_size;
  const int r = row % moe_block_size;

  const int32_t sorted = sorted_ids[g * moe_block_size + r];
  if (sorted < 0 || sorted >= prob_m * top_k) {
    return;
  }

  const int64_t row_base = static_cast<int64_t>(row) * prob_n;
  const float gate = static_cast<float>(C_grouped[row_base + col]);
  const float up = static_cast<float>(C_grouped[row_base + col + out_n]);
  const float silu_gate = gate / (1.0f + expf(-gate));
  C_out[static_cast<int64_t>(sorted) * out_n + col] = MmaType(silu_gate * up);
}

template <typename GemmTypes>
bool launch_grouped_gemm(
    int groups, int moe_block_size, int prob_n, int prob_k, int group_size,
    const typename GemmTypes::ElementA* A_grouped,
    const typename GemmTypes::ElementB* B_base,
    const typename GemmTypes::ElementScale* scale_base,
    typename GemmTypes::ElementC* C_grouped, const int32_t* expert_ids,
    size_t expert_b_bytes, size_t expert_scale_bytes, cudaStream_t stream,
    Cutlass69GemmProfile* profile = nullptr) {
  using Gemm = typename GemmTypes::Gemm;
  using ElementA = typename GemmTypes::ElementA;
  using ElementC = typename GemmTypes::ElementC;
  using ElementScale = typename GemmTypes::ElementScale;
  using LayoutB_Reordered = typename GemmTypes::LayoutB_Reordered;
  using StrideA = typename GemmTypes::StrideA;
  using StrideB = typename GemmTypes::StrideB;
  using StrideC = typename GemmTypes::StrideC;
  using StrideD = typename GemmTypes::StrideD;
  using StrideS = typename GemmTypes::StrideS;

  const int scale_k = cutlass::ceil_div(prob_k, group_size > 0 ? group_size : prob_k);
  const auto host_setup_begin = std::chrono::steady_clock::now();

  std::vector<int32_t> expert_ids_host(groups);
  CUDA_CHECK(cudaMemcpyAsync(expert_ids_host.data(), expert_ids,
                             groups * sizeof(int32_t),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  struct ExpertProblem {
    int expert_id;
    int start_group;
    int num_groups;
  };
  std::vector<ExpertProblem> expert_problems;
  for (int g = 0; g < groups; ++g) {
    if (expert_problems.empty() || expert_problems.back().expert_id != expert_ids_host[g]) {
      expert_problems.push_back({expert_ids_host[g], g, 1});
    } else {
      expert_problems.back().num_groups++;
    }
  }
  int num_expert_problems = expert_problems.size();

  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host(num_expert_problems);
  std::vector<const ElementA*> ptr_A_host(num_expert_problems);
  std::vector<const typename GemmTypes::ElementB*> ptr_B_host(num_expert_problems);
  std::vector<const ElementScale*> ptr_scale_host(num_expert_problems);
  std::vector<const ElementC*> ptr_C_host(num_expert_problems);
  std::vector<ElementC*> ptr_D_host(num_expert_problems);
  std::vector<StrideA> stride_A_host(num_expert_problems);
  std::vector<StrideB> stride_B_host(num_expert_problems);
  std::vector<StrideC> stride_C_host(num_expert_problems);
  std::vector<StrideD> stride_D_host(num_expert_problems);
  std::vector<StrideS> stride_S_host(num_expert_problems);
  std::vector<LayoutB_Reordered> layout_B_reordered_host(num_expert_problems);

  const int64_t group_a_elems = static_cast<int64_t>(moe_block_size) * prob_k;
  const int64_t group_c_elems = static_cast<int64_t>(moe_block_size) * prob_n;
  const int64_t b_elems_per_expert = static_cast<int64_t>(prob_k) * prob_n / 2;

  cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;
  cutlass::DeviceAllocation<const ElementA*> ptr_A;
  cutlass::DeviceAllocation<const typename GemmTypes::ElementB*> ptr_B;
  cutlass::DeviceAllocation<const ElementScale*> ptr_scale;
  cutlass::DeviceAllocation<const ElementC*> ptr_C;
  cutlass::DeviceAllocation<ElementC*> ptr_D;
  cutlass::DeviceAllocation<StrideA> stride_A;
  cutlass::DeviceAllocation<StrideB> stride_B;
  cutlass::DeviceAllocation<StrideC> stride_C;
  cutlass::DeviceAllocation<StrideD> stride_D;
  cutlass::DeviceAllocation<StrideS> stride_S;
  cutlass::DeviceAllocation<LayoutB_Reordered> layout_B_reordered;

  for (int p = 0; p < num_expert_problems; ++p) {
    const int expert = expert_problems[p].expert_id;
    const int m = expert_problems[p].num_groups * moe_block_size;
    problem_sizes_host[p] = make_tuple(prob_n, m, prob_k);

    ptr_A_host[p] = const_cast<ElementA*>(A_grouped) + expert_problems[p].start_group * group_a_elems;
    ptr_C_host[p] = C_grouped + expert_problems[p].start_group * group_c_elems;
    ptr_D_host[p] = C_grouped + expert_problems[p].start_group * group_c_elems;

    stride_A_host[p] = cutlass::make_cute_packed_stride(StrideA{}, {m, prob_k, 1});
    stride_B_host[p] = cutlass::make_cute_packed_stride(StrideB{}, {prob_n, prob_k, 1});
    if constexpr (GemmTypes::kRowMajorD) {
      stride_C_host[p] = cutlass::make_cute_packed_stride(StrideC{}, {m, prob_n, 1});
      stride_D_host[p] = cutlass::make_cute_packed_stride(StrideD{}, {m, prob_n, 1});
    } else {
      stride_C_host[p] = cutlass::make_cute_packed_stride(StrideC{}, {prob_n, m, 1});
      stride_D_host[p] = cutlass::make_cute_packed_stride(StrideD{}, {prob_n, m, 1});
    }
    stride_S_host[p] = cutlass::make_cute_packed_stride(StrideS{}, {prob_n, scale_k, 1});

    const auto* expert_b = reinterpret_cast<const typename GemmTypes::ElementB*>(B_base) +
                           static_cast<int64_t>(expert) * b_elems_per_expert;
    const auto* expert_scale = scale_base + static_cast<int64_t>(expert) * (prob_n * scale_k);

    auto shape_B = cute::make_shape(prob_n, prob_k, Int<1>{});
    layout_B_reordered_host[p] = tile_to_shape(typename GemmTypes::LayoutAtomQuant{}, shape_B);

    ptr_B_host[p] = expert_b;
    ptr_scale_host[p] = expert_scale;
  }

  problem_sizes.reset(num_expert_problems);
  problem_sizes.copy_from_host(problem_sizes_host.data());
  ptr_A.reset(num_expert_problems);
  ptr_A.copy_from_host(ptr_A_host.data());
  ptr_B.reset(num_expert_problems);
  ptr_B.copy_from_host(ptr_B_host.data());
  ptr_scale.reset(num_expert_problems);
  ptr_scale.copy_from_host(ptr_scale_host.data());
  ptr_C.reset(num_expert_problems);
  ptr_C.copy_from_host(ptr_C_host.data());
  ptr_D.reset(num_expert_problems);
  ptr_D.copy_from_host(ptr_D_host.data());
  stride_A.reset(num_expert_problems);
  stride_A.copy_from_host(stride_A_host.data());
  stride_B.reset(num_expert_problems);
  stride_B.copy_from_host(stride_B_host.data());
  stride_C.reset(num_expert_problems);
  stride_C.copy_from_host(stride_C_host.data());
  stride_D.reset(num_expert_problems);
  stride_D.copy_from_host(stride_D_host.data());
  stride_S.reset(num_expert_problems);
  stride_S.copy_from_host(stride_S_host.data());
  layout_B_reordered.reset(num_expert_problems);
  layout_B_reordered.copy_from_host(layout_B_reordered_host.data());

  if (profile != nullptr) {
    const auto host_setup_end = std::chrono::steady_clock::now();
    profile->host_setup_ms =
        std::chrono::duration<double, std::milli>(host_setup_end - host_setup_begin)
            .count();
    profile->num_expert_problems = num_expert_problems;
  }

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename Gemm::Arguments arguments;
  decltype(arguments.epilogue.thread) fusion_args;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
  fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

  arguments = typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_expert_problems, problem_sizes.get(), nullptr},
      {ptr_B.get(), layout_B_reordered.get(), ptr_A.get(), stride_A.get(),
       ptr_scale.get(), stride_S.get(), kScaleChunk},
      {fusion_args, ptr_C.get(), stride_C.get(), ptr_D.get(), stride_D.get()},
      hw_info};

  const auto gemm_init_begin = std::chrono::steady_clock::now();
  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get(), stream));
  if (profile != nullptr) {
    const auto gemm_init_end = std::chrono::steady_clock::now();
    profile->gemm_init_ms =
        std::chrono::duration<double, std::milli>(gemm_init_end - gemm_init_begin)
            .count();
  }

  cudaEvent_t gemm_run_start{};
  cudaEvent_t gemm_run_stop{};
  const bool time_gemm_run = profile != nullptr;
  if (time_gemm_run) {
    CUDA_CHECK(cudaEventCreate(&gemm_run_start));
    CUDA_CHECK(cudaEventCreate(&gemm_run_stop));
    CUDA_CHECK(cudaEventRecord(gemm_run_start, stream));
  }
  CUTLASS_CHECK(gemm.run(stream));
  if (time_gemm_run) {
    CUDA_CHECK(cudaEventRecord(gemm_run_stop, stream));
    profile->gemm_run_ms = cutlass69_elapsed_ms(gemm_run_start, gemm_run_stop);
    CUDA_CHECK(cudaEventDestroy(gemm_run_start));
    CUDA_CHECK(cudaEventDestroy(gemm_run_stop));
  }
  return true;
}

template <typename GemmTypes>
bool launch_grouped_gemm_expert_packed(
    int num_experts, const std::vector<int32_t>& expert_offsets,
    const std::vector<int32_t>& expert_counts, int prob_n, int prob_k,
    int group_size, const typename GemmTypes::ElementA* A_packed,
    const typename GemmTypes::ElementB* B_base,
    const typename GemmTypes::ElementScale* scale_base,
    typename GemmTypes::ElementC* C_packed, cudaStream_t stream,
    Cutlass69GemmProfile* profile = nullptr) {
  using Gemm = typename GemmTypes::Gemm;
  using ElementA = typename GemmTypes::ElementA;
  using ElementC = typename GemmTypes::ElementC;
  using ElementScale = typename GemmTypes::ElementScale;
  using LayoutB_Reordered = typename GemmTypes::LayoutB_Reordered;
  using StrideA = typename GemmTypes::StrideA;
  using StrideB = typename GemmTypes::StrideB;
  using StrideC = typename GemmTypes::StrideC;
  using StrideD = typename GemmTypes::StrideD;
  using StrideS = typename GemmTypes::StrideS;

  const int scale_k = cutlass::ceil_div(prob_k, group_size > 0 ? group_size : prob_k);
  const auto host_setup_begin = std::chrono::steady_clock::now();

  std::vector<int> active_experts;
  active_experts.reserve(num_experts);
  for (int e = 0; e < num_experts; ++e) {
    if (expert_counts[e] > 0) {
      active_experts.push_back(e);
    }
  }
  const int num_expert_problems = static_cast<int>(active_experts.size());
  if (num_expert_problems == 0) {
    return false;
  }

  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host(
      num_expert_problems);
  std::vector<const ElementA*> ptr_A_host(num_expert_problems);
  std::vector<const typename GemmTypes::ElementB*> ptr_B_host(num_expert_problems);
  std::vector<const ElementScale*> ptr_scale_host(num_expert_problems);
  std::vector<const ElementC*> ptr_C_host(num_expert_problems);
  std::vector<ElementC*> ptr_D_host(num_expert_problems);
  std::vector<StrideA> stride_A_host(num_expert_problems);
  std::vector<StrideB> stride_B_host(num_expert_problems);
  std::vector<StrideC> stride_C_host(num_expert_problems);
  std::vector<StrideD> stride_D_host(num_expert_problems);
  std::vector<StrideS> stride_S_host(num_expert_problems);
  std::vector<LayoutB_Reordered> layout_B_reordered_host(num_expert_problems);

  const int64_t b_elems_per_expert = static_cast<int64_t>(prob_k) * prob_n / 2;

  for (int p = 0; p < num_expert_problems; ++p) {
    const int expert = active_experts[p];
    const int m = expert_counts[expert];
    const int64_t row_offset = expert_offsets[expert];
    problem_sizes_host[p] = make_tuple(prob_n, m, prob_k);

    ptr_A_host[p] = A_packed + row_offset * prob_k;
    ptr_C_host[p] = C_packed + row_offset * prob_n;
    ptr_D_host[p] = C_packed + row_offset * prob_n;

    stride_A_host[p] = cutlass::make_cute_packed_stride(StrideA{}, {m, prob_k, 1});
    stride_B_host[p] = cutlass::make_cute_packed_stride(StrideB{}, {prob_n, prob_k, 1});
    stride_C_host[p] = cutlass::make_cute_packed_stride(StrideC{}, {m, prob_n, 1});
    stride_D_host[p] = cutlass::make_cute_packed_stride(StrideD{}, {m, prob_n, 1});
    stride_S_host[p] = cutlass::make_cute_packed_stride(StrideS{}, {prob_n, scale_k, 1});

    const auto* expert_b =
        reinterpret_cast<const typename GemmTypes::ElementB*>(B_base) +
        static_cast<int64_t>(expert) * b_elems_per_expert;
    const auto* expert_scale =
        scale_base + static_cast<int64_t>(expert) * (prob_n * scale_k);

    auto shape_B = cute::make_shape(prob_n, prob_k, Int<1>{});
    layout_B_reordered_host[p] =
        tile_to_shape(typename GemmTypes::LayoutAtomQuant{}, shape_B);

    ptr_B_host[p] = expert_b;
    ptr_scale_host[p] = expert_scale;
  }

  cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;
  cutlass::DeviceAllocation<const ElementA*> ptr_A;
  cutlass::DeviceAllocation<const typename GemmTypes::ElementB*> ptr_B;
  cutlass::DeviceAllocation<const ElementScale*> ptr_scale;
  cutlass::DeviceAllocation<const ElementC*> ptr_C;
  cutlass::DeviceAllocation<ElementC*> ptr_D;
  cutlass::DeviceAllocation<StrideA> stride_A;
  cutlass::DeviceAllocation<StrideB> stride_B;
  cutlass::DeviceAllocation<StrideC> stride_C;
  cutlass::DeviceAllocation<StrideD> stride_D;
  cutlass::DeviceAllocation<StrideS> stride_S;
  cutlass::DeviceAllocation<LayoutB_Reordered> layout_B_reordered;

  problem_sizes.reset(num_expert_problems);
  problem_sizes.copy_from_host(problem_sizes_host.data());
  ptr_A.reset(num_expert_problems);
  ptr_A.copy_from_host(ptr_A_host.data());
  ptr_B.reset(num_expert_problems);
  ptr_B.copy_from_host(ptr_B_host.data());
  ptr_scale.reset(num_expert_problems);
  ptr_scale.copy_from_host(ptr_scale_host.data());
  ptr_C.reset(num_expert_problems);
  ptr_C.copy_from_host(ptr_C_host.data());
  ptr_D.reset(num_expert_problems);
  ptr_D.copy_from_host(ptr_D_host.data());
  stride_A.reset(num_expert_problems);
  stride_A.copy_from_host(stride_A_host.data());
  stride_B.reset(num_expert_problems);
  stride_B.copy_from_host(stride_B_host.data());
  stride_C.reset(num_expert_problems);
  stride_C.copy_from_host(stride_C_host.data());
  stride_D.reset(num_expert_problems);
  stride_D.copy_from_host(stride_D_host.data());
  stride_S.reset(num_expert_problems);
  stride_S.copy_from_host(stride_S_host.data());
  layout_B_reordered.reset(num_expert_problems);
  layout_B_reordered.copy_from_host(layout_B_reordered_host.data());

  if (profile != nullptr) {
    const auto host_setup_end = std::chrono::steady_clock::now();
    profile->host_setup_ms =
        std::chrono::duration<double, std::milli>(host_setup_end - host_setup_begin)
            .count();
    profile->num_expert_problems = num_expert_problems;
  }

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  typename Gemm::Arguments arguments;
  decltype(arguments.epilogue.thread) fusion_args;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
  fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

  arguments = typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_expert_problems, problem_sizes.get(), nullptr},
      {ptr_B.get(), layout_B_reordered.get(), ptr_A.get(), stride_A.get(),
       ptr_scale.get(), stride_S.get(), kScaleChunk},
      {fusion_args, ptr_C.get(), stride_C.get(), ptr_D.get(), stride_D.get()},
      hw_info};

  const auto gemm_init_begin = std::chrono::steady_clock::now();
  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get(), stream));
  if (profile != nullptr) {
    const auto gemm_init_end = std::chrono::steady_clock::now();
    profile->gemm_init_ms =
        std::chrono::duration<double, std::milli>(gemm_init_end - gemm_init_begin)
            .count();
  }

  cudaEvent_t gemm_run_start{};
  cudaEvent_t gemm_run_stop{};
  const bool time_gemm_run = profile != nullptr;
  if (time_gemm_run) {
    CUDA_CHECK(cudaEventCreate(&gemm_run_start));
    CUDA_CHECK(cudaEventCreate(&gemm_run_stop));
    CUDA_CHECK(cudaEventRecord(gemm_run_start, stream));
  }
  CUTLASS_CHECK(gemm.run(stream));
  if (time_gemm_run) {
    CUDA_CHECK(cudaEventRecord(gemm_run_stop, stream));
    profile->gemm_run_ms = cutlass69_elapsed_ms(gemm_run_start, gemm_run_stop);
    CUDA_CHECK(cudaEventDestroy(gemm_run_start));
    CUDA_CHECK(cudaEventDestroy(gemm_run_stop));
  }
  return true;
}

template <typename MmaType, int TileN, int ClusterN, bool RowMajorD>
void dispatch_fused_gemm1_bfull_typed(
    const void* A, const void* B, void* C, const void* b_scales,
    const int32_t* sorted_token_ids, const int32_t* expert_ids, int groups,
    int moe_block_size, int top_k, int prob_m, int prob_n, int prob_k,
    int group_size, int num_experts, cudaStream_t stream, double host_prep_ms) {
  using GemmTypes =
      Cutlass69GroupedGemmTypes<MmaType, MmaType, TileN, ClusterN, RowMajorD>;
  const size_t expert_b_bytes = static_cast<size_t>(prob_k) * prob_n / 2;
  const size_t expert_scale_bytes =
      static_cast<size_t>(prob_n) *
      cutlass::ceil_div(prob_k, kScaleChunk) * sizeof(MmaType);

  const bool profile = cutlass69_profile_enabled();
  Cutlass69FusedProfile profile_data{};
  static int profile_call_id = 0;
  if (profile) {
    profile_data.groups = groups;
    profile_data.padded_m = groups * moe_block_size;
    profile_data.host_prep_ms = host_prep_ms;
    profile_data.bfull = true;
  }

  cutlass::DeviceAllocation<int32_t> expert_counts;
  expert_counts.reset(num_experts);
  CUDA_CHECK(cudaMemsetAsync(expert_counts.get(), 0,
                             static_cast<size_t>(num_experts) * sizeof(int32_t),
                             stream));

  const int count_threads = 256;
  const int64_t total_slots = static_cast<int64_t>(groups) * moe_block_size;
  const int count_blocks =
      static_cast<int>((total_slots + count_threads - 1) / count_threads);
  count_expert_tokens_kernel<MmaType><<<count_blocks, count_threads, 0,
                                        stream>>>(
      expert_counts.get(), sorted_token_ids, expert_ids, groups, moe_block_size,
      prob_m, top_k, num_experts);

  std::vector<int32_t> counts_host(num_experts);
  CUDA_CHECK(cudaMemcpyAsync(counts_host.data(), expert_counts.get(),
                             num_experts * sizeof(int32_t),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<int32_t> offsets_host(num_experts + 1);
  int total_packed = 0;
  for (int e = 0; e < num_experts; ++e) {
    offsets_host[e] = total_packed;
    total_packed += counts_host[e];
  }
  offsets_host[num_experts] = total_packed;
  if (profile) {
    profile_data.packed_m = total_packed;
  }

  const auto buffer_alloc_begin = std::chrono::steady_clock::now();
  cutlass::DeviceAllocation<MmaType> A_packed;
  cutlass::DeviceAllocation<MmaType> C_packed;
  cutlass::DeviceAllocation<int32_t> packed_to_sorted;
  cutlass::DeviceAllocation<int32_t> expert_write_cursor;
  if (total_packed > 0) {
    A_packed.reset(static_cast<int64_t>(total_packed) * prob_k);
    C_packed.reset(static_cast<int64_t>(total_packed) * prob_n);
    packed_to_sorted.reset(total_packed);
    expert_write_cursor.reset(num_experts);
    CUDA_CHECK(cudaMemcpyAsync(expert_write_cursor.get(), offsets_host.data(),
                               num_experts * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream));
  }
  if (profile) {
    const auto buffer_alloc_end = std::chrono::steady_clock::now();
    profile_data.buffer_alloc_ms =
        std::chrono::duration<double, std::milli>(buffer_alloc_end -
                                                 buffer_alloc_begin)
            .count();
  }

  if (total_packed > 0) {
    cudaEvent_t pack_start{};
    cudaEvent_t pack_stop{};
    if (profile) {
      CUDA_CHECK(cudaEventCreate(&pack_start));
      CUDA_CHECK(cudaEventCreate(&pack_stop));
      CUDA_CHECK(cudaEventRecord(pack_start, stream));
    }
    const int pack_threads = 256;
    const int pack_blocks = static_cast<int>(total_slots);
    pack_a_expert_rows_kernel<MmaType><<<pack_blocks, pack_threads, 0,
                                         stream>>>(
        reinterpret_cast<const MmaType*>(A), A_packed.get(),
        expert_write_cursor.get(), packed_to_sorted.get(), sorted_token_ids,
        expert_ids, groups, moe_block_size, prob_m, top_k, prob_k, num_experts);
    if (profile) {
      CUDA_CHECK(cudaEventRecord(pack_stop, stream));
      profile_data.gather_a_ms = cutlass69_elapsed_ms(pack_start, pack_stop);
      CUDA_CHECK(cudaEventDestroy(pack_start));
      CUDA_CHECK(cudaEventDestroy(pack_stop));
    }

    Cutlass69GemmProfile* gemm_profile = profile ? &profile_data.gemm : nullptr;
    launch_grouped_gemm_expert_packed<GemmTypes>(
        num_experts, offsets_host, counts_host, prob_n, prob_k, group_size,
        A_packed.get(), reinterpret_cast<const typename GemmTypes::ElementB*>(B),
        reinterpret_cast<const MmaType*>(b_scales), C_packed.get(), stream,
        gemm_profile);

    const int out_n = prob_n / 2;
    const int scatter_threads = 256;
    const int64_t scatter_total = static_cast<int64_t>(total_packed) * out_n;
    const int scatter_blocks = static_cast<int>(
        (scatter_total + scatter_threads - 1) / scatter_threads);

    cudaEvent_t scatter_start{};
    cudaEvent_t scatter_stop{};
    if (profile) {
      CUDA_CHECK(cudaEventCreate(&scatter_start));
      CUDA_CHECK(cudaEventCreate(&scatter_stop));
      CUDA_CHECK(cudaEventRecord(scatter_start, stream));
    }
    scatter_packed_fused_silu_rowmajor_kernel<MmaType>
        <<<scatter_blocks, scatter_threads, 0, stream>>>(
            C_packed.get(), reinterpret_cast<MmaType*>(C), packed_to_sorted.get(),
            total_packed, prob_n);
    if (profile) {
      CUDA_CHECK(cudaEventRecord(scatter_stop, stream));
      profile_data.fused_silu_scatter_ms =
          cutlass69_elapsed_ms(scatter_start, scatter_stop);
      CUDA_CHECK(cudaEventDestroy(scatter_start));
      CUDA_CHECK(cudaEventDestroy(scatter_stop));
    }
  }

  if (profile) {
    cutlass69_print_fused_profile(++profile_call_id, profile_data, prob_m, prob_n,
                                  prob_k);
  }
  CUDA_CHECK(cudaGetLastError());
}

template <typename MmaType, int TileN, int ClusterN, bool RowMajorD>
void dispatch_fused_gemm1_typed(
    const void* A, const void* B, void* C, const void* b_scales,
    const int32_t* sorted_token_ids, const int32_t* expert_ids, int groups,
    int moe_block_size, int top_k, int prob_m, int prob_n, int prob_k,
    int group_size, cudaStream_t stream, double host_prep_ms) {
  using GemmTypes =
      Cutlass69GroupedGemmTypes<MmaType, MmaType, TileN, ClusterN, RowMajorD>;
  const size_t expert_b_bytes = static_cast<size_t>(prob_k) * prob_n / 2;
  const size_t expert_scale_bytes =
      static_cast<size_t>(prob_n) *
      cutlass::ceil_div(prob_k, kScaleChunk) * sizeof(MmaType);

  const bool profile = cutlass69_profile_enabled();
  Cutlass69FusedProfile profile_data{};
  static int profile_call_id = 0;
  if (profile) {
    profile_data.groups = groups;
    profile_data.padded_m = groups * moe_block_size;
    profile_data.host_prep_ms = host_prep_ms;
  }

  const auto buffer_alloc_begin = std::chrono::steady_clock::now();
  cutlass::DeviceAllocation<MmaType> A_grouped;
  A_grouped.reset(static_cast<int64_t>(groups) * moe_block_size * prob_k);
  cutlass::DeviceAllocation<MmaType> C_grouped;
  C_grouped.reset(static_cast<int64_t>(groups) * moe_block_size * prob_n);
  if (profile) {
    const auto buffer_alloc_end = std::chrono::steady_clock::now();
    profile_data.buffer_alloc_ms =
        std::chrono::duration<double, std::milli>(buffer_alloc_end -
                                                 buffer_alloc_begin)
            .count();
  }

  const int threads = 256;
  const int blocks = static_cast<int>(
      (static_cast<int64_t>(groups) * moe_block_size * prob_k + threads - 1) /
      threads);

  cudaEvent_t gather_start{};
  cudaEvent_t gather_stop{};
  if (profile) {
    CUDA_CHECK(cudaEventCreate(&gather_start));
    CUDA_CHECK(cudaEventCreate(&gather_stop));
    CUDA_CHECK(cudaEventRecord(gather_start, stream));
  }
  gather_moe_a_kernel<MmaType><<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const MmaType*>(A), A_grouped.get(), sorted_token_ids,
      groups, moe_block_size, prob_m, top_k, prob_k);
  if (profile) {
    CUDA_CHECK(cudaEventRecord(gather_stop, stream));
    profile_data.gather_a_ms = cutlass69_elapsed_ms(gather_start, gather_stop);
    CUDA_CHECK(cudaEventDestroy(gather_start));
    CUDA_CHECK(cudaEventDestroy(gather_stop));
  }

  Cutlass69GemmProfile* gemm_profile = profile ? &profile_data.gemm : nullptr;
  launch_grouped_gemm<GemmTypes>(
      groups, moe_block_size, prob_n, prob_k, group_size, A_grouped.get(),
      reinterpret_cast<const typename GemmTypes::ElementB*>(B),
      reinterpret_cast<const MmaType*>(b_scales), C_grouped.get(), expert_ids,
      expert_b_bytes, expert_scale_bytes, stream, gemm_profile);

  const int out_n = prob_n / 2;
  const int scatter_threads = 256;
  const int64_t scatter_total =
      static_cast<int64_t>(groups) * moe_block_size * out_n;
  const int scatter_blocks =
      static_cast<int>((scatter_total + scatter_threads - 1) / scatter_threads);

  cudaEvent_t scatter_start{};
  cudaEvent_t scatter_stop{};
  if (profile) {
    CUDA_CHECK(cudaEventCreate(&scatter_start));
    CUDA_CHECK(cudaEventCreate(&scatter_stop));
    CUDA_CHECK(cudaEventRecord(scatter_start, stream));
  }
  if constexpr (RowMajorD) {
    scatter_moe_c_fused_silu_rowmajor_kernel<MmaType>
        <<<scatter_blocks, scatter_threads, 0, stream>>>(
            C_grouped.get(), reinterpret_cast<MmaType*>(C), sorted_token_ids,
            groups, moe_block_size, prob_n, prob_m, top_k);
  } else {
    scatter_moe_c_fused_silu_kernel<MmaType><<<scatter_blocks, scatter_threads, 0,
                                               stream>>>(
        C_grouped.get(), reinterpret_cast<MmaType*>(C), sorted_token_ids,
        groups, moe_block_size, prob_n, prob_m, top_k);
  }
  if (profile) {
    CUDA_CHECK(cudaEventRecord(scatter_stop, stream));
    profile_data.fused_silu_scatter_ms =
        cutlass69_elapsed_ms(scatter_start, scatter_stop);
    CUDA_CHECK(cudaEventDestroy(scatter_start));
    CUDA_CHECK(cudaEventDestroy(scatter_stop));
    cutlass69_print_fused_profile(++profile_call_id, profile_data, prob_m,
                                  prob_n, prob_k);
  }
  CUDA_CHECK(cudaGetLastError());
}

#endif  // CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED

}  // namespace

void dispatch_marlin_moe_cutlass69(
    const void* A, const void* B, void* C, const void* b_scales,
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    const int32_t* num_tokens_past_padded, const float* topk_weights,
    int moe_block_size, int num_experts, int top_k, bool mul_topk_weights,
    int prob_m, int prob_n, int prob_k, vllm::ScalarType const& a_type,
    vllm::ScalarType const& b_type, vllm::ScalarType const& c_type,
    int group_size, int dev, cudaStream_t stream) {
  (void)topk_weights;
  (void)mul_topk_weights;
  (void)dev;

#if !defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
  TORCH_CHECK(false,
              "CUTLASS example-69 kernels require sm_90a compilation.");
#else
  int32_t num_tokens_past_padded_host = 0;
  CUDA_CHECK(cudaMemcpyAsync(&num_tokens_past_padded_host,
                             num_tokens_past_padded, sizeof(int32_t),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const int groups = num_tokens_past_padded_host / moe_block_size;
  TORCH_CHECK(groups > 0, "CUTLASS69 requires at least one MoE block.");

  const int scale_k = cutlass::ceil_div(prob_k, kScaleChunk);
  const size_t expert_b_bytes = static_cast<size_t>(prob_k) * prob_n / 2;

  if (c_type == vllm::kFloat16) {
    using MmaType = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using GemmTypes =
        Cutlass69GroupedGemmTypes<MmaType, ElementC, 16, 1, false>;
    const size_t expert_scale_bytes =
        static_cast<size_t>(prob_n) * scale_k * sizeof(MmaType);

    cutlass::DeviceAllocation<MmaType> A_grouped;
    A_grouped.reset(static_cast<int64_t>(groups) * moe_block_size * prob_k);

    const int threads = 256;
    const int blocks =
        static_cast<int>((static_cast<int64_t>(groups) * moe_block_size *
                              prob_k +
                          threads - 1) /
                         threads);
    gather_moe_a_kernel<MmaType><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const MmaType*>(A), A_grouped.get(), sorted_token_ids,
        groups, moe_block_size, prob_m, top_k, prob_k);

    cutlass::DeviceAllocation<MmaType> C_grouped;
    C_grouped.reset(static_cast<int64_t>(groups) * moe_block_size * prob_n);

    launch_grouped_gemm<GemmTypes>(
        groups, moe_block_size, prob_n, prob_k, group_size, A_grouped.get(),
        reinterpret_cast<const typename GemmTypes::ElementB*>(B),
        reinterpret_cast<const MmaType*>(b_scales), C_grouped.get(),
        expert_ids, expert_b_bytes, expert_scale_bytes, stream);

    const int scatter_threads = 256;
    const int64_t scatter_total =
        static_cast<int64_t>(groups) * moe_block_size * prob_n;
    const int scatter_blocks = static_cast<int>(
        (scatter_total + scatter_threads - 1) / scatter_threads);
    scatter_moe_c_kernel<MmaType><<<scatter_blocks, scatter_threads, 0,
                                     stream>>>(
        C_grouped.get(), reinterpret_cast<MmaType*>(C), sorted_token_ids,
        groups, moe_block_size, prob_n, prob_m, top_k);
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  if (c_type == vllm::kBFloat16) {
    using MmaType = cutlass::bfloat16_t;
    using ElementC = cutlass::bfloat16_t;
    using GemmTypes =
        Cutlass69GroupedGemmTypes<MmaType, ElementC, 16, 1, false>;
    const size_t expert_scale_bytes =
        static_cast<size_t>(prob_n) * scale_k * sizeof(MmaType);

    cutlass::DeviceAllocation<MmaType> A_grouped;
    A_grouped.reset(static_cast<int64_t>(groups) * moe_block_size * prob_k);

    const int threads = 256;
    const int blocks =
        static_cast<int>((static_cast<int64_t>(groups) * moe_block_size *
                              prob_k +
                          threads - 1) /
                         threads);
    gather_moe_a_kernel<MmaType><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const MmaType*>(A), A_grouped.get(), sorted_token_ids,
        groups, moe_block_size, prob_m, top_k, prob_k);

    cutlass::DeviceAllocation<MmaType> C_grouped;
    C_grouped.reset(static_cast<int64_t>(groups) * moe_block_size * prob_n);

    launch_grouped_gemm<GemmTypes>(
        groups, moe_block_size, prob_n, prob_k, group_size, A_grouped.get(),
        reinterpret_cast<const typename GemmTypes::ElementB*>(B),
        reinterpret_cast<const MmaType*>(b_scales), C_grouped.get(),
        expert_ids, expert_b_bytes, expert_scale_bytes, stream);

    const int scatter_threads = 256;
    const int64_t scatter_total =
        static_cast<int64_t>(groups) * moe_block_size * prob_n;
    const int scatter_blocks = static_cast<int>(
        (scatter_total + scatter_threads - 1) / scatter_threads);
    scatter_moe_c_kernel<MmaType><<<scatter_blocks, scatter_threads, 0,
                                     stream>>>(
        C_grouped.get(), reinterpret_cast<MmaType*>(C), sorted_token_ids,
        groups, moe_block_size, prob_n, prob_m, top_k);
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  TORCH_CHECK(false, "Unsupported activation dtype for CUTLASS69 path.");
#endif
}

void dispatch_marlin_moe_cutlass69_fused_gemm1(
    const void* A, const void* B, void* C, const void* b_scales,
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    const int32_t* num_tokens_past_padded, int moe_block_size, int num_experts,
    int top_k, int prob_m, int prob_n, int prob_k,
    vllm::ScalarType const& a_type, vllm::ScalarType const& b_type,
    vllm::ScalarType const& c_type, int group_size, int dev,
    cudaStream_t stream) {
  (void)num_experts;
  (void)a_type;
  (void)b_type;
  (void)dev;

#if !defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
  TORCH_CHECK(false,
              "CUTLASS example-69 kernels require sm_90a compilation.");
#else
  const auto host_prep_begin = std::chrono::steady_clock::now();
  int32_t num_tokens_past_padded_host = 0;
  CUDA_CHECK(cudaMemcpyAsync(&num_tokens_past_padded_host,
                             num_tokens_past_padded, sizeof(int32_t),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const int groups = num_tokens_past_padded_host / moe_block_size;
  TORCH_CHECK(groups > 0, "CUTLASS69 requires at least one MoE block.");
  const auto host_prep_end = std::chrono::steady_clock::now();
  const double host_prep_ms =
      std::chrono::duration<double, std::milli>(host_prep_end - host_prep_begin)
          .count();

  const bool use_bfull = cutlass69_bfull_enabled(prob_m);
  const bool use_large_tile = prob_m >= 1024;

  if (c_type == vllm::kFloat16) {
    if (use_bfull) {
      dispatch_fused_gemm1_bfull_typed<cutlass::half_t, 256, 2, true>(
          A, B, C, b_scales, sorted_token_ids, expert_ids, groups,
          moe_block_size, top_k, prob_m, prob_n, prob_k, group_size,
          num_experts, stream, host_prep_ms);
    } else if (use_large_tile) {
      dispatch_fused_gemm1_typed<cutlass::half_t, 256, 2, true>(
          A, B, C, b_scales, sorted_token_ids, expert_ids, groups,
          moe_block_size, top_k, prob_m, prob_n, prob_k, group_size, stream,
          host_prep_ms);
    } else {
      dispatch_fused_gemm1_typed<cutlass::half_t, 16, 1, false>(
          A, B, C, b_scales, sorted_token_ids, expert_ids, groups,
          moe_block_size, top_k, prob_m, prob_n, prob_k, group_size, stream,
          host_prep_ms);
    }
    return;
  }

  if (c_type == vllm::kBFloat16) {
    if (use_bfull) {
      dispatch_fused_gemm1_bfull_typed<cutlass::bfloat16_t, 256, 2, true>(
          A, B, C, b_scales, sorted_token_ids, expert_ids, groups,
          moe_block_size, top_k, prob_m, prob_n, prob_k, group_size,
          num_experts, stream, host_prep_ms);
    } else if (use_large_tile) {
      dispatch_fused_gemm1_typed<cutlass::bfloat16_t, 256, 2, true>(
          A, B, C, b_scales, sorted_token_ids, expert_ids, groups,
          moe_block_size, top_k, prob_m, prob_n, prob_k, group_size, stream,
          host_prep_ms);
    } else {
      dispatch_fused_gemm1_typed<cutlass::bfloat16_t, 16, 1, false>(
          A, B, C, b_scales, sorted_token_ids, expert_ids, groups,
          moe_block_size, top_k, prob_m, prob_n, prob_k, group_size, stream,
          host_prep_ms);
    }
    return;
  }

  TORCH_CHECK(false, "Unsupported activation dtype for CUTLASS69 path.");
#endif
}

#endif  // MARLIN_MOE_HAS_CUTLASS

}  // namespace marlin_moe_cutlass69_host
