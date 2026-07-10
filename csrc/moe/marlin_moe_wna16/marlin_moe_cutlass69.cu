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
  if (prob_n != 256 || prob_k != 6144 || prob_m < 32 || prob_m > 8192) {
    return {false,
            "CUTLASS example-69 path is scoped to N=256, K=6144, "
            "and 32 <= M <= 8192."};
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

#include "gated_stride.hpp"
#include "gated_builder.hpp"
#include "sm90_visitor_gated_act.hpp"

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

namespace marlin_moe_cutlass69_host {

using namespace cute;

namespace {

using ProblemShape =
    cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using QuantType = cutlass::int4b_t;
constexpr int kScaleChunk = 128;

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

template <typename MmaType, typename ElementC_>
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
  using TileShape = Shape<_128, _16, Int<TileShapeK>>;
  using ClusterShape = Shape<_1, _1, _1>;
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, TileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementAccumulator,
          ElementC, typename cutlass::layout::LayoutTranspose<LayoutC>::type*,
          AlignmentC,
          ElementD, typename cutlass::layout::LayoutTranspose<LayoutD>::type*,
          AlignmentD, EpilogueSchedule>::CollectiveOp;

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
  using StrideD = cute::remove_pointer_t<typename GemmKernel::CollectiveEpilogue::FusionCallbacks::Operation::GmemLayoutTagAux>;
  using StrideS = typename CollectiveMainloop::StrideScale;
};

template <typename MmaType, typename ElementC_>
struct Cutlass69FusedGroupedGemmTypes {
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

  using StrideA_Transpose =
      cute::remove_pointer_t<cutlass::gemm::TagToStrideB_t<LayoutA_Transpose*>>;
  using GatedStrideA_Transpose = StrideA_Transpose;

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
  using TileShape = Shape<_128, _16, Int<TileShapeK>>;
  using ClusterShape = Shape<_1, _1, _1>;
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;

  using CollectiveEpilogueBuilder =
      cutlass::epilogue::collective::Sm90CollectiveBuilderGated<
          OperatorClass, TileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, float, float, MmaType,
          ElementC, typename cutlass::layout::LayoutTranspose<LayoutC>::type*, AlignmentC,
          ElementD, typename cutlass::layout::LayoutTranspose<LayoutD>::type*, AlignmentD,
          EpilogueSchedule, cutlass::epilogue::thread::SiLu, false, 1>;

  using CollectiveEpilogue = typename CollectiveEpilogueBuilder::CollectiveOp;

  using GatedTileShape = decltype(cutlass::sm90_make_gated_shape<1>(TileShape{}));

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass,
          cute::tuple<ElementB, ElementScale>, LayoutB_Reordered*, AlignmentB,
          ElementA, StrideA_Transpose*, AlignmentA, ElementAccumulator,
          GatedTileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GatedProblemShape = decltype(cutlass::sm90_make_gated_shape<1>(Shape<int, int, int>{}));
  using GatedGroupProblemShape = cutlass::gemm::GroupProblemShape<GatedProblemShape>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      GatedGroupProblemShape, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::GroupScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideC = typename CollectiveEpilogueBuilder::StrideC;
  using StrideD = typename CollectiveEpilogueBuilder::StrideD;
  using StrideDAux = typename CollectiveEpilogueBuilder::StrideDAux;
  using StrideS = typename CollectiveMainloop::StrideScale;
};

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

template <typename GemmTypes>
bool launch_grouped_gemm(
    int groups, int moe_block_size, int prob_n, int prob_k, int group_size,
    const typename GemmTypes::ElementA* A_grouped,
    const typename GemmTypes::ElementB* B_base,
    const typename GemmTypes::ElementScale* scale_base,
    typename GemmTypes::ElementC* C_out, const int32_t* expert_ids,
    size_t expert_b_bytes, size_t expert_scale_bytes, cudaStream_t stream) {
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

  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host(
      groups);
  std::vector<ElementA*> ptr_A_host(groups);
  std::vector<const typename GemmTypes::ElementB*> ptr_B_host(groups);
  std::vector<const ElementScale*> ptr_scale_host(groups);
  std::vector<const ElementC*> ptr_C_host(groups);
  std::vector<ElementC*> ptr_D_host(groups);
  std::vector<StrideA> stride_A_host(groups);
  std::vector<StrideB> stride_B_host(groups);
  std::vector<StrideC> stride_C_host(groups);
  std::vector<StrideD> stride_D_host(groups);
  std::vector<StrideS> stride_S_host(groups);
  std::vector<LayoutB_Reordered> layout_B_reordered_host(groups);
  std::vector<int32_t> expert_ids_host(groups);

  const int64_t group_a_elems =
      static_cast<int64_t>(moe_block_size) * prob_k;
  const int64_t group_c_elems =
      static_cast<int64_t>(moe_block_size) * prob_n;

  cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape>
      problem_sizes;
  cutlass::DeviceAllocation<ElementA*> ptr_A;
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

  cutlass::DeviceAllocation<typename GemmTypes::ElementB> b_staging;
  const int64_t b_elems_per_expert = static_cast<int64_t>(prob_k) * prob_n / 2;
  b_staging.reset(groups * b_elems_per_expert);
  CUDA_CHECK(cudaMemcpyAsync(expert_ids_host.data(), expert_ids,
                             groups * sizeof(int32_t),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  for (int g = 0; g < groups; ++g) {
    const int expert = expert_ids_host[g];
    problem_sizes_host[g] = make_tuple(prob_n, moe_block_size, prob_k);

    ptr_A_host[g] = const_cast<ElementA*>(A_grouped) + g * group_a_elems;
    ptr_C_host[g] = C_out + g * group_c_elems;
    ptr_D_host[g] = C_out + g * group_c_elems;

    stride_A_host[g] =
        cutlass::make_cute_packed_stride(StrideA{}, {moe_block_size, prob_k, 1});
    stride_B_host[g] =
        cutlass::make_cute_packed_stride(StrideB{}, {prob_n, prob_k, 1});
    stride_C_host[g] =
        cutlass::make_cute_packed_stride(StrideC{}, {prob_n, moe_block_size, 1});
    stride_D_host[g] =
        cutlass::make_cute_packed_stride(StrideD{}, {prob_n, moe_block_size, 1});
    stride_S_host[g] = cutlass::make_cute_packed_stride(
        StrideS{}, {prob_n, scale_k, 1});

    typename GemmTypes::ElementB* group_b =
        b_staging.get() + g * b_elems_per_expert;
    const auto* expert_b =
        reinterpret_cast<const typename GemmTypes::ElementB*>(B_base) +
        static_cast<int64_t>(expert) * b_elems_per_expert;
    const auto* expert_scale =
        scale_base + static_cast<int64_t>(expert) * (prob_n * scale_k);

    // Experimental: expects per-expert B in CUTLASS column-major int4 layout.
    // Marlin-packed weights still need a dedicated repack pass.
    CUDA_CHECK(cudaMemcpyAsync(group_b, expert_b, expert_b_bytes,
                               cudaMemcpyDeviceToDevice, stream));

    auto shape_B = cute::make_shape(prob_n, prob_k, Int<1>{});
    auto layout_B = make_layout(shape_B, stride_B_host[g]);
    layout_B_reordered_host[g] =
        tile_to_shape(typename GemmTypes::LayoutAtomQuant{}, shape_B);
    cutlass::reorder_tensor(group_b, layout_B, layout_B_reordered_host[g]);

    ptr_B_host[g] = group_b;
    ptr_scale_host[g] = expert_scale;
  }

  problem_sizes.reset(groups);
  problem_sizes.copy_from_host(problem_sizes_host.data());
  ptr_A.reset(groups);
  ptr_A.copy_from_host(ptr_A_host.data());
  ptr_B.reset(groups);
  ptr_B.copy_from_host(ptr_B_host.data());
  ptr_scale.reset(groups);
  ptr_scale.copy_from_host(ptr_scale_host.data());
  ptr_C.reset(groups);
  ptr_C.copy_from_host(ptr_C_host.data());
  ptr_D.reset(groups);
  ptr_D.copy_from_host(ptr_D_host.data());
  stride_A.reset(groups);
  stride_A.copy_from_host(stride_A_host.data());
  stride_B.reset(groups);
  stride_B.copy_from_host(stride_B_host.data());
  stride_C.reset(groups);
  stride_C.copy_from_host(stride_C_host.data());
  stride_D.reset(groups);
  stride_D.copy_from_host(stride_D_host.data());
  stride_S.reset(groups);
  stride_S.copy_from_host(stride_S_host.data());
  layout_B_reordered.reset(groups);
  layout_B_reordered.copy_from_host(layout_B_reordered_host.data());

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  typename Gemm::Arguments::EpilogueArguments::ThreadArguments fusion_args;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
  fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {groups, problem_sizes.get(), nullptr},
      {ptr_B.get(), layout_B_reordered.get(), ptr_A.get(), stride_A.get(),
       ptr_scale.get(), stride_S.get(), kScaleChunk},
      {fusion_args, ptr_C.get(), stride_C.get(), ptr_D.get(), stride_D.get()},
      hw_info};

  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get(), stream));
  CUTLASS_CHECK(gemm.run(stream));
  return true;
}

template <typename GemmTypes>
bool launch_fused_grouped_gemm(
    int groups, int moe_block_size, int prob_n, int prob_k, int group_size,
    const typename GemmTypes::ElementA* A_grouped,
    const typename GemmTypes::ElementB* B_base,
    const typename GemmTypes::ElementScale* scale_base,
    typename GemmTypes::ElementD* C_out, const int32_t* expert_ids,
    size_t expert_b_bytes, size_t expert_scale_bytes, cudaStream_t stream) {
  using Gemm = typename GemmTypes::Gemm;
  using ElementA = typename GemmTypes::ElementA;
  using ElementC = typename GemmTypes::ElementC;
  using ElementD = typename GemmTypes::ElementD;
  using ElementScale = typename GemmTypes::ElementScale;
  using LayoutB_Reordered = typename GemmTypes::LayoutB_Reordered;

  using StrideA_Internal = typename cute::remove_pointer_t<typename GemmTypes::StrideB>;
  using StrideB_Internal = typename cute::remove_pointer_t<typename GemmTypes::StrideA>;
  using StrideC_Internal = typename cute::remove_pointer_t<typename GemmTypes::StrideC>;
  using StrideD_Internal = typename cute::remove_pointer_t<typename GemmTypes::StrideDAux>;
  using StrideS_Internal = typename cute::remove_pointer_t<typename GemmTypes::StrideS>;

  const int scale_k = cutlass::ceil_div(prob_k, group_size > 0 ? group_size : prob_k);

  std::vector<typename GemmTypes::GatedProblemShape> problem_sizes_host(groups);
  std::vector<ElementA*> ptr_A_host(groups);
  std::vector<const typename GemmTypes::ElementB*> ptr_B_host(groups);
  std::vector<const ElementScale*> ptr_scale_host(groups);
  std::vector<const ElementC*> ptr_C_host(groups);
  std::vector<ElementD*> ptr_D_host(groups);
  
  std::vector<StrideA_Internal> stride_A_host(groups);
  std::vector<StrideB_Internal> stride_B_host(groups);
  std::vector<StrideC_Internal> stride_C_host(groups);
  std::vector<StrideD_Internal> stride_D_host(groups);
  std::vector<StrideS_Internal> stride_S_host(groups);
  std::vector<LayoutB_Reordered> layout_B_reordered_host(groups);
  std::vector<int32_t> expert_ids_host(groups);

  const int64_t group_a_elems = static_cast<int64_t>(moe_block_size) * prob_k;
  const int64_t group_d_elems = static_cast<int64_t>(moe_block_size) * (prob_n / 2);
  const int64_t b_elems_per_expert = static_cast<int64_t>(prob_k) * prob_n / 2;

  cutlass::DeviceAllocation<typename GemmTypes::GatedProblemShape> problem_sizes;
  cutlass::DeviceAllocation<ElementA*> ptr_A;
  cutlass::DeviceAllocation<const typename GemmTypes::ElementB*> ptr_B;
  cutlass::DeviceAllocation<const ElementScale*> ptr_scale;
  cutlass::DeviceAllocation<const ElementC*> ptr_C;
  cutlass::DeviceAllocation<ElementD*> ptr_D;
  
  cutlass::DeviceAllocation<StrideA_Internal> stride_A;
  cutlass::DeviceAllocation<StrideB_Internal> stride_B;
  cutlass::DeviceAllocation<StrideC_Internal> stride_C;
  cutlass::DeviceAllocation<StrideD_Internal> stride_D;
  cutlass::DeviceAllocation<StrideS_Internal> stride_S;
  cutlass::DeviceAllocation<LayoutB_Reordered> layout_B_reordered;

  cutlass::DeviceAllocation<typename GemmTypes::ElementB> b_staging;
  b_staging.reset(groups * b_elems_per_expert);
  
  CUDA_CHECK(cudaMemcpyAsync(expert_ids_host.data(), expert_ids,
                             groups * sizeof(int32_t),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  for (int g = 0; g < groups; ++g) {
    const int expert = expert_ids_host[g];
    const Shape<int, int, int> shape_unfused{prob_n, moe_block_size, prob_k};
    problem_sizes_host[g] = cutlass::sm90_make_gated_shape<1>(shape_unfused);

    ptr_A_host[g] = const_cast<ElementA*>(A_grouped) + g * group_a_elems;
    ptr_C_host[g] = nullptr;
    ptr_D_host[g] = reinterpret_cast<ElementD*>(reinterpret_cast<uint8_t*>(C_out) + g * group_d_elems * sizeof(ElementD));

    stride_A_host[g] = make_stride(static_cast<int64_t>(moe_block_size), make_stride(Int<1>{}, static_cast<int64_t>(0), Int<8>{}), Int<0>{});
    stride_B_host[g] = make_stride(static_cast<int64_t>(prob_k), Int<1>{}, Int<0>{});
    stride_C_host[g] = make_stride(static_cast<int64_t>(moe_block_size), make_stride(Int<1>{}, Int<8>{}), Int<0>{});
    stride_D_host[g] = make_stride(static_cast<int64_t>(moe_block_size), make_stride(Int<1>{}, Int<8>{}), Int<0>{});
    stride_S_host[g] = cutlass::make_cute_packed_stride(StrideS_Internal{}, {prob_n, scale_k, 1});

    typename GemmTypes::ElementB* group_b = b_staging.get() + g * b_elems_per_expert;
    const auto* expert_b = reinterpret_cast<const typename GemmTypes::ElementB*>(B_base) +
                           static_cast<int64_t>(expert) * b_elems_per_expert;
    const auto* expert_scale = scale_base + static_cast<int64_t>(expert) * (prob_n * scale_k);

    CUDA_CHECK(cudaMemcpyAsync(group_b, expert_b, expert_b_bytes,
                               cudaMemcpyDeviceToDevice, stream));

    auto shape_B = cute::make_shape(prob_n, prob_k, Int<1>{});
    auto stride_B_non_gated = cutlass::make_cute_packed_stride(cute::Stride<cute::Int<1>, int64_t, cute::Int<0>>{}, {prob_n, prob_k, 1});
    auto layout_B = make_layout(shape_B, stride_B_non_gated);
    layout_B_reordered_host[g] = tile_to_shape(typename GemmTypes::LayoutAtomQuant{}, shape_B);
    cutlass::reorder_tensor(group_b, layout_B, layout_B_reordered_host[g]);

    ptr_B_host[g] = group_b;
    ptr_scale_host[g] = expert_scale;
  }

  problem_sizes.reset(groups);
  problem_sizes.copy_from_host(problem_sizes_host.data());
  ptr_A.reset(groups);
  ptr_A.copy_from_host(ptr_A_host.data());
  ptr_B.reset(groups);
  ptr_B.copy_from_host(ptr_B_host.data());
  ptr_scale.reset(groups);
  ptr_scale.copy_from_host(ptr_scale_host.data());
  ptr_C.reset(groups);
  ptr_C.copy_from_host(ptr_C_host.data());
  ptr_D.reset(groups);
  ptr_D.copy_from_host(ptr_D_host.data());
  stride_A.reset(groups);
  stride_A.copy_from_host(stride_A_host.data());
  stride_B.reset(groups);
  stride_B.copy_from_host(stride_B_host.data());
  stride_C.reset(groups);
  stride_C.copy_from_host(stride_C_host.data());
  stride_D.reset(groups);
  stride_D.copy_from_host(stride_D_host.data());
  stride_S.reset(groups);
  stride_S.copy_from_host(stride_S_host.data());
  layout_B_reordered.reset(groups);
  layout_B_reordered.copy_from_host(layout_B_reordered_host.data());

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename GemmTypes::CollectiveEpilogue::FusionCallbacks::Arguments fusion_args;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;
  fusion_args.ptr_D = reinterpret_cast<typename GemmTypes::ElementD**>(ptr_D.get());
  fusion_args.dD = stride_D.get();
  fusion_args.sm_count = hw_info.sm_count;

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {groups, problem_sizes.get(), nullptr},
      {ptr_B.get(), layout_B_reordered.get(), ptr_A.get(), stride_A.get(),
       ptr_scale.get(), stride_S.get(), kScaleChunk},
      {fusion_args, ptr_C.get(), stride_C.get(), {}, {}},
      hw_info};

  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get(), stream));
  CUTLASS_CHECK(gemm.run(stream));
  return true;
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
    using GemmTypes = Cutlass69GroupedGemmTypes<MmaType, ElementC>;
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

    launch_grouped_gemm<GemmTypes>(
        groups, moe_block_size, prob_n, prob_k, group_size, A_grouped.get(),
        reinterpret_cast<const typename GemmTypes::ElementB*>(B),
        reinterpret_cast<const MmaType*>(b_scales),
        reinterpret_cast<ElementC*>(C), expert_ids, expert_b_bytes,
        expert_scale_bytes, stream);
    return;
  }

  if (c_type == vllm::kBFloat16) {
    using MmaType = cutlass::bfloat16_t;
    using ElementC = cutlass::bfloat16_t;
    using GemmTypes = Cutlass69GroupedGemmTypes<MmaType, ElementC>;
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

    launch_grouped_gemm<GemmTypes>(
        groups, moe_block_size, prob_n, prob_k, group_size, A_grouped.get(),
        reinterpret_cast<const typename GemmTypes::ElementB*>(B),
        reinterpret_cast<const MmaType*>(b_scales),
        reinterpret_cast<ElementC*>(C), expert_ids, expert_b_bytes,
        expert_scale_bytes, stream);
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
    using ElementD = cutlass::half_t;
    using GemmTypes = Cutlass69FusedGroupedGemmTypes<MmaType, ElementD>;
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

    launch_fused_grouped_gemm<GemmTypes>(
        groups, moe_block_size, prob_n, prob_k, group_size, A_grouped.get(),
        reinterpret_cast<const typename GemmTypes::ElementB*>(B),
        reinterpret_cast<const MmaType*>(b_scales),
        reinterpret_cast<typename GemmTypes::ElementD*>(C), expert_ids, expert_b_bytes,
        expert_scale_bytes, stream);
    return;
  }

  if (c_type == vllm::kBFloat16) {
    using MmaType = cutlass::bfloat16_t;
    using ElementD = cutlass::bfloat16_t;
    using GemmTypes = Cutlass69FusedGroupedGemmTypes<MmaType, ElementD>;
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

    launch_fused_grouped_gemm<GemmTypes>(
        groups, moe_block_size, prob_n, prob_k, group_size, A_grouped.get(),
        reinterpret_cast<const typename GemmTypes::ElementB*>(B),
        reinterpret_cast<const MmaType*>(b_scales),
        reinterpret_cast<typename GemmTypes::ElementD*>(C), expert_ids, expert_b_bytes,
        expert_scale_bytes, stream);
    return;
  }

  TORCH_CHECK(false, "Unsupported activation dtype for CUTLASS69 path.");
#endif
}

#endif  // MARLIN_MOE_HAS_CUTLASS

}  // namespace marlin_moe_cutlass69_host
