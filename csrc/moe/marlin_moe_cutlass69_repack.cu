#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

#ifdef MARLIN_MOE_HAS_CUTLASS
#include "cutlass/cutlass.h"
#include "cutlass/detail/layout.hpp"
#include "cutlass/util/mixed_dtype_utils.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"

using namespace cute;

using MmaType = cutlass::half_t;
using ValueShuffle = Layout<Shape<_2, _4>, Stride<_4, _1>>;
using MmaAtomShape = Layout<Shape<_1, Int<1>>>;
using LayoutAtomQuant = decltype(
    cutlass::compute_memory_reordering_atom<MmaType, MmaAtomShape,
                                          ValueShuffle>());

using LayoutBTag = cutlass::layout::ColumnMajor;
using StrideB =
    cute::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutBTag*>>;
using LayoutScaleTag = cutlass::layout::RowMajor;
using StrideS =
    cute::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutScaleTag*>>;

// Pack uint4b8 [N,K] int8 into ColumnMajor int4b physical storage (pair along N).
__global__ void pack_marlin_uint4_to_cutlass_int4(
    const int8_t* __restrict__ src, uint8_t* __restrict__ dst, int N, int K) {
  const int n_pair = blockIdx.x * blockDim.x + threadIdx.x;
  const int k = blockIdx.y * blockDim.y + threadIdx.y;
  const int n0 = n_pair * 2;
  const int n1 = n0 + 1;

  if (n1 < N && k < K) {
    const int8_t v0 = static_cast<int8_t>((src[n0 * K + k] & 0xF) - 8);
    const int8_t v1 = static_cast<int8_t>((src[n1 * K + k] & 0xF) - 8);
    const uint8_t packed =
        static_cast<uint8_t>(v0 & 0xF) |
        (static_cast<uint8_t>(v1 & 0xF) << 4);
    const int linear = n0 + k * N;
    dst[linear / 2] = packed;
  }
}

namespace {

void pack_expert_int8_to_columnmajor(const int8_t* src, uint8_t* dst, int N,
                                     int K, cudaStream_t stream) {
  const dim3 threads(16, 16);
  const dim3 blocks((N / 2 + threads.x - 1) / threads.x,
                    (K + threads.y - 1) / threads.y);
  pack_marlin_uint4_to_cutlass_int4<<<blocks, threads, 0, stream>>>(src, dst,
                                                                     N, K);
}

auto make_layout_b(int N, int K) {
  const auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  return make_layout(make_shape(N, K, Int<1>{}), stride_B);
}

auto make_layout_b_reordered(int N, int K) {
  return tile_to_shape(LayoutAtomQuant{}, make_shape(N, K, Int<1>{}));
}

auto make_layout_scale(int N, int scale_k) {
  const auto stride_S =
      cutlass::make_cute_packed_stride(StrideS{}, {N, scale_k, 1});
  return make_layout(make_shape(N, scale_k, Int<1>{}), stride_S);
}

}  // namespace

torch::Tensor cutlass69_pack_only(torch::Tensor q_weight_int8) {
  TORCH_CHECK(q_weight_int8.is_contiguous(), "q_weight_int8 must be contiguous");
  TORCH_CHECK(q_weight_int8.dtype() == torch::kInt8,
              "q_weight_int8 must be int8");
  const int num_experts = static_cast<int>(q_weight_int8.size(0));
  const int N = static_cast<int>(q_weight_int8.size(1));
  const int K = static_cast<int>(q_weight_int8.size(2));
  TORCH_CHECK(N % 2 == 0, "N must be even for int4 packing");
  TORCH_CHECK(K % 2 == 0, "K must be even for int4 packing");

  auto options =
      torch::TensorOptions().dtype(torch::kUInt8).device(q_weight_int8.device());
  torch::Tensor packed = torch::empty({num_experts, N, K / 2}, options);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  for (int e = 0; e < num_experts; ++e) {
    pack_expert_int8_to_columnmajor(q_weight_int8[e].data_ptr<int8_t>(),
                                    packed[e].data_ptr<uint8_t>(), N, K,
                                    stream);
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
              "pack_marlin_uint4_to_cutlass_int4 failed");
  return packed;
}

torch::Tensor cutlass69_pack_and_reorder(torch::Tensor q_weight_int8) {
  TORCH_CHECK(q_weight_int8.is_contiguous(), "q_weight_int8 must be contiguous");
  TORCH_CHECK(q_weight_int8.dtype() == torch::kInt8,
              "q_weight_int8 must be int8");
  const int num_experts = static_cast<int>(q_weight_int8.size(0));
  const int N = static_cast<int>(q_weight_int8.size(1));
  const int K = static_cast<int>(q_weight_int8.size(2));
  TORCH_CHECK(N % 2 == 0, "N must be even for int4 packing");
  TORCH_CHECK(K % 2 == 0, "K must be even for int4 packing");

  torch::Tensor packed = cutlass69_pack_only(q_weight_int8);
  torch::Tensor reordered = torch::empty_like(packed);

  const auto layout_B = make_layout_b(N, K);
  const auto layout_B_reordered = make_layout_b_reordered(N, K);

  for (int e = 0; e < num_experts; ++e) {
    cutlass::reorder_tensor(
        reinterpret_cast<cutlass::int4b_t*>(packed[e].data_ptr<uint8_t>()),
        layout_B,
        reinterpret_cast<cutlass::int4b_t*>(reordered[e].data_ptr<uint8_t>()),
        layout_B_reordered);
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "reorder_tensor failed");

  return reordered;
}

// Dequantize packed uint4 weights with the given CuTe layout.
// Returns [num_experts, K, N] fp16 in logical GEMM weight orientation.
static torch::Tensor dequant_packed_with_layout(
    torch::Tensor packed, torch::Tensor scales, int64_t group_size,
    auto layout_operand) {
  TORCH_CHECK(packed.is_contiguous());
  TORCH_CHECK(scales.is_contiguous());
  TORCH_CHECK(packed.dtype() == torch::kUInt8);
  const int num_experts = static_cast<int>(packed.size(0));
  const int N = static_cast<int>(packed.size(1));
  const int K = static_cast<int>(packed.size(2) * 2);
  const int gs = static_cast<int>(group_size);
  const int scale_k = cutlass::ceil_div(K, gs);

  TORCH_CHECK(scales.size(0) == num_experts);
  TORCH_CHECK(scales.size(1) == N);
  TORCH_CHECK(scales.size(2) == scale_k);

  auto dq_options = torch::TensorOptions()
                        .dtype(scales.dtype())
                        .device(packed.device());
  torch::Tensor dq_nk = torch::empty({num_experts, N, K}, dq_options);

  const auto layout_scale = make_layout_scale(N, scale_k);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  for (int e = 0; e < num_experts; ++e) {
    cutlass::dequantize(
        reinterpret_cast<MmaType*>(dq_nk[e].data_ptr()),
        reinterpret_cast<const cutlass::int4b_t*>(
            packed[e].data_ptr<uint8_t>()),
        layout_operand,
        reinterpret_cast<const MmaType*>(scales[e].data_ptr()),
        static_cast<const MmaType*>(nullptr),
        layout_scale,
        gs,
        stream);
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "dequantize failed");
  return dq_nk.transpose(1, 2).contiguous();
}

torch::Tensor cutlass69_dequant_packed(torch::Tensor packed,
                                       torch::Tensor scales,
                                       int64_t group_size) {
  const int N = static_cast<int>(packed.size(1));
  const int K = static_cast<int>(packed.size(2) * 2);
  return dequant_packed_with_layout(packed, scales, group_size,
                                    make_layout_b(N, K));
}

torch::Tensor cutlass69_dequant_reordered(torch::Tensor reordered,
                                        torch::Tensor scales,
                                        int64_t group_size) {
  const int N = static_cast<int>(reordered.size(1));
  const int K = static_cast<int>(reordered.size(2) * 2);
  return dequant_packed_with_layout(reordered, scales, group_size,
                                    make_layout_b_reordered(N, K));
}
#else
torch::Tensor cutlass69_pack_only(torch::Tensor q_weight_int8) {
  TORCH_CHECK(false, "CUTLASS not enabled");
  return q_weight_int8;
}

torch::Tensor cutlass69_pack_and_reorder(torch::Tensor q_weight_int8) {
  TORCH_CHECK(false, "CUTLASS not enabled");
  return q_weight_int8;
}

torch::Tensor cutlass69_dequant_packed(torch::Tensor packed, torch::Tensor scales,
                                       int64_t group_size) {
  TORCH_CHECK(false, "CUTLASS not enabled");
  return packed;
}

torch::Tensor cutlass69_dequant_reordered(torch::Tensor reordered,
                                          torch::Tensor scales,
                                          int64_t group_size) {
  TORCH_CHECK(false, "CUTLASS not enabled");
  return reordered;
}
#endif
