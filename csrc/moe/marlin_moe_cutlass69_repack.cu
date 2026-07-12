#include <torch/all.h>
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

__global__ void pack_marlin_uint4_to_cutlass_int4(
    const int8_t* __restrict__ src, uint8_t* __restrict__ dst, int N, int K) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int k_pair = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = k_pair * 2;

  if (n < N && k < K) {
    // Marlin uint4b8 stores (signed + 8); CUTLASS int4b_t expects signed int4.
    const int8_t v0 = static_cast<int8_t>((src[n * K + k] & 0xF) - 8);
    const int8_t v1 = static_cast<int8_t>((src[n * K + k + 1] & 0xF) - 8);
    const uint8_t packed =
        static_cast<uint8_t>(v0 & 0xF) |
        (static_cast<uint8_t>(v1 & 0xF) << 4);
    dst[(n * K + k) / 2] = packed;
  }
}

torch::Tensor cutlass69_pack_and_reorder(torch::Tensor q_weight_int8) {
  // q_weight_int8: [num_experts, N, K] row-major contiguous
  TORCH_CHECK(q_weight_int8.is_contiguous(), "q_weight_int8 must be contiguous");
  TORCH_CHECK(q_weight_int8.dtype() == torch::kInt8,
              "q_weight_int8 must be int8");
  const int num_experts = static_cast<int>(q_weight_int8.size(0));
  const int N = static_cast<int>(q_weight_int8.size(1));
  const int K = static_cast<int>(q_weight_int8.size(2));
  TORCH_CHECK(K % 2 == 0, "K must be even for int4 packing");

  auto options =
      torch::TensorOptions().dtype(torch::kUInt8).device(q_weight_int8.device());
  torch::Tensor packed = torch::empty({num_experts, N, K / 2}, options);

  const dim3 threads(16, 16);
  const dim3 blocks((N + threads.x - 1) / threads.x,
                    (K / 2 + threads.y - 1) / threads.y);

  for (int e = 0; e < num_experts; ++e) {
    pack_marlin_uint4_to_cutlass_int4<<<blocks, threads>>>(
        q_weight_int8[e].data_ptr<int8_t>(), packed[e].data_ptr<uint8_t>(), N,
        K);
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
              "pack_marlin_uint4_to_cutlass_int4 failed");

  torch::Tensor reordered = torch::empty_like(packed);

  using LayoutB = cutlass::layout::ColumnMajor;
  using StrideB =
      cute::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutB*>>;
  const auto stride_B =
      cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  const auto layout_B = make_layout(make_shape(N, K, Int<1>{}), stride_B);
  const auto layout_B_reordered = tile_to_shape(LayoutAtomQuant{}, make_shape(N, K, Int<1>{}));

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
#else
torch::Tensor cutlass69_pack_and_reorder(torch::Tensor q_weight_int8) {
  TORCH_CHECK(false, "CUTLASS not enabled");
  return q_weight_int8;
}
#endif
