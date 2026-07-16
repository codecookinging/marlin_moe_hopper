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

bool cutlass69_fused_moe_full_env_enabled() {
  const char* env = std::getenv("MARLIN_MOE_CUTLASS69_FUSED_MOE_FULL");
  if (env != nullptr) {
    return env[0] != '0';
  }
  const char* gemm2_env = std::getenv("MARLIN_MOE_CUTLASS69_GEMM2");
  return gemm2_env != nullptr && gemm2_env[0] == '1';
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
    return {false, "CUTLASS GEMM1 path supports N=256 or N=512."};
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

HostSupport select_gemm2_host_path(int major_capability, int a_bits, int b_bits,
                                   int prob_m, int prob_n, int prob_k,
                                   bool has_act_order, bool has_zp,
                                   int moe_block_size, int group_size) {
  if (!cutlass69_compiled()) {
    return {false,
            "CUTLASS was not found at build time. Rebuild with CUTLASS_DIR "
            "pointing to an NVIDIA/cutlass checkout."};
  }
  if (major_capability < 9) {
    return {false, "CUTLASS GEMM2 path requires SM90 or newer."};
  }
  if (prob_k != 256 || prob_m < 16 || prob_m > 65536) {
    return {false,
            "CUTLASS GEMM2 path is scoped to K=256 and "
            "16 <= M <= 65536."};
  }
  if (prob_n != 6144) {
    return {false, "CUTLASS GEMM2 path currently supports N=6144."};
  }
  if (a_bits != 16) {
    return {false, "CUTLASS GEMM2 path is scoped to WNA16 activations."};
  }
  if (b_bits != 4) {
    return {false, "CUTLASS GEMM2 path currently supports INT4 B only."};
  }
  if (has_act_order || has_zp) {
    return {false,
            "act_order/zero-point are not supported on the CUTLASS69 path."};
  }
  if (moe_block_size != 16 && moe_block_size != 32 && moe_block_size != 64) {
    return {false,
            "CUTLASS GEMM2 path supports moe_block_size in {16, 32, 64}."};
  }
  if (group_size != 128 && group_size != -1) {
    return {false,
            "CUTLASS GEMM2 path expects group_size=128 or channelwise (-1)."};
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

void dispatch_marlin_moe_cutlass69_gemm2(
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
              "MARLIN_MOE_USE_CUTLASS69 GEMM2 requires CUTLASS. "
              "Set CUTLASS_DIR and rebuild.");
}

void dispatch_marlin_moe_cutlass69_fused_moe_full(
    const void* hidden, const void* B1, const void* B2, void* output,
    const void* b_scales1, const void* b_scales2, const float* topk_weights,
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    const int32_t* num_tokens_past_padded, int moe_block_size, int num_experts,
    int top_k, int prob_m, int prob_n1, int prob_k1, int prob_n2, int prob_k2,
    vllm::ScalarType const& c_type, int group_size, int dev,
    cudaStream_t stream) {
  (void)hidden;
  (void)B1;
  (void)B2;
  (void)output;
  (void)b_scales1;
  (void)b_scales2;
  (void)topk_weights;
  (void)sorted_token_ids;
  (void)expert_ids;
  (void)num_tokens_past_padded;
  (void)moe_block_size;
  (void)num_experts;
  (void)top_k;
  (void)prob_m;
  (void)prob_n1;
  (void)prob_k1;
  (void)prob_n2;
  (void)prob_k2;
  (void)c_type;
  (void)group_size;
  (void)dev;
  (void)stream;
  TORCH_CHECK(false,
              "CUTLASS69 fused MoE full requires CUTLASS. "
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

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
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

struct Cutlass69FullMoeProfile {
  double host_prep_ms = 0.0;
  double routing_ms = 0.0;
  double buffer_alloc_ms = 0.0;
  double pack_a_ms = 0.0;
  Cutlass69GemmProfile gemm1{};
  Cutlass69GemmProfile gemm2{};
  double silu_a2_ms = 0.0;
  double reduce_ms = 0.0;
  int groups = 0;
  int padded_m = 0;
  int packed_m = 0;
};

bool cutlass69_bfull_enabled(int prob_m) {
  const char* env = std::getenv("MARLIN_MOE_CUTLASS69_BFULL");
  if (env != nullptr) {
    return env[0] != '0';
  }
  return prob_m >= 1024;
}

bool cutlass69_gemm2_enabled() {
  const char* env = std::getenv("MARLIN_MOE_CUTLASS69_GEMM2");
  if (env != nullptr) {
    return env[0] != '0';
  }
  return true;
}

bool cutlass69_profile_enabled() {
  const char* env = std::getenv("MARLIN_MOE_CUTLASS69_PROFILE");
  return env != nullptr && env[0] == '1';
}

struct FullMoeTileConfig {
  int gemm1_tile_n = 128;
  int gemm1_cluster_n = 2;
  int gemm2_tile_n = 256;
  int gemm2_cluster_n = 2;
};

FullMoeTileConfig cutlass69_full_moe_tile_config() {
  FullMoeTileConfig cfg;
  const char* global_cluster = std::getenv("MARLIN_MOE_CUTLASS69_CLUSTER_N");
  const char* gemm1_cluster = std::getenv("MARLIN_MOE_CUTLASS69_GEMM1_CLUSTER_N");
  const char* gemm2_cluster = std::getenv("MARLIN_MOE_CUTLASS69_GEMM2_CLUSTER_N");
  const char* gemm1_tile = std::getenv("MARLIN_MOE_CUTLASS69_GEMM1_TILE_N");
  const char* gemm2_tile = std::getenv("MARLIN_MOE_CUTLASS69_GEMM2_TILE_N");

  auto parse_int = [](const char* env, int fallback) {
    if (env == nullptr || env[0] == '\0') {
      return fallback;
    }
    return std::atoi(env);
  };

  const int global_c =
      global_cluster != nullptr ? parse_int(global_cluster, 0) : 0;
  cfg.gemm1_tile_n = parse_int(gemm1_tile, 128);
  cfg.gemm2_tile_n = parse_int(gemm2_tile, 256);
  cfg.gemm1_cluster_n =
      parse_int(gemm1_cluster, global_c > 0 ? global_c : 2);
  cfg.gemm2_cluster_n =
      parse_int(gemm2_cluster, global_c > 0 ? global_c : 2);
  return cfg;
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

void cutlass69_print_gemm2_profile(int call_id, const Cutlass69FusedProfile& profile,
                                   int prob_m, int prob_n, int prob_k) {
  const double gemm_total = profile.gemm.host_setup_ms + profile.gemm.gemm_init_ms +
                            profile.gemm.gemm_run_ms;
  const double total = profile.host_prep_ms + profile.buffer_alloc_ms +
                       profile.gather_a_ms + gemm_total +
                       profile.fused_silu_scatter_ms;
  const double denom = total > 0.0 ? total : 1.0;
  fprintf(stderr,
          "[CUTLASS69 GEMM2 profile #%d] M=%d N=%d K=%d groups=%d padded_M=%d "
          "packed_M=%d bfull=%d expert_problems=%d total=%.3f ms\n",
          call_id, prob_m, prob_n, prob_k, profile.groups, profile.padded_m,
          profile.packed_m, profile.bfull ? 1 : 0,
          profile.gemm.num_expert_problems, total);
  fprintf(stderr, "  expert_pack_a:                   %8.3f ms (%5.1f%%)\n",
          profile.gather_a_ms, 100.0 * profile.gather_a_ms / denom);
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
  fprintf(stderr, "  weighted_scatter:                %8.3f ms (%5.1f%%)\n",
          profile.fused_silu_scatter_ms,
          100.0 * profile.fused_silu_scatter_ms / denom);
  fprintf(stderr, "  --- gemm substages sum:          %8.3f ms\n", gemm_total);
}

void cutlass69_print_full_moe_profile(int call_id,
                                      const Cutlass69FullMoeProfile& profile,
                                      int prob_m, int prob_n1, int prob_k1,
                                      int prob_n2, int prob_k2) {
  const double gemm1_total = profile.gemm1.host_setup_ms +
                             profile.gemm1.gemm_init_ms +
                             profile.gemm1.gemm_run_ms;
  const double gemm2_total = profile.gemm2.host_setup_ms +
                             profile.gemm2.gemm_init_ms +
                             profile.gemm2.gemm_run_ms;
  const double total = profile.host_prep_ms + profile.routing_ms +
                       profile.buffer_alloc_ms + profile.pack_a_ms +
                       gemm1_total + profile.silu_a2_ms + gemm2_total +
                       profile.reduce_ms;
  const double denom = total > 0.0 ? total : 1.0;
  fprintf(stderr,
          "[CUTLASS69 fused MoE full profile #%d] M=%d N1=%d K1=%d N2=%d K2=%d "
          "groups=%d padded_M=%d packed_M=%d total=%.3f ms\n",
          call_id, prob_m, prob_n1, prob_k1, prob_n2, prob_k2, profile.groups,
          profile.padded_m, profile.packed_m, total);
  fprintf(stderr, "  host_prep (num_tokens D2H+sync): %8.3f ms (%5.1f%%)\n",
          profile.host_prep_ms, 100.0 * profile.host_prep_ms / denom);
  fprintf(stderr, "  routing (count+slot_dst):        %8.3f ms (%5.1f%%)\n",
          profile.routing_ms, 100.0 * profile.routing_ms / denom);
  fprintf(stderr, "  buffer_alloc:                    %8.3f ms (%5.1f%%)\n",
          profile.buffer_alloc_ms, 100.0 * profile.buffer_alloc_ms / denom);
  fprintf(stderr, "  pack_a:                          %8.3f ms (%5.1f%%)\n",
          profile.pack_a_ms, 100.0 * profile.pack_a_ms / denom);
  fprintf(stderr, "  gemm1_host_setup:                %8.3f ms (%5.1f%%)\n",
          profile.gemm1.host_setup_ms,
          100.0 * profile.gemm1.host_setup_ms / denom);
  fprintf(stderr, "  gemm1_init:                      %8.3f ms (%5.1f%%)\n",
          profile.gemm1.gemm_init_ms,
          100.0 * profile.gemm1.gemm_init_ms / denom);
  fprintf(stderr, "  gemm1_run:                       %8.3f ms (%5.1f%%)\n",
          profile.gemm1.gemm_run_ms,
          100.0 * profile.gemm1.gemm_run_ms / denom);
  fprintf(stderr, "  silu_a2 (packed):                %8.3f ms (%5.1f%%)\n",
          profile.silu_a2_ms, 100.0 * profile.silu_a2_ms / denom);
  fprintf(stderr, "  gemm2_host_setup:                %8.3f ms (%5.1f%%)\n",
          profile.gemm2.host_setup_ms,
          100.0 * profile.gemm2.host_setup_ms / denom);
  fprintf(stderr, "  gemm2_init:                      %8.3f ms (%5.1f%%)\n",
          profile.gemm2.gemm_init_ms,
          100.0 * profile.gemm2.gemm_init_ms / denom);
  fprintf(stderr, "  gemm2_run:                       %8.3f ms (%5.1f%%)\n",
          profile.gemm2.gemm_run_ms,
          100.0 * profile.gemm2.gemm_run_ms / denom);
  fprintf(stderr, "  weighted_reduce:                 %8.3f ms (%5.1f%%)\n",
          profile.reduce_ms, 100.0 * profile.reduce_ms / denom);
  fprintf(stderr, "  --- gemm1 substages sum:         %8.3f ms\n", gemm1_total);
  fprintf(stderr, "  --- gemm2 substages sum:         %8.3f ms\n", gemm2_total);
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

  const MmaType val =
      C_grouped[static_cast<int64_t>(col) +
                static_cast<int64_t>(row) * prob_n];
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
  const float gate = static_cast<float>(
      C_grouped[static_cast<int64_t>(col) +
                row_off * static_cast<int64_t>(prob_n)]);
  const float up = static_cast<float>(
      C_grouped[static_cast<int64_t>(col + out_n) +
                row_off * static_cast<int64_t>(prob_n)]);
  const float silu_gate = gate / (1.0f + expf(-gate));
  C_out[static_cast<int64_t>(sorted) * out_n + col] = MmaType(silu_gate * up);
}

template <typename MmaType>
__global__ void scatter_packed_colmajor_weighted_kernel(
    const MmaType* __restrict__ C_packed, MmaType* __restrict__ C_out,
    const int32_t* __restrict__ packed_to_sorted, int packed_m, int prob_n,
    const float* __restrict__ topk_weights, bool mul_topk_weights) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(packed_m) * prob_n;
  if (idx >= total) {
    return;
  }
  const int row = static_cast<int>(idx / prob_n);
  const int col = static_cast<int>(idx % prob_n);
  const int32_t sorted = packed_to_sorted[row];
  if (sorted < 0) {
    return;
  }
  float val = static_cast<float>(
      C_packed[static_cast<int64_t>(col) +
               static_cast<int64_t>(row) * prob_n]);
  if (mul_topk_weights) {
    val *= topk_weights[sorted];
  }
  C_out[static_cast<int64_t>(sorted) * prob_n + col] = MmaType(val);
}

template <typename MmaType>
__global__ void scatter_packed_rowmajor_weighted_kernel(
    const MmaType* __restrict__ C_packed, MmaType* __restrict__ C_out,
    const int32_t* __restrict__ packed_to_sorted, int packed_m, int prob_n,
    const float* __restrict__ topk_weights, bool mul_topk_weights) {
  const int row = blockIdx.x;
  if (row >= packed_m) {
    return;
  }
  const int32_t sorted = packed_to_sorted[row];
  if (sorted < 0) {
    return;
  }
  const float weight =
      mul_topk_weights ? topk_weights[sorted] : 1.0f;
  const int64_t row_base = static_cast<int64_t>(row) * prob_n;
  const int64_t out_base = static_cast<int64_t>(sorted) * prob_n;
  for (int col = threadIdx.x; col < prob_n; col += blockDim.x) {
    const float val = static_cast<float>(C_packed[row_base + col]);
    C_out[out_base + col] = MmaType(val * weight);
  }
}

template <typename MmaType>
__global__ void scatter_packed_rowmajor_weighted_vec_kernel(
    const MmaType* __restrict__ C_packed, MmaType* __restrict__ C_out,
    const int32_t* __restrict__ packed_to_sorted, int packed_m, int prob_n,
    const float* __restrict__ topk_weights, bool mul_topk_weights) {
  constexpr int kVec = 16 / sizeof(MmaType);
  const int row = blockIdx.x;
  if (row >= packed_m) {
    return;
  }
  const int32_t sorted = packed_to_sorted[row];
  if (sorted < 0) {
    return;
  }
  const float weight =
      mul_topk_weights ? topk_weights[sorted] : 1.0f;
  const int64_t row_base = static_cast<int64_t>(row) * prob_n;
  const int64_t out_base = static_cast<int64_t>(sorted) * prob_n;
  for (int col = threadIdx.x * kVec; col < prob_n; col += blockDim.x * kVec) {
    if (col + kVec <= prob_n) {
      uint4 v = *reinterpret_cast<const uint4*>(C_packed + row_base + col);
      MmaType* src = reinterpret_cast<MmaType*>(&v);
      MmaType out_local[kVec];
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        out_local[i] = MmaType(static_cast<float>(src[i]) * weight);
      }
      *reinterpret_cast<uint4*>(C_out + out_base + col) =
          *reinterpret_cast<const uint4*>(out_local);
    } else {
      for (int i = col; i < prob_n; ++i) {
        C_out[out_base + i] =
            MmaType(static_cast<float>(C_packed[row_base + i]) * weight);
      }
    }
  }
}

template <typename MmaType>
__global__ void silu_colmajor_to_a2_kernel(
    const MmaType* __restrict__ C1_packed, MmaType* __restrict__ A2_packed,
    int packed_m, int prob_n) {
  const int out_n = prob_n / 2;
  const int row = blockIdx.x;
  if (row >= packed_m) {
    return;
  }
  for (int col = threadIdx.x; col < out_n; col += blockDim.x) {
    const float gate = static_cast<float>(
        C1_packed[static_cast<int64_t>(col) +
                  static_cast<int64_t>(row) * prob_n]);
    const float up = static_cast<float>(
        C1_packed[static_cast<int64_t>(col + out_n) +
                  static_cast<int64_t>(row) * prob_n]);
    const float silu_gate = gate / (1.0f + expf(-gate));
    A2_packed[static_cast<int64_t>(row) * out_n + col] =
        MmaType(silu_gate * up);
  }
}

template <typename MmaType>
__global__ void silu_colmajor_to_a2_vec_kernel(
    const MmaType* __restrict__ C1_packed, MmaType* __restrict__ A2_packed,
    int packed_m, int prob_n) {
  constexpr int kVec = 16 / sizeof(MmaType);
  const int out_n = prob_n / 2;
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(packed_m) * out_n;
  if (idx >= total) {
    return;
  }
  const int row = static_cast<int>(idx / out_n);
  const int col = static_cast<int>(idx % out_n);
  const float gate = static_cast<float>(
      C1_packed[static_cast<int64_t>(col) +
                static_cast<int64_t>(row) * prob_n]);
  const float up = static_cast<float>(
      C1_packed[static_cast<int64_t>(col + out_n) +
                static_cast<int64_t>(row) * prob_n]);
  const float silu_gate = gate / (1.0f + expf(-gate));
  A2_packed[static_cast<int64_t>(row) * out_n + col] =
      MmaType(silu_gate * up);
}

template <typename MmaType, int TopK>
__global__ void reduce_packed_weighted_rowmajor_vec_kernel(
    const MmaType* __restrict__ C2_packed, MmaType* __restrict__ output,
    const int32_t* __restrict__ sorted_to_packed, int packed_m, int prob_n,
    int num_tokens, int top_k, const float* __restrict__ topk_weights,
    bool mul_topk_weights) {
  constexpr int kVec = 16 / sizeof(MmaType);
  const int token = blockIdx.x;
  if (token >= num_tokens) {
    return;
  }

  __shared__ int32_t s_packed_rows[TopK];
  __shared__ float s_weights[TopK];
  if (threadIdx.x < top_k) {
    const int sorted = token * top_k + threadIdx.x;
    s_packed_rows[threadIdx.x] = sorted_to_packed[sorted];
    s_weights[threadIdx.x] =
        mul_topk_weights ? topk_weights[sorted] : 1.0f;
  }
  __syncthreads();

  const int vec_cols = (prob_n + kVec - 1) / kVec;
  for (int vec_col = threadIdx.x; vec_col < vec_cols; vec_col += blockDim.x) {
    const int col = vec_col * kVec;
    const int64_t out_base = static_cast<int64_t>(token) * prob_n + col;
    if (col + kVec <= prob_n) {
      float acc[kVec];
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        acc[i] = 0.f;
      }
#pragma unroll
      for (int k = 0; k < TopK; ++k) {
        if (k >= top_k) {
          continue;
        }
        const int packed_row = s_packed_rows[k];
        if (packed_row < 0 || packed_row >= packed_m) {
          continue;
        }
        const float weight = s_weights[k];
        const int64_t row_base =
            static_cast<int64_t>(packed_row) * prob_n + col;
        const uint4 vals =
            *reinterpret_cast<const uint4*>(C2_packed + row_base);
        const MmaType* elems = reinterpret_cast<const MmaType*>(&vals);
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
          acc[i] += static_cast<float>(elems[i]) * weight;
        }
      }
      MmaType out_local[kVec];
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        out_local[i] = MmaType(acc[i]);
      }
      *reinterpret_cast<uint4*>(output + out_base) =
          *reinterpret_cast<const uint4*>(out_local);
    } else {
      for (int i = col; i < prob_n; ++i) {
        float sum = 0.f;
#pragma unroll
        for (int k = 0; k < TopK; ++k) {
          if (k >= top_k) {
            continue;
          }
          const int packed_row = s_packed_rows[k];
          if (packed_row < 0 || packed_row >= packed_m) {
            continue;
          }
          sum += static_cast<float>(
                     C2_packed[static_cast<int64_t>(packed_row) * prob_n + i]) *
                 s_weights[k];
        }
        output[static_cast<int64_t>(token) * prob_n + i] = MmaType(sum);
      }
    }
  }
}

template <typename MmaType>
__global__ void reduce_packed_weighted_rowmajor_kernel(
    const MmaType* __restrict__ C2_packed, MmaType* __restrict__ output,
    const int32_t* __restrict__ sorted_to_packed, int packed_m, int prob_n,
    int num_tokens, int top_k, const float* __restrict__ topk_weights,
    bool mul_topk_weights) {
  const int token = blockIdx.x;
  if (token >= num_tokens) {
    return;
  }
  for (int col = threadIdx.x; col < prob_n; col += blockDim.x) {
    float sum = 0.f;
    for (int k = 0; k < top_k; ++k) {
      const int sorted = token * top_k + k;
      const int packed_row = sorted_to_packed[sorted];
      if (packed_row < 0 || packed_row >= packed_m) {
        continue;
      }
      float val = static_cast<float>(
          C2_packed[static_cast<int64_t>(packed_row) * prob_n + col]);
      if (mul_topk_weights) {
        val *= topk_weights[sorted];
      }
      sum += val;
    }
    output[static_cast<int64_t>(token) * prob_n + col] = MmaType(sum);
  }
}

template <typename MmaType>
void launch_reduce_packed_weighted_rowmajor(
    const MmaType* C2_packed, MmaType* output,
    const int32_t* sorted_to_packed, int packed_m, int prob_n, int num_tokens,
    int top_k, const float* topk_weights, bool mul_topk_weights,
    cudaStream_t stream) {
  constexpr int kVec = 16 / sizeof(MmaType);
  const int threads = 256;
  if (top_k <= 8) {
    reduce_packed_weighted_rowmajor_vec_kernel<MmaType, 8>
        <<<num_tokens, threads, 0, stream>>>(
            C2_packed, output, sorted_to_packed, packed_m, prob_n, num_tokens,
            top_k, topk_weights, mul_topk_weights);
  } else {
    reduce_packed_weighted_rowmajor_kernel<MmaType>
        <<<num_tokens, threads, 0, stream>>>(
            C2_packed, output, sorted_to_packed, packed_m, prob_n, num_tokens,
            top_k, topk_weights, mul_topk_weights);
  }
}

template <typename MmaType>
__global__ void silu_rowmajor_to_a2_vec_kernel(
    const MmaType* __restrict__ C1_packed, MmaType* __restrict__ A2_packed,
    int packed_m, int prob_n) {
  const int out_n = prob_n / 2;
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(packed_m) * out_n;
  if (idx >= total) {
    return;
  }
  const int row = static_cast<int>(idx / out_n);
  const int col = static_cast<int>(idx % out_n);
  const int64_t row_base = static_cast<int64_t>(row) * prob_n;
  const float gate = static_cast<float>(C1_packed[row_base + col]);
  const float up = static_cast<float>(C1_packed[row_base + col + out_n]);
  const float silu_gate = gate / (1.0f + expf(-gate));
  A2_packed[static_cast<int64_t>(row) * out_n + col] =
      MmaType(silu_gate * up);
}

template <typename MmaType>
void launch_silu_to_a2(const MmaType* C1_packed, MmaType* A2_packed,
                       int packed_m, int prob_n, bool row_major,
                       cudaStream_t stream) {
  const int out_n = prob_n / 2;
  const int threads = 256;
  const int64_t total = static_cast<int64_t>(packed_m) * out_n;
  const int blocks =
      static_cast<int>((total + threads - 1) / threads);
  if (row_major) {
    silu_rowmajor_to_a2_vec_kernel<MmaType><<<blocks, threads, 0, stream>>>(
        C1_packed, A2_packed, packed_m, prob_n);
  } else {
    silu_colmajor_to_a2_vec_kernel<MmaType><<<blocks, threads, 0, stream>>>(
        C1_packed, A2_packed, packed_m, prob_n);
  }
}

template <typename MmaType>
void launch_silu_colmajor_to_a2(const MmaType* C1_packed, MmaType* A2_packed,
                                int packed_m, int prob_n, cudaStream_t stream) {
  const int out_n = prob_n / 2;
  const int threads = 256;
  const int64_t total = static_cast<int64_t>(packed_m) * out_n;
  const int blocks =
      static_cast<int>((total + threads - 1) / threads);
  silu_colmajor_to_a2_vec_kernel<MmaType><<<blocks, threads, 0, stream>>>(
      C1_packed, A2_packed, packed_m, prob_n);
}

template <typename MmaType>
struct FusedGemm1BufferCache {
  cutlass::DeviceAllocation<MmaType> A_grouped;
  cutlass::DeviceAllocation<MmaType> C_grouped;
  int64_t a_cap = 0;
  int64_t c_cap = 0;

  MmaType* ensure_a(int64_t elems) {
    if (elems > a_cap) {
      A_grouped.reset(elems);
      a_cap = elems;
    }
    return A_grouped.get();
  }

  MmaType* ensure_c(int64_t elems) {
    if (elems > c_cap) {
      C_grouped.reset(elems);
      c_cap = elems;
    }
    return C_grouped.get();
  }
};

template <typename MmaType>
__global__ void gather_moe_a_vec_kernel(const MmaType* __restrict__ A,
                                        MmaType* __restrict__ A_grouped,
                                        const int32_t* __restrict__ sorted_ids,
                                        int groups, int moe_block_size,
                                        int prob_m, int top_k, int prob_k) {
  constexpr int kVec = 16 / sizeof(MmaType);
  static_assert(kVec >= 1 && (16 % sizeof(MmaType)) == 0,
                "128-bit vector requires 16-byte element grouping");
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t vec_total =
      (static_cast<int64_t>(groups) * moe_block_size * prob_k + kVec - 1) /
      kVec;
  if (idx >= vec_total) {
    return;
  }
  const int64_t base = idx * kVec;
  const int g = static_cast<int>(base / (moe_block_size * prob_k));
  const int rem = static_cast<int>(base % (moe_block_size * prob_k));
  const int row = rem / prob_k;
  const int col = rem % prob_k;

  MmaType* dst = A_grouped +
                 (static_cast<int64_t>(g) * moe_block_size + row) * prob_k +
                 col;
  const int32_t sorted = sorted_ids[g * moe_block_size + row];
  if (sorted < 0 || sorted >= prob_m * top_k) {
    if (col + kVec <= prob_k) {
      *reinterpret_cast<uint4*>(dst) = uint4{0, 0, 0, 0};
    } else {
      for (int i = 0; i < kVec && col + i < prob_k; ++i) {
        dst[i] = MmaType(0);
      }
    }
    return;
  }
  const int64_t token = static_cast<int64_t>(sorted) / top_k;
  const MmaType* src = A + token * prob_k + col;
  if (col + kVec <= prob_k) {
    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
  } else {
    for (int i = 0; i < kVec && col + i < prob_k; ++i) {
      dst[i] = src[i];
    }
  }
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

__global__ void build_slot_dst_kernel(
    int32_t* __restrict__ slot_dst, const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ sorted_ids, int32_t* __restrict__ expert_next_row,
    int groups, int moe_block_size, int prob_m, int top_k, int num_experts) {
  const int64_t total = static_cast<int64_t>(groups) * moe_block_size;
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int g = static_cast<int>(idx / moe_block_size);
  const int expert = expert_ids[g];
  if (expert < 0 || expert >= num_experts) {
    slot_dst[idx] = -1;
    return;
  }
  const int32_t sorted = sorted_ids[idx];
  if (sorted < 0 || sorted >= prob_m * top_k) {
    slot_dst[idx] = -1;
    return;
  }
  slot_dst[idx] = atomicAdd(&expert_next_row[expert], 1);
}

template <typename MmaType>
struct BfullBufferCache {
  cutlass::DeviceAllocation<MmaType> A_packed;
  cutlass::DeviceAllocation<MmaType> C_packed;
  cutlass::DeviceAllocation<MmaType> C2_packed;
  cutlass::DeviceAllocation<MmaType> A2_packed;
  cutlass::DeviceAllocation<int32_t> packed_to_sorted;
  cutlass::DeviceAllocation<int32_t> sorted_to_packed;
  cutlass::DeviceAllocation<int32_t> slot_dst;
  cutlass::DeviceAllocation<int32_t> expert_next_row;
  int64_t a_cap = 0;
  int64_t c_cap = 0;
  int64_t c2_cap = 0;
  int64_t a2_cap = 0;
  int64_t packed_cap = 0;
  int64_t sorted_cap = 0;
  int64_t slot_cap = 0;
  int64_t expert_next_cap = 0;

  MmaType* ensure_a(int64_t elems) {
    if (elems > a_cap) {
      A_packed.reset(elems);
      a_cap = elems;
    }
    return A_packed.get();
  }

  MmaType* ensure_c(int64_t elems) {
    if (elems > c_cap) {
      C_packed.reset(elems);
      c_cap = elems;
    }
    return C_packed.get();
  }

  MmaType* ensure_c2(int64_t elems) {
    if (elems > c2_cap) {
      C2_packed.reset(elems);
      c2_cap = elems;
    }
    return C2_packed.get();
  }

  int32_t* ensure_packed_map(int64_t packed_elems) {
    if (packed_elems > packed_cap) {
      packed_to_sorted.reset(packed_elems);
      packed_cap = packed_elems;
    }
    return packed_to_sorted.get();
  }

  int32_t* ensure_slot_map(int64_t slot_elems) {
    if (slot_elems > slot_cap) {
      slot_dst.reset(slot_elems);
      slot_cap = slot_elems;
    }
    return slot_dst.get();
  }

  int32_t* ensure_sorted_to_packed(int64_t sorted_elems) {
    if (sorted_elems > sorted_cap) {
      sorted_to_packed.reset(sorted_elems);
      sorted_cap = sorted_elems;
    }
    return sorted_to_packed.get();
  }

  MmaType* ensure_a2(int64_t elems) {
    if (elems > a2_cap) {
      A2_packed.reset(elems);
      a2_cap = elems;
    }
    return A2_packed.get();
  }

  int32_t* ensure_expert_next_row(int num_experts) {
    if (num_experts > expert_next_cap) {
      expert_next_row.reset(num_experts);
      expert_next_cap = num_experts;
    }
    return expert_next_row.get();
  }
};

template <typename GemmTypes>
struct ExpertPackedGemmLayoutCache {
  static constexpr int kMaxExperts = 256;
  cutlass::DeviceAllocation<typename GemmTypes::LayoutB_Reordered> layout_B;
  typename GemmTypes::LayoutB_Reordered host_layout{};
  int cached_prob_n = 0;
  int cached_prob_k = 0;

  const typename GemmTypes::LayoutB_Reordered* ensure(int prob_n, int prob_k) {
    if (prob_n != cached_prob_n || prob_k != cached_prob_k) {
      auto shape_B = cute::make_shape(prob_n, prob_k, Int<1>{});
      host_layout = cute::tile_to_shape(typename GemmTypes::LayoutAtomQuant{},
                                        shape_B);
      std::vector<typename GemmTypes::LayoutB_Reordered> layouts(
          kMaxExperts, host_layout);
      layout_B.reset(kMaxExperts);
      layout_B.copy_from_host(layouts.data());
      cached_prob_n = prob_n;
      cached_prob_k = prob_k;
    }
    return layout_B.get();
  }
};

template <typename MmaType>
__global__ void pack_a_expert_deterministic_kernel(
    const MmaType* __restrict__ A, MmaType* __restrict__ A_packed,
    int32_t* __restrict__ packed_to_sorted, const int32_t* __restrict__ slot_dst,
    const int32_t* __restrict__ sorted_ids, int total_slots, int prob_m,
    int top_k, int prob_k, int32_t* __restrict__ sorted_to_packed) {
  const int64_t slot =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (slot >= total_slots) {
    return;
  }
  const int32_t dst_row = slot_dst[slot];
  if (dst_row < 0) {
    return;
  }
  const int32_t sorted = sorted_ids[slot];
  if (sorted < 0 || sorted >= prob_m * top_k) {
    return;
  }
  if (packed_to_sorted != nullptr) {
    packed_to_sorted[dst_row] = sorted;
  }
  if (sorted_to_packed != nullptr) {
    sorted_to_packed[sorted] = dst_row;
  }
  const int64_t token = static_cast<int64_t>(sorted) / top_k;
  const MmaType* src = A + token * prob_k;
  MmaType* dst = A_packed + static_cast<int64_t>(dst_row) * prob_k;
  constexpr int kVec = 16 / sizeof(MmaType);
  const int vec_k = prob_k / kVec;
  for (int kv = threadIdx.x; kv < vec_k; kv += blockDim.x) {
    *reinterpret_cast<uint4*>(dst + kv * kVec) =
        *reinterpret_cast<const uint4*>(src + kv * kVec);
  }
  for (int k = vec_k * kVec + threadIdx.x; k < prob_k; k += blockDim.x) {
    dst[k] = src[k];
  }
}

template <typename MmaType>
__global__ void scatter_packed_fused_silu_colmajor_kernel(
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
  if (sorted < 0) {
    return;
  }
  const float gate = static_cast<float>(
      C_packed[static_cast<int64_t>(col) +
               static_cast<int64_t>(row) * prob_n]);
  const float up = static_cast<float>(
      C_packed[static_cast<int64_t>(col + out_n) +
               static_cast<int64_t>(row) * prob_n]);
  const float silu_gate = gate / (1.0f + expf(-gate));
  C_out[static_cast<int64_t>(sorted) * out_n + col] = MmaType(silu_gate * up);
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
struct ExpertPackedGemmHostBundle {
  int num_expert_problems = 0;
  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;
  std::vector<const typename GemmTypes::ElementA*> ptr_A_host;
  std::vector<const typename GemmTypes::ElementB*> ptr_B_host;
  std::vector<const typename GemmTypes::ElementScale*> ptr_scale_host;
  std::vector<const typename GemmTypes::ElementC*> ptr_C_host;
  std::vector<typename GemmTypes::ElementC*> ptr_D_host;
  std::vector<typename GemmTypes::StrideA> stride_A_host;
  std::vector<typename GemmTypes::StrideB> stride_B_host;
  std::vector<typename GemmTypes::StrideC> stride_C_host;
  std::vector<typename GemmTypes::StrideD> stride_D_host;
  std::vector<typename GemmTypes::StrideS> stride_S_host;
  std::vector<typename GemmTypes::LayoutB_Reordered> layout_B_reordered_host;
};

template <typename GemmTypes>
struct GroupedGemmLaunchCache {
  cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape>
      problem_sizes;
  cutlass::DeviceAllocation<const typename GemmTypes::ElementA*> ptr_A;
  cutlass::DeviceAllocation<const typename GemmTypes::ElementB*> ptr_B;
  cutlass::DeviceAllocation<const typename GemmTypes::ElementScale*> ptr_scale;
  cutlass::DeviceAllocation<const typename GemmTypes::ElementC*> ptr_C;
  cutlass::DeviceAllocation<typename GemmTypes::ElementC*> ptr_D;
  cutlass::DeviceAllocation<typename GemmTypes::StrideA> stride_A;
  cutlass::DeviceAllocation<typename GemmTypes::StrideB> stride_B;
  cutlass::DeviceAllocation<typename GemmTypes::StrideC> stride_C;
  cutlass::DeviceAllocation<typename GemmTypes::StrideD> stride_D;
  cutlass::DeviceAllocation<typename GemmTypes::StrideS> stride_S;
  cutlass::DeviceAllocation<typename GemmTypes::LayoutB_Reordered>
      layout_B_reordered;
  cutlass::DeviceAllocation<uint8_t> workspace;
  int problem_cap = 0;
  size_t workspace_bytes = 0;
};

template <typename GemmTypes>
ExpertPackedGemmHostBundle<GemmTypes> build_expert_packed_gemm_host_bundle(
    int num_experts, const std::vector<int32_t>& expert_offsets,
    const std::vector<int32_t>& expert_counts, int prob_n, int prob_k,
    int group_size, const typename GemmTypes::ElementA* A_packed,
    const typename GemmTypes::ElementB* B_base,
    const typename GemmTypes::ElementScale* scale_base,
    typename GemmTypes::ElementC* C_packed) {
  using ElementA = typename GemmTypes::ElementA;
  using ElementC = typename GemmTypes::ElementC;
  using ElementScale = typename GemmTypes::ElementScale;
  using LayoutB_Reordered = typename GemmTypes::LayoutB_Reordered;
  using StrideA = typename GemmTypes::StrideA;
  using StrideB = typename GemmTypes::StrideB;
  using StrideC = typename GemmTypes::StrideC;
  using StrideD = typename GemmTypes::StrideD;
  using StrideS = typename GemmTypes::StrideS;

  ExpertPackedGemmHostBundle<GemmTypes> bundle;
  const int scale_k =
      cutlass::ceil_div(prob_k, group_size > 0 ? group_size : prob_k);
  const int64_t b_elems_per_expert =
      static_cast<int64_t>(prob_k) * prob_n / 2;

  std::vector<int> active_experts;
  active_experts.reserve(num_experts);
  for (int e = 0; e < num_experts; ++e) {
    if (expert_counts[e] > 0) {
      active_experts.push_back(e);
    }
  }
  bundle.num_expert_problems = static_cast<int>(active_experts.size());
  if (bundle.num_expert_problems == 0) {
    return bundle;
  }

  bundle.problem_sizes_host.resize(bundle.num_expert_problems);
  bundle.ptr_A_host.resize(bundle.num_expert_problems);
  bundle.ptr_B_host.resize(bundle.num_expert_problems);
  bundle.ptr_scale_host.resize(bundle.num_expert_problems);
  bundle.ptr_C_host.resize(bundle.num_expert_problems);
  bundle.ptr_D_host.resize(bundle.num_expert_problems);
  bundle.stride_A_host.resize(bundle.num_expert_problems);
  bundle.stride_B_host.resize(bundle.num_expert_problems);
  bundle.stride_C_host.resize(bundle.num_expert_problems);
  bundle.stride_D_host.resize(bundle.num_expert_problems);
  bundle.stride_S_host.resize(bundle.num_expert_problems);
  bundle.layout_B_reordered_host.resize(bundle.num_expert_problems);

  static ExpertPackedGemmLayoutCache<GemmTypes> layout_cache;
  layout_cache.ensure(prob_n, prob_k);

  for (int p = 0; p < bundle.num_expert_problems; ++p) {
    const int expert = active_experts[p];
    const int m = expert_counts[expert];
    const int64_t row_offset = expert_offsets[expert];
    bundle.problem_sizes_host[p] = make_tuple(prob_n, m, prob_k);

    bundle.ptr_A_host[p] = A_packed + row_offset * prob_k;
    bundle.ptr_C_host[p] = C_packed + row_offset * prob_n;
    bundle.ptr_D_host[p] = C_packed + row_offset * prob_n;

    bundle.stride_A_host[p] =
        cutlass::make_cute_packed_stride(StrideA{}, {m, prob_k, 1});
    bundle.stride_B_host[p] =
        cutlass::make_cute_packed_stride(StrideB{}, {prob_n, prob_k, 1});
    if constexpr (GemmTypes::kRowMajorD) {
      bundle.stride_C_host[p] =
          cutlass::make_cute_packed_stride(StrideC{}, {m, prob_n, 1});
      bundle.stride_D_host[p] =
          cutlass::make_cute_packed_stride(StrideD{}, {m, prob_n, 1});
    } else {
      bundle.stride_C_host[p] =
          cutlass::make_cute_packed_stride(StrideC{}, {prob_n, m, 1});
      bundle.stride_D_host[p] =
          cutlass::make_cute_packed_stride(StrideD{}, {prob_n, m, 1});
    }
    bundle.stride_S_host[p] =
        cutlass::make_cute_packed_stride(StrideS{}, {prob_n, scale_k, 1});

    bundle.ptr_B_host[p] =
        reinterpret_cast<const typename GemmTypes::ElementB*>(B_base) +
        static_cast<int64_t>(expert) * b_elems_per_expert;
    bundle.ptr_scale_host[p] =
        scale_base + static_cast<int64_t>(expert) * (prob_n * scale_k);
    bundle.layout_B_reordered_host[p] = layout_cache.host_layout;
  }
  return bundle;
}

template <typename GemmTypes>
bool launch_expert_packed_gemm_from_bundle(
    const ExpertPackedGemmHostBundle<GemmTypes>& bundle, cudaStream_t stream,
    Cutlass69GemmProfile* profile = nullptr) {
  using Gemm = typename GemmTypes::Gemm;
  using ElementA = typename GemmTypes::ElementA;
  using LayoutB_Reordered = typename GemmTypes::LayoutB_Reordered;
  using StrideA = typename GemmTypes::StrideA;

  const int num_expert_problems = bundle.num_expert_problems;
  if (num_expert_problems == 0) {
    return false;
  }

  static GroupedGemmLaunchCache<GemmTypes> cache;
  if (num_expert_problems > cache.problem_cap) {
    cache.problem_sizes.reset(num_expert_problems);
    cache.ptr_A.reset(num_expert_problems);
    cache.ptr_B.reset(num_expert_problems);
    cache.ptr_scale.reset(num_expert_problems);
    cache.ptr_C.reset(num_expert_problems);
    cache.ptr_D.reset(num_expert_problems);
    cache.stride_A.reset(num_expert_problems);
    cache.stride_B.reset(num_expert_problems);
    cache.stride_C.reset(num_expert_problems);
    cache.stride_D.reset(num_expert_problems);
    cache.stride_S.reset(num_expert_problems);
    cache.layout_B_reordered.reset(num_expert_problems);
    cache.problem_cap = num_expert_problems;
  }

  cache.problem_sizes.copy_from_host(bundle.problem_sizes_host.data());
  cache.ptr_A.copy_from_host(bundle.ptr_A_host.data());
  cache.ptr_B.copy_from_host(bundle.ptr_B_host.data());
  cache.ptr_scale.copy_from_host(bundle.ptr_scale_host.data());
  cache.ptr_C.copy_from_host(bundle.ptr_C_host.data());
  cache.ptr_D.copy_from_host(bundle.ptr_D_host.data());
  cache.stride_A.copy_from_host(bundle.stride_A_host.data());
  cache.stride_B.copy_from_host(bundle.stride_B_host.data());
  cache.stride_C.copy_from_host(bundle.stride_C_host.data());
  cache.stride_D.copy_from_host(bundle.stride_D_host.data());
  cache.stride_S.copy_from_host(bundle.stride_S_host.data());
  cache.layout_B_reordered.copy_from_host(
      bundle.layout_B_reordered_host.data());

  if (profile != nullptr) {
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
      {num_expert_problems, cache.problem_sizes.get(), nullptr},
      {cache.ptr_B.get(), cache.layout_B_reordered.get(), cache.ptr_A.get(),
       cache.stride_A.get(), cache.ptr_scale.get(), cache.stride_S.get(),
       kScaleChunk},
      {fusion_args, cache.ptr_C.get(), cache.stride_C.get(),
       cache.ptr_D.get(), cache.stride_D.get()},
      hw_info};

  const auto gemm_init_begin = std::chrono::steady_clock::now();
  static Gemm gemm;
  static size_t cached_workspace_bytes = 0;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  if (workspace_size > cache.workspace_bytes) {
    cache.workspace.reset(workspace_size);
    cache.workspace_bytes = workspace_size;
    cached_workspace_bytes = 0;
  }
  CUTLASS_CHECK(gemm.can_implement(arguments));
  if (cached_workspace_bytes == cache.workspace_bytes &&
      cache.workspace_bytes > 0) {
    CUTLASS_CHECK(gemm.update(arguments, cache.workspace.get()));
  } else {
    CUTLASS_CHECK(
        gemm.initialize(arguments, cache.workspace.get(), stream));
    cached_workspace_bytes = cache.workspace_bytes;
  }
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
  const auto host_setup_begin = std::chrono::steady_clock::now();
  auto bundle = build_expert_packed_gemm_host_bundle<GemmTypes>(
      num_experts, expert_offsets, expert_counts, prob_n, prob_k, group_size,
      A_packed, B_base, scale_base, C_packed);
  if (profile != nullptr) {
    const auto host_setup_end = std::chrono::steady_clock::now();
    profile->host_setup_ms =
        std::chrono::duration<double, std::milli>(host_setup_end -
                                                 host_setup_begin)
            .count();
  }
  return launch_expert_packed_gemm_from_bundle<GemmTypes>(bundle, stream,
                                                          profile);
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

  std::vector<int32_t> expert_ids_host(groups);
  std::vector<int32_t> sorted_host(total_slots);
  CUDA_CHECK(cudaMemcpyAsync(expert_ids_host.data(), expert_ids,
                             groups * sizeof(int32_t), cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(sorted_host.data(), sorted_token_ids,
                             total_slots * sizeof(int32_t),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<int32_t> slot_dst_host(total_slots, -1);
  std::vector<int32_t> next_row(num_experts);
  for (int e = 0; e < num_experts; ++e) {
    next_row[e] = offsets_host[e];
  }
  for (int64_t slot = 0; slot < total_slots; ++slot) {
    const int g = static_cast<int>(slot / moe_block_size);
    const int expert = expert_ids_host[g];
    if (expert < 0 || expert >= num_experts) {
      continue;
    }
    const int32_t sorted = sorted_host[slot];
    if (sorted < 0 || sorted >= prob_m * top_k) {
      continue;
    }
    slot_dst_host[slot] = next_row[expert]++;
  }

  const auto buffer_alloc_begin = std::chrono::steady_clock::now();
  static BfullBufferCache<MmaType> buffer_cache;
  MmaType* A_packed_ptr = nullptr;
  MmaType* C_packed_ptr = nullptr;
  int32_t* packed_to_sorted_ptr = nullptr;
  int32_t* slot_dst_ptr = nullptr;
  if (total_packed > 0) {
    A_packed_ptr = buffer_cache.ensure_a(static_cast<int64_t>(total_packed) * prob_k);
    C_packed_ptr = buffer_cache.ensure_c(static_cast<int64_t>(total_packed) * prob_n);
    packed_to_sorted_ptr = buffer_cache.ensure_packed_map(total_packed);
    slot_dst_ptr = buffer_cache.ensure_slot_map(total_slots);
    CUDA_CHECK(cudaMemsetAsync(packed_to_sorted_ptr, 0xff,
                             static_cast<size_t>(total_packed) * sizeof(int32_t),
                             stream));
    CUDA_CHECK(cudaMemcpyAsync(slot_dst_ptr, slot_dst_host.data(),
                               total_slots * sizeof(int32_t),
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
    const int pack_blocks = static_cast<int>(
        (total_slots + pack_threads - 1) / pack_threads);
    pack_a_expert_deterministic_kernel<MmaType><<<pack_blocks, pack_threads, 0,
                                                stream>>>(
        reinterpret_cast<const MmaType*>(A), A_packed_ptr, packed_to_sorted_ptr,
        slot_dst_ptr, sorted_token_ids, static_cast<int>(total_slots), prob_m,
        top_k, prob_k, nullptr);
    if (profile) {
      CUDA_CHECK(cudaEventRecord(pack_stop, stream));
      profile_data.gather_a_ms = cutlass69_elapsed_ms(pack_start, pack_stop);
      CUDA_CHECK(cudaEventDestroy(pack_start));
      CUDA_CHECK(cudaEventDestroy(pack_stop));
    }

    Cutlass69GemmProfile* gemm_profile = profile ? &profile_data.gemm : nullptr;
    launch_grouped_gemm_expert_packed<GemmTypes>(
        num_experts, offsets_host, counts_host, prob_n, prob_k, group_size,
        A_packed_ptr, reinterpret_cast<const typename GemmTypes::ElementB*>(B),
        reinterpret_cast<const MmaType*>(b_scales), C_packed_ptr, stream,
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
    if constexpr (RowMajorD) {
      scatter_packed_fused_silu_rowmajor_kernel<MmaType>
          <<<scatter_blocks, scatter_threads, 0, stream>>>(
              C_packed_ptr, reinterpret_cast<MmaType*>(C), packed_to_sorted_ptr,
              total_packed, prob_n);
    } else {
      scatter_packed_fused_silu_colmajor_kernel<MmaType>
          <<<scatter_blocks, scatter_threads, 0, stream>>>(
              C_packed_ptr, reinterpret_cast<MmaType*>(C), packed_to_sorted_ptr,
              total_packed, prob_n);
    }
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
void dispatch_gemm2_bfull_typed(
    const void* A, const void* B, void* C, const void* b_scales,
    const int32_t* sorted_token_ids, const int32_t* expert_ids, int groups,
    int moe_block_size, int top_k, int prob_m, int prob_n, int prob_k,
    int group_size, int num_experts, const float* topk_weights,
    bool mul_topk_weights, cudaStream_t stream, double host_prep_ms) {
  using GemmTypes =
      Cutlass69GroupedGemmTypes<MmaType, MmaType, TileN, ClusterN, RowMajorD>;

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

  std::vector<int32_t> expert_ids_host(groups);
  std::vector<int32_t> sorted_host(total_slots);
  CUDA_CHECK(cudaMemcpyAsync(expert_ids_host.data(), expert_ids,
                             groups * sizeof(int32_t), cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(sorted_host.data(), sorted_token_ids,
                             total_slots * sizeof(int32_t),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<int32_t> slot_dst_host(total_slots, -1);
  std::vector<int32_t> next_row(num_experts);
  for (int e = 0; e < num_experts; ++e) {
    next_row[e] = offsets_host[e];
  }
  for (int64_t slot = 0; slot < total_slots; ++slot) {
    const int g = static_cast<int>(slot / moe_block_size);
    const int expert = expert_ids_host[g];
    if (expert < 0 || expert >= num_experts) {
      continue;
    }
    const int32_t sorted = sorted_host[slot];
    if (sorted < 0 || sorted >= prob_m * top_k) {
      continue;
    }
    slot_dst_host[slot] = next_row[expert]++;
  }

  const auto buffer_alloc_begin = std::chrono::steady_clock::now();
  static BfullBufferCache<MmaType> buffer_cache;
  MmaType* A_packed_ptr = nullptr;
  MmaType* C_packed_ptr = nullptr;
  int32_t* packed_to_sorted_ptr = nullptr;
  int32_t* slot_dst_ptr = nullptr;
  if (total_packed > 0) {
    A_packed_ptr = buffer_cache.ensure_a(static_cast<int64_t>(total_packed) * prob_k);
    C_packed_ptr = buffer_cache.ensure_c(static_cast<int64_t>(total_packed) * prob_n);
    packed_to_sorted_ptr = buffer_cache.ensure_packed_map(total_packed);
    slot_dst_ptr = buffer_cache.ensure_slot_map(total_slots);
    CUDA_CHECK(cudaMemsetAsync(packed_to_sorted_ptr, 0xff,
                               static_cast<size_t>(total_packed) * sizeof(int32_t),
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(slot_dst_ptr, slot_dst_host.data(),
                               total_slots * sizeof(int32_t),
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
    const int pack_blocks = static_cast<int>(
        (total_slots + pack_threads - 1) / pack_threads);
    pack_a_expert_deterministic_kernel<MmaType><<<pack_blocks, pack_threads, 0,
                                                stream>>>(
        reinterpret_cast<const MmaType*>(A), A_packed_ptr, packed_to_sorted_ptr,
        slot_dst_ptr, sorted_token_ids, static_cast<int>(total_slots), prob_m,
        top_k, prob_k, nullptr);
    if (profile) {
      CUDA_CHECK(cudaEventRecord(pack_stop, stream));
      profile_data.gather_a_ms = cutlass69_elapsed_ms(pack_start, pack_stop);
      CUDA_CHECK(cudaEventDestroy(pack_start));
      CUDA_CHECK(cudaEventDestroy(pack_stop));
    }

    Cutlass69GemmProfile* gemm_profile = profile ? &profile_data.gemm : nullptr;
    launch_grouped_gemm_expert_packed<GemmTypes>(
        num_experts, offsets_host, counts_host, prob_n, prob_k, group_size,
        A_packed_ptr, reinterpret_cast<const typename GemmTypes::ElementB*>(B),
        reinterpret_cast<const MmaType*>(b_scales), C_packed_ptr, stream,
        gemm_profile);

    const int scatter_threads = 256;
    const int scatter_blocks = total_packed;

    cudaEvent_t scatter_start{};
    cudaEvent_t scatter_stop{};
    if (profile) {
      CUDA_CHECK(cudaEventCreate(&scatter_start));
      CUDA_CHECK(cudaEventCreate(&scatter_stop));
      CUDA_CHECK(cudaEventRecord(scatter_start, stream));
    }
    if constexpr (RowMajorD) {
      scatter_packed_rowmajor_weighted_vec_kernel<MmaType>
          <<<scatter_blocks, scatter_threads, 0, stream>>>(
              C_packed_ptr, reinterpret_cast<MmaType*>(C), packed_to_sorted_ptr,
              total_packed, prob_n, topk_weights, mul_topk_weights);
    } else {
      scatter_packed_colmajor_weighted_kernel<MmaType>
          <<<static_cast<int>((static_cast<int64_t>(total_packed) * prob_n +
                              scatter_threads - 1) /
                             scatter_threads),
           scatter_threads, 0, stream>>>(
              C_packed_ptr, reinterpret_cast<MmaType*>(C), packed_to_sorted_ptr,
              total_packed, prob_n, topk_weights, mul_topk_weights);
    }
    if (profile) {
      CUDA_CHECK(cudaEventRecord(scatter_stop, stream));
      profile_data.fused_silu_scatter_ms =
          cutlass69_elapsed_ms(scatter_start, scatter_stop);
      CUDA_CHECK(cudaEventDestroy(scatter_start));
      CUDA_CHECK(cudaEventDestroy(scatter_stop));
    }
  }

  if (profile) {
    cutlass69_print_gemm2_profile(++profile_call_id, profile_data, prob_m, prob_n,
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
  static FusedGemm1BufferCache<MmaType> buffer_cache;
  const int64_t a_elems =
      static_cast<int64_t>(groups) * moe_block_size * prob_k;
  const int64_t c_elems =
      static_cast<int64_t>(groups) * moe_block_size * prob_n;
  MmaType* A_grouped = buffer_cache.ensure_a(a_elems);
  MmaType* C_grouped = buffer_cache.ensure_c(c_elems);
  if (profile) {
    const auto buffer_alloc_end = std::chrono::steady_clock::now();
    profile_data.buffer_alloc_ms =
        std::chrono::duration<double, std::milli>(buffer_alloc_end -
                                                 buffer_alloc_begin)
            .count();
  }

  const int threads = 256;
  const int64_t gather_elems =
      static_cast<int64_t>(groups) * moe_block_size * prob_k;
  const bool use_vec_gather = prob_m >= 1024 && (prob_k % 8 == 0);

  cudaEvent_t gather_start{};
  cudaEvent_t gather_stop{};
  if (profile) {
    CUDA_CHECK(cudaEventCreate(&gather_start));
    CUDA_CHECK(cudaEventCreate(&gather_stop));
    CUDA_CHECK(cudaEventRecord(gather_start, stream));
  }
  if (use_vec_gather) {
    constexpr int kVec = 16 / sizeof(MmaType);
    const int64_t vec_total = (gather_elems + kVec - 1) / kVec;
    const int blocks =
        static_cast<int>((vec_total + threads - 1) / threads);
    gather_moe_a_vec_kernel<MmaType><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const MmaType*>(A), A_grouped, sorted_token_ids,
        groups, moe_block_size, prob_m, top_k, prob_k);
  } else {
    const int blocks =
        static_cast<int>((gather_elems + threads - 1) / threads);
    gather_moe_a_kernel<MmaType><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const MmaType*>(A), A_grouped, sorted_token_ids,
        groups, moe_block_size, prob_m, top_k, prob_k);
  }
  if (profile) {
    CUDA_CHECK(cudaEventRecord(gather_stop, stream));
    profile_data.gather_a_ms = cutlass69_elapsed_ms(gather_start, gather_stop);
    CUDA_CHECK(cudaEventDestroy(gather_start));
    CUDA_CHECK(cudaEventDestroy(gather_stop));
  }

  Cutlass69GemmProfile* gemm_profile = profile ? &profile_data.gemm : nullptr;
  launch_grouped_gemm<GemmTypes>(
      groups, moe_block_size, prob_n, prob_k, group_size, A_grouped,
      reinterpret_cast<const typename GemmTypes::ElementB*>(B),
      reinterpret_cast<const MmaType*>(b_scales), C_grouped, expert_ids,
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
            C_grouped, reinterpret_cast<MmaType*>(C), sorted_token_ids,
            groups, moe_block_size, prob_n, prob_m, top_k);
  } else {
    scatter_moe_c_fused_silu_kernel<MmaType><<<scatter_blocks, scatter_threads, 0,
                                               stream>>>(
        C_grouped, reinterpret_cast<MmaType*>(C), sorted_token_ids,
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
      dispatch_fused_gemm1_bfull_typed<cutlass::half_t, 256, 2, false>(
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
      dispatch_fused_gemm1_bfull_typed<cutlass::bfloat16_t, 256, 2, false>(
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

template <typename MmaType, int TileN1, int ClusterN1, int TileN2, int ClusterN2>
void dispatch_fused_moe_full_impl(
    const void* hidden, const void* B1, const void* B2, void* output,
    const void* b_scales1, const void* b_scales2, const float* topk_weights,
    const int32_t* sorted_token_ids, const int32_t* expert_ids, int groups,
    int moe_block_size, int top_k, int prob_m, int prob_n1, int prob_k1,
    int prob_n2, int prob_k2, int group_size, int num_experts,
    cudaStream_t stream, double host_prep_ms) {
  using Gemm1Types =
      Cutlass69GroupedGemmTypes<MmaType, MmaType, TileN1, ClusterN1, true>;
  using Gemm2Types =
      Cutlass69GroupedGemmTypes<MmaType, MmaType, TileN2, ClusterN2, true>;
  const int out_n = prob_n1 / 2;
  const int64_t total_slots = static_cast<int64_t>(groups) * moe_block_size;
  const int sorted_elems = prob_m * top_k;

  const bool profile = cutlass69_profile_enabled();
  Cutlass69FullMoeProfile profile_data{};
  static int profile_call_id = 0;
  if (profile) {
    profile_data.host_prep_ms = host_prep_ms;
    profile_data.groups = groups;
    profile_data.padded_m = static_cast<int>(total_slots);
  }

  const auto routing_begin = std::chrono::steady_clock::now();

  cutlass::DeviceAllocation<int32_t> expert_counts;
  expert_counts.reset(num_experts);
  CUDA_CHECK(cudaMemsetAsync(expert_counts.get(), 0,
                             static_cast<size_t>(num_experts) * sizeof(int32_t),
                             stream));

  const int count_threads = 256;
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
  if (total_packed == 0) {
    return;
  }
  if (profile) {
    profile_data.packed_m = total_packed;
  }

  static BfullBufferCache<MmaType> buffer_cache;
  int32_t* expert_next_row_ptr =
      buffer_cache.ensure_expert_next_row(num_experts);
  int32_t* slot_dst_ptr = buffer_cache.ensure_slot_map(total_slots);
  CUDA_CHECK(cudaMemcpyAsync(expert_next_row_ptr, offsets_host.data(),
                             static_cast<size_t>(num_experts) * sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream));
  build_slot_dst_kernel<<<count_blocks, count_threads, 0, stream>>>(
      slot_dst_ptr, expert_ids, sorted_token_ids, expert_next_row_ptr, groups,
      moe_block_size, prob_m, top_k, num_experts);

  if (profile) {
    const auto routing_end = std::chrono::steady_clock::now();
    profile_data.routing_ms =
        std::chrono::duration<double, std::milli>(routing_end - routing_begin)
            .count();
  }

  const auto buffer_alloc_begin = std::chrono::steady_clock::now();
  MmaType* A1_packed =
      buffer_cache.ensure_a(static_cast<int64_t>(total_packed) * prob_k1);
  MmaType* C1_packed =
      buffer_cache.ensure_c(static_cast<int64_t>(total_packed) * prob_n1);
  MmaType* A2_packed =
      buffer_cache.ensure_a2(static_cast<int64_t>(total_packed) * out_n);
  MmaType* C2_packed =
      buffer_cache.ensure_c2(static_cast<int64_t>(total_packed) * prob_n2);
  int32_t* sorted_to_packed_ptr =
      buffer_cache.ensure_sorted_to_packed(sorted_elems);

  CUDA_CHECK(cudaMemsetAsync(sorted_to_packed_ptr, 0xff,
                             static_cast<size_t>(sorted_elems) * sizeof(int32_t),
                             stream));
  if (profile) {
    const auto buffer_alloc_end = std::chrono::steady_clock::now();
    profile_data.buffer_alloc_ms =
        std::chrono::duration<double, std::milli>(buffer_alloc_end -
                                                 buffer_alloc_begin)
            .count();
  }

  auto gemm1_bundle = build_expert_packed_gemm_host_bundle<Gemm1Types>(
      num_experts, offsets_host, counts_host, prob_n1, prob_k1, group_size,
      A1_packed, reinterpret_cast<const typename Gemm1Types::ElementB*>(B1),
      reinterpret_cast<const MmaType*>(b_scales1), C1_packed);
  auto gemm2_bundle = build_expert_packed_gemm_host_bundle<Gemm2Types>(
      num_experts, offsets_host, counts_host, prob_n2, prob_k2, group_size,
      A2_packed, reinterpret_cast<const typename Gemm2Types::ElementB*>(B2),
      reinterpret_cast<const MmaType*>(b_scales2), C2_packed);

  const int pack_threads = 256;
  const int pack_blocks =
      static_cast<int>((total_slots + pack_threads - 1) / pack_threads);

  cudaEvent_t pack_start{};
  cudaEvent_t pack_stop{};
  if (profile) {
    CUDA_CHECK(cudaEventCreate(&pack_start));
    CUDA_CHECK(cudaEventCreate(&pack_stop));
    CUDA_CHECK(cudaEventRecord(pack_start, stream));
  }
  pack_a_expert_deterministic_kernel<MmaType><<<pack_blocks, pack_threads, 0,
                                              stream>>>(
      reinterpret_cast<const MmaType*>(hidden), A1_packed, nullptr,
      slot_dst_ptr, sorted_token_ids, static_cast<int>(total_slots), prob_m,
      top_k, prob_k1, sorted_to_packed_ptr);
  if (profile) {
    CUDA_CHECK(cudaEventRecord(pack_stop, stream));
    profile_data.pack_a_ms = cutlass69_elapsed_ms(pack_start, pack_stop);
    CUDA_CHECK(cudaEventDestroy(pack_start));
    CUDA_CHECK(cudaEventDestroy(pack_stop));
  }

  Cutlass69GemmProfile* gemm1_profile = profile ? &profile_data.gemm1 : nullptr;
  if (profile) {
    profile_data.gemm1.host_setup_ms = 0.0;
  }
  launch_expert_packed_gemm_from_bundle<Gemm1Types>(gemm1_bundle, stream,
                                                    gemm1_profile);

  cudaEvent_t silu_start{};
  cudaEvent_t silu_stop{};
  if (profile) {
    CUDA_CHECK(cudaEventCreate(&silu_start));
    CUDA_CHECK(cudaEventCreate(&silu_stop));
    CUDA_CHECK(cudaEventRecord(silu_start, stream));
  }
  launch_silu_to_a2(C1_packed, A2_packed, total_packed, prob_n1, true,
                    stream);
  if (profile) {
    CUDA_CHECK(cudaEventRecord(silu_stop, stream));
    profile_data.silu_a2_ms = cutlass69_elapsed_ms(silu_start, silu_stop);
    CUDA_CHECK(cudaEventDestroy(silu_start));
    CUDA_CHECK(cudaEventDestroy(silu_stop));
  }

  Cutlass69GemmProfile* gemm2_profile = profile ? &profile_data.gemm2 : nullptr;
  if (profile) {
    profile_data.gemm2.host_setup_ms = 0.0;
  }
  launch_expert_packed_gemm_from_bundle<Gemm2Types>(gemm2_bundle, stream,
                                                    gemm2_profile);

  cudaEvent_t reduce_start{};
  cudaEvent_t reduce_stop{};
  if (profile) {
    CUDA_CHECK(cudaEventCreate(&reduce_start));
    CUDA_CHECK(cudaEventCreate(&reduce_stop));
    CUDA_CHECK(cudaEventRecord(reduce_start, stream));
  }
  launch_reduce_packed_weighted_rowmajor(
      C2_packed, reinterpret_cast<MmaType*>(output), sorted_to_packed_ptr,
      total_packed, prob_n2, prob_m, top_k, topk_weights, true, stream);
  if (profile) {
    CUDA_CHECK(cudaEventRecord(reduce_stop, stream));
    profile_data.reduce_ms = cutlass69_elapsed_ms(reduce_start, reduce_stop);
    CUDA_CHECK(cudaEventDestroy(reduce_start));
    CUDA_CHECK(cudaEventDestroy(reduce_stop));
  }

  if (profile) {
    cutlass69_print_full_moe_profile(++profile_call_id, profile_data, prob_m,
                                     prob_n1, prob_k1, prob_n2, prob_k2);
  }
  CUDA_CHECK(cudaGetLastError());
}

template <typename MmaType>
void dispatch_fused_moe_full_typed(
    const void* hidden, const void* B1, const void* B2, void* output,
    const void* b_scales1, const void* b_scales2, const float* topk_weights,
    const int32_t* sorted_token_ids, const int32_t* expert_ids, int groups,
    int moe_block_size, int top_k, int prob_m, int prob_n1, int prob_k1,
    int prob_n2, int prob_k2, int group_size, int num_experts,
    cudaStream_t stream, double host_prep_ms) {
  const FullMoeTileConfig cfg = cutlass69_full_moe_tile_config();
  if (cutlass69_profile_enabled()) {
    fprintf(stderr,
            "[CUTLASS69 full MoE tile] GEMM1 tile=%d cluster=%d  "
            "GEMM2 tile=%d cluster=%d\n",
            cfg.gemm1_tile_n, cfg.gemm1_cluster_n, cfg.gemm2_tile_n,
            cfg.gemm2_cluster_n);
  }

#define DISPATCH_FULL_MOE_TILE(TN1, CN1, TN2, CN2)                           \
  if (cfg.gemm1_tile_n == (TN1) && cfg.gemm1_cluster_n == (CN1) &&           \
      cfg.gemm2_tile_n == (TN2) && cfg.gemm2_cluster_n == (CN2)) {           \
    dispatch_fused_moe_full_impl<MmaType, (TN1), (CN1), (TN2), (CN2)>(       \
        hidden, B1, B2, output, b_scales1, b_scales2, topk_weights,           \
        sorted_token_ids, expert_ids, groups, moe_block_size, top_k, prob_m,  \
        prob_n1, prob_k1, prob_n2, prob_k2, group_size, num_experts, stream,  \
        host_prep_ms);                                                         \
    return;                                                                  \
  }

  DISPATCH_FULL_MOE_TILE(256, 2, 256, 2);
  DISPATCH_FULL_MOE_TILE(256, 2, 256, 4);
  DISPATCH_FULL_MOE_TILE(256, 4, 256, 4);
  DISPATCH_FULL_MOE_TILE(256, 4, 256, 2);
  DISPATCH_FULL_MOE_TILE(128, 2, 256, 4);
  DISPATCH_FULL_MOE_TILE(128, 2, 256, 2);
  DISPATCH_FULL_MOE_TILE(128, 4, 256, 4);
  DISPATCH_FULL_MOE_TILE(128, 4, 256, 2);

#undef DISPATCH_FULL_MOE_TILE

  TORCH_CHECK(false,
              "Unsupported CUTLASS69 full MoE tile/cluster config: GEMM1 tile=",
              cfg.gemm1_tile_n, " cluster=", cfg.gemm1_cluster_n,
              " GEMM2 tile=", cfg.gemm2_tile_n,
              " cluster=", cfg.gemm2_cluster_n,
              ". Supported presets: 256x2/256x2, 256x2/256x4, 256x4/256x4, "
              "256x4/256x2, 128x2/256x4, 128x2/256x2, 128x4/256x4, 128x4/256x2");
}

void dispatch_marlin_moe_cutlass69_fused_moe_full(
    const void* hidden, const void* B1, const void* B2, void* output,
    const void* b_scales1, const void* b_scales2, const float* topk_weights,
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    const int32_t* num_tokens_past_padded, int moe_block_size, int num_experts,
    int top_k, int prob_m, int prob_n1, int prob_k1, int prob_n2, int prob_k2,
    vllm::ScalarType const& c_type, int group_size, int dev,
    cudaStream_t stream) {
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
  TORCH_CHECK(groups > 0, "CUTLASS69 fused MoE requires at least one block.");
  const auto host_prep_end = std::chrono::steady_clock::now();
  const double host_prep_ms =
      std::chrono::duration<double, std::milli>(host_prep_end - host_prep_begin)
          .count();

  if (c_type == vllm::kFloat16) {
    dispatch_fused_moe_full_typed<cutlass::half_t>(
        hidden, B1, B2, output, b_scales1, b_scales2, topk_weights,
        sorted_token_ids, expert_ids, groups, moe_block_size, top_k, prob_m,
        prob_n1, prob_k1, prob_n2, prob_k2, group_size, num_experts, stream,
        host_prep_ms);
    return;
  }
  if (c_type == vllm::kBFloat16) {
    dispatch_fused_moe_full_typed<cutlass::bfloat16_t>(
        hidden, B1, B2, output, b_scales1, b_scales2, topk_weights,
        sorted_token_ids, expert_ids, groups, moe_block_size, top_k, prob_m,
        prob_n1, prob_k1, prob_n2, prob_k2, group_size, num_experts, stream,
        host_prep_ms);
    return;
  }
  TORCH_CHECK(false, "Unsupported dtype for CUTLASS69 fused MoE full.");
#endif
}

void dispatch_marlin_moe_cutlass69_gemm2(
    const void* A, const void* B, void* C, const void* b_scales,
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    const int32_t* num_tokens_past_padded, const float* topk_weights,
    int moe_block_size, int num_experts, int top_k, bool mul_topk_weights,
    int prob_m, int prob_n, int prob_k, vllm::ScalarType const& a_type,
    vllm::ScalarType const& b_type, vllm::ScalarType const& c_type,
    int group_size, int dev, cudaStream_t stream) {
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
  TORCH_CHECK(groups > 0, "CUTLASS69 GEMM2 requires at least one MoE block.");
  const auto host_prep_end = std::chrono::steady_clock::now();
  const double host_prep_ms =
      std::chrono::duration<double, std::milli>(host_prep_end - host_prep_begin)
          .count();

  const bool use_bfull = cutlass69_bfull_enabled(prob_m);

  if (c_type == vllm::kFloat16) {
    if (use_bfull) {
      dispatch_gemm2_bfull_typed<cutlass::half_t, 256, 2, true>(
          A, B, C, b_scales, sorted_token_ids, expert_ids, groups,
          moe_block_size, top_k, prob_m, prob_n, prob_k, group_size,
          num_experts, topk_weights, mul_topk_weights, stream, host_prep_ms);
    } else {
      TORCH_CHECK(false, "CUTLASS69 GEMM2 requires B-full expert pack.");
    }
    return;
  }

  if (c_type == vllm::kBFloat16) {
    if (use_bfull) {
      dispatch_gemm2_bfull_typed<cutlass::bfloat16_t, 256, 2, true>(
          A, B, C, b_scales, sorted_token_ids, expert_ids, groups,
          moe_block_size, top_k, prob_m, prob_n, prob_k, group_size,
          num_experts, topk_weights, mul_topk_weights, stream, host_prep_ms);
    } else {
      TORCH_CHECK(false, "CUTLASS69 GEMM2 requires B-full expert pack.");
    }
    return;
  }

  TORCH_CHECK(false, "Unsupported activation dtype for CUTLASS69 GEMM2.");
#endif
}

#endif  // MARLIN_MOE_HAS_CUTLASS

}  // namespace marlin_moe_cutlass69_host
