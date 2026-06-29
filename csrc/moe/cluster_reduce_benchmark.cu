// SPDX-License-Identifier: Apache-2.0
// Microbenchmark: cluster DSMEM Stream-K reduce vs atomic global reduce.

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/all.h>

#include <cooperative_groups.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "core/registration.h"
#include "moe/marlin_moe_wna16/marlin_hopper.cuh"

namespace {

__device__ __forceinline__ void bench_wait_negative_and_add(int* lock) {
  if (threadIdx.x == 0) {
    int state = 0;
    int delay = 32;
    do {
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
      if (state >= 0) {
        // Back off so follower CTAs do not monopolize SMs and starve the
        // leader CTA that must publish lock = -1 (otherwise the grid hangs).
        __nanosleep(delay);
        delay = delay < 8192 ? delay * 2 : 8192;
      }
    } while (state >= 0);
    atomicAdd(lock, 1);
  }
  __syncthreads();
}

template <int NumFloats, int NumThreads>
__device__ __forceinline__ void load_partial(const float* partials, int pair_id,
                                             int slice_idx, float* frag_c) {
  const float* src = partials + (pair_id * 2 + slice_idx) * (NumFloats * NumThreads);
  constexpr int Vecs = NumFloats / 4;
  float4* dst4 = reinterpret_cast<float4*>(frag_c);
  const float4* src4 = reinterpret_cast<const float4*>(src);
  for (int i = 0; i < Vecs; ++i) {
    dst4[i] = src4[i * NumThreads + threadIdx.x];
  }
  __syncthreads();
}

template <int NumFloats, int NumThreads>
__device__ __forceinline__ void store_result(float* output, int pair_id,
                                             const float* frag_c) {
  float* dst = output + pair_id * (NumFloats * NumThreads);
  constexpr int Vecs = NumFloats / 4;
  float4* dst4 = reinterpret_cast<float4*>(dst);
  const float4* src4 = reinterpret_cast<const float4*>(frag_c);
  for (int i = 0; i < Vecs; ++i) {
    dst4[i * NumThreads + threadIdx.x] = src4[i];
  }
}

// Mirrors Marlin MoE atomic Stream-K tail: leader zeros output + sets lock,
// follower spins, both atomicAdd partial tiles into global output.
template <int NumFloats, int NumThreads>
__global__ void atomic_streamk_reduce_bench_kernel(float* __restrict__ output,
                                                   const float* __restrict__ partials,
                                                   int* __restrict__ locks,
                                                   int num_pairs, int iters) {
  const int pair_id = blockIdx.x / 2;
  const int slice_idx = blockIdx.x % 2;
  if (pair_id >= num_pairs) {
    return;
  }

  float frag_c[NumFloats];
  int* lock = locks + pair_id;
  float* out_base = output + pair_id * (NumFloats * NumThreads);

  for (int iter = 0; iter < iters; ++iter) {
    load_partial<NumFloats, NumThreads>(partials, pair_id, slice_idx, frag_c);

    if (slice_idx == 0) {
      constexpr int Vecs = NumFloats / 4;
      float4* out4 = reinterpret_cast<float4*>(out_base);
      const float4 zero = make_float4(0.f, 0.f, 0.f, 0.f);
      for (int i = 0; i < Vecs; ++i) {
        out4[i * NumThreads + threadIdx.x] = zero;
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        const int val = -1;
        asm volatile("st.global.release.gpu.b32 [%0], %1;\n" ::"l"(lock), "r"(val));
      }
    }

    __syncthreads();

    if (slice_idx != 0) {
      bench_wait_negative_and_add(lock);
    }

    constexpr int Vecs = NumFloats / 4;
    float4* frag4 = reinterpret_cast<float4*>(frag_c);
    float4* out4 = reinterpret_cast<float4*>(out_base);
    for (int i = 0; i < Vecs; ++i) {
      float4 v = frag4[i];
      atomicAdd(reinterpret_cast<float*>(&out4[i * NumThreads + threadIdx.x].x), v.x);
      atomicAdd(reinterpret_cast<float*>(&out4[i * NumThreads + threadIdx.x].y), v.y);
      atomicAdd(reinterpret_cast<float*>(&out4[i * NumThreads + threadIdx.x].z), v.z);
      atomicAdd(reinterpret_cast<float*>(&out4[i * NumThreads + threadIdx.x].w), v.w);
    }
    __syncthreads();
  }
}

template <int NumFloats, int NumThreads>
__global__ void cluster_streamk_reduce_bench_kernel(
    float* __restrict__ output, const float* __restrict__ partials, int num_pairs,
    int iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  extern __shared__ int4 sh_raw[];
  int4* sh_red = sh_raw;

  cooperative_groups::cluster_group cluster =
      cooperative_groups::this_cluster();
  // Match atomic kernel indexing: clusterDim.x == 2 groups (0,1), (2,3), ...
  const int pair_id = blockIdx.x / 2;
  const int slice_idx = blockIdx.x % 2;
  if (pair_id >= num_pairs) {
    return;
  }
  // Avoid cluster.sync() deadlock when the launch did not form 2-CTA clusters.
  if (cluster.num_blocks() != 2) {
    return;
  }

  float frag_c[NumFloats];

  for (int iter = 0; iter < iters; ++iter) {
    load_partial<NumFloats, NumThreads>(partials, pair_id, slice_idx, frag_c);

    marlin_hopper::cluster_streamk_reduce<NumFloats>(
        frag_c, sh_red, slice_idx, /*slice_count=*/2);

    if (slice_idx == 0) {
      store_result<NumFloats, NumThreads>(output, pair_id, frag_c);
    }
    __syncthreads();
  }
#endif
}

template <int NumFloats, int NumThreads>
void launch_atomic_bench(float* output, const float* partials, int* locks,
                         int num_pairs, int iters, cudaStream_t stream) {
  const int blocks = num_pairs * 2;
  atomic_streamk_reduce_bench_kernel<NumFloats, NumThreads>
      <<<blocks, NumThreads, 0, stream>>>(output, partials, locks, num_pairs,
                                             iters);
}

template <int NumFloats, int NumThreads>
void launch_cluster_bench(float* output, const float* partials, int num_pairs,
                          int iters, cudaStream_t stream) {
  auto* kernel = &cluster_streamk_reduce_bench_kernel<NumFloats, NumThreads>;

  const int cluster_size = 2;
  const int blocks = num_pairs * cluster_size;
  const int smem = NumFloats * NumThreads * sizeof(float);

  cudaLaunchConfig_t config{};
  config.gridDim = blocks;
  config.blockDim = NumThreads;
  config.dynamicSmemBytes = smem;
  config.stream = stream;

  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = cluster_size;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;
  config.attrs = &attr;
  config.numAttrs = 1;

  C10_CUDA_CHECK(cudaFuncSetAttribute(
      (cluster_streamk_reduce_bench_kernel<NumFloats, NumThreads>),
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      (cluster_streamk_reduce_bench_kernel<NumFloats, NumThreads>),
      cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
#endif

  C10_CUDA_CHECK(cudaLaunchKernelEx(
      &config, kernel, output, partials, num_pairs, iters));
}

template <typename LaunchFn>
double time_kernel(LaunchFn launch, int warmup_iters, int bench_iters,
                   cudaStream_t stream) {
  cudaEvent_t start{};
  cudaEvent_t stop{};
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < warmup_iters; ++i) {
    launch(1);
  }
  cudaStreamSynchronize(stream);

  cudaEventRecord(start, stream);
  launch(bench_iters);
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);

  float ms = 0.f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return static_cast<double>(ms) * 1e6 / static_cast<double>(bench_iters);
}

constexpr int kDefaultChunkPairs = 8;

int chunk_pairs_for_launch(int num_pairs) {
  int chunk = kDefaultChunkPairs;
  if (const char* env = std::getenv("MARLIN_REDUCE_BENCH_CHUNK")) {
    chunk = std::max(1, std::atoi(env));
  }
  return std::min(chunk, num_pairs);
}

bool cluster_bench_enabled() {
  if (const char* env = std::getenv("MARLIN_REDUCE_BENCH_CLUSTER")) {
    return std::strcmp(env, "0") != 0;
  }
  return true;
}

template <int NumFloats, int NumThreads>
at::Tensor run_one_config(int num_pairs, int warmup_iters, int bench_iters,
                            bool run_verify, at::Device device) {
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);

  auto partials = torch::randn({num_pairs, 2, NumFloats * NumThreads}, options);
  auto output_atomic = torch::zeros({num_pairs, NumFloats * NumThreads}, options);
  auto output_cluster = torch::zeros({num_pairs, NumFloats * NumThreads}, options);
  auto locks = torch::zeros({num_pairs}, options.dtype(torch::kInt32));

  const float* partials_ptr = partials.data_ptr<float>();
  float* out_atomic_ptr = output_atomic.data_ptr<float>();
  float* out_cluster_ptr = output_cluster.data_ptr<float>();
  int* locks_ptr = locks.data_ptr<int>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int chunk_pairs = chunk_pairs_for_launch(num_pairs);
  const bool run_cluster = cluster_bench_enabled();

  auto launch_atomic_chunk = [&](int pair_off, int pairs_in_chunk, int iters) {
    launch_atomic_bench<NumFloats, NumThreads>(
        out_atomic_ptr + pair_off * (NumFloats * NumThreads),
        partials_ptr + pair_off * 2 * (NumFloats * NumThreads), locks_ptr + pair_off,
        pairs_in_chunk, iters, stream);
  };
  auto launch_cluster_chunk = [&](int pair_off, int pairs_in_chunk, int iters) {
    launch_cluster_bench<NumFloats, NumThreads>(
        out_cluster_ptr + pair_off * (NumFloats * NumThreads),
        partials_ptr + pair_off * 2 * (NumFloats * NumThreads), pairs_in_chunk, iters, stream);
  };

  // One reduce per launch keeps concurrent follower spinners bounded (per chunk)
  // and matches the cluster launch model for apples-to-apples timing.
  auto launch_atomic = [&](int iters) {
    for (int rep = 0; rep < iters; ++rep) {
      for (int off = 0; off < num_pairs; off += chunk_pairs) {
        const int n = std::min(chunk_pairs, num_pairs - off);
        launch_atomic_chunk(off, n, /*iters=*/1);
      }
    }
  };
  auto launch_cluster = [&](int iters) {
    if (!run_cluster) {
      return;
    }
    for (int rep = 0; rep < iters; ++rep) {
      for (int off = 0; off < num_pairs; off += chunk_pairs) {
        const int n = std::min(chunk_pairs, num_pairs - off);
        launch_cluster_chunk(off, n, /*iters=*/1);
      }
    }
  };

  double max_abs_diff = 0.0;
  if (run_verify) {
    output_atomic.zero_();
    output_cluster.zero_();
    locks.zero_();
    launch_atomic(1);
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (run_cluster) {
      launch_cluster(1);
      C10_CUDA_CHECK(cudaStreamSynchronize(stream));
      max_abs_diff =
          (output_cluster - output_atomic).abs().max().item<double>();
    }

    output_atomic.zero_();
    output_cluster.zero_();
    locks.zero_();
  }

  const double atomic_ns = time_kernel(launch_atomic, warmup_iters, bench_iters,
                                       stream);

  double cluster_ns = 0.0;
  if (run_cluster) {
    cluster_ns =
        time_kernel(launch_cluster, warmup_iters, bench_iters, stream);
  }

  C10_CUDA_CHECK(cudaGetLastError());

  // [num_floats, num_threads, atomic_ns_per_pair, cluster_ns_per_pair,
  //  max_abs_diff, num_pairs, atomic_ns_per_launch]
  auto result = torch::empty({7}, torch::TensorOptions().dtype(torch::kFloat32));
  float* r = result.data_ptr<float>();
  r[0] = static_cast<float>(NumFloats);
  r[1] = static_cast<float>(NumThreads);
  r[2] = static_cast<float>(atomic_ns / static_cast<double>(num_pairs));
  r[3] = static_cast<float>(cluster_ns / static_cast<double>(num_pairs));
  r[4] = static_cast<float>(max_abs_diff);
  r[5] = static_cast<float>(num_pairs);
  r[6] = static_cast<float>(atomic_ns);
  return result;
}

template <int NumFloats>
at::Tensor dispatch_threads(int num_threads, int num_pairs, int warmup_iters,
                            int bench_iters, bool run_verify,
                            at::Device device) {
  if (num_threads == 128) {
    return run_one_config<NumFloats, 128>(num_pairs, warmup_iters, bench_iters,
                                          run_verify, device);
  }
  if (num_threads == 256) {
    return run_one_config<NumFloats, 256>(num_pairs, warmup_iters, bench_iters,
                                          run_verify, device);
  }
  TORCH_CHECK(false, "Unsupported num_threads=", num_threads,
              " (supported: 128, 256)");
}

at::Tensor dispatch_floats(int num_floats, int num_threads, int num_pairs,
                           int warmup_iters, int bench_iters, bool run_verify,
                           at::Device device) {
  if (num_floats == 16) {
    return dispatch_threads<16>(num_threads, num_pairs, warmup_iters,
                                bench_iters, run_verify, device);
  }
  if (num_floats == 32) {
    return dispatch_threads<32>(num_threads, num_pairs, warmup_iters,
                                bench_iters, run_verify, device);
  }
  if (num_floats == 64) {
    return dispatch_threads<64>(num_threads, num_pairs, warmup_iters,
                                bench_iters, run_verify, device);
  }
  if (num_floats == 512) {
    return dispatch_threads<512>(num_threads, num_pairs, warmup_iters,
                               bench_iters, run_verify, device);
  }
  TORCH_CHECK(false, "Unsupported num_floats=", num_floats,
              " (supported: 16, 32, 64, 512)");
}

int device_major_capability(at::Device device) {
  int major = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                         device.index());
  return major;
}

}  // namespace

at::Tensor benchmark_streamk_reduce(const at::Tensor& device_guard,
                                    int64_t num_floats, int64_t num_threads,
                                    int64_t num_pairs, int64_t warmup_iters,
                                    int64_t bench_iters, bool run_verify) {
  TORCH_CHECK(device_guard.is_cuda(),
              "benchmark_streamk_reduce requires a CUDA device_guard tensor");
  TORCH_CHECK(torch::cuda::is_available(), "CUDA is required");
  at::Device device = device_guard.device();

  TORCH_CHECK(num_pairs > 0, "num_pairs must be positive");
  TORCH_CHECK(warmup_iters >= 0, "warmup_iters must be non-negative");
  TORCH_CHECK(bench_iters > 0, "bench_iters must be positive");

  if (device_major_capability(device) < 9) {
    TORCH_CHECK(false,
                "benchmark_streamk_reduce requires SM90+ for cluster path");
  }

  const at::cuda::OptionalCUDAGuard cuda_guard(device);
  return dispatch_floats(static_cast<int>(num_floats),
                         static_cast<int>(num_threads),
                         static_cast<int>(num_pairs),
                         static_cast<int>(warmup_iters),
                         static_cast<int>(bench_iters), run_verify, device);
}
