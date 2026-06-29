// SPDX-License-Identifier: Apache-2.0
// Microbenchmark: cluster DSMEM Stream-K reduce vs atomic global reduce.
//
// Modes (via slices_per_tile + in_kernel_iters):
//   - slices_per_tile=2: one tile per 2 CTAs (cluster + atomic comparable)
//   - slices_per_tile>2: many CTAs reduce the same output tile (Stream-K style
//     atomic contention); cluster path is only run when slices_per_tile==2
//   - in_kernel_iters=true: amortize launch inside the kernel (closer to Marlin)

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/all.h>

#include <cooperative_groups.h>
#include <cooperative_groups/cluster.h>

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
        __nanosleep(delay);
        delay = delay < 8192 ? delay * 2 : 8192;
      }
    } while (state >= 0);
    atomicAdd(lock, 1);
  }
  __syncthreads();
}

template <int NumFloats>
__device__ __forceinline__ void load_partial_strided(const float* src,
                                                     float* frag) {
  constexpr int Vecs = NumFloats / 4;
  const float4* src4 = reinterpret_cast<const float4*>(src);
  float4* dst4 = reinterpret_cast<float4*>(frag);
  for (int v = threadIdx.x; v < Vecs; v += blockDim.x) {
    dst4[v] = src4[v];
  }
  __syncthreads();
}

template <int NumFloats>
__device__ __forceinline__ void store_result_strided(const float* frag,
                                                     float* dst) {
  constexpr int Vecs = NumFloats / 4;
  const float4* src4 = reinterpret_cast<const float4*>(src);
  float4* dst4 = reinterpret_cast<float4*>(dst);
  for (int v = threadIdx.x; v < Vecs; v += blockDim.x) {
    dst4[v] = src4[v];
  }
}

template <int NumFloats>
__device__ __forceinline__ void zero_output_strided(float* dst) {
  constexpr int Vecs = NumFloats / 4;
  float4* dst4 = reinterpret_cast<float4*>(dst);
  const float4 zero = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int v = threadIdx.x; v < Vecs; v += blockDim.x) {
    dst4[v] = zero;
  }
}

template <int NumFloats>
__device__ __forceinline__ void atomic_accumulate_strided(const float* frag,
                                                          float* dst) {
  constexpr int Vecs = NumFloats / 4;
  const float4* frag4 = reinterpret_cast<const float4*>(frag);
  float4* dst4 = reinterpret_cast<float4*>(dst);
  for (int v = threadIdx.x; v < Vecs; v += blockDim.x) {
    float4 val = frag4[v];
    atomicAdd(reinterpret_cast<float*>(&dst4[v].x), val.x);
    atomicAdd(reinterpret_cast<float*>(&dst4[v].y), val.y);
    atomicAdd(reinterpret_cast<float*>(&dst4[v].z), val.z);
    atomicAdd(reinterpret_cast<float*>(&dst4[v].w), val.w);
  }
}

// Marlin-style multi-slice atomic reduce to one output tile per lock.
template <int NumFloats, int NumThreads, int SlicesPerTile>
__global__ void atomic_streamk_reduce_bench_kernel(float* __restrict__ output,
                                                   const float* __restrict__ partials,
                                                   int* __restrict__ locks,
                                                   int num_tiles, int iters) {
  extern __shared__ float sh_frag[];
  const int tile_id = blockIdx.x / SlicesPerTile;
  const int slice_idx = blockIdx.x % SlicesPerTile;
  if (tile_id >= num_tiles) {
    return;
  }

  int* lock = locks + tile_id;
  float* out_base = output + tile_id * NumFloats;
  const float* partial_base =
      partials + (tile_id * SlicesPerTile + slice_idx) * NumFloats;

  for (int iter = 0; iter < iters; ++iter) {
    load_partial_strided<NumFloats>(partial_base, sh_frag);

    if (slice_idx == 0) {
      zero_output_strided<NumFloats>(out_base);
      __syncthreads();
      if (threadIdx.x == 0) {
        const int val = 1 - SlicesPerTile;
        asm volatile("st.global.release.gpu.b32 [%0], %1;\n" ::"l"(lock), "r"(val));
      }
    }

    __syncthreads();

    if (slice_idx != 0) {
      bench_wait_negative_and_add(lock);
    }

    atomic_accumulate_strided<NumFloats>(sh_frag, out_base);
    __syncthreads();
  }
}

template <int NumFloats, int NumThreads>
__global__ void cluster_streamk_reduce_bench_kernel(
    float* __restrict__ output, const float* __restrict__ partials, int num_tiles,
    int iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  extern __shared__ float sh_pool[];
  float* sh_frag = sh_pool;
  int4* sh_red = reinterpret_cast<int4*>(sh_pool + NumFloats);

  cooperative_groups::cluster_group cluster =
      cooperative_groups::this_cluster();
  constexpr int kSlicesPerTile = 2;
  const int tile_id = blockIdx.x / kSlicesPerTile;
  const int slice_idx = blockIdx.x % kSlicesPerTile;
  if (tile_id >= num_tiles) {
    return;
  }
  if (cluster.num_blocks() != kSlicesPerTile) {
    return;
  }

  const float* partial_base =
      partials + (tile_id * kSlicesPerTile + slice_idx) * NumFloats;

  for (int iter = 0; iter < iters; ++iter) {
    load_partial_strided<NumFloats>(partial_base, sh_frag);

    marlin_hopper::cluster_streamk_reduce<NumFloats>(
        sh_frag, sh_red, slice_idx, /*slice_count=*/kSlicesPerTile);

    if (slice_idx == 0) {
      store_result_strided<NumFloats>(sh_frag, output + tile_id * NumFloats);
    }
    __syncthreads();
  }
#endif
}

template <int NumFloats, int NumThreads, int SlicesPerTile>
int bench_smem_bytes_atomic() {
  return NumFloats * static_cast<int>(sizeof(float));
}

template <int NumFloats, int NumThreads>
int bench_smem_bytes_cluster() {
  return NumFloats * static_cast<int>(sizeof(float)) * 2;
}

template <int NumFloats, int NumThreads, int SlicesPerTile>
void launch_atomic_bench(float* output, const float* partials, int* locks,
                         int num_tiles, int iters, cudaStream_t stream) {
  const int blocks = num_tiles * SlicesPerTile;
  const int smem = bench_smem_bytes_atomic<NumFloats, NumThreads, SlicesPerTile>();
  atomic_streamk_reduce_bench_kernel<NumFloats, NumThreads, SlicesPerTile>
      <<<blocks, NumThreads, smem, stream>>>(output, partials, locks, num_tiles,
                                             iters);
}

template <int NumFloats, int NumThreads>
void launch_cluster_bench(float* output, const float* partials, int num_tiles,
                          int iters, cudaStream_t stream) {
  auto* kernel = &cluster_streamk_reduce_bench_kernel<NumFloats, NumThreads>;

  const int cluster_size = 2;
  const int blocks = num_tiles * cluster_size;
  const int smem = bench_smem_bytes_cluster<NumFloats, NumThreads>();

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
      &config, kernel, output, partials, num_tiles, iters));
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

constexpr int kDefaultChunkTiles = 8;

int chunk_tiles_for_launch(int num_tiles) {
  int chunk = kDefaultChunkTiles;
  if (const char* env = std::getenv("MARLIN_REDUCE_BENCH_CHUNK")) {
    chunk = std::max(1, std::atoi(env));
  }
  return std::min(chunk, num_tiles);
}

bool cluster_bench_enabled() {
  if (const char* env = std::getenv("MARLIN_REDUCE_BENCH_CLUSTER")) {
    return std::strcmp(env, "0") != 0;
  }
  return true;
}

template <int NumFloats, int NumThreads, int SlicesPerTile>
at::Tensor run_one_config(int num_tiles, int warmup_iters, int bench_iters,
                          bool run_verify, bool in_kernel_iters,
                          at::Device device) {
  constexpr int kClusterSlices = 2;
  const bool run_cluster =
      cluster_bench_enabled() && SlicesPerTile == kClusterSlices;

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);

  auto partials =
      torch::randn({num_tiles, SlicesPerTile, NumFloats}, options);
  auto output_atomic = torch::zeros({num_tiles, NumFloats}, options);
  auto output_cluster = torch::zeros({num_tiles, NumFloats}, options);
  auto locks = torch::zeros({num_tiles}, options.dtype(torch::kInt32));

  const float* partials_ptr = partials.data_ptr<float>();
  float* out_atomic_ptr = output_atomic.data_ptr<float>();
  float* out_cluster_ptr = output_cluster.data_ptr<float>();
  int* locks_ptr = locks.data_ptr<int>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int chunk_tiles = chunk_tiles_for_launch(num_tiles);

  auto launch_atomic_chunk = [&](int tile_off, int tiles_in_chunk, int iters) {
    launch_atomic_bench<NumFloats, NumThreads, SlicesPerTile>(
        out_atomic_ptr + tile_off * NumFloats,
        partials_ptr + tile_off * SlicesPerTile * NumFloats,
        locks_ptr + tile_off, tiles_in_chunk, iters, stream);
  };
  auto launch_cluster_chunk = [&](int tile_off, int tiles_in_chunk, int iters) {
    launch_cluster_bench<NumFloats, NumThreads>(
        out_cluster_ptr + tile_off * NumFloats,
        partials_ptr + tile_off * kClusterSlices * NumFloats, tiles_in_chunk,
        iters, stream);
  };

  auto launch_atomic = [&](int iters) {
    if (in_kernel_iters) {
      for (int off = 0; off < num_tiles; off += chunk_tiles) {
        const int n = std::min(chunk_tiles, num_tiles - off);
        launch_atomic_chunk(off, n, iters);
      }
    } else {
      for (int rep = 0; rep < iters; ++rep) {
        for (int off = 0; off < num_tiles; off += chunk_tiles) {
          const int n = std::min(chunk_tiles, num_tiles - off);
          launch_atomic_chunk(off, n, /*inner_iters=*/1);
        }
      }
    }
  };

  auto launch_cluster = [&](int iters) {
    if (!run_cluster) {
      return;
    }
    if (in_kernel_iters) {
      for (int off = 0; off < num_tiles; off += chunk_tiles) {
        const int n = std::min(chunk_tiles, num_tiles - off);
        launch_cluster_chunk(off, n, iters);
      }
    } else {
      for (int rep = 0; rep < iters; ++rep) {
        for (int off = 0; off < num_tiles; off += chunk_tiles) {
          const int n = std::min(chunk_tiles, num_tiles - off);
          launch_cluster_chunk(off, n, /*inner_iters=*/1);
        }
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

  // [num_floats, num_threads, atomic_ns/tile, cluster_ns/tile, max_abs_diff,
  //  num_tiles, atomic_ns/launch, slices_per_tile, in_kernel_iters]
  auto result = torch::empty({9}, torch::TensorOptions().dtype(torch::kFloat32));
  float* r = result.data_ptr<float>();
  r[0] = static_cast<float>(NumFloats);
  r[1] = static_cast<float>(NumThreads);
  r[2] = static_cast<float>(atomic_ns / static_cast<double>(num_tiles));
  r[3] = static_cast<float>(cluster_ns / static_cast<double>(num_tiles));
  r[4] = static_cast<float>(max_abs_diff);
  r[5] = static_cast<float>(num_tiles);
  r[6] = static_cast<float>(atomic_ns);
  r[7] = static_cast<float>(SlicesPerTile);
  r[8] = in_kernel_iters ? 1.f : 0.f;
  return result;
}

template <int NumFloats, int SlicesPerTile>
at::Tensor dispatch_threads(int num_threads, int num_tiles, int warmup_iters,
                            int bench_iters, bool run_verify,
                            bool in_kernel_iters, at::Device device) {
  if (num_threads == 128) {
    return run_one_config<NumFloats, 128, SlicesPerTile>(
        num_tiles, warmup_iters, bench_iters, run_verify, in_kernel_iters,
        device);
  }
  if (num_threads == 256) {
    return run_one_config<NumFloats, 256, SlicesPerTile>(
        num_tiles, warmup_iters, bench_iters, run_verify, in_kernel_iters,
        device);
  }
  TORCH_CHECK(false, "Unsupported num_threads=", num_threads,
              " (supported: 128, 256)");
}

template <int NumFloats>
at::Tensor dispatch_slices(int slices_per_tile, int num_threads, int num_tiles,
                           int warmup_iters, int bench_iters, bool run_verify,
                           bool in_kernel_iters, at::Device device) {
  if (slices_per_tile == 2) {
    return dispatch_threads<NumFloats, 2>(num_threads, num_tiles, warmup_iters,
                                          bench_iters, run_verify,
                                          in_kernel_iters, device);
  }
  if (slices_per_tile == 4) {
    return dispatch_threads<NumFloats, 4>(num_threads, num_tiles, warmup_iters,
                                          bench_iters, run_verify,
                                          in_kernel_iters, device);
  }
  if (slices_per_tile == 8) {
    return dispatch_threads<NumFloats, 8>(num_threads, num_tiles, warmup_iters,
                                          bench_iters, run_verify,
                                          in_kernel_iters, device);
  }
  if (slices_per_tile == 16) {
    return dispatch_threads<NumFloats, 16>(num_threads, num_tiles, warmup_iters,
                                           bench_iters, run_verify,
                                           in_kernel_iters, device);
  }
  TORCH_CHECK(false, "Unsupported slices_per_tile=", slices_per_tile,
              " (supported: 2, 4, 8, 16)");
}

at::Tensor dispatch_floats(int num_floats, int num_threads, int num_tiles,
                           int slices_per_tile, int warmup_iters,
                           int bench_iters, bool run_verify,
                           bool in_kernel_iters, at::Device device) {
  if (num_floats == 16) {
    return dispatch_slices<16>(slices_per_tile, num_threads, num_tiles,
                               warmup_iters, bench_iters, run_verify,
                               in_kernel_iters, device);
  }
  if (num_floats == 32) {
    return dispatch_slices<32>(slices_per_tile, num_threads, num_tiles,
                               warmup_iters, bench_iters, run_verify,
                               in_kernel_iters, device);
  }
  if (num_floats == 64) {
    return dispatch_slices<64>(slices_per_tile, num_threads, num_tiles,
                               warmup_iters, bench_iters, run_verify,
                               in_kernel_iters, device);
  }
  if (num_floats == 128) {
    return dispatch_slices<128>(slices_per_tile, num_threads, num_tiles,
                                warmup_iters, bench_iters, run_verify,
                                in_kernel_iters, device);
  }
  if (num_floats == 256) {
    return dispatch_slices<256>(slices_per_tile, num_threads, num_tiles,
                                warmup_iters, bench_iters, run_verify,
                                in_kernel_iters, device);
  }
  if (num_floats == 1024) {
    return dispatch_slices<1024>(slices_per_tile, num_threads, num_tiles,
                                 warmup_iters, bench_iters, run_verify,
                                 in_kernel_iters, device);
  }
  TORCH_CHECK(false, "Unsupported num_floats=", num_floats,
              " (supported: 16, 32, 64, 128, 256, 1024)");
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
                                    int64_t num_tiles, int64_t warmup_iters,
                                    int64_t bench_iters, bool run_verify,
                                    int64_t slices_per_tile,
                                    bool in_kernel_iters) {
  TORCH_CHECK(device_guard.is_cuda(),
              "benchmark_streamk_reduce requires a CUDA device_guard tensor");
  TORCH_CHECK(torch::cuda::is_available(), "CUDA is required");
  at::Device device = device_guard.device();

  TORCH_CHECK(num_tiles > 0, "num_tiles must be positive");
  TORCH_CHECK(warmup_iters >= 0, "warmup_iters must be non-negative");
  TORCH_CHECK(bench_iters > 0, "bench_iters must be positive");
  TORCH_CHECK(slices_per_tile >= 2, "slices_per_tile must be >= 2");

  if (device_major_capability(device) < 9) {
    TORCH_CHECK(false,
                "benchmark_streamk_reduce requires SM90+ for cluster path");
  }

  const at::cuda::OptionalCUDAGuard cuda_guard(device);
  return dispatch_floats(
      static_cast<int>(num_floats), static_cast<int>(num_threads),
      static_cast<int>(num_tiles), static_cast<int>(slices_per_tile),
      static_cast<int>(warmup_iters), static_cast<int>(bench_iters),
      run_verify, in_kernel_iters, device);
}
