#pragma once

#include <cstdint>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>

namespace marlin_moe_host {

struct ClusterOccCacheEntry {
  const void* kernel = nullptr;
  int num_threads = 0;
  int max_shared_mem = 0;
  int max_cluster = 0;
  bool valid = false;
};

inline std::vector<ClusterOccCacheEntry>& cluster_occ_cache_entries() {
  static std::vector<ClusterOccCacheEntry> entries;
  return entries;
}

inline std::unordered_set<const void*>& cluster_attr_set_kernels() {
  static std::unordered_set<const void*> kernels;
  return kernels;
}

inline int query_max_cluster_size_cached(const void* kernel, int num_threads,
                                         int max_shared_mem) {
  if (kernel == nullptr) {
    return 0;
  }

  auto& entries = cluster_occ_cache_entries();
  for (const auto& entry : entries) {
    if (entry.valid && entry.kernel == kernel &&
        entry.num_threads == num_threads &&
        entry.max_shared_mem == max_shared_mem) {
      return entry.max_cluster;
    }
  }

  int max_cluster = 0;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
  cudaLaunchConfig_t occ_cfg{};
  occ_cfg.blockDim = num_threads;
  occ_cfg.dynamicSmemBytes = max_shared_mem;
  if (cudaOccupancyMaxPotentialClusterSize(
          &max_cluster, const_cast<void*>(kernel), &occ_cfg) != cudaSuccess) {
    max_cluster = 0;
  }
#endif

  entries.push_back(
      ClusterOccCacheEntry{kernel, num_threads, max_shared_mem, max_cluster,
                           true});
  return max_cluster;
}

inline void ensure_non_portable_cluster_attr_cached(const void* kernel) {
  if (kernel == nullptr) {
    return;
  }
  auto& kernels = cluster_attr_set_kernels();
  if (kernels.find(kernel) != kernels.end()) {
    return;
  }
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
  cudaFuncSetAttribute(const_cast<void*>(kernel),
                       cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
#endif
  kernels.insert(kernel);
}

}  // namespace marlin_moe_host
