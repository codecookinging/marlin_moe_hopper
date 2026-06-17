#pragma once

#include <cstdint>

// Open-sieve offline policy selection (Stream-K++).  Generated tables live in
// generated/marlin_streamk_{lut,bloom}.inc — regenerate with:
//   python tools/generate_streamk_opensieve.py

namespace marlin_schedule {

enum StreamKPolicy : int {
  kDataParallel = 0,
  kDpOneTileSk = 1,
  kTwoTileSkDp = 2,
  kThreeTileSkDp = 3,
  kFourTileSkDp = 4,
  kFiveTileSkDp = 5,
  kSixTileSkDp = 6,
};

namespace opensieve {

// Bucket bounds — keep in sync with tools/generate_streamk_opensieve.py
constexpr int kMnBounds[] = {1, 4, 16, 64, 256, 1024, 4096};
constexpr int kKBounds[] = {1, 4, 16, 64, 256, 1024};
constexpr int kGridBounds[] = {64, 132, 264, 528, 2048};
constexpr int kMnBucketCount = sizeof(kMnBounds) / sizeof(kMnBounds[0]) - 1;
constexpr int kKBucketCount = sizeof(kKBounds) / sizeof(kKBounds[0]) - 1;
constexpr int kGridBucketCount =
    sizeof(kGridBounds) / sizeof(kGridBounds[0]) - 1;

#include "generated/marlin_streamk_lut.inc"
#include "generated/marlin_streamk_bloom.inc"

inline int bucket_index(int value, const int* bounds, int bound_count) {
  int idx = 0;
  while (idx < bound_count - 1 && value > bounds[idx]) {
    ++idx;
  }
  if (idx >= bound_count - 1) {
    return bound_count - 2;
  }
  return idx;
}

inline uint32_t make_bucket_key(int global_mn_tiles, int k_tiles, int grid_dim) {
  const int mn = global_mn_tiles > 0 ? global_mn_tiles : 1;
  const int k = k_tiles > 0 ? k_tiles : 1;
  const int g = grid_dim > 0 ? grid_dim : 1;
  const int mn_b = bucket_index(mn, kMnBounds, kMnBucketCount + 1);
  const int k_b = bucket_index(k, kKBounds, kKBucketCount + 1);
  const int g_b = bucket_index(g, kGridBounds, kGridBucketCount + 1);
  return (static_cast<uint32_t>(mn_b) << 16) |
         (static_cast<uint32_t>(k_b) << 8) |
         static_cast<uint32_t>(g_b);
}

inline int lut_index(int global_mn_tiles, int k_tiles, int grid_dim) {
  const int mn_b =
      bucket_index(global_mn_tiles > 0 ? global_mn_tiles : 1, kMnBounds,
                   kMnBucketCount + 1);
  const int k_b = bucket_index(k_tiles > 0 ? k_tiles : 1, kKBounds,
                               kKBucketCount + 1);
  const int g_b = bucket_index(grid_dim > 0 ? grid_dim : 1, kGridBounds,
                               kGridBucketCount + 1);
  return mn_b * kKBucketCount * kGridBucketCount + k_b * kGridBucketCount + g_b;
}

inline uint32_t streamk_hash(uint32_t key, uint32_t seed) {
  key ^= seed;
  key *= 0xCC9E2D51u;
  key ^= key >> 16;
  key *= 0x1B873593u;
  key ^= key >> 13;
  return key;
}

inline bool bloom_might_contain(const uint8_t* bits, uint32_t key,
                                uint32_t seed) {
  for (int i = 0; i < kStreamKBloomHashes; ++i) {
    const uint32_t h =
        streamk_hash(key, seed + static_cast<uint32_t>(i) * 0x9E3779B9u) %
        static_cast<uint32_t>(kStreamKBloomBits);
    if ((bits[h / 8] & (1u << (h % 8))) == 0) {
      return false;
    }
  }
  return true;
}

inline StreamKPolicy lut_policy(int global_mn_tiles, int k_tiles, int grid_dim) {
  const int idx = lut_index(global_mn_tiles, k_tiles, grid_dim);
  if (idx < 0 || idx >= kStreamKLutSize) {
    return kTwoTileSkDp;
  }
  const int raw = static_cast<int>(kStreamKLut[idx]);
  if (raw < kDataParallel || raw > kSixTileSkDp) {
    return kTwoTileSkDp;
  }
  return static_cast<StreamKPolicy>(raw);
}

inline StreamKPolicy select_streamk_policy(int global_mn_tiles, int k_tiles,
                                             int grid_dim) {
  if (global_mn_tiles <= grid_dim) {
    return kDataParallel;
  }

  const uint32_t key = make_bucket_key(global_mn_tiles, k_tiles, grid_dim);
  const StreamKPolicy lut = lut_policy(global_mn_tiles, k_tiles, grid_dim);

  int candidate_count = 0;
  StreamKPolicy first_candidate = lut;
  for (int p = 0; p < kStreamKBloomPolicyCount; ++p) {
    if (!bloom_might_contain(kStreamKBloomFilters[p], key,
                             kStreamKBloomSeeds[p])) {
      continue;
    }
    if (candidate_count == 0) {
      first_candidate = static_cast<StreamKPolicy>(p);
    }
    ++candidate_count;
    if (static_cast<StreamKPolicy>(p) == lut) {
      return lut;
    }
  }

  if (candidate_count == 1) {
    return first_candidate;
  }
  return lut;
}

}  // namespace opensieve
}  // namespace marlin_schedule
