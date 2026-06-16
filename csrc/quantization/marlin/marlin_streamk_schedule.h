#pragma once

#include <algorithm>

// Host-side Stream-K++ inspired schedule selection for Marlin's DP + split-K
// hybrid (see vLLM #24722 and Stream-K / Stream-K++ papers).  Maps the seven
// policy families to part2 MN tail sizing; full Bloom-filter tuning can replace
// select_streamk_policy() later.

namespace marlin_schedule {

inline int div_ceil(int a, int b) { return (a + b - 1) / b; }

enum StreamKPolicy : int {
  kDataParallel = 0,
  kDpOneTileSk = 1,
  kTwoTileSkDp = 2,
  kThreeTileSkDp = 3,
  kFourTileSkDp = 4,
  kFiveTileSkDp = 5,
  kSixTileSkDp = 6,
};

struct MarlinStreamKSchedule {
  int part2_mn_tiles;
  int part1_mn_iters;
  int slice_iters;
  StreamKPolicy policy;
};

inline int streamk_expand_threshold(StreamKPolicy policy) {
  switch (policy) {
    case kDataParallel:
      return 0;
    case kDpOneTileSk:
      return 1;
    case kTwoTileSkDp:
      return 3;
    case kThreeTileSkDp:
      return 4;
    case kFourTileSkDp:
      return 5;
    case kFiveTileSkDp:
      return 6;
    case kSixTileSkDp:
      return 7;
    default:
      return 3;
  }
}

inline StreamKPolicy select_streamk_policy(int global_mn_tiles, int k_tiles,
                                             int grid_dim) {
  if (global_mn_tiles <= grid_dim) {
    return kDataParallel;
  }

  const int tail = global_mn_tiles % grid_dim;
  const int mn_waves = div_ceil(global_mn_tiles, grid_dim);
  const int total_k_iters = k_tiles * global_mn_tiles;
  const int iters_per_block = div_ceil(total_k_iters, grid_dim);

  if (tail == 0 && mn_waves >= 2) {
    return kDpOneTileSk;
  }
  if (k_tiles >= 32 && mn_waves >= 4) {
    return kFourTileSkDp;
  }
  if (k_tiles >= 16 && mn_waves >= 3 && iters_per_block >= k_tiles / 2) {
    return kThreeTileSkDp;
  }
  if (k_tiles <= 4) {
    return kTwoTileSkDp;
  }
  if (mn_waves >= 2 && iters_per_block >= k_tiles) {
    return kFiveTileSkDp;
  }
  return kTwoTileSkDp;
}

inline MarlinStreamKSchedule compute_marlin_streamk_schedule(
    int global_mn_tiles, int k_tiles, int grid_dim, int group_blocks,
    int thread_k_blocks, bool has_act_order) {
  MarlinStreamKSchedule out{};
  out.part2_mn_tiles = global_mn_tiles;
  out.part1_mn_iters = 0;
  out.slice_iters = k_tiles;
  out.policy = kDataParallel;

  if (global_mn_tiles <= 0 || grid_dim <= 0 || k_tiles <= 0) {
    return out;
  }

  if (global_mn_tiles > grid_dim) {
    out.policy = select_streamk_policy(global_mn_tiles, k_tiles, grid_dim);
    const int thresh = streamk_expand_threshold(out.policy);
    out.part2_mn_tiles = global_mn_tiles % grid_dim;
    if (thresh > 0 && out.part2_mn_tiles * thresh <= grid_dim) {
      out.part2_mn_tiles += grid_dim;
    }
    out.part1_mn_iters = (global_mn_tiles - out.part2_mn_tiles) / grid_dim;
  }

  out.slice_iters = div_ceil(k_tiles * out.part2_mn_tiles, grid_dim);

  if (!has_act_order && group_blocks != -1 && group_blocks >= thread_k_blocks &&
      thread_k_blocks > 0) {
    const int gb = group_blocks / thread_k_blocks;
    if (gb > 0) {
      out.slice_iters = gb * div_ceil(out.slice_iters, gb);
    }
  }

  out.part2_mn_tiles = std::max(out.part2_mn_tiles, 0);
  out.part1_mn_iters = std::max(out.part1_mn_iters, 0);
  out.slice_iters = std::max(out.slice_iters, 1);
  return out;
}

}  // namespace marlin_schedule
