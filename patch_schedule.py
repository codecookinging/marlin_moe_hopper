import re

with open('csrc/quantization/marlin/marlin_streamk_schedule.h', 'r') as f:
    content = f.read()

old_code = """inline StreamKPolicy select_streamk_policy(int global_mn_tiles, int k_tiles,
                                             int grid_dim) {
  return opensieve::select_streamk_policy(global_mn_tiles, k_tiles, grid_dim);
}"""

new_code = """inline StreamKPolicy select_streamk_policy(int global_mn_tiles, int k_tiles,
                                             int grid_dim) {
  // Reverted to start branch behavior: default to DP + two-tile StreamK
  // instead of using the opensieve LUT which is optimized for Dense GEMM.
  return kTwoTileSkDp;
}"""

content = content.replace(old_code, new_code)

with open('csrc/quantization/marlin/marlin_streamk_schedule.h', 'w') as f:
    f.write(content)

