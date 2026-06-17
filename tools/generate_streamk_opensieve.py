#!/usr/bin/env python3
"""Generate Stream-K++ Open-sieve LUT and Bloom filters for Marlin scheduling.

Runs an analytical shape sweep (proxy for on-GPU benchmark) over Marlin schedule
inputs (global_mn_tiles, k_tiles, grid_dim), buckets the winners, and emits C++
include files consumed by marlin_streamk_opensieve.h.

Re-run on H100 after wiring optional CUDA timings to refresh the table:

    python tools/generate_streamk_opensieve.py
    python tools/generate_streamk_opensieve.py --output-dir csrc/quantization/marlin/generated
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

# Keep in sync with marlin_streamk_opensieve.h
MN_BOUNDS = (1, 4, 16, 64, 256, 1024, 4096)
K_BOUNDS = (1, 4, 16, 64, 256, 1024)
GRID_BOUNDS = (64, 132, 264, 528, 2048)

NUM_POLICIES = 7
BLOOM_BITS = 512
BLOOM_HASHES = 3

POLICY_THRESHOLDS = (0, 1, 3, 4, 5, 6, 7)


def div_ceil(a: int, b: int) -> int:
    return (a + b - 1) // b if b > 0 else 0


def bucket_index(value: int, bounds: tuple[int, ...]) -> int:
    idx = 0
    while idx < len(bounds) - 1 and value > bounds[idx]:
        idx += 1
    return min(idx, len(bounds) - 2)


def bucket_key(global_mn: int, k_tiles: int, grid: int) -> int:
    mn_b = bucket_index(max(global_mn, 1), MN_BOUNDS)
    k_b = bucket_index(max(k_tiles, 1), K_BOUNDS)
    g_b = bucket_index(max(grid, 1), GRID_BOUNDS)
    return (mn_b << 16) | (k_b << 8) | g_b


def lut_index(global_mn: int, k_tiles: int, grid: int) -> int:
    mn_b = bucket_index(max(global_mn, 1), MN_BOUNDS)
    k_b = bucket_index(max(k_tiles, 1), K_BOUNDS)
    g_b = bucket_index(max(grid, 1), GRID_BOUNDS)
    k_buckets = len(K_BOUNDS) - 1
    g_buckets = len(GRID_BOUNDS) - 1
    return mn_b * k_buckets * g_buckets + k_b * g_buckets + g_b


def lut_size() -> int:
    return (len(MN_BOUNDS) - 1) * (len(K_BOUNDS) - 1) * (len(GRID_BOUNDS) - 1)


def marlin_schedule(policy: int, global_mn: int, k_tiles: int, grid: int) -> tuple[int, int, int]:
    if global_mn <= 0 or k_tiles <= 0 or grid <= 0:
        return global_mn, 0, max(k_tiles, 1)
    if global_mn <= grid:
        return global_mn, 0, div_ceil(k_tiles * global_mn, grid)

    thresh = POLICY_THRESHOLDS[policy]
    part2 = global_mn % grid
    if thresh > 0 and part2 * thresh <= grid:
        part2 += grid
    part1 = (global_mn - part2) // grid
    iters = div_ceil(k_tiles * part2, grid)
    return part2, part1, iters


def pick_best_policy(global_mn: int, k_tiles: int, grid: int) -> int:
    """Analytical oracle for offline tuning (replace with GPU timings on H100)."""
    if global_mn <= grid:
        return 0

    tail = global_mn % grid
    mn_waves = div_ceil(global_mn, grid)
    iters_per_block = div_ceil(k_tiles * global_mn, grid)

    if tail == 0 and mn_waves >= 2:
        return 1
    if k_tiles >= 64 and mn_waves >= 4:
        return 4
    if k_tiles >= 32 and mn_waves >= 3 and iters_per_block >= max(k_tiles // 2, 1):
        return 3
    if k_tiles <= 4:
        return 2
    if mn_waves >= 2 and iters_per_block >= k_tiles:
        return 5
    if mn_waves >= 3 and k_tiles >= 16:
        return 6
    return 2


def iter_shape_sweep() -> list[tuple[int, int, int]]:
    mn_values = [1, 2, 3, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048]
    k_values = [1, 2, 3, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
    grid_values = [64, 108, 132, 216, 264, 528]
    shapes: list[tuple[int, int, int]] = []
    for mn in mn_values:
        for k in k_values:
            for grid in grid_values:
                shapes.append((mn, k, grid))
    return shapes


def build_lut() -> tuple[list[int], dict[int, set[int]]]:
    """Return LUT entries and policy->bucket_keys for Bloom insertion."""
    default_policy = 2  # kTwoTileSkDp
    lut = [default_policy] * lut_size()
    bucket_votes: list[Counter[int]] = [Counter() for _ in range(lut_size())]
    policy_keys: dict[int, set[int]] = {p: set() for p in range(NUM_POLICIES)}

    for global_mn, k_tiles, grid in iter_shape_sweep():
        policy = pick_best_policy(global_mn, k_tiles, grid)
        idx = lut_index(global_mn, k_tiles, grid)
        key = bucket_key(global_mn, k_tiles, grid)
        if global_mn > grid:
            bucket_votes[idx][policy] += 1
        policy_keys[policy].add(key)

    for idx, votes in enumerate(bucket_votes):
        if votes:
            lut[idx] = votes.most_common(1)[0][0]

    for idx, policy in enumerate(lut):
        mn_b = idx // ((len(K_BOUNDS) - 1) * (len(GRID_BOUNDS) - 1))
        rem = idx % ((len(K_BOUNDS) - 1) * (len(GRID_BOUNDS) - 1))
        k_b = rem // (len(GRID_BOUNDS) - 1)
        g_b = rem % (len(GRID_BOUNDS) - 1)
        key = (mn_b << 16) | (k_b << 8) | g_b
        policy_keys[policy].add(key)

    return lut, policy_keys


def streamk_hash(key: int, seed: int) -> int:
    key &= 0xFFFFFFFF
    key ^= seed & 0xFFFFFFFF
    key = (key * 0xCC9E2D51) & 0xFFFFFFFF
    key ^= key >> 16
    key = (key * 0x1B873593) & 0xFFFFFFFF
    key ^= key >> 13
    return key


def bloom_insert(bits: bytearray, key: int, seed: int) -> None:
    for i in range(BLOOM_HASHES):
        h = streamk_hash(key, seed + i * 0x9E3779B9) % BLOOM_BITS
        bits[h // 8] |= 1 << (h % 8)


def build_bloom_filters(policy_keys: dict[int, set[int]]) -> list[bytearray]:
    filters: list[bytearray] = []
    for policy in range(NUM_POLICIES):
        bits = bytearray(BLOOM_BITS // 8)
        seed = 0xF1EA5EED + policy * 0x85EBCA77
        for key in policy_keys[policy]:
            bloom_insert(bits, key, seed)
        filters.append(bits)
    return filters


def format_byte_array(data: bytes, name: str, per_line: int = 16) -> str:
    lines = [f"static const uint8_t {name}[] = {{"]
    for i in range(0, len(data), per_line):
        chunk = ", ".join(f"0x{b:02x}" for b in data[i : i + per_line])
        lines.append(f"  {chunk},")
    lines.append("};")
    return "\n".join(lines)


def emit_files(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lut, policy_keys = build_lut()
    blooms = build_bloom_filters(policy_keys)

    lut_path = output_dir / "marlin_streamk_lut.inc"
    lut_path.write_text(
        "// Auto-generated by tools/generate_streamk_opensieve.py\n"
        f"static const int kStreamKLutSize = {len(lut)};\n"
        f"static const uint8_t kStreamKLut[kStreamKLutSize] = {{\n"
        + ",\n".join(f"  {v}" for v in lut)
        + "\n};\n",
        encoding="utf-8",
    )

    bloom_path = output_dir / "marlin_streamk_bloom.inc"
    bloom_chunks = []
    for policy, bits in enumerate(blooms):
        bloom_chunks.append(format_byte_array(bytes(bits), f"kStreamKBloom{policy}"))
    bloom_path.write_text(
        "// Auto-generated by tools/generate_streamk_opensieve.py\n"
        f"static const int kStreamKBloomBits = {BLOOM_BITS};\n"
        f"static const int kStreamKBloomHashes = {BLOOM_HASHES};\n"
        f"static const int kStreamKBloomPolicyCount = {NUM_POLICIES};\n"
        + "\n\n".join(bloom_chunks)
        + "\n\n"
        "static const uint8_t* const kStreamKBloomFilters[kStreamKBloomPolicyCount] = {\n"
        + ",\n".join(f"  kStreamKBloom{p}" for p in range(NUM_POLICIES))
        + ",\n};\n\n"
        "static const uint32_t kStreamKBloomSeeds[kStreamKBloomPolicyCount] = {\n"
        + ",\n".join(f"  0x{0xF1EA5EED + p * 0x85EBCA77:08x}u" for p in range(NUM_POLICIES))
        + ",\n};\n",
        encoding="utf-8",
    )

    print(f"Wrote {lut_path} ({len(lut)} entries)")
    print(f"Wrote {bloom_path} ({NUM_POLICIES} filters x {BLOOM_BITS} bits)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "csrc"
        / "quantization"
        / "marlin"
        / "generated",
    )
    args = parser.parse_args()
    emit_files(args.output_dir)


if __name__ == "__main__":
    main()
