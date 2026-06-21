"""Grid-selection cost model for DP + two-tile Split-K on N=256, K=6144.

This mirrors compute_marlin_streamk_schedule() and the kernel's launch/runtime
behaviour closely enough to *rank* candidate grids. It is a design aid, not a
cycle-accurate simulator: absolute numbers are arbitrary units, only relative
ordering across grids for the same (m) matters.
"""

import math

SMS = 132          # H100 SM count
N = 256
K = 6144
TOPK = 8
MOE_BLOCK = 16

# Tile config we are designing for (SM90, N=256 -> thread_n=128 -> n_tiles=2).
THREAD_N = 128
THREAD_K = 64
N_TILES = N // THREAD_N          # 2
K_TILES = K // THREAD_K          # 96
BMAX = 6                         # occupancy ceiling (blocks/SM) at this tile/stage

# two-tile SK merge threshold (matches streamk_expand_threshold(kTwoTileSkDp))
THRESH = 3


def div_ceil(a, b):
    return (a + b - 1) // b


def parallel_moe_blocks(m):
    """Idealised balanced-routing block count. Real MoE padding inflates this
    for small m, but the *grid policy* is driven by global_mn_tiles ordering,
    which this preserves."""
    return max(1, round(m * TOPK / MOE_BLOCK))


def schedule(global_mn_tiles, grid, k_tiles):
    """Port of compute_marlin_streamk_schedule (two-tile SK + DP)."""
    part2 = global_mn_tiles
    part1_iters = 0
    if global_mn_tiles > grid:
        part2 = global_mn_tiles % grid
        if part2 * THRESH <= grid:
            part2 += grid
        part1_iters = (global_mn_tiles - part2) // grid
    slice_iters = max(1, div_ceil(k_tiles * part2, grid))
    return part1_iters, part2, slice_iters


def runtime_units(global_mn_tiles, grid, k_tiles, sms=SMS, bmax=BMAX):
    """Relative runtime estimate for one launch of `grid` CTAs."""
    part1_iters, part2, slice_iters = schedule(global_mn_tiles, grid, k_tiles)

    # Per-CTA K-iteration load. Every resident CTA runs part1_iters full tiles
    # (k_tiles each) then participates in the SK tail (slice_iters).
    sk_active = part2 > 0 and slice_iters < k_tiles
    per_cta_iters = part1_iters * k_tiles + (slice_iters if part2 > 0 else 0)

    # Hardware-wave serialisation: at most sms*bmax CTAs resident at once.
    resident = min(grid, sms * bmax)
    hw_waves = div_ceil(grid, resident)
    # The launch's makespan is bounded by the busiest hardware wave; with
    # near-uniform per-CTA load it is per_cta_iters scaled by wave count.
    busy = per_cta_iters * hw_waves

    # --- penalties -------------------------------------------------------
    # (1) idle SMs: launching fewer than sms CTAs wastes the machine.
    idle_penalty = 0.0
    if grid < sms:
        idle_penalty = (sms - grid) / sms * k_tiles * 0.5

    # (2) split-K reduction: a global merge across CTAs sharing a tile.
    #     Cost grows with how many CTAs cooperate per split tile.
    red_penalty = 0.0
    if sk_active:
        splits_per_tile = grid / max(1, part2)
        red_penalty = 6.0 + 1.5 * splits_per_tile

    # (3) smem pressure: more co-resident blocks/SM -> shallower effective
    #     pipeline -> each iteration slightly more expensive.
    bps = div_ceil(min(grid, sms * bmax), sms)
    smem_factor = 1.0 + 0.04 * max(0, bps - 1)

    # (4) tail wave-quantization in the last hardware wave (partial occupancy).
    last_wave = grid - (hw_waves - 1) * resident
    waveq_penalty = 0.0
    if last_wave < sms and hw_waves >= 1 and grid >= sms:
        waveq_penalty = (sms - last_wave) / sms * per_cta_iters * 0.15

    return busy * smem_factor + idle_penalty + red_penalty + waveq_penalty


def best_grid(m, k_tiles=K_TILES):
    pmb = parallel_moe_blocks(m)
    T = pmb * N_TILES
    # candidate grids: every SM-multiple up to Bmax, plus T itself and a few
    # data-parallel-friendly divisors.
    cands = set()
    for b in range(1, BMAX + 1):
        cands.add(SMS * b)
    cands.add(T)
    cands.add(max(SMS, T))
    # grids that make T a clean multiple (minimise DP tail)
    for waves in range(1, 4 * BMAX + 1):
        g = div_ceil(T, waves)
        if SMS <= g <= SMS * BMAX:
            cands.add(g)
    cands = sorted(c for c in cands if c >= 1)

    scored = [(runtime_units(T, g, k_tiles), g) for g in cands]
    scored.sort()
    return T, scored[0], scored


MIN_SLICE = 8  # min k-tiles per SK slice (avoid micro-slices on this K-heavy shape)


def proposed_grid(T, sms=SMS, bmax=BMAX, k_tiles=K_TILES):
    """The three-regime DP + two-tile Split-K policy we are designing."""
    target = sms * bmax
    if T >= target:
        return target, "A:DP+SKtail"
    if T >= sms:
        return T, "B:pureDP"
    # Regime C: T < sms  ->  split-K to fill the machine, minimal even depth.
    d = div_ceil(sms, T)                  # min slices/tile to reach >= sms CTAs
    d = min(d, bmax)                      # respect occupancy ceiling
    d = max(1, min(d, k_tiles // MIN_SLICE))  # avoid micro-slices
    grid = T * d
    grid = max(grid, sms)                 # never leave SMs idle
    grid = min(grid, target)
    return grid, "C:splitK"


def compare(m):
    pmb = parallel_moe_blocks(m)
    T = pmb * N_TILES
    g, regime = proposed_grid(T)
    cost = runtime_units(T, g, K_TILES)
    p1, p2, sl = schedule(T, g, K_TILES)
    fixed = SMS * BMAX
    fc = runtime_units(T, fixed, K_TILES)
    # current work-aware grid (regime B only; C keeps fixed)
    if SMS <= T < fixed:
        wg = T
    else:
        wg = fixed
    wc = runtime_units(T, wg, K_TILES)
    bps = div_ceil(min(g, fixed), SMS)
    print(f"m={m:5d} T={T:5d} | proposed g={g:4d} {regime:11s} bps={bps} "
          f"p2={p2:4d} sl={sl:3d} cost={cost:7.1f} | "
          f"fixed792={fc:7.1f}(+{100*(fc-cost)/fc:4.1f}%) "
          f"workaware={wc:7.1f}(+{100*(wc-cost)/wc:4.1f}%)")


def describe(m):
    T, (cost, g), scored = best_grid(m)
    p1, p2, sl = schedule(T, g, K_TILES)
    fixed = SMS * BMAX
    fc = runtime_units(T, fixed, K_TILES)
    bps = div_ceil(min(g, SMS * BMAX), SMS)
    policy = "DP" if p2 == 0 or sl >= K_TILES else ("DP+SK" if p1 > 0 else "pureSK")
    print(f"m={m:5d}  T={T:5d}  best_grid={g:4d} (={g/SMS:.2f}*SMS, bps={bps})  "
          f"policy={policy:6s} p1={p1} p2={p2} sl={sl}  "
          f"cost={cost:8.1f}  fixed({fixed})={fc:8.1f}  gain={100*(fc-cost)/fc:5.1f}%")


if __name__ == "__main__":
    print(f"SMS={SMS} N={N} K={K} thread_n={THREAD_N} thread_k={THREAD_K} "
          f"n_tiles={N_TILES} k_tiles={K_TILES} Bmax={BMAX}")
    print("\n--- proposed three-regime policy vs fixed-792 / current work-aware ---")
    for m in [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
              1536, 2048, 3072, 4096, 6144, 8192]:
        compare(m)
