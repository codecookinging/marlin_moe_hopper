import math

SMS = 132          # H100 SM count
N = 256
K = 6144
TOPK = 8
MOE_BLOCK = 16

THREAD_N = 128
THREAD_K = 64
N_TILES = N // THREAD_N          # 2
K_TILES = K // THREAD_K          # 96
BMAX = 6                         # occupancy ceiling (blocks/SM) at this tile/stage

def div_ceil(a, b):
    return (a + b - 1) // b

def parallel_moe_blocks(m):
    return max(1, round(m * TOPK / MOE_BLOCK))

def schedule(global_mn_tiles, grid, k_tiles):
    part2 = global_mn_tiles
    part1_iters = 0
    if global_mn_tiles > grid:
        part2 = global_mn_tiles % grid
        if part2 * 3 <= grid:
            part2 += grid
        part1_iters = (global_mn_tiles - part2) // grid
    slice_iters = max(1, div_ceil(k_tiles * part2, grid))
    return part1_iters, part2, slice_iters

def runtime_units(global_mn_tiles, grid, k_tiles, sms=SMS, bmax=BMAX):
    part1_iters, part2, slice_iters = schedule(global_mn_tiles, grid, k_tiles)
    sk_active = part2 > 0 and slice_iters < k_tiles
    per_cta_iters = part1_iters * k_tiles + (slice_iters if part2 > 0 else 0)
    resident = min(grid, sms * bmax)
    hw_waves = div_ceil(grid, resident)
    busy = per_cta_iters * hw_waves

    idle_penalty = 0.0
    if grid < sms:
        idle_penalty = (sms - grid) / sms * k_tiles * 0.5

    red_penalty = 0.0
    if sk_active:
        splits_per_tile = grid / max(1, part2)
        red_penalty = 6.0 + 1.5 * splits_per_tile

    bps = div_ceil(min(grid, sms * bmax), sms)
    smem_factor = 1.0 + 0.04 * max(0, bps - 1)

    last_wave = grid - (hw_waves - 1) * resident
    waveq_penalty = 0.0
    if last_wave < sms and hw_waves >= 1 and grid >= sms:
        waveq_penalty = (sms - last_wave) / sms * per_cta_iters * 0.15

    return busy * smem_factor + idle_penalty + red_penalty + waveq_penalty

def best_grid(m, k_tiles=K_TILES):
    pmb = parallel_moe_blocks(m)
    T = pmb * N_TILES
    cands = set()
    for b in range(1, BMAX + 1):
        cands.add(SMS * b)
    cands.add(T)
    cands.add(max(SMS, T))
    for waves in range(1, 4 * BMAX + 1):
        g = div_ceil(T, waves)
        if SMS <= g <= SMS * BMAX:
            cands.add(g)
    cands = sorted(c for c in cands if c >= 1)

    scored = [(runtime_units(T, g, k_tiles), g) for g in cands]
    scored.sort()
    return T, scored[0], scored

m = 256
pmb = parallel_moe_blocks(m)
T = pmb * N_TILES
_, (cost, g), _ = best_grid(m)
print(f"M={m} T={T} Best Grid={g} Cost={cost}")
