// marlin_sm90_advanced.cuh
#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace marlin_sm90_advanced {

// ============================================================================
// Dimension 1: Register-Sourced A for WGMMA
// ============================================================================
// By keeping A in registers, we save SMEM allocation and avoid SMEM write/read
// overhead for the A matrix.
__device__ __forceinline__ void warpgroup_mma_accumulate_regA(
    const uint32_t reg_A[4], uint64_t b_desc, float accum[64]) {
  // wgmma.mma_async with A in registers (4x32bit = 8 halfs per thread)
  // trans-b = 1 (B is row-major in SMEM)
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
      " {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n"
      : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]),
        "+f"(accum[4]), "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7]),
        "+f"(accum[8]), "+f"(accum[9]), "+f"(accum[10]), "+f"(accum[11]),
        "+f"(accum[12]), "+f"(accum[13]), "+f"(accum[14]), "+f"(accum[15]),
        "+f"(accum[16]), "+f"(accum[17]), "+f"(accum[18]), "+f"(accum[19]),
        "+f"(accum[20]), "+f"(accum[21]), "+f"(accum[22]), "+f"(accum[23]),
        "+f"(accum[24]), "+f"(accum[25]), "+f"(accum[26]), "+f"(accum[27]),
        "+f"(accum[28]), "+f"(accum[29]), "+f"(accum[30]), "+f"(accum[31]),
        "+f"(accum[32]), "+f"(accum[33]), "+f"(accum[34]), "+f"(accum[35]),
        "+f"(accum[36]), "+f"(accum[37]), "+f"(accum[38]), "+f"(accum[39]),
        "+f"(accum[40]), "+f"(accum[41]), "+f"(accum[42]), "+f"(accum[43]),
        "+f"(accum[44]), "+f"(accum[45]), "+f"(accum[46]), "+f"(accum[47]),
        "+f"(accum[48]), "+f"(accum[49]), "+f"(accum[50]), "+f"(accum[51]),
        "+f"(accum[52]), "+f"(accum[53]), "+f"(accum[54]), "+f"(accum[55]),
        "+f"(accum[56]), "+f"(accum[57]), "+f"(accum[58]), "+f"(accum[59]),
        "+f"(accum[60]), "+f"(accum[61]), "+f"(accum[62]), "+f"(accum[63])
      : "r"(reg_A[0]), "r"(reg_A[1]), "r"(reg_A[2]), "r"(reg_A[3]),
        "l"(b_desc));
}

// ============================================================================
// Dimension 2: Hardware Dequantization (INT4 WGMMA)
// ============================================================================
// Assuming A is also quantized to INT4 (W4A4), we can use s32.s4.s4 WGMMA.
// This completely eliminates the SMEM dequantization roundtrip.
__device__ __forceinline__ void warpgroup_mma_accumulate_int4(
    uint64_t a_desc, uint64_t b_desc, int32_t accum[64]) {
  // m64n128k32 for s4 (requires K=32 per instruction)
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n128k32.s32.s4.s4 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
      " %64, %65, 1, 1, 1, 0, 1;\n"
      : "+r"(accum[0]), "+r"(accum[1]), "+r"(accum[2]), "+r"(accum[3]),
        "+r"(accum[4]), "+r"(accum[5]), "+r"(accum[6]), "+r"(accum[7]),
        "+r"(accum[8]), "+r"(accum[9]), "+r"(accum[10]), "+r"(accum[11]),
        "+r"(accum[12]), "+r"(accum[13]), "+r"(accum[14]), "+r"(accum[15]),
        "+r"(accum[16]), "+r"(accum[17]), "+r"(accum[18]), "+r"(accum[19]),
        "+r"(accum[20]), "+r"(accum[21]), "+r"(accum[22]), "+r"(accum[23]),
        "+r"(accum[24]), "+r"(accum[25]), "+r"(accum[26]), "+r"(accum[27]),
        "+r"(accum[28]), "+r"(accum[29]), "+r"(accum[30]), "+r"(accum[31]),
        "+r"(accum[32]), "+r"(accum[33]), "+r"(accum[34]), "+r"(accum[35]),
        "+r"(accum[36]), "+r"(accum[37]), "+r"(accum[38]), "+r"(accum[39]),
        "+r"(accum[40]), "+r"(accum[41]), "+r"(accum[42]), "+r"(accum[43]),
        "+r"(accum[44]), "+r"(accum[45]), "+r"(accum[46]), "+r"(accum[47]),
        "+r"(accum[48]), "+r"(accum[49]), "+r"(accum[50]), "+r"(accum[51]),
        "+r"(accum[52]), "+r"(accum[53]), "+r"(accum[54]), "+r"(accum[55]),
        "+r"(accum[56]), "+r"(accum[57]), "+r"(accum[58]), "+r"(accum[59]),
        "+r"(accum[60]), "+r"(accum[61]), "+r"(accum[62]), "+r"(accum[63])
      : "l"(a_desc), "l"(b_desc));
}

// ============================================================================
// Dimension 3: TMA Multicast
// ============================================================================
// Broadcasts the same weight tile to multiple CTAs in the cluster.
__device__ __forceinline__ void tma_load_2d_b_tile_multicast(
    void* dst_smem, const void* tensor_map, int packed_col, int packed_row,
    uint64_t* barrier, uint16_t mcast_mask) {
  uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
  uint32_t bar = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::multicast::"
      "bytes [%0], [%1, {%2, %3}], [%4], %5;\n" ::"r"(dst),
      "l"(tensor_map), "r"(packed_col), "r"(packed_row), "r"(bar), "h"(mcast_mask)
      : "memory");
}

// ============================================================================
// Dimension 4: W1 + W2 DSMEM Fusion
// ============================================================================
// W1 CTA writes intermediate activation directly to W2 CTA's SMEM using DSMEM.
__device__ __forceinline__ void store_to_dsmem(
    float* local_accum, void* remote_smem_ptr, int remote_rank) {
  // Map remote SMEM pointer to DSMEM address space
  uint32_t remote_smem = static_cast<uint32_t>(__cvta_generic_to_shared(remote_smem_ptr));
  
  // In PTX, map.shared::cluster can be used to get the remote pointer
  uint32_t dsmem_addr;
  asm volatile("map.shared::cluster.u32 %0, %1, %2;\n"
               : "=r"(dsmem_addr)
               : "r"(remote_smem), "r"(remote_rank));
  
  // Store to remote SMEM asynchronously
  asm volatile(
      "st.async.shared::cluster.m32.b32 [%0], %1;\n"
      :: "r"(dsmem_addr), "r"(*(reinterpret_cast<uint32_t*>(local_accum))));
}

} // namespace marlin_sm90_advanced
