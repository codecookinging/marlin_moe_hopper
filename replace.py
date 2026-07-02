import re

with open("csrc/moe/marlin_moe_wna16/ops.cu", "r") as f:
    content = f.read()

pattern = r'TORCH_CHECK\(false,\s*"SM90 TMA/WGMMA dataflow/tile path is selected[^;]+;\s*\}'

replacement = """
    CUtensorMap tma_map_host;
    uint64_t globalDim[2] = {
        static_cast<uint64_t>(prob_n * b_type.size_bits() / 32),
        static_cast<uint64_t>(num_experts * prob_k)
    };
    uint64_t globalStrides[1] = { globalDim[0] * sizeof(uint32_t) };
    uint32_t boxDim[2] = {
        static_cast<uint32_t>(128 * b_type.size_bits() / 32),
        static_cast<uint32_t>(64)
    };
    uint32_t elementStrides[2] = {1, 1};

    CUresult res = cuTensorMapEncodeTiled(
        &tma_map_host,
        CU_TENSOR_MAP_DATA_TYPE_UINT32,
        2,
        const_cast<void*>(B),
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    TORCH_CHECK(res == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed with error code ", res);

    void* tma_map_dev = nullptr;
    cudaMallocAsync(&tma_map_dev, sizeof(CUtensorMap), stream);
    cudaMemcpyAsync(tma_map_dev, &tma_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);

    marlin_sm90_tma_wgmma::Params params;
    params.A = reinterpret_cast<const int4*>(A);
    params.B = reinterpret_cast<const int4*>(B);
    params.B_tma_map = tma_map_dev;
    params.C = reinterpret_cast<int4*>(C);
    params.C_tmp = reinterpret_cast<int4*>(C_tmp);
    params.scales = reinterpret_cast<const int4*>(b_s);
    params.sorted_token_ids = reinterpret_cast<const int32_t*>(sorted_token_ids);
    params.expert_ids = reinterpret_cast<const int32_t*>(expert_ids);
    params.num_tokens_past_padded = reinterpret_cast<const int32_t*>(num_tokens_past_padded);
    params.topk_weights = reinterpret_cast<const float*>(topk_weights);
    params.locks = reinterpret_cast<int*>(workspace);
    params.prob_m = prob_m;
    params.prob_n = prob_n;
    params.prob_k = prob_k;
    params.top_k = top_k;
    params.moe_block_size = moe_block_size;
    params.num_groups = num_groups;
    params.group_size = group_size;
    params.sk_slice_count = 1;
    params.sk_slice_idx = 0;
    params.mul_topk_weights = mul_topk_weights;
    params.use_fp32_reduce = use_fp32_reduce;
    params.use_tma_load = true;

    int smem_size = marlin_sm90_tma_wgmma::required_shared_memory_bytes(moe_block_size, b_type.size_bits());
    int max_parallel = prob_m * top_k / moe_block_size;
    int total_tiles = marlin_sm90_tma_wgmma::logical_mn_tiles(max_parallel, prob_n);
    
    if (b_type.size_bits() == 4) {
      if (moe_block_size == 16) {
        auto kernel = marlin_sm90_tma_wgmma::MarlinSm90TmaWgmmaKernel<16, 4, 3>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        kernel<<<total_tiles, 128, smem_size, stream>>>(params);
      } else if (moe_block_size == 32) {
        auto kernel = marlin_sm90_tma_wgmma::MarlinSm90TmaWgmmaKernel<32, 4, 3>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        kernel<<<total_tiles, 128, smem_size, stream>>>(params);
      } else if (moe_block_size == 64) {
        auto kernel = marlin_sm90_tma_wgmma::MarlinSm90TmaWgmmaKernel<64, 4, 3>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        kernel<<<total_tiles, 128, smem_size, stream>>>(params);
      } else {
        TORCH_CHECK(false, "Unsupported moe_block_size for TMA/WGMMA: ", moe_block_size);
      }
    } else if (b_type.size_bits() == 8) {
      if (moe_block_size == 16) {
        auto kernel = marlin_sm90_tma_wgmma::MarlinSm90TmaWgmmaKernel<16, 8, 3>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        kernel<<<total_tiles, 128, smem_size, stream>>>(params);
      } else if (moe_block_size == 32) {
        auto kernel = marlin_sm90_tma_wgmma::MarlinSm90TmaWgmmaKernel<32, 8, 3>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        kernel<<<total_tiles, 128, smem_size, stream>>>(params);
      } else if (moe_block_size == 64) {
        auto kernel = marlin_sm90_tma_wgmma::MarlinSm90TmaWgmmaKernel<64, 8, 3>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        kernel<<<total_tiles, 128, smem_size, stream>>>(params);
      } else {
        TORCH_CHECK(false, "Unsupported moe_block_size for TMA/WGMMA: ", moe_block_size);
      }
    } else {
      TORCH_CHECK(false, "Unsupported b_bits for TMA/WGMMA: ", b_type.size_bits());
    }

    cudaFreeAsync(tma_map_dev, stream);
    return;
  }"""

if re.search(pattern, content):
    new_content = re.sub(pattern, replacement, content)
    with open("csrc/moe/marlin_moe_wna16/ops.cu", "w") as f:
        f.write(new_content)
    print("Replaced successfully.")
else:
    print("Pattern not found.")
