import torch

from .helpers import _quantize_unsigned_with_bias, ScalarType


def cutlass69_quantize_experts(
    weights: torch.Tensor,
    quant_type: ScalarType,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize expert weights for CUTLASS69 int4 grouped GEMM.

    weights: [num_experts, size_k, size_n]
    Returns:
      q_weights: [num_experts, size_n, size_k] int8 in Marlin uint4b8 encoding (0..15)
      scales: [num_experts, size_n, scale_k] fp16/bf16 row-major for CUTLASS
    """
    num_experts, size_k, size_n = weights.shape
    q_weights = []
    scales = []
    scale_k = size_k // group_size if group_size > 0 else 1
    for expert in range(num_experts):
        q_weight, scale = _quantize_unsigned_with_bias(
            weights[expert], group_size, quant_type.bias
        )
        # CUTLASS column-major B uses logical [N, K]; store row-major [N, K].
        q_weights.append(q_weight.t().contiguous().to(torch.int8))
        # CUTLASS scales are [N, scale_k] row-major.
        scales.append(scale.t().contiguous())
    return torch.stack(q_weights), torch.stack(scales)
