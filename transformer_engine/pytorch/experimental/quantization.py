# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations
import enum
import torch
from typing import Optional, Tuple
import dataclasses

from transformer_engine.pytorch.tensor.quantized_tensor import QuantizedTensorBase, Quantizer


@enum.unique
class GEMMType(enum.Enum):
    """Type of GEMM operation being performed."""

    FPROP = "fprop"
    DGRAD = "dgrad"
    WGRAD = "wgrad"


@dataclasses.dataclass(frozen=True)
class MMParams:
    """Matrix multiplication parameters."""

    out_dtype: torch.dtype | None = None
    # Use split accumulator for more accurate FP8 GEMM
    use_split_accumulator: bool = True


@dataclasses.dataclass
class ExperimentalQuantizedTensor(QuantizedTensorBase):
    """Base class for experimental quantized tensor containers.

    An experimental container to hold quantization result, including quantized tensor, optional
    transposed quantized tensor, and corresponding decoding scales.

    data: torch.Tensor
        the quantized tensor.
    scale: torch.Tensor
        the decoding scale for the quantized tensor. Shape depends on the scaling granularity.
        - if scaling type is PER_TENSOR, it should be a 1D scalar tensor.
    data_t: torch.Tensor
        the transposed quantized tensor (computed lazily if needed).
    scale_t: torch.Tensor
        the decoding scale for the transposed quantized tensor.
    dtype: torch.dtype
        nominal tensor datatype.
    device: torch.device
        device of the tensor.
    original_shape: Tuple[int, ...]
        original shape of the tensor.
    _quantizer: Quantizer
        Builder class for quantized tensor.
    """

    data: Optional[torch.Tensor] = None
    scale: Optional[torch.Tensor] = None
    data_t: Optional[torch.Tensor] = None
    scale_t: Optional[torch.Tensor] = None

    dtype: Optional[torch.dtype] = None
    device: Optional[torch.device] = None
    original_shape: Optional[Tuple[int, ...]] = None
    _quantizer: Optional[ExperimentalQuantizer] = None

    @property
    def experimental(self) -> bool:
        return True

    def prepare_for_saving(
        self,
    ) -> Tuple[list[Optional[torch.Tensor]], ExperimentalQuantizedTensor]:
        """Prepare the quantization result for saving for backward"""
        tensors = [self.data, self.data_t, self.scale, self.scale_t]
        self.data = None
        self.data_t = None
        self.scale = None
        self.scale_t = None
        return tensors, self

    def restore_from_saved(
        self, tensors: list[Optional[torch.Tensor]]
    ) -> list[Optional[torch.Tensor]]:
        """Restore the quantization result from the saved tensors"""
        self.data = tensors[0]
        self.data_t = tensors[1]
        self.scale = tensors[2]
        self.scale_t = tensors[3]
        return tensors[4:]


class ExperimentalQuantizer(Quantizer):
    """Experimental Quantizer class

    Defines the interface for experimental quantizers.
    """

    def __init__(self, *, rowwise: bool, columnwise: bool) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.internal = True

    @property
    def experimental(self) -> bool:
        return True

    def qgemm(
        self,
        qx: torch.Tensor,
        qw: torch.Tensor,
        m_params: MMParams,
        out_dtype: torch.dtype,
        sx: torch.Tensor,
        sw: torch.Tensor,
        bias: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        accumulate: bool = False,
        gemm_type: GEMMType = GEMMType.FPROP,
        qresult_x: ExperimentalQuantizedTensor | None = None,
        qresult_w: ExperimentalQuantizedTensor | None = None,
    ) -> torch.Tensor:
        """Quantized GEMM interface."""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement qgemm function"
        )


def _scale_from_amax_tensor(
    x_dtype: torch.dtype,
    amax: torch.Tensor,
    quant_dtype: torch.dtype,
    *,
    eps: float,
    pow_2_scales: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derives quantization and dequantization from amax and options.
    Reference implementation for scale calculation.
    Returns:
    - scale: quantization scales
    - scale_inv: dequantization scales
    - amax: Amax tensor with updates made for extrema values.
    """
    assert amax.dtype == torch.float, "amax must be a float tensor."
    fp8_max = torch.finfo(quant_dtype).max
    # Clamping amax to avoid division by small numbers
    amax = torch.max(amax, torch.tensor(eps))

    # Compute scale factor
    scale = torch.div(fp8_max, amax)
    # Note frexp doesn't give back inf for exponent with an inf input
    # We take care of inf before pow_2_scales
    scale = torch.where(scale == torch.inf, torch.finfo(x_dtype).max, scale)
    if pow_2_scales:
        # Calculate rounded down exponent
        _, exp = torch.frexp(scale)
        # Positive numbers are always returned as mant, exp with
        # a mantissa in [0.5, 1.0). Because a normal float has a mantissa with
        # hidden bit in [1.0, 2.0), the exponent will be off by exactly one because
        # of the shift. Subnormal and zero cases need not be considered because
        # the smallest possible result of fp8_max / amax is still normal.
        exp = exp - 1
        # No subnormals and zero.
        assert (exp > -127).all()
        # TODO: If/when adding a URM option an option is to cap to 126
        # rather than allowing the full range of FP32 (2 - 2^23) x 2^127
        # addresses cases where adding a mantissa overflows into inf scales.
        # Not necessary currently without additional scale smudging options.
        unity = torch.tensor([1.0], device=exp.device)
        torch.ldexp(unity, exp, out=scale)
        # Case where amax is inf. The frexp, ldexp logic changes 0.0 scales
        # Return 0.0 for 0.0 scale for consistency with non-pow2 scale
        # calculation.
        scale = torch.where(amax == float("inf"), 0.0, scale)

    # Handle overflow cases for amax zero causing NaN
    scale = torch.where(amax == 0, 1.0, scale)

    # Compute scale_inv
    scale_inv = torch.reciprocal(scale)

    return scale, scale_inv, amax
