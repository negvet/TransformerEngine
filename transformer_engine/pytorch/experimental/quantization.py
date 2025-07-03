# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import abc
import torch
from typing import Optional, Tuple, Union
import dataclasses

from transformer_engine.pytorch.experimental import utils


def _compute_scale_fp4fp8(
    x_dtype: torch.dtype,
    amax: torch.Tensor,
    quant_dtype: Union[utils.Fp4Formats, torch.dtype],  # FP4 format enum or FP8 torch.dtype
    *,
    eps: float,
    pow_2_scales: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derives quantization and dequantization scales from amax for FP4/FP8 formats.

    Reference implementation for scale calculation.
    This does not follow recommended implementation when decoding factor is calculated first.
    This follows the implementation in kitchen/quantization.py::_scale_from_amax_tensor.
    The same approach is used for FP4 and FP8 per tensor quantization.

    Returns:
    - scale: quantization scales
    - scale_inv: dequantization scales
    - amax: Amax tensor with updates made for extrema values.
    """
    assert amax.dtype == torch.float, "amax must be a float tensor."
    assert quant_dtype in utils.FP4_DTYPES + utils.FP8_DTYPES, f"Unsupported quant dtype {quant_dtype}."

    # Clamping amax to avoid division by small numbers
    amax = torch.max(amax, torch.tensor(eps))

    # Get max values for different dtypes
    if quant_dtype in utils.FP4_DTYPES:
        # FP4 max values
        if quant_dtype == utils.Fp4Formats.E2M1:
            max_value = utils.FP4_E2M1_MAXVAL
        elif quant_dtype == utils.Fp4Formats.E0M3:
            max_value = utils.FP4_E0M3_MAXVAL
        elif quant_dtype == utils.Fp4Formats.E3M0:
            max_value = utils.FP4_E3M0_MAXVAL
        else:
            raise ValueError(f"Unsupported FP4 quant_dtype {quant_dtype}")
    elif quant_dtype in utils.FP8_DTYPES:
        # FP8 max values
        max_value = torch.finfo(quant_dtype).max
    else:
        raise ValueError(f"Unsupported quant_dtype {quant_dtype}")

    # Compute scale factor
    scale = torch.div(max_value, amax)

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


@dataclasses.dataclass
class QuantizeResult:
    """A container to hold quantization result, including quantized tensor, optional
    transposed quantized tensor, and corresponding decoding scales.

    data: the quantized tensor.
    scale: the decoding scale for the quantized tensor. Shape depends on the scaling granularity.
        - if scaling type is PER_TENSOR, it should be a 1D scalar tensor.
    data_t: the transposed quantized tensor (computed lazily if needed).
    scale_t: the decoding scale for the transposed quantized tensor.
    """

    data: torch.Tensor | None
    scale: torch.Tensor | None
    data_t: torch.Tensor | None = None
    scale_t: torch.Tensor | None = None


class Quantizer(abc.ABC):
    """Builder class for quantized tensors.

    This class is typically used to convert a high-precision tensor
    (e.g. in FP32 or BF16) into a quantized tensor (e.g. in FP8).

    """

    """Whether to construct quantized tensors with "row-wise usage"

    Hand-wave explanation: Consider the matrix multiplication C = A *
    B^T (used in linear forward). Tensor Cores prefer "TN GEMMs" (in
    Fortran-style column-major order), so A and B should be in
    row-major order.

    """
    rowwise_usage: bool

    """Whether to construct quantized tensors with "column-wise usage"

    Hand-wave explanation: Consider the matrix multiplication C = A^T
    * B (used in linear backward wgrad). Tensor Cores prefer "TN
    GEMMs" (in Fortran-style column-major order), so A and B should be
    in column-major order.

    """
    columnwise_usage: bool

    """Whether to instantiates tensor for purely internal usage

    Internal tensors are storage classes with minimal logic. They have
    less overhead than PyTorch tensor sub-classes, but are not
    compatible with PyTorch's autograd infrastructure nor PyTorch
    operations.

    """
    internal: bool

    def __init__(self, *, rowwise: bool, columnwise: bool) -> None:
        self.rowwise_usage = rowwise
        self.columnwise_usage = columnwise
        self.internal = False

    @abc.abstractmethod
    def quantize(self, tensor: torch.Tensor) -> QuantizeResult:
        """Quantize tensor"""
        pass

    def __call__(self, tensor: torch.Tensor) -> QuantizeResult:
        """Quantize tensor"""
        return self.quantize(tensor)

    def set_usage(
        self, *, rowwise: Optional[bool] = None, columnwise: Optional[bool] = None
    ) -> None:
        """Set how the quantized tensor is expected to be used"""
        if rowwise is not None:
            self.rowwise_usage = rowwise
        if columnwise is not None:
            self.columnwise_usage = columnwise


class Float4Float8CurrentScalingEmulationRefQuantizer(Quantizer):
    """FP4/FP8 quantizer with current scaling and fake quantization"""

    """FP4/FP8 datatype"""
    dtype: Union[utils.Fp4Formats, torch.dtype]
    """Options about how to quantize the tensor"""
    force_pow_2_scales: bool
    amax_epsilon: float

    def __init__(
        self,
        dtype: Union[utils.Fp4Formats, torch.dtype],
        rowwise: bool = True,
        columnwise: bool = True,
        force_pow_2_scales: bool = False,
        amax_epsilon: float = 0.0,
    ):
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.dtype = dtype
        self.force_pow_2_scales = force_pow_2_scales
        self.amax_epsilon = amax_epsilon

    def _cast_to_quantized_format(self, tensor: torch.Tensor) -> torch.Tensor:
        """Cast tensor to quantized format (FP4/FP8) and back to float32"""
        if self.dtype in utils.FP4_DTYPES:
            if self.dtype == utils.Fp4Formats.E2M1:
                return utils.cast_to_fp4_e2m1(tensor)
            elif self.dtype == utils.Fp4Formats.E0M3:
                return utils.cast_to_fp4_e0m3(tensor)
            elif self.dtype == utils.Fp4Formats.E3M0:
                return utils.cast_to_fp4_e3m0(tensor)
            else:
                raise ValueError(f"Unsupported FP4 format: {self.dtype}")
        elif self.dtype in utils.FP8_DTYPES:
            return tensor.to(self.dtype).to(torch.float32)
        else:
            raise ValueError(f"Unsupported quantization format: {self.dtype}")

    def _fake_quantize_tensor(self, tensor: torch.Tensor, scale: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
        """Perform fake quantization: scale -> cast -> dequantize"""
        if tensor.numel() == 0:
            return tensor.clone()

        scaled_tensor = tensor.float() * scale
        quantized_data = self._cast_to_quantized_format(scaled_tensor)
        return (quantized_data * scale_inv).to(tensor.dtype)

    def _compute_scale(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute quantization scales for the input tensor"""
        x_fp32 = x.to(torch.float32)

        if x_fp32.numel() == 0:
            amax = torch.empty(1, dtype=torch.float32, device=x.device)
        else:
            amax = torch.amax(torch.abs(x_fp32)).view(1)

        return _compute_scale_fp4fp8(
            x.dtype,
            amax=amax,
            quant_dtype=self.dtype,
            eps=self.amax_epsilon,
            pow_2_scales=self.force_pow_2_scales,
        )

    def quantize(self, x: torch.Tensor) -> QuantizeResult:
        """
        Python implementation of quantization.
        Fake quantize tensor to FP4/FP8 and back - returns regular tensor.
        Note: c++ kernel can be used as an option instead of this python implementation.
        """
        scale, scale_inv, _ = self._compute_scale(x)
        
        # Initialize outputs
        qx, sx = None, None
        qx_t, sx_t = None, None
        
        # Empty scale tensor for fake quantization (data is already dequantized)
        empty_scale = torch.empty(0)

        # Compute identity (original layout) if needed
        if self.rowwise_usage:
            qx = self._fake_quantize_tensor(x, scale, scale_inv)
            sx = empty_scale

        # Compute transpose if needed
        if self.columnwise_usage:
            if self.rowwise_usage and qx is not None:
                # For per-tensor quantization, just transpose the result
                qx_t = qx.t().contiguous()
                sx_t = empty_scale  # Scale is empty because data is already dequantized
            else:
                # Transpose and quantize the transposed tensor
                x_t = x.t().contiguous()
                qx_t = self._fake_quantize_tensor(x_t, scale, scale_inv)
                sx_t = empty_scale  # Scale is empty because data is already dequantized

        return QuantizeResult(data=qx, scale=sx, data_t=qx_t, scale_t=sx_t)
