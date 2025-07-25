# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations
import abc
import enum
import torch
from typing import Optional, Tuple, Union
import dataclasses

from transformer_engine.pytorch.tensor.quantized_tensor import QuantizedTensorBase
from transformer_engine.pytorch.experimental import utils


@enum.unique
class ScalingType(enum.Enum):
    """Scaling granularity for quantization."""

    PER_TENSOR = "per_tensor"
    """Activations and weights are quantized based on a global amax statistic"""
    PER_CHANNEL = "per_channel"
    """Activations and weights are quantized using row-wise and column-wise
    scaling tiles for fwd and backward respectively."""
    VECTOR_TILED_X_AND_G_BLOCK_TILED_W = "vector_tiled_x_and_g_block_tiled_w"
    """Activations and gradients and their transposes are scaled using 1xB
    tilings, while weights and their transpose are scaled using BxB tilings
    where B is a block extent such as 128."""
    PER_1D_BLOCK = "per_1d_block"
    """Activations and weights are quantized using a 1D block scaling scheme."""


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


@enum.unique
class DequantizeLayout(enum.Enum):
    """Scaling granularity for quantization."""

    AS_ORIGINAL = "as_original"
    """Dequantize method is responsible to return same shape as pre-quantized tensor"""
    CALLER_CHECKS = "caller_checks"
    """
    Dequantize method may return the transpose of the pre-quantized tensor shape
    if this is convenient to reduce work and perf of dequantize. Caller takes
    responsibility.
    """


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


# TODO: reuse/import from TE
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

    def __init__(self, *, rowwise: bool, columnwise: bool) -> None:
        self.rowwise_usage = rowwise
        self.columnwise_usage = columnwise

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"rowwise_usage={self.rowwise_usage}, "
            f"columnwise_usage={self.columnwise_usage}, "
            ")"
        )

    @abc.abstractmethod
    def quantize(self, tensor: torch.Tensor, **kwargs) -> QuantizedTensorBase:
        """Quantize tensor"""
        pass

    def __call__(self, tensor: torch.Tensor) -> QuantizedTensorBase:
        """Quantize tensor"""
        return self.quantize(tensor)

    def set_usage(
        self,
        rowwise: Optional[bool] = None,
        columnwise: Optional[bool] = None,
    ) -> None:
        """Set how the quantized tensor is expected to be used"""
        if rowwise is not None:
            self.rowwise_usage = rowwise
        if columnwise is not None:
            self.columnwise_usage = columnwise


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
    low_precision_dtype: Union[utils.Fp4Formats, torch.dtype]
        low precision tensor datatype.
    original_shape: Tuple[int, ...]
        original shape of the tensor.
    quantizer: ExperimentalQuantizer
        Builder class for quantized tensor.
    """

    data: Optional[torch.Tensor] = None
    scale: Optional[torch.Tensor] = None
    data_t: Optional[torch.Tensor] = None
    scale_t: Optional[torch.Tensor] = None

    dtype: Optional[torch.dtype] = None
    device: Optional[torch.device] = None
    low_precision_dtype: Optional[Union[utils.Fp4Formats, torch.dtype]] = None
    original_shape: Optional[Tuple[int, ...]] = None
    quantizer: Optional[ExperimentalQuantizer] = None

    @property
    def experimental(self) -> bool:
        """Flag for upstreaming to TE"""
        return True

    # Compatibility
    @property
    def _data(self):
        return self.data

    @_data.setter
    def _data(self, value):
        self.data = value

    @property
    def _scale_inv(self):
        return self.scale

    @_scale_inv.setter
    def _scale_inv(self, value):
        self.scale = value


class ExperimentalQuantizer(Quantizer):
    """Experimental Quantizer class

    Defines the interface for experimental quantizers.
    """

    def __init__(self, *, rowwise: bool, columnwise: bool) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.internal = True

    @property
    def experimental(self) -> bool:
        """Flag for upstreaming to TE"""
        return True

    @abc.abstractmethod
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


def is_experimental(x: Union[Quantizer, QuantizedTensorBase]) -> bool:
    """Check if a object is experimental"""
    # assert isinstance(x, (Quantizer, QuantizedTensorBase)), "Object must be a Quantizer or QuantizedTensorBase instance"

    return hasattr(x, "experimental") and x.experimental
