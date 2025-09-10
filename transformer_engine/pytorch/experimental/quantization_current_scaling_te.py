# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
from typing import Optional, Tuple

from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.tensor.quantized_tensor import Quantizer
from transformer_engine.pytorch.tensor.base.float8_tensor_base import Float8TensorBase


class Float8CurrentScalingQuantizerRef(Quantizer):
    """Reference implementation of FP8 quantizer with current scaling"""

    """FP8 datatype (torch FP8 dtype, e.g., torch.float8_e4m3fn)"""
    dtype: torch.dtype
    """amax reduction options"""
    with_amax_reduction: bool
    amax_reduction_group: Optional[torch.distributed.ProcessGroup]
    """Options about how to quantize the tensor"""
    force_pow_2_scales: bool
    amax_epsilon: float

    def __init__(
        self,
        fp8_dtype: torch.dtype,
        rowwise: bool = True,
        columnwise: bool = True,
        force_pow_2_scales: bool = False,
        amax_epsilon: float = 0.0,
    ):
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.internal = True

        self.dtype = fp8_dtype
        self.with_amax_reduction = False
        self.amax_reduction_group = None
        self.force_pow_2_scales = force_pow_2_scales
        self.amax_epsilon = amax_epsilon

    @staticmethod
    def _scale_from_amax_tensor(
        x_dtype: torch.dtype,
        amax: torch.Tensor,
        quant_dtype: torch.dtype,
        *,
        amax_epsilon: float,
        force_pow_2_scales: bool,
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
        # Clamp on-device
        amax = amax.clamp_min(amax_epsilon)

        # Compute scale factor
        scale = torch.div(fp8_max, amax)
        # Note frexp doesn't give back inf for exponent with an inf input
        # We take care of inf before pow_2_scales
        scale = torch.where(scale == torch.inf, torch.finfo(x_dtype).max, scale)
        if force_pow_2_scales:
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

    def compute_scale(
        self,
        x: torch.Tensor,
        quant_dtype: torch.dtype,
        amax_epsilon=0.0,
        force_pow_2_scales: bool = False,
    ):
        # Use float32 for computation
        x_fp32 = x.to(torch.float32)

        if x_fp32.numel() == 0:
            amax = torch.empty(1, dtype=torch.float32, device=x.device)
        else:
            amax = torch.amax(torch.abs(x_fp32)).view(1)

        return self._scale_from_amax_tensor(
            x.dtype,
            amax=amax,
            quant_dtype=quant_dtype,
            amax_epsilon=amax_epsilon,
            force_pow_2_scales=force_pow_2_scales,
        )

    def _quantize(self, tensor: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Python reference quantization.
        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor to quantize. Supports 2D and ND; rowwise data retains input shape.
            If columnwise data is requested, a 2D transpose buffer of shape
            [inner_dim, numel/inner_dim] is produced by flattening all leading dims.

        Returns
        -------
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]
            (qx, sx, qx_t, sx_t) where:
            - qx: quantized data in row-major order (if rowwise_usage), None otherwise
            - sx: dequantization scale_inv for qx (if rowwise_usage), None otherwise
            - qx_t: quantized data in column-major order (if columnwise_usage), None otherwise
            - sx_t: dequantization scale_inv for qx_t (if columnwise_usage), None otherwise
        """
        # Handle amax reduction if enabled
        if self.with_amax_reduction:
            assert self.amax_reduction_group is not None, "amax_reduction_group must be set when with_amax_reduction is True"

            # Compute local amax
            if tensor.numel() == 0:
                amax = torch.empty(1, dtype=torch.float32, device=tensor.device)
            else:
                amax = torch.amax(torch.abs(tensor)).view(1).to(torch.float32)

            # Reduce amax across all ranks
            torch.distributed.all_reduce(
                amax, group=self.amax_reduction_group, op=torch.distributed.ReduceOp.MAX
            )

            # Compute scale using the global amax
            scale, scale_inv, _ = self._scale_from_amax_tensor(
                tensor.dtype,
                amax=amax,
                quant_dtype=self.dtype,
                amax_epsilon=self.amax_epsilon,
                force_pow_2_scales=self.force_pow_2_scales,
            )
        else:
            # compute scale factor using local amax
            scale, scale_inv, _ = self.compute_scale(
                tensor,
                self.dtype,
                amax_epsilon=self.amax_epsilon,
                force_pow_2_scales=self.force_pow_2_scales,
            )

        # Quantize to FP8 (preserve original shape for rowwise data)
        qx_full: Optional[torch.Tensor] = (tensor.float() * scale).to(self.dtype)
        sx: Optional[torch.Tensor] = scale_inv

        # Compute 2D transpose buffer with shape [inner_dim, numel/inner_dim] if needed
        if self.columnwise_usage:
            assert qx_full is not None
            last_dim = qx_full.shape[-1]
            qx_2d = qx_full.view(-1, last_dim)
            qx_t = qx_2d.t().contiguous()
            sx_t = sx
        else:
            qx_t, sx_t = None, None

        if not self.rowwise_usage:
            qx_full = None
            sx = None

        return qx_full, sx, qx_t, sx_t

    def quantize_impl(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> Float8TensorBase:
        # sanity checks
        # assert x.dtype in utils.HIGH_PRECISION_FLOAT_DTYPES, "Unsupported input dtype."

        qx, sx, qx_t, sx_t = self._quantize(x)

        return Float8TensorBase(
            data=qx,
            fp8_scale_inv=sx,
            data_transpose=qx_t,
            fp8_dtype=TE_DType[self.dtype],
            quantizer=self,
        )

    def dequantize(self, tensor: torch.Tensor, scale_inv: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Dequantize the quantized tensor"""
        tensor = (tensor.to(torch.float32) * scale_inv)
        if dtype is None:
            return tensor
        return tensor.to(dtype)

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: Float8TensorBase,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> Float8TensorBase:
        """Update the quantized tensor with the given tensor in-place
        Parameters
        ----------
        src: torch.Tensor
            Source tensor to copy from
        dst: ExperimentalQuantizedTensor
            Destination ExperimentalQuantizedTensor to update
        noop_flag: torch.Tensor, optional
            float32 flag indicating whether to avoid performing update
        """
        # Handle noop flag
        if noop_flag is not None and noop_flag.item() != 0:
            return dst

        # Make sure input is in expected format
        if not src.is_contiguous():
            src = src.contiguous()

        qx, sx, qx_t, sx_t = self._quantize(src)

        # Update the destination with new data
        dst._data = qx
        dst._scale_inv = sx
        dst._transpose = qx_t
        dst._transpose_invalid = qx_t is None
        dst._fp8_dtype = TE_DType[self.dtype]

        return dst
