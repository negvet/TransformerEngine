# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import math
import torch
from typing import Iterable, Tuple, Optional, Union

from transformer_engine.pytorch.experimental.quantization import ExperimentalQuantizerBase, MMParams, GEMMType, ExperimentalQuantizedTensorBase
from transformer_engine.pytorch.experimental import utils


class PerTensorExperimentalQuantizedTensor(ExperimentalQuantizedTensorBase):
    """Experimental quantized tensor container for per-tensor (tensorwise) scaling."""

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"dtype={self.dtype}, "
            f"device={self.device}, "
            f"low_precision_dtype={self.low_precision_dtype}, "
            f"data={self.dequantize(dtype=self.dtype)}, "
            f"original_shape={self.original_shape}"
            ")"
        )

    def quantize_(
        self,
        tensor: torch.Tensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> ExperimentalQuantizedTensorBase:
        """In-place update of quantized data

        Parameters
        ----------
        tensor: torch.Tensor
            Tensor to copy from
        noop_flag: torch.Tensor, optional
            float32 flag indicating whether to avoid performing update

        """
        if isinstance(tensor, ExperimentalQuantizedTensorBase):
            return self.quantize_(tensor.dequantize(), noop_flag=noop_flag)
        self.get_quantizer().update_quantized(tensor, self, noop_flag=noop_flag)
        return self

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Construct plain PyTorch tensor from QuantizedExperimentalTensorBase
        """
        if dtype is None:
            dtype = self.dtype

        # TODO: what to do with data_t ?
        assert self.data is not None, "QuantizedExperimentalTensor has no valid tensor data"
        assert self.scale is not None, "QuantizedExperimentalTensor has no valid scale"
        tensor_data = self.data
        tensor_scale = self.scale
        return self.get_quantizer().dequantize(tensor_data, tensor_scale, dtype=dtype)

    def get_quantizer(self) -> ExperimentalQuantizerBase:
        """Get builder for QuantizedExperimentalTensor

        Quantizer can be used for in-place operations.

        """
        if self.quantizer is not None:
            return self.quantizer
        raise ValueError("Quantizer is not set")

    def prepare_for_saving(self) -> Tuple[list[Optional[torch.Tensor]], ExperimentalQuantizedTensorBase]:
        """Prepare the quantization result for saving for backward"""
        tensors = [self.data, self.data_t, self.scale, self.scale_t]
        self.data = None
        self.data_t = None
        self.scale = None
        self.scale_t = None
        return tensors, self

    def restore_from_saved(self, tensors: list[Optional[torch.Tensor]]) -> list[Optional[torch.Tensor]]:
        """Restore the quantization result from the saved tensors"""
        self.data = tensors[0]
        self.data_t = tensors[1]
        self.scale = tensors[2]
        self.scale_t = tensors[3]
        return tensors[4:]

    def update_usage(
        self,
        rowwise_usage: Optional[bool] = None,
        columnwise_usage: Optional[bool] = None,
    ):
        """Generate or remove quantized data based on provided usage."""
        has_data = self.data is not None
        has_data_transpose = self.data_t is not None
        needs_data = has_data
        needs_data_transpose = has_data_transpose

        if rowwise_usage is not None:
            needs_data = rowwise_usage
        if columnwise_usage is not None:
            needs_data_transpose = columnwise_usage
        
        # Generate data that is required
        if needs_data and not has_data:
            raise RuntimeError("Cannot generate FP8 data, even from FP8 data transpose")
        if needs_data_transpose and not has_data_transpose:
            if not has_data:
                raise RuntimeError("FP8 data is required to generate FP8 data transpose")
            self._create_transpose()

        # Delete data that is not required
        if not needs_data:
            self.data = None
        if not needs_data_transpose:
            self.data_t = None

    def _create_transpose(self):
        """Create transposed quantized tensor"""
        if not self.data.is_contiguous():
            self.data = self.data.contiguous()
        self.data_t = self.data.t().contiguous()
        self.scale_t = self.scale

    def size(self, *args, **kwargs):
        if self.data is not None:
            return self.data.size(*args, **kwargs)
        size = self.data_t.size(*args, **kwargs)
        return torch.Size([size[-1], math.prod(size[:-1])])


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


class PerTensorExperimentalQuantizer(ExperimentalQuantizerBase):
    """Per-tensor experimental quantizer"""


class Float8CurrentScalingRefQuantizer(PerTensorExperimentalQuantizer):
    """FP8 quantizer with current scaling"""

    """FP8 datatype"""
    dtype: torch.dtype
    """Options about how to quantize the tensor"""
    pow_2_scales: bool
    eps: float

    def __init__(
        self,
        dtype: torch.dtype,
        rowwise: bool = True,
        columnwise: bool = True,
        pow_2_scales: bool = False,
        eps: float = 0.0,
    ):
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.dtype = dtype
        self.pow_2_scales = pow_2_scales
        self.eps = eps

    @classmethod
    def compute_scale(
        cls,
        x: torch.Tensor,
        quant_dtype: torch.dtype,
        eps=0.0,
        pow_2_scales: bool = False,
    ):
        # Use float32 for computation
        x_fp32 = x.to(torch.float32)

        if x_fp32.numel() == 0:
            amax = torch.empty(1, dtype=torch.float32, device=x.device)
        else:
            amax = torch.amax(torch.abs(x_fp32)).view(1)

        return _scale_from_amax_tensor(
            x.dtype,
            amax=amax,
            quant_dtype=quant_dtype,
            eps=eps,
            pow_2_scales=pow_2_scales,
        )

    def _quantize(self, tensor: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Python implementation of quantization (c++ kernel can be used as an option instead).
        Fake quantize tensor to FP4/FP8 and back - returns regular tensor.
        
        Common quantization logic used by both quantize and update_quantized methods.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor to quantize (should be 2D)
            
        Returns
        -------
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]
            (qx, sx, qx_t, sx_t) where:
            - qx: quantized data in row-major order (if rowwise_usage), None otherwise
            - sx: empty scale tensor for qx (if rowwise_usage), None otherwise
            - qx_t: quantized data in column-major order (if columnwise_usage), None otherwise
            - sx_t: empty scale tensor for qx_t (if columnwise_usage), None otherwise
        """
        # compute scale factor
        scale, scale_inv, _ = self.compute_scale(
            tensor,
            self.dtype,
            eps=self.eps,
            pow_2_scales=self.pow_2_scales,
        )

        qx: Optional[torch.Tensor] = (tensor.float() * scale).to(self.dtype)
        sx: Optional[torch.Tensor] = scale_inv

        # transpose if needed
        if self.columnwise_usage:
            assert qx is not None
            qx_t = qx.t().contiguous()
            sx_t = sx
        else:
            qx_t, sx_t = None, None

        if not self.rowwise_usage:
            qx = None
            sx = None

        return qx, sx, qx_t, sx_t

    def quantize(self, tensor: torch.Tensor, **kwargs) -> PerTensorExperimentalQuantizedTensor:
        """Quantize tensor"""
        original_shape = tensor.shape
        if tensor.ndim > 2:
            tensor = tensor.view(-1, tensor.shape[-1])

        # sanity check
        assert tensor.dtype in utils.HIGH_PRECISION_FLOAT_DTYPES, "Unsupported input dtype."

        qx, sx, qx_t, sx_t = self._quantize(tensor)

        return PerTensorExperimentalQuantizedTensor(
            data=qx,
            scale=sx,
            data_t=qx_t,
            scale_t=sx_t,
            dtype=tensor.dtype,
            device=tensor.device,
            low_precision_dtype=self.dtype,
            quantizer=self,
            original_shape=original_shape,
        )

    def dequantize(self, tensor: torch.Tensor, scale: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Dequantize the quantized tensor"""
        return (tensor.to(torch.float32) * scale).to(dtype)

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
        qresult_x: ExperimentalQuantizedTensorBase | None = None,
        qresult_w: ExperimentalQuantizedTensorBase | None = None,
    ) -> torch.Tensor:
        """Quantized GEMM interface."""
        M, K = qx.shape
        N, K_B = qw.shape

        if M == 0 or K == 0 or N == 0:
            if accumulate:
                assert out is not None
                y = out
            else:
                y = torch.zeros((M, N), dtype=out_dtype, device=qx.device)
            if bias is not None:
                y += bias
            return y

        # cublas fp8 gemm does not support fp32 bias
        use_bias_in_gemm = (
            bias is not None
            and out_dtype != torch.float32
            and bias.dtype != torch.float32
        )

        # Run quantized gemm: y = qw * qx
        scaled_mm_res = torch._scaled_mm(
            qx,
            qw.transpose(-1, -2),
            scale_a=sx,
            scale_b=sw,
            out_dtype=out_dtype,
            use_fast_accum=not m_params.use_split_accumulator,
            bias=bias if use_bias_in_gemm else None,
        )
        y = scaled_mm_res[0] if isinstance(scaled_mm_res, tuple) else scaled_mm_res

        if bias is not None and not use_bias_in_gemm:
            # Check number of elements in bias tensor because it can be an empty tensor
            if bias.numel():
                y += bias

        if accumulate:
            assert out is not None, "Output tensor must be provided for accumulation."
            out.add_(y)
            y = out
        else:
            assert out is None, "Output tensor should be None when accumulate is False."

        return y

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: ExperimentalQuantizedTensorBase,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> ExperimentalQuantizedTensorBase:
        """Update the quantized tensor with the given tensor in-place

        Parameters
        ----------
        src: torch.Tensor
            Source tensor to copy from
        dst: QuantizedExperimentalTensorBase
            Destination QuantizedExperimentalTensorBase to update
        noop_flag: torch.Tensor, optional
            float32 flag indicating whether to avoid performing update
        """
        # Handle noop flag
        if noop_flag is not None and noop_flag.item() != 0:
            return dst

        # Make sure input is in expected format
        if not src.is_contiguous():
            src = src.contiguous()

        # Store the original shape and reshape for processing
        original_shape = src.shape
        if src.ndim > 2:
            src = src.view(-1, src.shape[-1])

        qx, sx, qx_t, sx_t = self._quantize(src)

        # Update the destination with new data
        dst.data = qx
        dst.scale = sx
        dst.data_t = qx_t
        dst.scale_t = sx_t
        dst.dtype = src.dtype
        dst.low_precision_dtype = self.dtype
        dst.original_shape = original_shape

        return dst

    def make_empty(
            self,
            shape: Iterable[int],
            *,
            dtype: torch.dtype = torch.float32,
            device: Optional[torch.device] = None,
            requires_grad: bool = False,
    ) -> PerTensorExperimentalQuantizedTensor:
        assert len(shape) == 2, "shape is not 2d"

        # Canonicalize tensor attributes
        if device is None:
            device = torch.device("cuda")

        # Empty scale tensor for fake quantization (data is already dequantized)
        empty_scale = torch.empty(0)

        # Allocate quantized data
        qx = torch.empty(shape, dtype=self.dtype, device=device)
        sx = torch.empty(1, dtype=torch.float32, device=device)

        # Allocate quantized data transpose if needed
        qx_t = None
        sx_t = None
        if self.columnwise_usage:
            inner_dim = qx.size(-1)
            qx_t = torch.empty(
                inner_dim,
                qx.numel() // inner_dim,
                dtype=torch.uint8,
                device=device,
            )
            sx_t = torch.empty(1, dtype=torch.float32, device=device)

        # Construct quantized tensor
        return PerTensorExperimentalQuantizedTensor(
            data=qx,
            scale=sx,
            data_t=qx_t,
            scale_t=sx_t,
            dtype=dtype,
            device=device,
            low_precision_dtype=self.dtype,
            quantizer=self,
            original_shape=shape,
        )


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


class Float4Float8CurrentScalingEmulationRefQuantizer(PerTensorExperimentalQuantizer):
    """FP4/FP8 quantizer with current scaling and fake quantization"""

    """FP4/FP8 datatype"""
    dtype: Union[utils.Fp4Formats, torch.dtype]
    """Options about how to quantize the tensor"""
    pow_2_scales: bool
    eps: float

    def __init__(
        self,
        dtype: Union[utils.Fp4Formats, torch.dtype],
        rowwise: bool = True,
        columnwise: bool = True,
        pow_2_scales: bool = False,
        eps: float = 0.0,
    ):
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.dtype = dtype
        self.pow_2_scales = pow_2_scales
        self.eps = eps

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
            eps=self.eps,
            pow_2_scales=self.pow_2_scales,
        )

    def _quantize(self, tensor: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Python implementation of quantization (c++ kernel can be used as an option instead).
        Fake quantize tensor to FP4/FP8 and back - returns regular tensor.
        
        Common quantization logic used by both quantize and update_quantized methods.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor to quantize (should be 2D)
            
        Returns
        -------
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]
            (qx, sx, qx_t, sx_t) where:
            - qx: quantized data in row-major order (if rowwise_usage), None otherwise
            - sx: empty scale tensor for qx (if rowwise_usage), None otherwise
            - qx_t: quantized data in column-major order (if columnwise_usage), None otherwise
            - sx_t: empty scale tensor for qx_t (if columnwise_usage), None otherwise
        """
        scale, scale_inv, _ = self._compute_scale(tensor)

        # Initialize outputs
        qx, sx = None, None
        qx_t, sx_t = None, None
        
        # Empty scale tensor for fake quantization (data is already dequantized)
        empty_scale = torch.empty(0)

        # Compute identity (original layout) if needed
        if self.rowwise_usage:
            qx = self._fake_quantize_tensor(tensor, scale, scale_inv)
            sx = empty_scale

        # Compute transpose if needed
        if self.columnwise_usage:
            if self.rowwise_usage and qx is not None:
                # For per-tensor quantization, just transpose the result
                qx_t = qx.t().contiguous()
                sx_t = empty_scale  # Scale is empty because data is already dequantized
            else:
                # Transpose and quantize the transposed tensor
                x_t = tensor.t().contiguous()
                qx_t = self._fake_quantize_tensor(x_t, scale, scale_inv)
                sx_t = empty_scale  # Scale is empty because data is already dequantized

        return qx, sx, qx_t, sx_t

    def quantize(self, tensor: torch.Tensor, **kwargs) -> PerTensorExperimentalQuantizedTensor:
        """Quantize tensor"""
        original_shape = tensor.shape
        if tensor.ndim > 2:
            tensor = tensor.view(-1, tensor.shape[-1])

        qx, sx, qx_t, sx_t = self._quantize(tensor)

        return PerTensorExperimentalQuantizedTensor(
            data=qx,
            scale=sx,
            data_t=qx_t,
            scale_t=sx_t,
            dtype=tensor.dtype,
            device=tensor.device,
            low_precision_dtype=self.dtype,
            quantizer=self,
            original_shape=original_shape,
        )

    def dequantize(self, tensor: torch.Tensor, scale: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Dequantize the quantized tensor"""
        # For fake quantization, data is already dequantized
        if dtype is None:
            return tensor
        return tensor.to(dtype)

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
        qresult_x: ExperimentalQuantizedTensorBase | None = None,
        qresult_w: ExperimentalQuantizedTensorBase | None = None,
    ) -> torch.Tensor:
        """Quantized GEMM interface."""
        assert (
            qx.dtype in utils.HIGH_PRECISION_FLOAT_DTYPES
        ), f"Unsupported input dtype, {qx.dtype}"
        assert (
            qw.dtype in utils.HIGH_PRECISION_FLOAT_DTYPES
        ), f"Unsupported weight dtype, {qw.dtype}"
        assert qx.is_contiguous(), "qx must be contiguous."
        assert qw.is_contiguous(), "qw must be contiguous."

        # For fake quantization, scales should be empty
        assert sx.numel() == 0, "FP4/FP8 emulation should have empty scale tensors"
        assert sw.numel() == 0, "FP4/FP8 emulation should have empty scale tensors"

        # Extract dimensions based on GEMM type
        if gemm_type == GEMMType.FPROP:
            M, K = qx.shape  # qx: (M, K)
            N, K_B = qw.shape  # qw: (N, K)
        elif gemm_type == GEMMType.DGRAD:
            M, N = qx.shape  # qx: (M, N) - dY
            K, N_B = qw.shape  # qw: (K, N) - W.t() (transposed weight)
            assert N == N_B, f"Shape mismatch: qx has N={N}, qw has N={N_B}"
        else:  # WGRAD
            N, M = qx.shape  # qx: (N, M) - dY.t()
            K, M_B = qw.shape  # qw: (K, M) - X.t() (transposed input)
            assert M == M_B, f"Shape mismatch: qx has M={M}, qw has M={M_B}"

        # Handle empty tensor cases
        if M == 0 or K == 0 or N == 0:
            if accumulate:
                assert out is not None
                y = out
            else:
                # Create output with correct shape based on GEMM type
                if gemm_type == GEMMType.FPROP:
                    y = torch.zeros((M, N), dtype=out_dtype, device=qx.device)
                elif gemm_type == GEMMType.DGRAD:
                    y = torch.zeros((M, K), dtype=out_dtype, device=qx.device)
                else:  # WGRAD
                    y = torch.zeros((N, K), dtype=out_dtype, device=qx.device)
            # Only add bias if we have valid dimensions for GEMM  
            if bias is not None and gemm_type == GEMMType.FPROP and K > 0:
                y += bias
            return y

        if gemm_type == GEMMType.FPROP:
            # fwd:    Y = x @ w.t()  # [m, k] x [k, n] = [m, n]
            if bias is not None:
                if out_dtype == torch.float32:
                    y = torch.matmul(qx, qw.t())
                    y += bias
                else:
                    y = torch.addmm(bias, qx, qw.t(), beta=1, alpha=1)
            else:
                y = torch.matmul(qx, qw.t())
        elif gemm_type == GEMMType.DGRAD:
            # dgrad: dX = dY @ W     # [m, n] x [n, k] = [m, k]
            # Note: qw is W.t() from linear.py, so we transpose to get W  
            y = torch.matmul(qx, qw.t())
        elif gemm_type == GEMMType.WGRAD:
            # wgrad: dW = dY.t() @ X # [n, m] x [m, k] = [n, k]
            # Note: qx is dY.t() with shape (N, M), qw is X.t() with shape (K, M)
            # So we need qw.t() to get X from X.t()
            if accumulate:
                assert (
                    out is not None
                ), "Output tensor must be provided for accumulation."
                # Convert tensors to same dtype as out for addmm
                qx_dtype = qx.to(out.dtype)
                qw_dtype = qw.t().to(out.dtype)  # Transpose qw to get X from X.t()
                torch.addmm(out, qx_dtype, qw_dtype, beta=1, alpha=1, out=out)
                y = out
            else:
                y = torch.matmul(qx, qw.t())  # Transpose qw to get X from X.t()
        else:
            raise NotImplementedError(f"Unsupported GEMM type: {gemm_type}")

        # Handle accumulation for non-WGRAD cases
        if accumulate and gemm_type != GEMMType.WGRAD:
            assert out is not None, "Output tensor must be provided for accumulation."
            out.add_(y.to(out.dtype))
            y = out

        return y.to(out_dtype)

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: ExperimentalQuantizedTensorBase,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> ExperimentalQuantizedTensorBase:
        """Update the quantized tensor with the given tensor in-place

        Parameters
        ----------
        src: torch.Tensor
            Source tensor to copy from
        dst: QuantizedExperimentalTensorBase
            Destination QuantizedExperimentalTensorBase to update
        noop_flag: torch.Tensor, optional
            float32 flag indicating whether to avoid performing update
        """
        # Handle noop flag
        if noop_flag is not None and noop_flag.item() != 0:
            return dst

        # Make sure input is in expected format
        if not src.is_contiguous():
            src = src.contiguous()

        # Store the original shape and reshape for processing
        original_shape = src.shape
        if src.ndim > 2:
            src = src.view(-1, src.shape[-1])

        qx, sx, qx_t, sx_t = self._quantize(src)

        # Update the destination with new data
        dst.data = qx
        dst.scale = sx
        dst.data_t = qx_t
        dst.scale_t = sx_t
        dst.dtype = src.dtype
        dst.low_precision_dtype = self.dtype
        dst.original_shape = original_shape

        return dst

    def make_empty(
            self,
            shape: Iterable[int],
            *,
            dtype: torch.dtype = torch.float32,
            device: Optional[torch.device] = None,
            requires_grad: bool = False,
    ) -> PerTensorExperimentalQuantizedTensor:
        assert len(shape) == 2, "shape is not 2d"

        # Canonicalize tensor attributes
        if device is None:
            device = torch.device("cuda")

        # Empty scale tensor for fake quantization (data is already dequantized)
        empty_scale = torch.empty(0)

        # Allocate quantized data
        qx = torch.empty(shape, dtype=dtype, device=device)
        sx = empty_scale

        # Allocate quantized data transpose if needed
        qx_t = None
        sx_t = None
        if self.columnwise_usage:
            inner_dim = qx.size(-1)
            qx_t = torch.empty(
                inner_dim,
                qx.numel() // inner_dim,
                dtype=dtype,
                device=device,
            )
            sx_t = empty_scale

        # Construct quantized tensor
        return PerTensorExperimentalQuantizedTensor(
            data=qx,
            scale=sx,
            data_t=qx_t,
            scale_t=sx_t,
            dtype=dtype,
            device=device,
            low_precision_dtype=self.dtype,
            quantizer=self,
            original_shape=shape,
        )
