# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
from typing import Tuple, Optional

from transformer_engine.pytorch.experimental import quantization
from transformer_engine.pytorch.experimental.quantization import ExperimentalQuantizedTensor, ExperimentalQuantizer
from transformer_engine.pytorch.experimental import utils


class PerTensorQuantizedTensor(ExperimentalQuantizedTensor):
    """Quantized tensor container for per-tensor (tensorwise) scaling."""

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"dtype={self.dtype}, "
            f"device={self.device}, "
            f"quant_dtype={self.quant_dtype}, "
            f"data={self.dequantize(dtype=self.dtype)}, "
            f"original_shape={self.original_shape}"
            ")"
        )

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


class Float8CurrentScalingQuantizerRef(ExperimentalQuantizer):
    """FP8 quantizer with current scaling"""

    """FP8 datatype"""
    dtype: torch.dtype
    """amax reduction options"""
    with_amax_reduction: bool
    amax_reduction_group: Optional[torch.distributed.ProcessGroup]
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
        self.with_amax_reduction = False
        self.amax_reduction_group = None
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

        return quantization._scale_from_amax_tensor(
            x.dtype,
            amax=amax,
            quant_dtype=quant_dtype,
            eps=eps,
            pow_2_scales=pow_2_scales,
        )

    def _quantize(self, tensor: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Python implementation of quantization (c++ kernel can be used as an option instead).
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
            scale, scale_inv, _ = quantization._scale_from_amax_tensor(
                tensor.dtype,
                amax=amax,
                quant_dtype=self.dtype,
                eps=self.eps,
                pow_2_scales=self.pow_2_scales,
            )
        else:
            # compute scale factor using local amax
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

    def quantize_impl(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> PerTensorQuantizedTensor:
        # sanity checks
        assert x.dtype in utils.HIGH_PRECISION_FLOAT_DTYPES, "Unsupported input dtype."

        # Make it work with 3D tensors
        original_shape = x.shape
        if x.ndim > 2:
            x = x.view(-1, x.shape[-1])

        qx, sx, qx_t, sx_t = self._quantize(x)

        return PerTensorQuantizedTensor(
            data=qx,
            scale=sx,
            data_t=qx_t,
            scale_t=sx_t,
            dtype=x.dtype,
            device=x.device,
            original_shape=original_shape,
            _quantizer=self,
        )

    def qgemm(
        self,
        qx: torch.Tensor,
        qw: torch.Tensor,
        m_params: quantization.MMParams,
        out_dtype: torch.dtype,
        sx: torch.Tensor,
        sw: torch.Tensor,
        bias: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        accumulate: bool = False,
        gemm_type: quantization.GEMMType = quantization.GEMMType.FPROP,
        qresult_x: quantization.ExperimentalQuantizedTensor | None = None,
        qresult_w: quantization.ExperimentalQuantizedTensor | None = None,
    ) -> torch.Tensor:
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
        dst: ExperimentalQuantizedTensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> ExperimentalQuantizedTensor:
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
        dst.quant_dtype = self.dtype
        dst.original_shape = original_shape

        return dst
