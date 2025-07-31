# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tensor class with NVFP4 data"""

import dataclasses
import enum
from typing import Iterable, Optional, Tuple, Union

import torch
from transformer_engine_torch import DType as TE_DType

from .quantized_tensor import Quantizer
from ._internal.nvfp4_tensor_base import NVFP4TensorBase
from transformer_engine.common.recipe import Recipe


HIGH_PRECISION_FLOAT_DTYPES = (
    torch.float,
    torch.float16,
    torch.bfloat16,
    torch.float32,
)


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


def cast_to_fp4x2(x: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(x, dtype=torch.uint8)
    result[(x >= 0.0) & (x <= 0.25)] = 0
    result[(x > 0.25) & (x < 0.75)] = 1
    result[(x >= 0.75) & (x <= 1.25)] = 2
    result[(x > 1.25) & (x < 1.75)] = 3
    result[(x >= 1.75) & (x <= 2.5)] = 4
    result[(x > 2.5) & (x < 3.5)] = 5
    result[(x >= 3.5) & (x <= 5.0)] = 6
    result[x > 5.0] = 7

    result[(x >= -0.25) & (x < -0.0)] = 8
    result[(x < -0.25) & (x > -0.75)] = 9
    result[(x <= -0.75) & (x >= -1.25)] = 10
    result[(x < -1.25) & (x > -1.75)] = 11
    result[(x <= -1.75) & (x >= -2.5)] = 12
    result[(x < -2.5) & (x > -3.5)] = 13
    result[(x <= -3.5) & (x >= -5.0)] = 14
    result[x < -5.0] = 15

    return result[:, ::2] + result[:, 1::2] * 16


def cast_from_fp4x2(x: torch.Tensor, dq_dtype: torch.dtype) -> torch.Tensor:
    fp4_values = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device=x.device,
        dtype=dq_dtype,
    )

    # Convert to long integers for indexing
    second_bit = torch.div(x, 16, rounding_mode="floor").to(torch.long)
    first_bit = (x - second_bit * 16).to(torch.long)

    # Use the long integers to index fp4_values
    first_bit_values = fp4_values[first_bit]
    second_bit_values = fp4_values[second_bit]

    result = torch.zeros(
        (first_bit_values.shape[0], first_bit_values.shape[1] * 2),
        device=x.device,
        dtype=dq_dtype,
    )
    result[:, ::2] = first_bit_values
    result[:, 1::2] = second_bit_values

    return result


def cast_to_e4m3(decode_scale: torch.Tensor, global_amax: torch.Tensor) -> torch.Tensor:
    decode_scale = decode_scale * global_amax
    FLOAT8_E4M3_MAX = torch.tensor(
        448.0, device=decode_scale.device, dtype=torch.float32
    )
    decode_scale = torch.clamp(decode_scale, min=-FLOAT8_E4M3_MAX, max=FLOAT8_E4M3_MAX)
    return decode_scale.to(torch.float8_e4m3fn)


def high_precision_gemm_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype,
    accumulate: bool = False,
    is_a_transposed: bool = False,
    is_b_transposed: bool = False,
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    scale_alpha: float = 1.0,
) -> torch.Tensor:
    
    # Handle transpositions
    mat1, mat2 = a, b
    if is_a_transposed:
        mat1 = a.T
    if is_b_transposed:
        mat2 = b.T
    
    # Ensure dtype compatibility for torch.addmm
    mat1 = mat1.to(out_dtype)
    mat2 = mat2.to(out_dtype)
    
    # Determine output shape
    y_shape = (mat1.size(0), mat2.size(1))
    
    if bias is not None:
        assert not accumulate, "Bias is not supported with accumulation"
        bias = bias.to(out_dtype)
        # With bias case
        if out_dtype == torch.float32:
            y_ref = torch.addmm(
                bias.repeat(mat1.size(0), 1), mat1, mat2, beta=1, alpha=1
            )
        else:
            y_ref = torch.addmm(bias, mat1, mat2, beta=1, alpha=scale_alpha)
    else:
        # Without bias case
        if accumulate and out is not None:
            y_ref = out.clone().to(out_dtype)
        else:
            y_ref = torch.zeros(y_shape, dtype=out_dtype, device=a.device)
        torch.addmm(y_ref, mat1, mat2, beta=1, alpha=scale_alpha, out=y_ref)
    
    return y_ref


class NVFP4Quantizer(Quantizer):
    """Builder class for NVFP4 tensors"""

    def __init__(
        self,
        dtype: Optional[TE_DType] = None,
        rowwise: bool = True,
        columnwise: bool = True,
        quant_tile_shape: Tuple[int, int] = (1, 16),
    ):
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.dtype = dtype
        self.quant_tile_shape = quant_tile_shape
        self.internal = True

    @classmethod
    def _quantize_vectorwise_reference(
        cls,
        x: torch.Tensor,
        global_amax: torch.Tensor,
        tile_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert x.ndim == 2
        m, n = x.shape
        x = torch.reshape(x, (m, n // tile_len, tile_len))
        vec_max = torch.max(torch.abs(x), dim=-1, keepdim=True)[0].to(torch.float32)
        FLOAT4_E2M1_MAX = torch.tensor(6.0, device=x.device, dtype=torch.float32)
        FLOAT8_E4M3_MAX = torch.tensor(448.0, device=x.device, dtype=torch.float32)
        decode_scale = torch.div(vec_max, FLOAT4_E2M1_MAX)

        global_encode_scale = torch.div(
            FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX, global_amax
        )
        global_encode_scale = torch.clamp(
            global_encode_scale,
            max=torch.finfo(torch.float32).max,
        )
        if global_encode_scale == torch.tensor(
            0.0, device=x.device, dtype=torch.float32
        ):
            global_encode_scale = torch.tensor(
                1.0, device=x.device, dtype=torch.float32
            )
        decode_scale = cast_to_e4m3(decode_scale, global_encode_scale)

        encode_scale = torch.div(
            global_encode_scale,
            decode_scale.to(torch.float32),
        )
        encode_scale = torch.clamp(encode_scale, max=torch.finfo(torch.float32).max)

        scaled_x = x.to(torch.float32) * encode_scale
        clipped_x = torch.clamp(scaled_x, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX).reshape(
            m, n
        )
        return cast_to_fp4x2(clipped_x), decode_scale.squeeze(-1)

    @staticmethod
    def _pad_tensor(
        tensor: torch.Tensor, row_divisor: Optional[int], col_divisor: Optional[int]
    ) -> torch.Tensor:

        assert tensor.dim() == 2, "only supports 2D tensors"
        M, N = tensor.shape
        padding_needed_rows = 0
        padding_needed_cols = 0

        if row_divisor is not None and M % row_divisor != 0:
            padding_needed_rows = row_divisor - (M % row_divisor)
        # Check and calculate column padding if col_divisor is provided
        if col_divisor is not None and N % col_divisor != 0:
            padding_needed_cols = col_divisor - (N % col_divisor)

        # Return original tensor if no padding is needed
        if padding_needed_rows == 0 and padding_needed_cols == 0:
            return tensor

        # pad the tensor
        out = torch.nn.functional.pad(
            tensor,
            (0, padding_needed_cols, 0, padding_needed_rows),
            mode="constant",
            value=0.0,
        ).contiguous()

        return out

    @staticmethod
    def _rm_pad_tensor(
        tensor: torch.Tensor, original_size: tuple[int, ...]
    ) -> torch.Tensor:

        assert tensor.dim() == 2, "only supports 2D tensors"
        M, N = original_size
        out = tensor[:M, :N].contiguous()
        return out

    def _quantize(
        self,
        tensor: torch.Tensor
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
    ]:
        """
        Python implementation of NVFP4 quantization.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor to quantize (should be 2D)

        Returns
        -------
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]
            (qx, sx, qx_t, sx_t, global_amax) where:
            - qx: quantized data in row-major order (if self.rowwise_usage), None otherwise
            - sx: scale tensor for qx (if self.rowwise_usage), None otherwise
            - qx_t: quantized data in column-major order (if self.columnwise_usage), None otherwise
            - sx_t: scale tensor for qx_t (if self.columnwise_usage), None otherwise
            - global_amax: global amax tensor
        """
        assert self.quant_tile_shape == (
            1,
            16,
        ), "NVFP4 only supports 1x16 tile shape."
        global_amax = torch.max(torch.abs(tensor)).to(torch.float32).view(1)

        transpose_scales = False

        M, N = tensor.shape
        if self.rowwise_usage:
            x_padded = self._pad_tensor(
                tensor, row_divisor=None, col_divisor=self.quant_tile_shape[1]
            )

            qx, sx = self._quantize_vectorwise_reference(
                x_padded,
                global_amax,
                self.quant_tile_shape[1],
            )
            if transpose_scales:
                sx = sx.T

            qx = self._rm_pad_tensor(qx, (M, N // 2))

        else:
            qx = None
            sx = None

        if self.columnwise_usage:
            x_t = tensor.t().contiguous()
            x_t_padded = self._pad_tensor(
                x_t, row_divisor=None, col_divisor=self.quant_tile_shape[1]
            )

            qx_t, sx_t = self._quantize_vectorwise_reference(
                x_t_padded,
                global_amax,
                self.quant_tile_shape[1],
            )

            qx_t = self._rm_pad_tensor(qx_t, (N, M // 2))

            if transpose_scales:
                sx_t = sx_t.T
        else:
            qx_t = None
            sx_t = None

        return qx, sx, qx_t, sx_t, global_amax

    def quantize(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> NVFP4TensorBase:
        # sanity checks
        assert x.dtype in HIGH_PRECISION_FLOAT_DTYPES, "Unsupported input dtype."

        # Make it work with 3D tensors
        original_shape = x.shape
        if x.ndim > 2:
            x = x.view(-1, x.shape[-1])

        qx, sx, qx_t, sx_t, global_amax = self._quantize(x)

        return NVFP4TensorBase(
            rowwise_data=qx,
            rowwise_scale_inv=sx,
            columnwise_data=qx_t,
            columnwise_scale_inv=sx_t,
            global_amax=global_amax,
            quantizer=self,
            original_shape=original_shape,
            quant_dtype=self.dtype,
        )

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: NVFP4TensorBase,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> NVFP4TensorBase:
        """Update the quantized tensor with the given tensor in-place

        Parameters
        ----------
        src: torch.Tensor
            Source tensor to copy from
        dst: NVFP4TensorBase
            Destination NVFP4TensorBase to update
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

        qx, sx, qx_t, sx_t, global_amax = self._quantize(src)

        # Update the destination with new data
        dst._rowwise_data = qx
        dst._rowwise_scale_inv = sx
        dst._columnwise_data = qx_t
        dst._columnwise_scale_inv = sx_t
        dst._global_amax = global_amax
        dst._dtype = src.dtype
        dst._quant_dtype = self.dtype
        dst._original_shape = original_shape

        return dst

    @property
    def supports_allgather_fp8(self) -> bool:
        return False

    def dequantize(self, tensor: torch.Tensor, scale: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Dequantize the quantized tensor"""
        raise NotImplementedError("Not implemented yet")

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
        qresult_x: NVFP4TensorBase | None = None,
        qresult_w: NVFP4TensorBase | None = None,
    ) -> torch.Tensor:
        assert bias is None, "Bias is not implemented for FP4 GEMM."

        high_precision_x = cast_from_fp4x2(qx, out_dtype)
        high_precision_w = cast_from_fp4x2(qw, out_dtype)

        assert qresult_x is not None
        assert qresult_w is not None

        assert qresult_x._global_amax is not None
        assert qresult_w._global_amax is not None

        sx = sx.to(torch.float32)
        sw = sw.to(torch.float32)

        factor = 6.0 * 6.0 * 448.0 * 448.0

        alpha = torch.div(
            qresult_x._global_amax * qresult_w._global_amax, factor
        ).squeeze(-1)

        M, K = high_precision_x.shape
        N, K_w = high_precision_w.shape
        assert K == K_w, "K dimension mismatch between qx and qw"

        assert K % 32 == 0, "K dimension must be divisible by 32"
        assert N % 8 == 0, "N dimension must be divisible by 8"

        block_length = 16

        grid_k = K // block_length

        assert sx.shape == (
            M,
            K // block_length,
        ), f"sx shape mismatch: expected ({M}, {K//block_length}), got {sx.shape}"
        assert sw.shape == (
            N,
            K // block_length,
        ), f"sw shape mismatch: expected ({N}, {K//block_length}), got {sw.shape}"

        y = torch.zeros(M, N, dtype=torch.float32, device=qx.device)

        # below implementation is to match the FP4 tensor core implementation
        # Each output element (i, j) is fp32 accumulation of (K // block_length) inner products
        # Each inner product is sx * sw * (1, block_length) x (block_length, 1) with precision in fp32
        # Then batch the computation in M, N dimension
        for k in range(grid_k):
            k_start = k * block_length
            k_end = k_start + block_length

            qx_block = high_precision_x[:, k_start:k_end].clone().contiguous()
            qw_block = high_precision_w[:, k_start:k_end].clone().contiguous()

            # Extract scaling factors for the current blocks
            sx_block = sx[:, k]
            sw_block = sw[:, k]

            y += torch.outer(sx_block, sw_block) * high_precision_gemm_ref(
                qx_block, qw_block, torch.float32, is_b_transposed=True
            )

        if K > 0:
            # only apply global scale for NVFP4 and non-empty cases
            y = alpha * y

        # accumulation happens at epilogue in float32
        if accumulate:
            assert out is not None, "Output tensor must be provided for accumulation."
            y += out.to(torch.float32)
        else:
            assert out is None, "Output tensor should be None when accumulate is False."

        y = y.to(out_dtype)
        return y

    def _get_compatible_recipe(self) -> Union[type[Recipe], None]:
        """Returns recipe class that is compatible with this quantizer"""
        raise NotImplementedError("Not implemented yet")

    def calibrate(self, tensor: torch.Tensor) -> None:
        """Calibrate quantizer state

        Updates quantization state as if quantizing a tensor, but
        without actually performing the quantization.

        """
        raise NotImplementedError("Not implemented yet")

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> NVFP4TensorBase:
        """Construct quantized tensor with uninitialized data"""
        raise NotImplementedError("Not implemented yet")
