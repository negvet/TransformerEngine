# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Container class holding data specific for NVFP4Tensor"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
from transformer_engine_torch import DType as TE_DType

from ..quantized_tensor import QuantizedTensorBase, Quantizer


class NVFP4TensorBase(QuantizedTensorBase):
    """Container that holds data attributes of NVFP4Tensor."""

    _rowwise_data: Optional[torch.Tensor]
    _columnwise_data: Optional[torch.Tensor]
    _quantizer: Optional[Quantizer]
    _rowwise_scale_inv: Optional[torch.Tensor]
    _columnwise_scale_inv: Optional[torch.Tensor]
    _global_amax: Optional[torch.Tensor]
    _original_shape: Optional[Tuple[int, ...]]
    _quant_dtype: Optional[TE_DType]

    def __init__(
        self,
        rowwise_data: Optional[torch.Tensor] = None,
        rowwise_scale_inv: Optional[torch.Tensor] = None,
        columnwise_data: Optional[torch.Tensor] = None,
        columnwise_scale_inv: Optional[torch.Tensor] = None,
        global_amax: Optional[torch.Tensor] = None,
        quantizer: Optional[Quantizer] = None,
        original_shape: Optional[Tuple[int, ...]] = None,
        quant_dtype: Optional[TE_DType] = None,
    ):
        self._rowwise_data = rowwise_data
        self._columnwise_data = columnwise_data
        self._rowwise_scale_inv = rowwise_scale_inv
        self._columnwise_scale_inv = columnwise_scale_inv
        self._global_amax = global_amax
        self._quantizer = quantizer
        self._original_shape = original_shape
        self._quant_dtype = quant_dtype

    def get_quantizer(self) -> Quantizer:
        """Get builder for NVFP4Tensor

        Quantizer can be used for in-place operations.

        """
        if self._quantizer is not None:
            return self._quantizer
        raise ValueError("Quantizer is not set")

    def quantize_(
        self,
        tensor: torch.Tensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> NVFP4TensorBase:
        """In-place update of NVFP4 data

        Parameters
        ----------
        tensor: torch.Tensor
            Tensor to copy from
        noop_flag: torch.Tensor, optional
            float32 flag indicating whether to avoid performing update

        """
        self.get_quantizer().update_quantized(tensor, self, noop_flag=noop_flag)
        return self

    def prepare_for_saving(self) -> Tuple[list[Optional[torch.Tensor]], NVFP4TensorBase]:
        """Prepare the quantization result for saving for backward"""
        tensors = [self._rowwise_data, self._columnwise_data, self._rowwise_scale_inv, self._columnwise_scale_inv]
        self._rowwise_data = None
        self._columnwise_data = None
        self._rowwise_scale_inv = None
        self._columnwise_scale_inv = None
        return tensors, self

    def restore_from_saved(self, tensors: list[Optional[torch.Tensor]]) -> list[Optional[torch.Tensor]]:
        """Restore the quantization result from the saved tensors"""
        self._rowwise_data = tensors[0]
        self._columnwise_data = tensors[1]
        self._rowwise_scale_inv = tensors[2]
        self._columnwise_scale_inv = tensors[3]
        return tensors[4:]

    def dequantize(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Dequantize to a higher precision."""
        raise NotImplementedError("Not implemented yet")

    def update_usage(
        self,
        rowwise_usage: Optional[bool] = None,
        columnwise_usage: Optional[bool] = None,
    ) -> None:
        """Generate or remove quantized data based on provided usage."""
        has_data = self._rowwise_data is not None
        has_data_transpose = self._columnwise_data is not None
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
            self._rowwise_data = None
        if not needs_data_transpose:
            self._columnwise_data = None

    def _create_transpose(self) -> None:
        """Create transposed quantized tensor"""
        if not self._rowwise_data.is_contiguous():
            self._rowwise_data = self._rowwise_data.contiguous()
        self._columnwise_data = self._rowwise_data.t().contiguous()
        self._columnwise_scale_inv = self._rowwise_scale_inv
