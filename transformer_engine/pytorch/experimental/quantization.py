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
class ExperimentalQuantizedTensorBase(QuantizedTensorBase):
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
    quantizer: Optional[ExperimentalQuantizerBase] = None

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


class ExperimentalQuantizerBase(abc.ABC):
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
        self.internal = True

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"rowwise_usage={self.rowwise_usage}, "
            f"columnwise_usage={self.columnwise_usage}, "
            f"internal={self.internal}, "
            ")"
        )

    @abc.abstractmethod
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

    @abc.abstractmethod
    def quantize(self, tensor: torch.Tensor, **kwargs) -> ExperimentalQuantizedTensorBase:
        """Quantize tensor"""
        pass

    @abc.abstractmethod
    def dequantize(self, tensor: torch.Tensor, scale: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Dequantize tensor"""
        pass

    def __call__(self, tensor: torch.Tensor) -> ExperimentalQuantizedTensorBase:
        """Quantize tensor"""
        return self.quantize(tensor)

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
        qresult_x: ExperimentalQuantizedTensorBase | None = None,
        qresult_w: ExperimentalQuantizedTensorBase | None = None,
    ) -> torch.Tensor:
        """Quantized GEMM interface."""

    def set_usage(
        self, *, rowwise: Optional[bool] = None, columnwise: Optional[bool] = None
    ) -> None:
        """Set how the quantized tensor is expected to be used"""
        if rowwise is not None:
            self.rowwise_usage = rowwise
        if columnwise is not None:
            self.columnwise_usage = columnwise
