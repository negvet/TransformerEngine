# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for GEMM extensions"""

from typing import Iterable, Optional, Tuple, Union, List
import os
import torch
import transformer_engine_torch as tex
from ..constants import TE_DType
from ..utils import get_sm_count, _empty_tensor

from ..tensor.quantized_tensor import Quantizer
from ..tensor._internal.float8_blockwise_tensor_base import Float8BlockwiseQTensorBase
from ..tensor._internal.nvfp4_tensor_base import NVFP4TensorBase
from ..tensor.nvfp4_tensor import GEMMType, MMParams
from ...debug.pytorch.debug_quantization import DebugQuantizer

__all__ = [
    "general_gemm",
    "general_grouped_gemm",
]


def _nvfp4_qgemm(
    A: NVFP4TensorBase,
    B: NVFP4TensorBase,
    workspace: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    quantization_params: Optional[Quantizer] = None,
    gelu: bool = False,
    gelu_in: torch.Tensor = None,
    accumulate: bool = False,
    layout: str = "TN",
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    use_split_accumulator: bool = False,
    grad: bool = False,
    ub: Union[tex.CommOverlap, tex.CommOverlapP2P] = None,
    ub_type: tex.CommOverlapType = None,
    extra_output: Optional[torch.Tensor] = None,
    bulk_overlap: bool = False,
) -> Iterable[Optional[torch.Tensor]]:
    """Dispatch GEMM to quantizer's qgemm method."""
    assert isinstance(A, NVFP4TensorBase) and isinstance(B, NVFP4TensorBase), "A and B must be NVFP4TensorBase instances"

    A, B = B, A

    # Determine GEMM type based on grad flag and layout
    if not grad:
        gemm_type = GEMMType.FPROP
    else:
        if layout == "NN":
            gemm_type = GEMMType.DGRAD
        elif layout == "NT":
            gemm_type = GEMMType.WGRAD
        else:
            # Default to FPROP for other layouts
            gemm_type = GEMMType.FPROP

    # Extract quantizer from QuantizedTensor to get qgemm logic
    # TODO: make it more flexible, what if we might want to use gemm logic from B.quantizer?
    quantizer = None
    if hasattr(A, '_quantizer') and A._quantizer is not None:
        quantizer = A._quantizer
    elif hasattr(B, '_quantizer') and B._quantizer is not None:
        quantizer = B._quantizer
    else:
        raise ValueError("No quantizer found in QuantizedETensor objects")

    # Create MMParams
    m_params = MMParams(
        out_dtype=out_dtype,
        use_split_accumulator=use_split_accumulator,
    )
    out_dtype = (
        A.dtype
        if m_params.out_dtype is None
        else m_params.out_dtype
    )

    if gemm_type == GEMMType.FPROP:
        qx, sx = A._rowwise_data, A._rowwise_scale_inv
        qw, sw = B._rowwise_data, B._rowwise_scale_inv
        assert qx is not None
        assert sx is not None
        assert qw is not None
        assert sw is not None
        assert A._original_shape is not None

        # Call quantizer's qgemm method
        result = quantizer.qgemm(
            qx,
            qw,
            m_params,
            out_dtype,
            sx,
            sw,
            bias,
            gemm_type=GEMMType.FPROP,
            qresult_x=A,
            qresult_w=B,
        )
        if len(A._original_shape) > 2:
            # Original input was 3D, so we need to reshape result back to 3D
            batch_size = A._original_shape[0]
            seq_len = A._original_shape[1]
            result = result.view(batch_size, seq_len, result.shape[-1])
    elif gemm_type == GEMMType.DGRAD:
        qdy, sdy = A._rowwise_data, A._rowwise_scale_inv
        qw_t, sw_t = B._columnwise_data, B._columnwise_scale_inv
        assert qdy is not None
        assert sdy is not None
        assert qw_t is not None
        assert sw_t is not None

        result = quantizer.qgemm(
            qdy,
            qw_t,
            m_params,
            out_dtype,
            sdy,
            sw_t,
            None,
            gemm_type=GEMMType.DGRAD,
            qresult_x=A,
            qresult_w=B,
        )
    elif gemm_type == GEMMType.WGRAD:
        qdy_t, sdy_t = A._columnwise_data, A._columnwise_scale_inv
        qx_t, sx_t = B._columnwise_data, B._columnwise_scale_inv
        assert qdy_t is not None
        assert sdy_t is not None
        assert qx_t is not None
        assert sx_t is not None

        result = quantizer.qgemm(
            qdy_t,
            qx_t,
            m_params,
            out_dtype,
            sdy_t,
            sx_t,
            None,
            gemm_type=GEMMType.WGRAD,
            qresult_x=A,
            qresult_w=B,
        )

    # Return in the same format as general_gemm
    return result, None, None, None


def general_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    workspace: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    quantization_params: Optional[Quantizer] = None,
    gelu: bool = False,
    gelu_in: torch.Tensor = None,
    accumulate: bool = False,
    layout: str = "TN",
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    use_split_accumulator: bool = False,
    grad: bool = False,
    ub: Union[tex.CommOverlap, tex.CommOverlapP2P] = None,
    ub_type: tex.CommOverlapType = None,
    extra_output: Optional[torch.Tensor] = None,
    bulk_overlap: bool = False,
) -> Iterable[Optional[torch.Tensor]]:
    """GEMM supporting fp8 inputs."""

    assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
    transa = layout[0] == "T"
    transb = layout[1] == "T"
    # assert quantization_params is None, "FP8 output not supported yet"

    if ub_type is not None:
        assert ub is not None, (
            f"{'AG+GEMM' if ub_type == tex.CommOverlapType.AG else 'GEMM+RS'} overlap requires"
            + "a valid `ub` communicator object."
        )

    if ub is not None:
        assert ub_type is not None, "Comm+GEMM overlap requires a valid `comm_type` argument."
        if ub_type == tex.CommOverlapType.RS:
            if not (bulk_overlap and not ub.is_fp8_ubuf()):
                assert extra_output is not None, "GEMM+RS overlap requires extra output tensor."

    if out is not None:
        if not out.is_contiguous():
            raise ValueError("Output tensor is not contiguous.")

    # If A or B are NVFP4TensorBase instances -> dispatch to quantizers's qgemm implementation
    if isinstance(A, NVFP4TensorBase) or isinstance(B, NVFP4TensorBase):
        return _nvfp4_qgemm(
            A, B, workspace, out_dtype, quantization_params, gelu, gelu_in, 
            accumulate, layout, out, bias, use_split_accumulator, grad, 
            ub, ub_type, extra_output, bulk_overlap
        )

    debug_quantizer = None
    if isinstance(quantization_params, DebugQuantizer):
        debug_quantizer = quantization_params
        quantization_params = quantization_params.parent_quantizer
        A = A.get_tensor(not transa)
        B = B.get_tensor(transb)

    # Use bfloat16 as default bias_dtype
    bias_dtype = TE_DType[torch.bfloat16 if bias is None else bias.dtype]

    if isinstance(A, Float8BlockwiseQTensorBase) or isinstance(B, Float8BlockwiseQTensorBase):
        # There is not use_split_accumulator == False
        # implementation for Float8BlockwiseQTensorBase GEMM
        use_split_accumulator = True

        # Check that data format is supported
        if (
            A._data_format != tex.Float8BlockScaleTensorFormat.GEMM_READY
            or B._data_format != tex.Float8BlockScaleTensorFormat.GEMM_READY
        ):
            raise RuntimeError("GEMM with Float8BlockwiseQTensor requires GEMM_READY format")

    args = (
        A,
        transa,  # transa
        B,
        transb,  # transb
        out,
        quantization_params,
        TE_DType[out_dtype] if out_dtype is not None else None,
        bias,
        bias_dtype,
        gelu,
        gelu_in,
        grad,  # grad
        workspace,
        workspace.shape[0],
        accumulate,
        use_split_accumulator,
    )
    kwargs = {
        "comm_overlap": ub,
        "comm_type": ub_type,
        "extra_output": extra_output,
        "bulk_overlap": bulk_overlap,
    }

    out, bias_grad, gelu_input, extra_output = tex.generic_gemm(*args, **kwargs)

    if debug_quantizer is not None:
        out = debug_quantizer.process_gemm_output(out)

    return out, bias_grad, gelu_input, extra_output


def general_grouped_gemm(
    A: List[torch.Tensor],
    B: List[torch.Tensor],
    out: List[torch.Tensor],
    out_dtype: torch.dtype,
    workspaces: List[torch.Tensor],
    layout: str = "TN",
    m_splits: Optional[List[int]] = None,
    gelu: bool = False,
    grad=False,
    accumulate: bool = False,
    bias: Optional[List[torch.Tensor]] = None,
    use_bias: bool = False,
    use_split_accumulator: bool = False,
    D_dtype: Optional[tex.DType] = None,
    single_output=False,
) -> Tuple[List[torch.Tensor], ...]:
    """
    TN layout Grouped GEMM with fp8 inputs.
    """
    num_gemms = len(A)

    transa = layout[0] == "T"
    transb = layout[1] == "T"

    empty_tensor = _empty_tensor()
    empty_tensors = [empty_tensor] * num_gemms

    # Use bfloat16 as default bias_dtype
    gelu_input = empty_tensors
    out_dtype = TE_DType[out[0].dtype] if D_dtype is None else D_dtype

    sm_count = get_sm_count()
    if grad and use_bias:
        grad_bias = [
            torch.empty(B[i].shape[1], dtype=out[0].dtype, device="cuda") for i in range(num_gemms)
        ]
    else:
        grad_bias = empty_tensors
    bias = bias if use_bias else empty_tensors
    if use_bias:
        bias_dtype = TE_DType[grad_bias[0].dtype] if grad else TE_DType[bias[0].dtype]
    else:
        bias_dtype = TE_DType[torch.bfloat16]

    if gelu:
        gelu_input = [
            torch.empty_like(o, dtype=bias_dtype, memory_format=torch.contiguous_format)
            for o in out
        ]  # this should differ with respect to single output

    bias = tex.te_general_grouped_gemm(
        A,
        transa,
        B,
        transb,
        out,
        out_dtype,
        m_splits,
        grad_bias if grad else bias,
        bias_dtype,
        single_output,
        gelu_input,  # this is pre_gelu_out
        grad,  # grad
        workspaces,
        workspaces[0].shape[0],
        accumulate,
        use_split_accumulator,
        sm_count - int(os.getenv("NVTE_EXT_MARGIN_SM", str(sm_count))),
    )

    return out, bias, gelu_input
