# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import dataclasses
import enum
import torch
from typing import Optional

from transformer_engine.pytorch.experimental import utils
from transformer_engine.pytorch.experimental import quantization


@dataclasses.dataclass(frozen=True)
class MMParams:
    """Matrix multiplication parameters."""

    out_dtype: torch.dtype | None = None
    # Use split accumulator for more accurate FP8 GEMM
    use_split_accumulator: bool = False


@dataclasses.dataclass()
class QLinearParams:
    """Quantization parameters of linear layer.

    Contains ready-to-use quantizers for input (x), weight (w), and gradient (g) tensors.
    """
    x_quantizer: Optional[quantization.Quantizer] = None
    w_quantizer: Optional[quantization.Quantizer] = None
    g_quantizer: Optional[quantization.Quantizer] = None

    mm_fprop: Optional[MMParams] = None
    mm_dgrad: Optional[MMParams] = None
    mm_wgrad: Optional[MMParams] = None


@enum.unique
class QuantizeRecipe(enum.Enum):
    """Pre-defined quantization recipes for linear layers."""

    NON_QUANTIZE = "non_quantize"
    FP8_CS_EMULATION = "fp8_current_scaling_emulation"
    FP4_CS_EMULATION = "fp4_current_scaling_emulation"


def get_qlinear_params_from_predefined(
    recipe: QuantizeRecipe,
) -> QLinearParams:
    """Get quantization parameters for linear layer based on recipe."""
    if recipe == QuantizeRecipe.NON_QUANTIZE:
        return QLinearParams()

    elif recipe == QuantizeRecipe.FP8_CS_EMULATION:
        return QLinearParams(
            x_quantizer=quantization.Float4Float8CurrentScalingEmulationRefQuantizer(
                dtype=torch.float8_e4m3fn,
            ),
            w_quantizer=quantization.Float4Float8CurrentScalingEmulationRefQuantizer(
                dtype=torch.float8_e4m3fn,
            ),
            g_quantizer=quantization.Float4Float8CurrentScalingEmulationRefQuantizer(
                dtype=torch.float8_e5m2,
            ),
        )
    elif recipe == QuantizeRecipe.FP4_CS_EMULATION:
        return QLinearParams(
            x_quantizer=quantization.Float4Float8CurrentScalingEmulationRefQuantizer(
                dtype=utils.Fp4Formats.E2M1,
            ),
            w_quantizer=quantization.Float4Float8CurrentScalingEmulationRefQuantizer(
                dtype=utils.Fp4Formats.E2M1,
            ),
            g_quantizer=quantization.Float4Float8CurrentScalingEmulationRefQuantizer(
                dtype=utils.Fp4Formats.E2M1,
            ),
        )
    else:
        raise ValueError(f"Unsupported quantize recipe: {recipe}")
