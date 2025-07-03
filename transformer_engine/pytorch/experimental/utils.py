# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import enum
import torch


HIGH_PRECISION_FLOAT_DTYPES = (
    torch.float,
    torch.float16,
    torch.bfloat16,
    torch.float32,
)


FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


FP4_E2M1_MAXVAL = 6.0
FP4_E0M3_MAXVAL = 7.0
FP4_E3M0_MAXVAL = 16.0
FP6_E3M2_MAXVAL = 28.0
FP6_E2M3_MAXVAL = 7.5
FP8_E4M3_MAXVAL = 448.0
NVFP4_BLOCK_SIZE = 16


class Fp4Formats(enum.Enum):
    E2M1 = "e2m1"
    E0M3 = "e0m3"
    E3M0 = "e3m0"


FP4_DTYPES = (Fp4Formats.E2M1, Fp4Formats.E0M3, Fp4Formats.E3M0)


def cast_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    # values = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    # bounds = {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5};
    sign = torch.sign(x)
    x = torch.abs(x)
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign


def cast_to_fp4_e0m3(x: torch.Tensor) -> torch.Tensor:
    # values = {0, 1, 2, 3, 4, 5, 6, 7};
    # bounds = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5};
    sign = torch.sign(x)
    x = torch.abs(x)
    x[(x >= 0.0) & (x <= 0.5)] = 0.0
    x[(x > 0.5) & (x < 1.5)] = 1.0
    x[(x >= 1.5) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 4.5)] = 4.0
    x[(x > 4.5) & (x < 5.5)] = 5.0
    x[(x >= 5.5) & (x <= 6.5)] = 6.0
    x[x > 6.5] = 7.0
    return x * sign


def cast_to_fp4_e3m0(x: torch.Tensor) -> torch.Tensor:
    # values = {0, 0.25, 0.5, 1, 2, 4, 8, 16}
    # bounds = {0.125, 0.375, 0.75, 1.5, 3, 6, 12}
    sign = torch.sign(x)
    x = torch.abs(x)
    x[(x >= 0.0) & (x <= 0.125)] = 0.0
    x[(x > 0.125) & (x < 0.375)] = 0.25
    x[(x >= 0.375) & (x <= 0.75)] = 0.5
    x[(x > 0.75) & (x < 1.5)] = 1.0
    x[(x >= 1.5) & (x <= 3.0)] = 2.0
    x[(x > 3.0) & (x < 6.0)] = 4.0
    x[(x >= 6.0) & (x <= 12.0)] = 8.0
    x[x > 12.0] = 16.0
    return x * sign
