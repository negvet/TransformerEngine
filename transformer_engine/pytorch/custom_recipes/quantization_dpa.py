#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DotProductAttention (DPA) custom recipe reference implementations.

This module is intended as a **simple starting point** for iterating on role-based
quantization for attention.

Custom recipes in TransformerEngine are driven by a user-provided factory:

    custom_recipe = transformer_engine.common.recipe.CustomRecipe(qfactory=my_factory)
    with transformer_engine.pytorch.autocast(enabled=True, recipe=custom_recipe):
        out = model(inp)

The factory is called as:

    quantizer = my_factory(role: str)

Where `role` is a stable semantic string describing *what tensor is being quantized*.

## DPA roles

When `DotProductAttention` is used with a `CustomRecipe`, it will request quantizers
for the following DPA-specific roles (stable contract):

- **Forward**
  - `dpa:qkv`  : QKV tensor slot used by attention backends
  - `dpa:o`    : O (attention output) tensor slot used by attention backends
  - `dpa:s`    : S (attention scores / logits) tensor slot used by attention backends

- **Backward**
  - `dpa:dqkv` : dQKV tensor slot used by attention backends
  - `dpa:do`   : dO tensor slot used by attention backends
  - `dpa:dp`   : dP tensor slot used by attention backends

Additional `dpa:*` roles may be requested internally for unused / structural slots.
Factories may safely return `None` for those roles.

Important: for the roles above that are *actually consumed by attention backends*,
returning `None` is not supported.
"""

from __future__ import annotations

from typing import Optional

import transformer_engine_torch as tex

from transformer_engine.common.recipe import Format
from transformer_engine.pytorch.tensor.float8_tensor import Float8CurrentScalingQuantizer
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer


def _split_role(role: str) -> tuple[str, str]:
    """Split `<scope>:<tensor_type>` role string."""
    if ":" not in role:
        raise ValueError(f"Invalid role: {role}, expected format: '<scope>:<tensor>'")
    scope, tensor_type = role.split(":", 1)
    return scope, tensor_type


def _get_fp8_te_dtype(fp8_format: Format, *, fwd: bool) -> tex.DType:
    """Match TE's per-tensor FP8 dtype selection."""
    if fp8_format == Format.E4M3 or (fp8_format == Format.HYBRID and fwd):
        return tex.DType.kFloat8E4M3
    return tex.DType.kFloat8E5M2


def nvfp4_quantizer_factory(role: str):
    """Native NVFP4 quantizer factory for GEMM-like roles.

    This factory is **scope-agnostic**: it works with roles like:
    - `linear:input`, `layernorm_linear:weight`, `layernorm_mlp:grad_output`, ...

    It follows the same high-level conventions as `NVFP4BlockScalingRecipeState`:
    - forward `input`/`output`: RHT enabled, no 2D quantization
    - forward `weight`: no RHT, 2D quantization enabled
    - backward gradients: RHT enabled, stochastic rounding enabled
    """
    _, tensor_type = _split_role(role)

    # Forward-ish roles
    if tensor_type in ("input", "output"):
        return NVFP4Quantizer(
            fp4_dtype=tex.DType.kFloat4E2M1,
            rowwise=True,
            columnwise=True,
            with_rht=True,
            with_post_rht_amax=True,
            with_2d_quantization=False,
            stochastic_rounding=False,
        )
    if tensor_type == "weight":
        return NVFP4Quantizer(
            fp4_dtype=tex.DType.kFloat4E2M1,
            rowwise=True,
            columnwise=True,
            with_rht=False,
            with_post_rht_amax=False,
            with_2d_quantization=True,
            stochastic_rounding=False,
        )

    # Backward-ish roles
    if tensor_type in ("grad_output", "grad_input"):
        return NVFP4Quantizer(
            fp4_dtype=tex.DType.kFloat4E2M1,
            rowwise=True,
            columnwise=True,
            with_rht=True,
            with_post_rht_amax=True,
            with_2d_quantization=False,
            stochastic_rounding=True,
        )

    return None


def dpa_fp8cs_quantizer_factory(role: str, *, fp8_format: Format = Format.HYBRID):
    """Minimal DPA factory: Float8CurrentScaling everywhere DPA needs it.

    This is the simplest workable example because current scaling quantizers are
    stateless w.r.t. amax history.

    Mapping:
    - forward  (`dpa:qkv`, `dpa:o`, `dpa:s`)    -> E4M3 (typical forward FP8)
    - backward (`dpa:dqkv`, `dpa:do`, `dpa:dp`) -> E5M2 (typical backward FP8)
    """
    if role in ("dpa:qkv", "dpa:o", "dpa:s"):
        return Float8CurrentScalingQuantizer(_get_fp8_te_dtype(fp8_format, fwd=True), device="cuda")
    if role in ("dpa:dqkv", "dpa:do", "dpa:dp"):
        return Float8CurrentScalingQuantizer(_get_fp8_te_dtype(fp8_format, fwd=False), device="cuda")
    return None


def make_nvfp4_linear_fp8cs_dpa_qfactory(*, fp8_format: Format = Format.HYBRID):
    """Create a `qfactory` for: non-DPA=NVFP4, DPA=FP8CS.

    This is a simplified mapping:
    - All GEMM-like modules/ops that use `*:input/weight/output/grad_*` roles use native NVFP4.
    - DPA uses FP8 current scaling for:
      `dpa:qkv`, `dpa:o`, `dpa:s`, `dpa:dqkv`, `dpa:do`, `dpa:dp`.

    Usage:
        qfactory = make_nvfp4_linear_fp8cs_dpa_qfactory(fp8_format=Format.HYBRID)
        custom_recipe = recipe.CustomRecipe(qfactory=qfactory)
    """

    def qfactory(role: str):
        if role.startswith("dpa:"):
            return dpa_fp8cs_quantizer_factory(role, fp8_format=fp8_format)
        return nvfp4_quantizer_factory(role)

    return qfactory
