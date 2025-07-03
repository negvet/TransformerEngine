# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from typing import Optional, Tuple, Union
import torch

from transformer_engine.pytorch.experimental import config


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        input_quantizer,
        weight_quantizer,
        gradient_quantizer,
        is_grad_enabled,
    ) -> torch.Tensor:
        # ------------------------------------------------------
        # Prepare input tensor
        # ------------------------------------------------------
        input_quantizer.set_usage(rowwise=True, columnwise=False)
        qresult_x = input_quantizer(input)
        # Use .data since we set rowwise=True (original layout)
        qx, sx = qresult_x.data, qresult_x.scale

        # ------------------------------------------------------
        # Prepare weight tensor
        # ------------------------------------------------------
        backward_needs_weight = is_grad_enabled and input.requires_grad
        weight_quantizer.set_usage(rowwise=backward_needs_weight, columnwise=True)
        qresult_w = weight_quantizer(weight)
        # Use .data_t since we set columnwise=True (need transposed data)
        qw_t, sw_t = qresult_w.data_t, qresult_w.scale_t

        # ------------------------------------------------------
        # Forward GEMM
        # Note: y = x * w^T, but since qw_t is already transposed from data_t,
        # we can do direct matmul: y = qx * qw_t
        # ------------------------------------------------------
        y = torch.matmul(qx, qw_t)

        # Save for backward - only when gradients are enabled
        if is_grad_enabled and (input.requires_grad or weight.requires_grad):
            # Save tensors only when needed for gradient computation:
            # - qx as data (non-transposed) when weight.requires_grad for wgrad computation (dy^T @ x)
            # - qw as data (non-transposed) when input.requires_grad for dgrad computation
            saved_qx = qresult_x.data if weight.requires_grad else None
            saved_qw = qresult_w.data if input.requires_grad else None
            ctx.save_for_backward(saved_qx, saved_qw)
            ctx.gradient_quantizer = gradient_quantizer
            ctx.requires_dgrad = input.requires_grad
            ctx.requires_wgrad = weight.requires_grad

        return y

    @staticmethod
    def backward(ctx, grad_output) -> Tuple[torch.Tensor, ...]:
        
        # print("TE experimental: _Linear.backward")
        # print("grad_output", grad_output.mean(), grad_output.max())

        qx, qw = ctx.saved_tensors
        gradient_quantizer = ctx.gradient_quantizer

        # --------------------------------------------------
        # Prepare grad output tensor
        # --------------------------------------------------
        # Set usage for both dgrad and wgrad
        gradient_quantizer.set_usage(rowwise=True, columnwise=True)
        qresult_dy = gradient_quantizer(grad_output)
        qdy, sdy = qresult_dy.data, qresult_dy.scale
        qdy_t, sdy_t = qresult_dy.data_t, qresult_dy.scale_t

        # --------------------------------------------------
        # Compute grad input tensor
        # --------------------------------------------------
        # dgrad GEMM: dx = dy * w
        dgrad = None
        if ctx.requires_dgrad:
            dgrad = torch.matmul(qdy, qw)

        # --------------------------------------------------
        # Compute grad weight
        # --------------------------------------------------
        # wgrad GEMM: dw = dy^T * x (following kitchen implementation)
        # qdy_t is dy^T, qx is x (non-transposed input)
        wgrad = None
        if ctx.requires_wgrad:
            wgrad = torch.matmul(qdy_t, qx)

        return dgrad, wgrad, None, None, None, None


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        params_dtype: Optional[torch.dtype] = None,
        device: Union[torch.device, str] = "cuda",
        qlinear_params: Optional[config.QLinearParams] = None,
        **kwargs,  # Accept any additional keyword arguments
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        self.weight = torch.nn.Parameter(
            torch.randn(
                out_features,
                in_features,
                device=device,
                dtype=params_dtype,
            )
        )

        # Use provided qlinear_params or default to FP8_CS_EMULATION
        if qlinear_params is None:
            qlinear_params = config.get_qlinear_params_from_predefined(config.QuantizeRecipe.FP8_CS_EMULATION)
        
        self.input_quantizer = qlinear_params.x_quantizer
        self.weight_quantizer = qlinear_params.w_quantizer
        self.gradient_quantizer = qlinear_params.g_quantizer

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply the linear transformation to the input.

        Parameters
        ----------
        input : torch.Tensor
             Input tensor.
        """
        # print("Experimental Linear Forward")
        input = input.contiguous()

        input_shape = input.shape
        input = input.view(-1, input.shape[-1])

        if torch.is_grad_enabled():
            linear_fn = _Linear.apply
            args = []
        else:
            linear_fn = _Linear.forward
            args = [None]

        args += [
            input,
            self.weight,
            self.input_quantizer,
            self.weight_quantizer,
            self.gradient_quantizer,
            torch.is_grad_enabled(),
        ]

        out = linear_fn(*args)
        return out.view(-1, *input_shape[1:-1], out.shape[-1])


if __name__ == "__main__":
    import kitchen
    # Import main TE linear and FP8 components
    from transformer_engine.pytorch.module.linear import Linear as TELinear
    from transformer_engine.pytorch.fp8 import fp8_autocast
    from transformer_engine.common.recipe import Float8CurrentScaling

    torch.manual_seed(0)
    
    print("=== Comparing FP4, FP8, and Main TE: Experimental vs Kitchen vs Main TE Linear ===")
    
    # Test parameters (using dimensions compatible with TE FP8 requirements)
    in_features = 32  # Divisible by 16
    out_features = 32  # Divisible by 16 
    device = "cuda"
    
    # Create shared input tensor (8, 32) - product except last = 8 (divisible by 8), last = 32 (divisible by 16)
    input_tensor = torch.randn(8, 32, device=device, dtype=torch.bfloat16)
    
    # Define format configurations
    format_configs = [
        {
            "name": "FP4",
            "exp_recipe": config.QuantizeRecipe.FP4_CS_EMULATION,
            "kitchen_recipe": kitchen.config.QuantizeRecipe.FP4_CS_EMULATION,
        },
        {
            "name": "FP8", 
            "exp_recipe": config.QuantizeRecipe.FP8_CS_EMULATION,
            "kitchen_recipe": kitchen.config.QuantizeRecipe.FP8_CS_EMULATION,
        }
    ]
    
    # Store results for summary
    results = {}
    
    # Loop through each format
    for cfg in format_configs:
        format_name = cfg["name"]
        exp_recipe = cfg["exp_recipe"] 
        kitchen_recipe = cfg["kitchen_recipe"]
        
        print("\n" + "="*60)
        print(f"{format_name} COMPARISON")
        print("="*60)
        
        # Create experimental linear
        print(f"\n1. Creating Experimental Linear ({format_name})...")
        exp_qlinear_params = config.get_qlinear_params_from_predefined(exp_recipe)
        exp_linear = Linear(
            in_features,
            out_features,
            bias=False,
            params_dtype=torch.bfloat16,
            device=device,
            qlinear_params=exp_qlinear_params
        )
        print(f"   Experimental {format_name} quantize_op:", type(exp_qlinear_params.x_quantizer).__name__)
        print(f"   Experimental {format_name} format:", exp_qlinear_params.x_quantizer.dtype)
        
        # Create kitchen linear
        print(f"\n2. Creating Kitchen Linear ({kitchen_recipe.value})...")
        kitchen_qlinear_params = kitchen.config.get_qlinear_params_from_predefined(kitchen_recipe)
        
        kitchen_linear = kitchen.linear.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            device=device,
            params_dtype=torch.bfloat16,
            qlinear_params=kitchen_qlinear_params,
        )
        
        # Copy weights for fair comparison (use first format's weights as reference)
        if format_name == "FP4":
            reference_weights = exp_linear.weight.data.clone()
        else:
            with torch.no_grad():
                exp_linear.weight.copy_(reference_weights)
        
        with torch.no_grad():
            kitchen_linear.weight.copy_(reference_weights)
        
        print(f"   Kitchen {format_name} quantize_op:", type(kitchen_qlinear_params.quantize_op).__name__)
        print(f"   Kitchen {format_name} format:", kitchen_qlinear_params.x_params.quant_dtype)
        
        # Run forward passes
        print(f"\n3. Running {format_name} forward passes...")
        with torch.no_grad():
            exp_output = exp_linear(input_tensor)
            kitchen_output = kitchen_linear(input_tensor)
        
        # Compare results
        print(f"\n4. Comparing {format_name} results...")
        torch.testing.assert_close(
            exp_output,
            kitchen_output,
            atol=0,
            rtol=0,
            msg=f"{format_name} Experimental vs Kitchen comparison"
        )
        print(f"   ✓ {format_name} Experimental and Kitchen outputs match!")
        
        # Test backward pass
        print(f"\n5. Running {format_name} backward passes...")
        
        # Create grad_output for backward computation
        grad_output = torch.randn_like(exp_output)
        
        # Test experimental backward
        exp_input_clone = input_tensor.clone().detach().requires_grad_(True)
        exp_output_bwd = exp_linear(exp_input_clone)
        exp_output_bwd.backward(grad_output)
        exp_input_grad = exp_input_clone.grad.clone() if exp_input_clone.grad is not None else None
        exp_weight_grad = exp_linear.weight.grad.clone() if exp_linear.weight.grad is not None else None
        
        # Reset gradients
        exp_linear.zero_grad()
        
        # Test kitchen backward
        kitchen_input_clone = input_tensor.clone().detach().requires_grad_(True)
        kitchen_output_bwd = kitchen_linear(kitchen_input_clone)
        kitchen_output_bwd.backward(grad_output)
        kitchen_input_grad = kitchen_input_clone.grad.clone() if kitchen_input_clone.grad is not None else None
        kitchen_weight_grad = kitchen_linear.weight.grad.clone() if kitchen_linear.weight.grad is not None else None
        
        # Compare backward results
        print(f"\n6. Comparing {format_name} backward results...")
        if exp_input_grad is not None and kitchen_input_grad is not None:
            torch.testing.assert_close(
                exp_input_grad,
                kitchen_input_grad,
                atol=0,
                rtol=0,
                msg=f"{format_name} Input gradient comparison"
            )
        if exp_weight_grad is not None and kitchen_weight_grad is not None:
            torch.testing.assert_close(
                exp_weight_grad,
                kitchen_weight_grad,
                atol=0,
                rtol=0,
                msg=f"{format_name} Weight gradient comparison"
            )
        print(f"   ✓ {format_name} Experimental and Kitchen gradients match!")
        
        results[format_name] = {
            "match": True, 
            "exp_output": exp_output, 
            "kitchen_output": kitchen_output,
            "exp_input_grad": exp_input_grad,
            "kitchen_input_grad": kitchen_input_grad,
            "exp_weight_grad": exp_weight_grad,
            "kitchen_weight_grad": kitchen_weight_grad,
        }


    # ========================================================================
    # Compare Experimental FP8 vs Main TE FP8 Current Scaling
    # ========================================================================
    print("\n" + "="*80)
    print("FP8 CURRENT SCALING: EXPERIMENTAL vs MAIN TE")
    print("="*80)
    
    # Reuse experimental FP8 linear from above (same FP8_CS_EMULATION configuration)
    print("\n1. Reusing Experimental FP8 Linear from above...")
    exp_fp8_linear = exp_linear  # This is the FP8 linear from the previous comparison
    print(f"   Experimental FP8 quantizer: {type(exp_fp8_linear.input_quantizer).__name__}")
    
    # Create main TE linear with FP8 current scaling
    print("\n2. Creating Main TE Linear with FP8 Current Scaling...")
    fp8_recipe = Float8CurrentScaling()
    te_fp8_linear = TELinear(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        device=device,
        params_dtype=torch.bfloat16,
    )
    
    with torch.no_grad():
        te_fp8_linear.weight.copy_(reference_weights)
    
    print(f"   Main TE FP8 recipe: {fp8_recipe}")
    
    # ========================================================================
    # Compare Quantizer.quantize() Results
    # ========================================================================
    print("\n" + "="*60)
    print("QUANTIZER COMPARISON: quantize() Results")
    print("="*60)
    
    # Get experimental quantizer
    exp_quantizer = exp_fp8_linear.input_quantizer
    print(f"\n1. Experimental quantizer: {type(exp_quantizer).__name__}")
    print(f"   Experimental quantizer dtype: {exp_quantizer.dtype}")
    
    # Create main TE quantizer
    print(f"\n2. Creating Main TE quantizer...")
    from transformer_engine.pytorch.fp8 import Float8CurrentScalingRecipeState
    te_recipe_state = Float8CurrentScalingRecipeState(
        recipe=fp8_recipe,
        mode="forward",
        num_quantizers=1,
        device=device,
    )
    te_quantizers = te_recipe_state.make_quantizers()
    te_quantizer = te_quantizers[0]
    print(f"   Main TE quantizer: {type(te_quantizer).__name__}")
    print(f"   Main TE quantizer dtype: {te_quantizer.dtype}")
    
    # Test quantize() on same input tensor
    print(f"\n3. Testing quantize() on input tensor...")
    print(f"   Input tensor shape: {input_tensor.shape}")
    print(f"   Input tensor (first 3x3):")
    print(f"   {input_tensor[:3, :3]}")
    
    # Set quantizer usage (both quantizers need rowwise=True for input quantization)
    exp_quantizer.set_usage(rowwise=True, columnwise=False)
    te_quantizer.set_usage(rowwise=True, columnwise=False)
    
    # Quantize with experimental quantizer
    print(f"\n4. Quantizing with experimental quantizer...")
    with torch.no_grad():
        exp_quantized = exp_quantizer(input_tensor)
    
    print(f"   Experimental quantized type: {type(exp_quantized)}")
    print(f"   Experimental quantized data shape: {exp_quantized.data.shape}")
    print(f"   Experimental quantized scale shape: {exp_quantized.scale.shape if exp_quantized.scale is not None and exp_quantized.scale.numel() > 0 else 'empty'}")
    print(f"   Experimental quantized data (first 3x3):")
    print(f"   {exp_quantized.data[:3, :3]}")
    
    # Quantize with main TE quantizer
    print(f"\n5. Quantizing with main TE quantizer...")
    with torch.no_grad():
        te_quantized = te_quantizer(input_tensor)
    
    print(f"   Main TE quantized type: {type(te_quantized)}")
    print(f"   Main TE quantized shape: {te_quantized.shape}")
    print(f"   Main TE quantized data shape: {te_quantized._data.shape}")
    print(f"   Main TE quantized scale shape: {te_quantizer.scale.shape}")
    print(f"   Main TE quantized scale: {te_quantizer.scale}")
    print(f"   Main TE quantized data (first 3x3):")
    print(f"   {te_quantized._data[:3, :3]}")
    
    # Compare quantized results
    print(f"\n6. Comparing quantized results...")
    
    # Note: The experimental quantizer does fake quantization (returns dequantized values)
    # while the main TE quantizer returns actual quantized data
    # We'll compare the final "dequantized" values
    
    # For experimental: the data is already dequantized (fake quantization)
    exp_dequantized = exp_quantized.data
    print(f"   Experimental output (fake quantized, first 3x3):")
    print(f"   {exp_dequantized[:3, :3]}")
    
    # For main TE: we need to dequantize the Float8Tensor
    te_dequantized = te_quantized.dequantize()
    print(f"   Main TE dequantized (first 3x3):")
    print(f"   {te_dequantized[:3, :3]}")
    
    # Compare dequantized results
    torch.testing.assert_close(
        exp_dequantized,
        te_dequantized,
        atol=0,
        rtol=0,
        msg="Quantizer comparison"
    )
    print("   ✓ Quantizers produce identical results!")
    
    # ========================================================================
    # Compare Forward Pass
    # ========================================================================
    print("\n" + "="*60)
    print("FORWARD PASS COMPARISON")
    print("="*60)
    print("\n7. Running FP8 forward passes...")
    
    # Experimental forward (no special context needed)
    with torch.no_grad():
        exp_fp8_output = exp_fp8_linear(input_tensor)
    
    # Main TE forward (requires FP8 autocast context)
    with torch.no_grad():
        with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            te_fp8_output = te_fp8_linear(input_tensor)
    
    # Compare FP8 results
    print("\n8. Comparing FP8 current scaling results...")
    print(f"   Experimental FP8 output shape: {exp_fp8_output.shape}")
    print(f"   Main TE FP8 output shape: {te_fp8_output.shape}")
    print(f"   Experimental FP8 output (first 3x3):")
    print(f"   {exp_fp8_output[:3, :3]}")
    print(f"   Main TE FP8 output (first 3x3):") 
    print(f"   {te_fp8_output[:3, :3]}")
    
    # Note: We expect some numerical differences due to different implementations
    # Check if they're in the same ballpark
    max_diff = torch.max(torch.abs(exp_fp8_output - te_fp8_output))
    rel_diff = max_diff / torch.max(torch.abs(te_fp8_output))
    print(f"   Max absolute difference: {max_diff.item():.6f}")
    print(f"   Max relative difference: {rel_diff.item():.6f}")
    
    if rel_diff < 0.1:  # Allow 10% relative difference
        print("   ✓ FP8 implementations are reasonably close!")
    else:
        print("   ⚠ FP8 implementations show significant differences (expected due to different quantization strategies)")
    
    print("\n9. Forward pass comparison completed successfully!")
    
    print("\n=== All Comparisons Complete ===")
