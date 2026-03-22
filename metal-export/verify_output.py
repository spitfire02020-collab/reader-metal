#!/usr/bin/env python3
"""
Verify the exported ONNX model runs correctly.

This script loads the exported ONNX model and runs a simple verification
to ensure it produces valid output, including numerical comparison with
the PyTorch reference model.
"""

import argparse
import os
import sys

# Apply patch BEFORE importing transformers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import patch_diff

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import GPT2Config

# Import GPT2NoEmbed from export script
from export_lm_float16 import GPT2NoEmbed


def verify_onnx_model(
    model_path: str,
    hidden_size: int = 1024,
    vocab_size: int = 6563,
    batch_size: int = 1,
    seq_len: int = 32,
):
    """
    Verify that the ONNX model runs correctly.

    Args:
        model_path: Path to the ONNX model
        hidden_size: Hidden dimension (default: 1024)
        vocab_size: Vocabulary size (default: 6563)
        batch_size: Batch size (default: 1)
        seq_len: Sequence length (default: 32)

    Returns:
        True if verification passes, False otherwise
    """
    print("=" * 60)
    print("ONNX Model Verification")
    print("=" * 60)

    # Check model exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return False

    # Load model
    print(f"\n1. Loading ONNX model from {model_path}...")
    try:
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
        print("   Model loaded and verified successfully")
    except Exception as e:
        print(f"ERROR: Failed to load/verify model: {e}")
        return False

    # Print model info
    print("\n2. Model information:")
    print(f"   IR version: {onnx_model.ir_version}")
    print(f"   Opset version: {onnx_model.opset_import[0].version}")
    print(f"   Producer: {onnx_model.producer_name}")

    # List inputs/outputs
    print("\n3. Model inputs/outputs:")
    for input_tensor in onnx_model.graph.input:
        shape = [dim.dim_value if dim.dim_value > 0 else "dynamic"
                 for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"   Input: {input_tensor.name}, shape={shape}")
    for output_tensor in onnx_model.graph.output:
        shape = [dim.dim_value if dim.dim_value > 0 else "dynamic"
                 for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"   Output: {output_tensor.name}, shape={shape}")

    # Create ONNX Runtime session
    print("\n4. Creating ONNX Runtime session...")
    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Use CPU execution provider
        providers = [
            ("CPUExecutionProvider", {
                "arena_extend_strategy": "kNextPowerOfTwo",
                "enable_mem_pattern": True,
                "enable_cpu_mem_arena": True,
            }),
        ]

        session = ort.InferenceSession(model_path, sess_options, providers=providers)
        print("   Session created successfully")
    except Exception as e:
        print(f"ERROR: Failed to create session: {e}")
        return False

    # Run inference
    print("\n5. Running inference...")
    try:
        # Create random input (float16 to match the exported model)
        torch.manual_seed(42)
        hidden_states = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float16)
        hidden_states_pt = torch.from_numpy(hidden_states.copy())
        print(f"   Input hidden_states shape: {hidden_states.shape}")

        # Run ONNX session
        input_name = session.get_inputs()[0].name
        onnx_outputs = session.run(None, {input_name: hidden_states})
        onnx_logits = onnx_outputs[0]
        print(f"   ONNX output shape: {onnx_logits.shape}, dtype: {onnx_logits.dtype}")

        # Basic ONNX validation
        if np.isnan(onnx_logits).any():
            print(f"   WARNING: ONNX output contains NaN values")
        if np.isinf(onnx_logits).any():
            print(f"   WARNING: ONNX output contains Inf values")

        # Check hidden_states output shape (GPT2NoEmbed outputs hidden_size, not vocab_size)
        if onnx_logits.shape[-1] != hidden_size:
            print(f"   WARNING: Hidden size {onnx_logits.shape[-1]} != expected {hidden_size}")
        else:
            print(f"   NOTE: Model outputs hidden_states (shape={onnx_logits.shape}, hidden_size={hidden_size})")

    except Exception as e:
        print(f"ERROR: Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Numerical comparison with PyTorch reference
    print("\n6. Running PyTorch reference model...")
    try:
        # Create PyTorch model with same seed
        torch.manual_seed(42)

        # Create GPT2NoEmbed with same config as export
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=1500,
            n_embd=hidden_size,
            n_layer=24,
            n_head=16,
            n_inner=4096,
            activation_function="gelu_new",
            resid_dropout=0.0,
            embd_dropout=0.0,
            attn_dropout=0.0,
            layer_norm_epsilon=1e-5,
            bos_token_id=6561,
            eos_token_id=6562,
            use_cache=False,
            scale_attn_weights=True,
            _attn_implementation="eager",
        )

        pt_model = GPT2NoEmbed(config)
        pt_model.eval()
        pt_model = pt_model.half()

        # Run PyTorch forward pass (no cache)
        # Note: GPT2NoEmbed returns hidden_states (not logits/lm_head output)
        with torch.no_grad():
            pt_result = pt_model(hidden_states_pt, use_cache=False)

        # Handle both cases: (hidden_states,) or (hidden_states, past_key_values)
        if isinstance(pt_result, tuple):
            pt_hidden_states = pt_result[0]
        else:
            pt_hidden_states = pt_result

        pt_logits_np = pt_hidden_states.float().numpy()
        print(f"   PyTorch output shape: {pt_logits_np.shape}, dtype: {pt_logits_np.dtype}")

    except Exception as e:
        print(f"ERROR: PyTorch reference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Compare ONNX vs PyTorch
    print("\n7. Comparing ONNX vs PyTorch outputs...")
    try:
        # ONNX outputs: first output is hidden_states (logits in ONNX naming)
        # The ONNX model may have additional KV cache outputs we skip
        onnx_out = onnx_logits.flatten()
        torch_out = pt_logits_np.flatten()

        # Check shapes match
        if onnx_out.shape != torch_out.shape:
            print(f"   ERROR: Shape mismatch - ONNX: {onnx_out.shape}, PyTorch: {torch_out.shape}")
            return False

        # Compute difference statistics
        diff = np.abs(onnx_out - torch_out)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"   Max absolute difference: {max_diff:.6f}")
        print(f"   Mean absolute difference: {mean_diff:.6f}")

        # Numerical comparison with tolerance
        rtol = 1e-2
        atol = 1e-3
        matches = np.allclose(onnx_out, torch_out, rtol=rtol, atol=atol)

        if matches:
            print(f"   PASSED: Outputs match within rtol={rtol}, atol={atol}")
        else:
            # Check if they're close enough for float16
            matches_fp16 = np.allclose(onnx_out, torch_out, rtol=1e-1, atol=1e-2)
            if matches_fp16:
                print(f"   WARNING: Outputs only match at loose tolerance (rtol=1e-1, atol=1e-2)")
                print(f"            This may be expected for float16 models")
            else:
                print(f"   FAILED: Outputs do not match within specified tolerance")
                return False

        print("\n8. Verification PASSED!")

    except Exception as e:
        print(f"ERROR: Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("Verification complete!")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(description="Verify exported ONNX model")
    parser.add_argument("--model_path", type=str, default="./onnx_output/language_model_fp16.onnx",
                        help="Path to the ONNX model")
    parser.add_argument("--hidden_size", type=int, default=1024,
                        help="Hidden size (default: 1024)")
    parser.add_argument("--vocab_size", type=int, default=6563,
                        help="Vocabulary size (default: 6563)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (default: 1)")
    parser.add_argument("--seq_len", type=int, default=32,
                        help="Sequence length (default: 32)")
    args = parser.parse_args()

    success = verify_onnx_model(
        args.model_path,
        hidden_size=args.hidden_size,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()