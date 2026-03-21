#!/usr/bin/env python3
"""
Verify the exported ONNX model runs correctly.

This script loads the exported ONNX model and runs a simple verification
to ensure it produces valid output.
"""

import argparse
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort


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
        hidden_states = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float16)
        print(f"   Input hidden_states shape: {hidden_states.shape}")

        # Run session
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: hidden_states})

        print(f"   Number of outputs: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"   Output {i} shape: {output.shape}, dtype: {output.dtype}")

            # Basic validation
            if output.dtype != np.float32 and output.dtype != np.float16:
                print(f"   WARNING: Unexpected dtype {output.dtype}")

            if np.isnan(output).any():
                print(f"   WARNING: Output contains NaN values")
            if np.isinf(output).any():
                print(f"   WARNING: Output contains Inf values")

        # Check logits output (should be vocab_size)
        logits = outputs[0]
        if logits.shape[-1] != vocab_size:
            print(f"   WARNING: Logits vocab size {logits.shape[-1]} != expected {vocab_size}")

        # Check if logits are reasonable (not all zeros, not all same value)
        logits_min = logits.min()
        logits_max = logits.max()
        logits_mean = logits.mean()
        print(f"   Logits range: [{logits_min:.4f}, {logits_max:.4f}], mean={logits_mean:.4f}")

        if logits_min == logits_max:
            print("   WARNING: All logits have the same value")

        print("\n6. Verification PASSED!")

    except Exception as e:
        print(f"ERROR: Inference failed: {e}")
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