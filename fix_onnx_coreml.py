#!/usr/bin/env python3
"""
Fix ONNX models to work with CoreML by providing flexible shapes.
CoreML needs bounded dimensions - we use large values but allow flexibility.
"""

import onnx
from onnx import helper, numpy_helper, shape_inference
import os

def get_models_dir():
    """Find the ChatterboxModels directory."""
    home = os.path.expanduser("~")
    base = os.path.join(home, "Library/Developer/CoreSimulator/Devices")

    for root, dirs, files in os.walk(base):
        if "ChatterboxModels" in dirs:
            return os.path.join(root, "ChatterboxModels")
    return None

def fix_model_shapes(model_path, max_seq=8192):
    """Fix ONNX model shapes for CoreML compatibility.

    Uses symbolic dimensions that ORT can handle while being large enough
    to accommodate typical audio lengths.
    """
    print(f"Processing {model_path}...")

    model = onnx.load(model_path)
    graph = model.graph

    # Find inputs with dynamic shapes
    for input_tensor in graph.input:
        shape = input_tensor.type.tensor_type.shape
        if shape.dim:
            for i, dim in enumerate(shape.dim):
                if dim.dim_param:  # Dynamic dimension
                    if i == 0:
                        # First dimension is batch - keep as 1
                        dim.dim_value = 1
                        print(f"  {input_tensor.name}[0]: dynamic -> 1 (batch)")
                    else:
                        # Subsequent dimensions - use large value for flexibility
                        dim.dim_value = max_seq
                        print(f"  {input_tensor.name}[{i}]: dynamic -> {max_seq} (sequence)")

    # Find outputs with dynamic shapes
    for output_tensor in graph.output:
        shape = output_tensor.type.tensor_type.shape
        if shape.dim:
            for i, dim in enumerate(shape.dim):
                if dim.dim_param:
                    if i == 0:
                        # First dimension is batch - keep as 1
                        dim.dim_value = 1
                        print(f"  output {output_tensor.name}[0]: dynamic -> 1 (batch)")
                    else:
                        # Subsequent dimensions - use large value
                        dim.dim_value = max_seq
                        print(f"  output {output_tensor.name}[{i}]: dynamic -> {max_seq} (sequence)")

    # Save the fixed model
    fixed_path = model_path.replace(".onnx", "_fixed.onnx")
    onnx.save(model, fixed_path)
    print(f"  Saved to {fixed_path}")

    return fixed_path

def main():
    models_dir = get_models_dir()
    if not models_dir:
        print("Error: Could not find ChatterboxModels")
        return 1

    print(f"Models dir: {models_dir}")

    # Use 4096 as max sequence length (covers most audio)
    max_seq = 4096

    # Fix each model
    models = [
        "speech_encoder_q4f16.onnx",
        "embed_tokens_q4f16.onnx",
        "language_model_q4f16.onnx",
        "conditional_decoder_q4f16.onnx"
    ]

    for model_name in models:
        model_path = os.path.join(models_dir, model_name)
        if os.path.exists(model_path):
            fix_model_shapes(model_path, max_seq=max_seq)

    print(f"\nDone! Fixed models use max sequence={max_seq}")
    print("Copy the _fixed.onnx files to your iOS project.")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
