#!/usr/bin/env python3
"""
Precompile ONNX models to CoreML format for faster loading.
Run this script before building to generate .mlmodelc files.
"""

import os
import sys
import subprocess

# Add coremltools
try:
    import coremltools as ct
except ImportError:
    print("Installing coremltools...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "coremltools"])
    import coremltools as ct

def get_models_dir():
    """Get the ChatterboxModels directory path."""
    # Find the models in DerivedData or use the source directory
    home = os.path.expanduser("~")
    possible_paths = [
        os.path.join(home, "Library/Developer/CoreSimulator/Devices"),
        os.path.join(os.getcwd(), "Reader"),
    ]

    for base in possible_paths:
        if os.path.exists(base):
            # Search for ChatterboxModels
            for root, dirs, files in os.walk(base):
                if "ChatterboxModels" in dirs:
                    return os.path.join(root, "ChatterboxModels")

    return None

def compile_model(onnx_path, output_path):
    """Compile a single ONNX model to CoreML."""
    print(f"Compiling {onnx_path}...")

    try:
        # Convert ONNX to CoreML directly
        coreml_model = ct.converters.onnx.convert(onnx_path)

        # Save as .mlmodelc (compiled directory)
        coreml_model.save(output_path)
        print(f"  -> Saved to {output_path}")
        return True

    except Exception as e:
        print(f"  -> Error: {e}")
        return False

def main():
    models_dir = get_models_dir()
    if not models_dir:
        print("Error: Could not find ChatterboxModels directory")
        sys.exit(1)

    print(f"Models directory: {models_dir}")

    # Define models to compile
    models = [
        ("speech_encoder_q4f16.onnx", "speech_encoder_q4f16.mlmodelc"),
        ("embed_tokens_q4f16.onnx", "embed_tokens_q4f16.mlmodelc"),
        ("language_model_q4f16.onnx", "language_model_q4f16.mlmodelc"),
        ("conditional_decoder_q4f16.onnx", "conditional_decoder_q4f16.mlmodelc"),
    ]

    success_count = 0
    for onnx_name, mlmodelc_name in models:
        onnx_path = os.path.join(models_dir, onnx_name)
        output_path = os.path.join(models_dir, mlmodelc_name)

        if not os.path.exists(onnx_path):
            print(f"Skipping {onnx_name} - not found")
            continue

        # Check if already compiled
        if os.path.exists(output_path):
            print(f"Skipping {onnx_name} - already compiled")
            success_count += 1
            continue

        if compile_model(onnx_path, output_path):
            success_count += 1

    print(f"\nCompiled {success_count}/{len(models)} models")
    return 0

if __name__ == "__main__":
    sys.exit(main())
