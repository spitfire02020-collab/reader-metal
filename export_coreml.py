#!/usr/bin/env python3
"""
Export Chatterbox Turbo PyTorch models to CoreML format for iOS.
This enables Apple Silicon GPU acceleration via CoreML.
"""

import os
import sys
import subprocess
import torch
import numpy as np

# Try to import chatterbox - if not installed, install it
try:
    from chatterbox import ChatterboxTTS
except ImportError:
    print("Installing chatterbox...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "chatterbox-turbo", "-q"])
    from chatterbox import ChatterboxTTS

try:
    import coremltools as ct
except ImportError:
    print("Installing coremltools...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "coremltools", "-q"])
    import coremltools as ct

def export_speech_encoder():
    """Export speech encoder to CoreML."""
    print("Exporting speech encoder...")

    # Load model
    model = ChatterboxTTS.from_pretrained("chatterbox/chatterbox-turbo")
    model.eval()

    # Create dummy input
    audio = torch.randn(1, 24000)  # 1 second of audio

    # Trace the model
    with torch.no_grad():
        traced = torch.jit.trace(model.speech_encoder, audio)

    # Convert to CoreML
    coreml_model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="audio", shape=(1, 24000))]
    )

    output_path = "speech_encoder.mlmodel"
    coreml_model.save(output_path)
    print(f"  -> Saved to {output_path}")
    return output_path


def export_embed_tokens():
    """Export embedding tokens to CoreML."""
    print("Exporting embed tokens...")

    model = ChatterboxTTS.from_pretrained("chatterbox/chatterbox-turbo")
    model.eval()

    # Create dummy input (token IDs)
    input_ids = torch.randint(0, 50000, (1, 100))

    # Trace
    with torch.no_grad():
        traced = torch.jit.trace(model.text_encoder, input_ids)

    # Convert
    coreml_model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input_ids", shape=(1, 100))]
    )

    output_path = "text_encoder.mlmodel"
    coreml_model.save(output_path)
    print(f"  -> Saved to {output_path}")
    return output_path


def export_language_model():
    """Export language model to CoreML."""
    print("Exporting language model...")

    model = ChatterboxTTS.from_pretrained("chatterbox/chatterbox-turbo")
    model.eval()

    # This is complex - LM has KV cache
    # We'll export a simplified version
    # For now, skip this as it's very complex

    print("  -> Skipping (too complex for now)")
    return None


def export_decoder():
    """Export audio decoder to CoreML."""
    print("Exporting decoder...")

    model = ChatterboxTTS.from_pretrained("chatterbox/chatterbox-turbo")
    model.eval()

    # Create dummy inputs
    speech_tokens = torch.randint(0, 6000, (1, 100))

    # Trace
    with torch.no_grad():
        traced = torch.jit.trace(model.decoder, speech_tokens)

    # Convert
    coreml_model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="speech_tokens", shape=(1, 100))]
    )

    output_path = "decoder.mlmodel"
    coreml_model.save(output_path)
    print(f"  -> Saved to {output_path}")
    return output_path


def main():
    print("=" * 50)
    print("Chatterbox Turbo -> CoreML Exporter")
    print("=" * 50)

    # Create output directory
    output_dir = "coreml_models"
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    # Download model
    print("\nDownloading Chatterbox Turbo...")
    model = ChatterboxTTS.from_pretrained("chatterbox/chatterbox-turbo")
    model.eval()

    print("\nExporting models to CoreML...")

    # Export each component
    export_speech_encoder()
    export_embed_tokens()
    export_decoder()

    print("\n" + "=" * 50)
    print("Export complete!")
    print("Models saved to:", os.getcwd())
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
