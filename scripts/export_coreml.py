import sys
import subprocess
import os

def install_deps():
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "coremltools==7.0", "onnx", "huggingface_hub", "numpy<2.0"])

# Force pure python protobuf to bypass macOS C++ extension builder issues with old protoc
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    import numpy as np
    # CRITICAL: Monkeypatch np.bool for coremltools compatibility with newer numpy
    if not hasattr(np, 'bool'):
        np.bool = np.bool_
        
    import coremltools as ct
    import onnx
    from huggingface_hub import snapshot_download
    
    # Test protobuf
    from coremltools.models import datatypes
except (ImportError, TypeError):
    install_deps()
    import numpy as np
    if not hasattr(np, 'bool'):
        np.bool = np.bool_
    import coremltools as ct
    import onnx
    from huggingface_hub import snapshot_download

print("Downloading float32 conditional_decoder and external data from HuggingFace...")
repo_path = snapshot_download(repo_id="ResembleAI/chatterbox-turbo-ONNX", allow_patterns=["onnx/conditional_decoder.onnx*"])
onnx_path = os.path.join(repo_path, "onnx", "conditional_decoder.onnx")
print(f"Downloaded to {onnx_path}")

print("Loading ONNX model...")
model = onnx.load(onnx_path)

print("\n--- Model Inputs ---")
for input in model.graph.input:
    shape = [d.dim_value if d.HasField("dim_value") else ("dynamic" if d.HasField("dim_param") else None) for d in input.type.tensor_type.shape.dim]
    print(f"Name: {input.name}, Type: {input.type.tensor_type.elem_type}, Shape: {shape}")

print("\nConfiguring CoreML input types with dynamic RangeDim...")
# Assuming standard shapes based on ChatterboxEngine
# speech_tokens: [1, seq_len]
# speaker_embeddings: [1, 192]
# speaker_features: [1, frame_len, 80]
inputs = [
    ct.TensorType(name="speech_tokens", shape=(1, ct.RangeDim(1, -1)), dtype=np.int64),
    ct.TensorType(name="speaker_embeddings", shape=(1, 192), dtype=np.float32),
    ct.TensorType(name="speaker_features", shape=(1, ct.RangeDim(1, -1), 80), dtype=np.float32)
]

print("\nStarting CoreML conversion (Computing FLOAT16 precision)...")
# Note: This step can take 1-3 minutes and appear 'stuck' while CoreML optimizes the graph
mlmodel = ct.convert(
    model,
    inputs=inputs,
    minimum_deployment_target=ct.target.iOS16,
    compute_precision=ct.precision.FLOAT16
)

out_dir = "/Users/rockymoon/Downloads/Reader/Reader/Resources/ChatterboxModels"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "conditional_decoder.mlpackage")
print(f"\nSaving MLPackage to {out_path}...")
mlmodel.save(out_path)
print("Done! Converted successfully.")
