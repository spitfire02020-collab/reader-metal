#!/usr/bin/env python3
"""
Standalone ONNX Builder

Loads ONNX models with external weight data (.onnx_data files) and re-saves
them as standalone files with all tensor data embedded in the ONNX protobuf.

This fixes the external data loading issue that causes OOM errors on iOS when
the .onnx_data file path resolution fails.

Usage:
    python3 merge_onnx_external_data.py

The script processes all .onnx files in the Resources/ChatterboxModels directory
that have corresponding .onnx_data files and replaces them with standalone versions.

IMPORTANT: Run this from the project root directory:
    cd /Users/rockymoon/Downloads/Reader
    .venv_onnx/bin/python3 merge_onnx_external_data.py
"""

import os
import sys
from pathlib import Path

# Ensure onnx is available
try:
    import onnx
    from onnx import external_data_helper
    ONNX_VERSION = onnx.__version__
except ImportError:
    print("ERROR: onnx not found!")
    print("Run: .venv_onnx/bin/python3 merge_onnx_external_data.py")
    sys.exit(1)


def load_model_with_external_data(model_path: str):
    """
    Load an ONNX model, resolving external data from .onnx_data files.

    ONNX models can store large tensor data in external files. This function
    handles loading those external data files correctly by:
    1. Reading the protobuf graph from the .onnx file
    2. Locating the corresponding .onnx_data file (same name, _data.onnx_data suffix)
    3. Mapping external data references to the actual file bytes

    Args:
        model_path: Path to the .onnx graph file

    Returns:
        onnx.ModelProto with external data loaded
    """
    model_path = Path(model_path)
    onnx_data_path = model_path.with_name(model_path.stem + "_data.onnx_data")

    if not onnx_data_path.exists():
        raise FileNotFoundError(f"External data not found: {onnx_data_path}")

    # Load the model protobuf (external data references remain as-is)
    model = onnx.load(str(model_path), load_external_data=False)

    # Check if model actually uses external data
    has_external = False
    for initializer in model.graph.initializer:
        if initializer.data_location == onnx.TensorProto.EXTERNAL:
            has_external = True
            break

    if not has_external:
        print(f"  ℹ️  Model does not use external data")
        return model

    # Load external data file bytes
    print(f"  📦 Loading external data: {onnx_data_path.name} ({onnx_data_path.stat().st_size / 1024**2:.1f} MB)")
    with open(onnx_data_path, 'rb') as f:
        raw_data = f.read()

    # For each external tensor, read its data from the external file
    for input_tensor in model.graph.input:
        if input_tensor.type.HasField('tensor_type'):
            pass  # Input tensors handled differently

    # Load external tensors
    external_tensors = {}
    for initializer in model.graph.initializer:
        if initializer.data_location == onnx.TensorProto.EXTERNAL:
            # Get external data info
            for entry in initializer.external_data:
                if entry.key == 'location':
                    location = entry.raw
                    if isinstance(location, bytes):
                        location = location.decode('utf-8')

                    # Read from external file
                    offset = 0
                    size = 0
                    for entry2 in initializer.external_data:
                        if entry2.key == 'offset':
                            offset = int(entry2.raw) if hasattr(entry2, 'raw') else int(entry2.i)
                        if entry2.key == 'length':
                            size = int(entry2.raw) if hasattr(entry2, 'raw') else int(entry2.i)

                    if location == onnx_data_path.name:
                        data = raw_data[offset:offset+size]
                        external_tensors[initializer.name] = data
                        break

    # We need to use onnx.external_data_helper to properly load external data
    # The key function is load_external_data_for_model
    try:
        model_with_data = external_data_helper.load_external_data_for_model(
            model, str(model_path.parent)
        )
        return model_with_data
    except Exception as e:
        print(f"  ⚠️  load_external_data_for_model failed: {e}")
        return model


def make_standalone(model, output_path: Path) -> bool:
    """
    Save a model as standalone (all tensor data embedded in the ONNX file).

    Args:
        model: onnx.ModelProto (with external data loaded)
        output_path: Path for the output standalone ONNX file

    Returns:
        True if successful
    """
    # Save the model - onnx.save embeds all data when external data was loaded
    onnx.save(model, str(output_path))
    return True


def process_file(model_path: Path, output_path: Path) -> bool:
    """
    Process a single ONNX file.

    Args:
        model_path: Path to the .onnx graph file
        output_path: Path for the output standalone .onnx file

    Returns:
        True if successful
    """
    print(f"\n{'='*60}")
    print(f"Processing: {model_path.name}")
    print(f"{'='*60}")

    graph_size = model_path.stat().st_size / 1024 / 1024
    data_path = model_path.with_name(model_path.stem + "_data.onnx_data")
    has_data = data_path.exists()
    data_size = data_path.stat().st_size / 1024 / 1024 if has_data else 0

    print(f"  Graph:  {graph_size:.1f} MB")
    print(f"  Data:   {data_size:.1f} MB")
    print(f"  Total:  {graph_size + data_size:.1f} MB")

    if not has_data:
        print(f"  ℹ️  No external data file found, copying as-is...")
        import shutil
        shutil.copy2(model_path, output_path)
        print(f"  ✅ Copied to: {output_path.name}")
        return True

    try:
        print(f"  🔄 Loading model with external data...")
        model = load_model_with_external_data(str(model_path))

        print(f"  🔄 Saving as standalone ONNX...")
        success = make_standalone(model, output_path)

        saved_size = output_path.stat().st_size / 1024 / 1024
        print(f"  ✅ SUCCESS!")
        print(f"     Standalone size: {saved_size:.1f} MB")
        print(f"     Output: {output_path.name}")

        # Verify
        verify = onnx.load(str(output_path))
        has_external = any(
            init.data_location == onnx.TensorProto.EXTERNAL
            for init in verify.graph.initializer
        )
        if has_external:
            print(f"  ⚠️  WARNING: Output still has external data references!")
        else:
            print(f"  ✓ Verified: No external data (fully standalone)")

        return True

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_bundle_models_dir() -> Path:
    """Find the ChatterboxModels directory in the Xcode bundle resources."""
    project_dir = Path(__file__).parent

    # Check Resources directory
    resources = project_dir / "Reader" / "Resources" / "ChatterboxModels"
    if resources.exists():
        return resources

    # Try device directories (Simulator)
    home = Path.home()
    library = home / "Library" / "Developer" / "CoreSimulator" / "Devices"
    if library.exists():
        for device_dir in library.iterdir():
            if device_dir.is_dir():
                data_dir = device_dir / "data" / "Containers" / "Data" / "Application"
                if data_dir.exists():
                    for app_dir in data_dir.iterdir():
                        models = app_dir / "Documents" / "ChatterboxModels"
                        if models.exists():
                            return models

    return resources  # Fall back to project dir (will fail if not exists)


def main():
    print(f"\n🔧 ONNX External Data → Standalone Converter")
    print(f"   onnx version: {ONNX_VERSION}")
    print(f"   Python: {sys.version}")

    # Find models directory
    models_dir = find_bundle_models_dir()
    print(f"\n📂 Models directory: {models_dir}")

    if not models_dir.exists():
        print(f"ERROR: Directory not found: {models_dir}")
        sys.exit(1)

    # Find all .onnx files
    onnx_files = sorted(models_dir.glob("*.onnx"))

    # Filter out already-merged and already-standalone files
    to_process = []
    for f in onnx_files:
        if "_standalone" in f.name:
            print(f"\nℹ️  Skipping (already standalone): {f.name}")
        elif "_merged" in f.name:
            print(f"\nℹ️  Skipping (already merged): {f.name}")
        else:
            to_process.append(f)

    if not to_process:
        print("\n✅ No files to process (all already standalone)")
        sys.exit(0)

    print(f"\n📋 Files to process: {len(to_process)}")
    for f in to_process:
        size = f.stat().st_size / 1024 / 1024
        data_path = f.with_name(f.stem + "_data.onnx_data")
        data_size = data_path.stat().st_size / 1024 / 1024 if data_path.exists() else 0
        print(f"   {f.name}: {size:.1f}MB + {data_size:.1f}MB = {size+data_size:.1f}MB")

    # Process each file
    results = []
    for model_path in to_process:
        output_path = model_path.with_name(
            model_path.stem.replace("_q4f16", "_q4f16_standalone") + ".onnx"
        )
        success = process_file(model_path, output_path)
        results.append((model_path.name, success))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    successful = sum(1 for _, s in results if s)
    failed = sum(1 for _, s in results if not s)

    for name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")

    print(f"\n  Total: {len(results)} | {successful} succeeded, {failed} failed")

    if successful > 0:
        print(f"\n📝 Next steps:")
        print(f"   1. The _standalone.onnx files are ready")
        print(f"   2. To use them, update ModelDownloadService.modelPath() to load")
        print(f"      the _standalone.onnx files instead of the original .onnx files")
        print(f"   3. Or copy the _standalone files over the originals (backup first!)")


if __name__ == "__main__":
    main()
