#!/bin/bash
# ONNX Runtime Float16 Patch Script
# Run this after `xcodebuild -resolvePackageDependencies` to patch the ONNX Runtime package

PATCH_DIR="$(dirname "$0")/ONNXRuntimePatches"
DERIVED_DATA="$HOME/Library/Developer/Xcode/DerivedData"

# Find the Reader DerivedData folder
ORT_PATH=$(find "$DERIVED_DATA" -name "onnxruntime-swift-package-manager" -type d 2>/dev/null | head -1)

if [ -z "$ORT_PATH" ]; then
    echo "ERROR: ONNX Runtime package not found. Run 'xcodebuild -resolvePackageDependencies' first."
    exit 1
fi

echo "Patching ONNX Runtime at: $ORT_PATH"

# Patch ort_enums.h - Add Float16 enum
if ! grep -q "ORTTensorElementDataTypeFloat16" "$ORT_PATH/objectivec/include/ort_enums.h"; then
    sed -i '' 's/ORTTensorElementDataTypeString,/ORTTensorElementDataTypeString,\n  ORTTensorElementDataTypeFloat16,/' "$ORT_PATH/objectivec/include/ort_enums.h"
    echo "Patched ort_enums.h"
else
    echo "ort_enums.h already patched"
fi

# Patch ort_enums.mm - Add Float16 mapping
if ! grep -q "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16" "$ORT_PATH/objectivec/ort_enums.mm"; then
    sed -i '' 's/ORTTensorElementDataTypeString, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, std::nullopt},/ORTTensorElementDataTypeString, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, std::nullopt},\n    {ORTTensorElementDataTypeFloat16, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, sizeof(uint16_t)},/' "$ORT_PATH/objectivec/ort_enums.mm"
    echo "Patched ort_enums.mm"
else
    echo "ort_enums.mm already patched"
fi

echo "Patch complete!"
