# ChatterboxMetalLM

Custom Metal compute shader implementation of the Chatterbox Turbo GPT2-Medium language model.

## Integration

1. Add `chatterbox-metal-lm/src/` to your Xcode target
2. Metal library loads from main bundle via `device.makeDefaultLibrary()`
3. Set `useMetalLM = true` on `ChatterboxEngine` to enable

## Files

- `LanguageModel.metal` — Q4F16 dequantization + LayerNorm kernels
- `MPSGraphEncoder.swift` — MPSGEMM, MPSDPA, LayerNormPipeline, GPT2BlockForward
- `Q4F16Dequant.swift` — Weight dequantization pipeline
- `WeightLoader.swift` — ONNX → Metal buffer loader
- `KVCacheBuffer.swift`, `KVCacheManager.swift` — KV cache management
- `MetalLMEncoder.swift` — MPS encoding orchestration
- `MetalLMForward.swift` — Single-step forward (TODO stub)
- `MetalLMDecode.swift` — Decode loop + greedy decode
- `ChatterboxMetalLM.swift` — Public API

## Build Notes

iOS Metal: `.metal` files are compiled automatically by Xcode. No manual `xcrun metallib` needed.

## Build Verification

On a Mac with Xcode and iOS Simulator:

1. Open `Reader.xcodeproj`
2. Ensure chatterbox-metal-lm/src/ is included in the Reader target
3. Build: `xcodebuild -project Reader.xcodeproj -scheme Reader -configuration Debug -destination 'platform=iOS Simulator,name=iPhone 17' compile 2>&1 | grep -E "error:|Build (succeeded|FAILED)"`
4. Enable Metal LM: Set `chatterboxEngine.useMetalLM = true` before calling `speak()`