# Performance Optimization Architecture Design
Date: 2026-03-20

## Objective
To significantly improve the speed and efficiency of the Reader application's Text-to-Speech (TTS) generation using the ResembleAI/chatterbox-turbo model without sacrificing audio quality. This is particularly targeted at bringing generation times closer to zero-latency on iOS devices.

## The Problem
The current implementation runs efficiently for small tests but suffers from extensive overhead during full sentence generation:
1. **CPU Bound Execution:** The `language_model` is quantized to `q4f16` (block quantized), causing Apple's CoreML framework to crash or fallback heavily because the ANE (Apple Neural Engine) lacks `GatherBlockQuantized` operations. The naive approach forces all four ONNX models to run on CPU.
2. **Swift/Obj-C Bridge Overhead:** The autoregressive loop runs ~500 times per second. Inside the loop, new Swift Arrays, `NSNumber`s, and `ORTValue`s are dynamically allocated and garbage-collected, creating massive memory thrashing and CPU overhead completely unrelated to matrix multiplication math.
3. **Wait-to-Play Bottleneck:** The audio player waits for the entire sentence to conclude generation before playing, leading to multi-second delays.

## Proposed 3-Phase Architecture

### Phase 1: The Swift Fast-Path (Targeting Model Binding overhead)
1. **Targeted CoreML Activation:**
   We will initialize `ORTSessionOptions` specifically customized for each model variant.
   - `speech_encoder`: CoreML Enabled (FP16/FP32).
   - `embed_tokens`: CoreML Enabled (FP16/FP32).
   - `conditional_decoder`: CoreML Enabled (FP16/FP32).
   - `language_model`: **CPU Only** (Simd-optimized). 
   This safely offloads 3/4 of the model architecture to the Apple Neural Engine/Metal while preserving the safety of the quantized LLM.

2. **Inner-Loop Tensor Pre-allocation:**
   Instead of dynamically instantiating input dictionaries and rebuilding arrays, we will:
   - Identify the exact buffer sizes needed for `attention_mask` and `position_ids`.
   - Update values directly via C-pointers (`UnsafeMutablePointer`) to avoid Swift Array reinstantiation.
   - Reuse KV-cache vectors where ORT API allows array-pointer mutation.

### Phase 2: Audio Streaming (Targeting Perceived Latency)
Modify the generation pipeline to operate on chunks. Instead of holding all generated TTS tokens until `config.stopSpeechToken` is reached, the model will flush tokens to the `conditional_decoder` every ~30 tokens. The output audio buffers will be piped immediately into an `AVAudioEngine` queue, allowing playback to begin while the remainder of the sentence computes asynchronously.

### Phase 3: Raw C++ Edge (Targeting Extreme Throughput)
Bypass `onnxruntime-objc` and write a custom pure C++ bridge (or integrate `llama.cpp` using GGML) strictly for the AR logic, minimizing Obj-C conversion delays.

## Next Steps
We are proceeding directly with **Phase 1 Implementation**. Once implemented, we will benchmark the tokens-per-second improvement to ensure ANE utilization is active for the target layers.
