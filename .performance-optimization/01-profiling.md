# Phase 1: Performance Profiling - Reader iOS App

## Target
Reader iOS App - Chatterbox TTS audio playback system

## Overview
This iOS app uses Chatterbox Turbo ONNX for text-to-speech synthesis. The performance profile is fundamentally different from web apps - it's focused on:
1. ONNX model inference performance
2. Audio processing and playback
3. Memory management for large models
4. Swift concurrency efficiency
5. SQLite database operations

## Key Performance-Sensitive Components

### 1. ChatterboxEngine.swift (ONNX Inference)
**Critical Path: Text → TTS Audio**

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Tokenizer encoding | O(n) | GPT-2 BPE, 50K vocab |
| embed_tokens ONNX | O(n × 1024) | Embedding lookup |
| speech_encoder ONNX | O(samples) | Audio feature extraction |
| language_model decode | O(T × 1024 × 24 layers) | Autoregressive, T=1500 tokens max |
| conditional_decoder ONNX | O(T × samples) | Waveform generation |
| Audio post-processing | O(n) | Clipping, normalization |

**Bottlenecks Identified:**
- **Line 767-900**: Language model autoregressive decode loop - O(T²) due to KV cache updates per step
- **Line 320-360**: Batch synthesis with TaskGroup - potential backpressure issues
- **Line 944**: Sample clipping loop - could use SIMD

### 2. AudioPlayerService.swift (Playback)
**Critical Path: Chunk loading → Playback → Waveform**

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Chunk polling | O(1) per 100ms | Timer-based |
| Waveform generation | O(samples) | Line 871-940 |
| Crossfade | O(n) per chunk | Lines 443-464 |
| AVAudioEngine routing | O(1) | Hardware-dependent |

**Bottlenecks Identified:**
- **Line 871-940**: Waveform generation in Task.detached - processes all samples
- **Line 1010**: Chunk polling with maxIterations=50 (5 seconds total wait)
- **Line 443-464**: Crossfade uses sequential asyncAfter calls

### 3. PlayerViewModel.swift (Orchestration)
**Critical Path: State management, caching**

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Text chunking | O(n) | TextChunker service |
| Paragraph caching | O(n) | Lines 358-384 |
| Attributed string creation | O(n × styles) | Per-paragraph |
| Chunk file management | O(n) | File I/O |

**Bottlenecks Identified:**
- **Line 358-384**: Creates attributed strings for all paragraphs upfront
- **Line 1494**: Sequential chunk URL loading with dropFirst()
- **Line 1504-1511**: Busy-wait polling for playback completion

### 4. SynthesisDatabase.swift (Persistence)
**Critical Path: Read/Write operations**

| Operation | Complexity | Notes |
|-----------|------------|-------|
| getAllItems() | O(n) | No pagination, line 221-242 |
| getItemByID | O(1) | Primary key lookup |
| createItem | O(1) | Insert |
| updateProgress | O(1) | Update |

**Bottlenecks Identified:**
- **Line 231**: Uses .map on database cursor - loads all results
- **Line 221-242**: No pagination for getAllItems()

## Memory Analysis

### ONNX Model Memory Footprint
- **speech_encoder_q4f16**: ~80MB (quantized)
- **embed_tokens_q4f16**: ~60MB
- **language_model_q4f16**: ~300MB
- **conditional_decoder_q4f16**: ~140MB
- **Total**: ~580MB (quantized models)

### Runtime Memory
- **Audio buffers**: 24kHz × 16-bit × chunk_duration
- **Token buffers**: maxNewTokens=1500 × 1024 × 4 bytes = ~6MB
- **KV cache**: 24 layers × 2 (K/V) × 1500 × 64 × 2 bytes = ~4.6MB

## Concurrency Analysis

### Current Pattern
- **MainActor** for UI updates
- **Task.detached** for ONNX inference (line 861)
- **TaskGroup** for parallel chunk synthesis (line 328)
- **DispatchQueue.main.async** for UI callbacks

### Issues
1. **Line 844-866**: Task.detached wraps entire synthesis - can't cancel mid-chunk
2. **Line 375-431**: Multiple MainActor.run calls for progress updates
3. **Line 1504-1511**: while loop with Task.sleep - CPU waste

## iOS-Specific Considerations

### Instruments Recommendations
1. **Time Profiler**: Focus on ChatterboxEngine.synthesizeChunk()
2. **Allocations**: Track ONNX session memory
3. **Core Animation**: Check for UI jank during synthesis
4. **Leaks**: Verify waveform Task cancellation works

### Optimization Priorities

| Priority | Area | Current State | Target |
|----------|------|---------------|-------|
| P0 | ONNX inference | Serial decode | Batch decode if possible |
| P0 | Memory usage | ~600MB models | Stay under 1GB |
| P1 | Waveform generation | Full sample processing | Downsample for display |
| P1 | Chunk polling | 5s timeout | Event-driven |
| P2 | Database queries | No pagination | Cursor-based |
| P2 | Text processing | Eager caching | Lazy loading |

## Deliverables Status
- [x] Performance profile analysis
- [x] Bottleneck identification
- [x] Memory analysis
- [x] Concurrency analysis
- [ ] Instruments profiling (requires Xcode)

## Next Steps
Proceed to Phase 2: Database & Backend Optimization (adapted for iOS: Database + Memory optimization)
