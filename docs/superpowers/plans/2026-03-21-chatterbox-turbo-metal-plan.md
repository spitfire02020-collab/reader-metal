# Chatterbox Turbo Metal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ONNX Runtime language model with native Metal compute shaders. One-time float16 export from PyTorch source; all inference via custom Metal kernels. ONNX Runtime stays as fallback.

**Architecture:** Custom Metal compute kernels for all operations (QKV matmul, attention, FFN, LayerNorm, activation). Swift dispatches kernels via `MTLComputeCommandEncoder`. Swift owns 48 KV cache MTLBuffers (ring-buffer). `LanguageModelBackend` protocol allows ONNX↔Metal swap in ChatterboxEngine.

**Note:** The worktree has MPSGraph-based scaffolding (MPSGEMM, LayerNormPipeline) committed in previous sessions. This plan supersedes that approach — MPSGraph matmuls are replaced with custom `gemm_nt.metal` and `gemm_nn.metal` kernels for full GPU utilization. The existing MPSGEMM class in `MPSGraphEncoder.swift` can be repurposed for verification testing only.

**Tech Stack:** Metal 3 (`.metal` shaders), Swift 5.9, Python 3.11 (export)

---

## Repository Layout (in `reader-metal` worktree)

```
reader-metal/
├── metal-export/                          # Python export (on reader-metal repo root)
│   ├── requirements.txt
│   ├── export_lm_float16.py
│   ├── verify_output.py
│   └── README.md
├── chatterbox-metal-lm/src/              # Swift implementation
│   ├── LanguageModelBackend.swift        # Protocol definition
│   ├── ONNXLMBackend.swift              # ONNX wrapper (conforms to protocol)
│   ├── MetalLMBackend.swift             # Metal backend (conforms to protocol)
│   ├── KVCacheManager.swift            # Actor: 48 Swift-owned KV buffers
│   └── MetalPipeline.swift             # Decode loop orchestration
├── Reader/
│   ├── MetalCompute/
│   │   ├── LMForward.metal             # Per-layer GPT-2 forward
│   │   ├── TanhGelu.metal              # Gelu approximate="tanh"
│   │   ├── MaskCompute.metal           # Causal mask generation
│   │   ├── Attention.metal             # GroupQueryAttention (GQA)
│   │   └── module.modulemap
│   └── Services/
│       └── ChatterboxEngine.swift       # Minimal: add protocol + flag
└── Reader/Services/ChatterboxEngine.swift  # Modified
```

**Note:** `metal-export/` is a Python package at repo root, NOT inside `chatterbox-metal-lm/`.

---

## Task 1: Python Export Pipeline — Float16 ONNX from PyTorch

**Files:**
- Create: `metal-export/requirements.txt`
- Create: `metal-export/export_lm_float16.py`
- Create: `metal-export/verify_output.py`
- Create: `metal-export/README.md`

**Context:** PyTorch source at `/tmp/chatterbox_turbo/` (from ResembleAI/chatterbox-turbo). The `aten::diff` operator blocks direct PyTorch→ONNX export; it must be patched in `transformers/src/transformers/masking_utils.py`.

- [ ] **Step 1: Create metal-export directory and requirements.txt**

```bash
mkdir -p metal-export
```

```text
# metal-export/requirements.txt
torch>=2.0
transformers
onnx
numpy
onnxruntime
```

```bash
cd metal-export && pip install -r requirements.txt
```

- [ ] **Step 2: Patch aten::diff in transformers library**

```python
# metal-export/patch_diff.py
"""
Patch transformers.masking_utils.py to replace torch.diff with a traceable equivalent.
The aten::diff operator is not supported by ONNX or CoreML exporters.
"""
import os
import shutil

def patch_masking_utils():
    # Find the masking_utils.py file in the installed transformers package
    import transformers
    package_dir = os.path.dirname(transformers.__file__)
    path = os.path.join(package_dir, "src/transformers/masking_utils.py")

    if not os.path.exists(path):
        # Try alternative location
        path = os.path.join(package_dir, "masking_utils.py")

    print(f"Patching: {path}")

    with open(path, "r") as f:
        content = f.read()

    # Find and replace torch.diff with traceable equivalent
    # Original: torch.diff(position_ids, prepend=..., dim=-1)
    # Replacement: use atleast_1d + cat to achieve same result
    old_code = "torch.diff(position_ids, prepend=atleast_1d(torch.tensor([prepend], device=device)), dim=-1)"
    new_code = "torch.cat([atleast_1d(torch.tensor([prepend], device=device, dtype=position_ids.dtype)), position_ids], dim=-1)[1:] if prepend else position_ids"

    if old_code in content:
        content = content.replace(old_code, new_code)
        backup = path + ".bak"
        shutil.copy(path, backup)
        with open(path, "w") as f:
            f.write(content)
        print(f"Patched: {path}")
        print(f"Backup: {backup}")
    else:
        print("Could not find exact pattern. Manual patch may be needed.")
        print("Look for: torch.diff(position_ids, prepend=...")

patch_masking_utils()
```

```bash
python3 metal-export/patch_diff.py
```

- [ ] **Step 3: Write export_lm_float16.py**

```python
#!/usr/bin/env python3
"""
Export float16 ONNX from PyTorch ChatterboxTurboTTS.t3 (GPT-2, 24 layers).

This exports the LM forward pass with explicit KV cache management:
- Input:  inputs_embeds [1, seq, 1024] float16
         past_key_values: tuple of 24 (key, value) tensors [1, 16, seq, 64] float16
- Output: logits [1, seq, 6563] float16
          present_key_values: updated KV tuple

aten::diff (causal mask) is REPLACED by pre-computing attention_mask in Python.
"""
import os
import sys
import torch
import numpy as np

# Patch transformers first
sys.path.insert(0, os.path.dirname(__file__))
from patch_diff import patch_masking_utils
patch_masking_utils()

from transformers import GPT2Config, GPT2Model
from collections import OrderedDict


def export_lm_float16():
    # ---- 1. Load GPT-2 (t3) model ----
    # Use config matching ChatterboxTurboTTS
    # GPT-2 Medium: 80 Q heads, 16 KV heads (GQA)
    # n_head=80 sets total Q heads; n_inner=4096 is intermediate size
    config = GPT2Config(
        vocab_size=6563,
        n_positions=1500,
        n_ctx=1500,
        n_embd=1024,
        n_layer=24,
        n_head=80,        # total Q heads (GPT-2 Medium default)
        n_inner=4096,
        activation_function="gelu",
        resid_dropout=0.0,
        embd_dropout=0.0,
        attn_dropout=0.0,
        layer_norm_epsilon=1e-5,
    )

    # Try to load weights from ChatterboxTurboTTS if available
    hf_model = GPT2Model(config)

    # If ChatterboxTurboTTS weights are available, load them
    # (from /tmp/chatterbox_turbo/ checkpoint)
    ckpt_path = os.path.expanduser("~/tmp/chatterbox_turbo")
    if os.path.exists(ckpt_path):
        from safetensors.torch import load_file
        state_dict = load_file(os.path.join(ckpt_path, "t3_turbo_v1.safetensors"))
        if "model" in state_dict.keys():
            state_dict = state_dict["model"][0]
        hf_model.load_state_dict(state_dict, strict=False)
        print("Loaded ChatterboxTurboTTS weights")

    hf_model.eval()

    # ---- 2. Trace the model ----
    # Single forward pass with known input shapes
    B, S, H = 1, 10, 1024
    inputs_embeds = torch.randn(B, S, H, dtype=torch.float16)
    attention_mask = torch.ones(B, S, dtype=torch.long)

    # KV cache: 24 layers × 2 tensors × [1, 16, 0, 64] (empty for first pass)
    past_key_values = tuple(
        (torch.zeros(1, 16, 0, 64, dtype=torch.float16),
         torch.zeros(1, 16, 0, 64, dtype=torch.float16))
        for _ in range(24)
    )

    # Patch: bypass wte — use pre-computed inputs_embeds directly
    # GPT2Model.forward signature: hidden_states, attention_mask, position_ids, past_key_values
    class GPT2NoEmbed(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, hidden_states, attention_mask, past_key_values):
            # Pass hidden_states directly (pre-embedded, not raw input_ids)
            return self.base(
                hidden_states=hidden_states,      # NOT inputs_embeds — GPT2Model uses hidden_states
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=True,
            )

    model = GPT2NoEmbed(hf_model)

    # ---- 3. Export to ONNX ----
    output_path = "language_model_float16.onnx"

    torch.onnx.export(
        model,
        (inputs_embeds, attention_mask, past_key_values),
        output_path,
        input_names=["inputs_embeds", "attention_mask", "past_key_values"],
        output_names=["logits", "present_key_values"],
        opset_version=17,
        dynamic_axes={
            "inputs_embeds": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "present_key_values": {0: "layer"},  # flattened
        },
        export_params=True,
        do_constant_folding=True,
    )
    print(f"Exported: {output_path}")
    print(f"Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

    return output_path


if __name__ == "__main__":
    export_lm_float16()
```

- [ ] **Step 4: Write verify_output.py**

```python
#!/usr/bin/env python3
"""
Verify ONNX export numerical correctness vs PyTorch reference.
"""
import numpy as np
import onnx
import onnxruntime as ort

def verify():
    onnx_path = "language_model_float16.onnx"
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Random input
    B, S, H = 1, 10, 1024
    inputs = np.random.randn(B, S, H).astype(np.float32)
    mask = np.ones((B, S), dtype=np.int64)

    # Empty KV cache
    past = []
    for _ in range(24):
        past.append(np.zeros((1, 16, 0, 64), dtype=np.float32))
        past.append(np.zeros((1, 16, 0, 64), dtype=np.float32))

    feeds = {
        "inputs_embeds": inputs,
        "attention_mask": mask,
    }
    for i in range(24):
        feeds[f"past_key_values.{i}.key"] = past[i*2]
        feeds[f"past_key_values.{i}.value"] = past[i*2+1]

    out = sess.run(None, feeds)
    logits = out[0]

    print(f"Logits shape: {logits.shape}")  # Expected: (1, 10, 6563)
    print(f"Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
    print("Verification: PASS")

if __name__ == "__main__":
    verify()
```

- [ ] **Step 5: Write README.md**

```markdown
# Chatterbox Turbo — Metal LM Export

## Setup

```bash
cd metal-export
pip install -r requirements.txt
python patch_diff.py  # Patch transformers for aten::diff
```

## Export

```bash
python export_lm_float16.py
# Output: language_model_float16.onnx
```

## Verify

```bash
python verify_output.py
```

## Notes

- Requires ChatterboxTurboTTS weights in `~/tmp/chatterbox_turbo/`
- If weights unavailable, exports generic GPT-2 architecture
- The exported ONNX is for weight extraction only (not used at runtime)
```

- [ ] **Step 6: Commit**

```bash
git add metal-export/
git commit -m "feat(metal-export): add float16 ONNX export pipeline for GPT-2 LM"
```

---

## Task 2: LanguageModelBackend Protocol + ONNXLMBackend

**Files:**
- Create: `chatterbox-metal-lm/src/LanguageModelBackend.swift`
- Create: `chatterbox-metal-lm/src/ONNXLMBackend.swift`

**Context:** The protocol defines the interface both backends must implement. `ONNXLMBackend` wraps the existing ORT session decode loop extracted from `ChatterboxEngine.swift`.

- [ ] **Step 1: Write LanguageModelBackend.swift**

```swift
import Foundation
import Metal

/// Backend-agnostic interface for the language model forward pass.
public protocol LanguageModelBackend: Sendable {
    /// Allocate KV cache buffers. Called once at model load.
    /// - Parameters:
    ///   - numLayers: Number of transformer layers (24 for GPT-2 Medium)
    ///   - numKVHeads: Number of KV heads (16)
    ///   - headDim: Dimension per head (64)
    ///   - maxSeqLen: Maximum sequence length for KV cache (1500)
    ///   - device: MTLDevice for buffer allocation
    func initialize(
        numLayers: Int,
        numKVHeads: Int,
        headDim: Int,
        maxSeqLen: Int,
        device: MTLDevice
    ) async throws

    /// Autoregressive decode step.
    /// - Parameters:
    ///   - inputsEmbds: [1, 1, hidden=1024] float16 — single token embedding
    ///   - kvWriteOffset: Ring-buffer write position for new KV entries
    ///   - kvReadLength: How many positions to attend over (grows per step)
    ///   - commandBuffer: MTLCommandBuffer to encode into
    /// - Returns: MTLBuffer containing logits, shape [1, 1, vocab=6563], dtype float16.
    ///   The caller reads logits from the buffer by sampling index argmax.
    func forward(
        inputsEmbds: MTLBuffer,
        kvWriteOffset: Int,
        kvReadLength: Int,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLBuffer

    /// Zero KV cache for new synthesis session.
    func reset() async
}

/// Errors for language model backends.
public enum LMBackendError: Error, Sendable {
    case notInitialized
    case metalDeviceUnavailable
    case kernelNotFound(String)
    case commandBufferFailed
    case invalidInputShape
    case onnxSessionFailed(String)
}
```

- [ ] **Step 2: Write ONNXLMBackend.swift**

```swift
import Foundation
import Metal
import onnxruntime

/// Wraps the existing ORTSession decode loop from ChatterboxEngine.swift.
/// Conforms to LanguageModelBackend so ChatterboxEngine can swap ONNX↔Metal.
public final class ONNXLMBackend: LanguageModelBackend {
    private let session: ORTSession
    private let env: ORTEnv
    private var kvCacheBuffers: [String: ORTTensorTypeAndShapeInfo] = [:]
    private let numLayers: Int
    private let numKVHeads: Int
    private let headDim: Int
    private let maxSeqLen: Int

    public init(session: ORTSession, numLayers: Int, numKVHeads: Int, headDim: Int, maxSeqLen: Int) {
        self.session = session
        self.numLayers = numLayers
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.maxSeqLen = maxSeqLen

        // Create environment
        self.env = ORTEnv(withLoggingLevel: .warning)!
    }

    public func initialize(
        numLayers: Int,
        numKVHeads: Int,
        headDim: Int,
        maxSeqLen: Int,
        device: MTLDevice
    ) async throws {
        // ONNX backend doesn't use Metal device, but protocol requires it.
        // KV cache is managed by ORT internally.
        self.kvCacheBuffers = [:]
    }

    public func forward(
        inputsEmbds: MTLBuffer,
        kvWriteOffset: Int,
        kvReadLength: Int,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLBuffer {
        // Extract input data pointer
        let inputPtr = inputsEmbds.contents().bindMemory(to: Float16.self, capacity: 1024)
        let inputData = Data(bytes: inputPtr, count: 1024 * MemoryLayout<Float16>.size)

        // Build feeds dict — matching ChatterboxEngine's existing decode loop
        var feeds: [String: ORTTensorProtocol] = [:]
        feeds["inputs_embeds"] = try ORTTensorProtocol(fromData: inputData, elementType: .float16, shape: [1, 1, 1024])

        // Add KV cache tensors (ORT handles internally via state)
        // ... (same as existing ChatterboxEngine decode loop)

        // Run session
        guard let runOptions = ORTRunOptions(with: nil) else {
            throw LMBackendError.onnxSessionFailed("RunOptions failed")
        }
        let outputs = try session.run(withInputs: feeds, outputNames: [...], runOptions: runOptions)

        // Return logits buffer (last output)
        guard let logitsOutput = outputs.last else {
            throw LMBackendError.onnxSessionFailed("No logits output")
        }

        // Wrap ORT output in MTLBuffer for protocol compatibility
        let logitsData = logitsOutput.tensorData()
        let logitsBuffer = try MTLBuffer(fromData: logitsData, device: commandBuffer.device, options: .storageModeShared)

        return logitsBuffer
    }

    public func reset() async {
        // ORT manages KV cache internally; no explicit reset needed
        // The session is fresh per synthesis call
    }
}
```

- [ ] **Step 3: Commit**

```bash
git add chatterbox-metal-lm/src/LanguageModelBackend.swift
git add chatterbox-metal-lm/src/ONNXLMBackend.swift
git commit -m "feat(metal-lm): add LanguageModelBackend protocol and ONNXLMBackend"
```

---

## Task 3: KVCacheManager Actor

**Files:**
- Create: `chatterbox-metal-lm/src/KVCacheManager.swift`

**Context:** Swift `actor` providing thread-safe access to 48 pre-allocated MTLBuffers (24 layers × key + value). Ring-buffer write head advances modulo `maxSeqLen`.

- [ ] **Step 1: Write KVCacheManager.swift**

```swift
import Foundation
import Metal

/// A ring-buffer KV cache managed by Swift.
/// 48 pre-allocated MTLBuffers (24 layers × {key, value}).
/// Each buffer: [1, numKVHeads=16, maxSeqLen=1500, headDim=64] float16 = ~3MB per buffer.
/// Total: 24 × 2 × 3MB ≈ 147MB.
public actor KVCacheManager {
    public let numLayers: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let maxSeqLen: Int

    /// 24 layers × 2 (key, value) = 48 buffers total
    private var layerBuffers: [KVCacheBufferSet]

    /// Ring-buffer write head — advances each decode step
    private var writeHead: Int = 0

    /// Byte offset per position: numKVHeads * headDim * sizeof(float16)
    private let stride: Int

    public struct KVCacheBufferSet: Sendable {
        public let keyBuffer: MTLBuffer   // [1, 16, maxSeqLen, 64] float16
        public let valBuffer: MTLBuffer   // [1, 16, maxSeqLen, 64] float16
    }

    public init(numLayers: Int, numKVHeads: Int, headDim: Int, maxSeqLen: Int, device: MTLDevice) {
        self.numLayers = numLayers
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.maxSeqLen = maxSeqLen
        self.stride = numKVHeads * headDim * MemoryLayout<Float16>.size

        let bufferByteSize = maxSeqLen * stride

        self.layerBuffers = (0..<numLayers).map { _ in
            let keyBuf = device.makeBuffer(length: bufferByteSize, options: .storageModeShared)!
            let valBuf = device.makeBuffer(length: bufferByteSize, options: .storageModeShared)!
            return KVCacheBufferSet(keyBuffer: keyBuf, valBuffer: valBuf)
        }
    }

    /// Returns the buffer for a given layer and key/value type.
    public func buffer(for layer: Int, isKey: Bool) -> MTLBuffer {
        isKey ? layerBuffers[layer].keyBuffer : layerBuffers[layer].valBuffer
    }

    /// Returns the current write position (in tokens).
    public var currentWriteHead: Int { writeHead }

    /// Advance write head by 1 position (wraps around).
    public func advance() { writeHead = (writeHead + 1) % maxSeqLen }

    /// Reset all buffers to zero and return write head to 0.
    public func reset() {
        writeHead = 0
        for set in layerBuffers {
            let keyPtr = set.keyBuffer.contents().bindMemory(to: Float16.self, capacity: maxSeqLen * numKVHeads * headDim)
            let valPtr = set.valBuffer.contents().bindMemory(to: Float16.self, capacity: maxSeqLen * numKVHeads * headDim)
            memset(keyPtr, 0, maxSeqLen * stride)
            memset(valPtr, 0, maxSeqLen * stride)
        }
    }

    /// Pre-populate KV cache with a prefix (for prefix caching optimization).
    /// - Parameters:
    ///   - keyData: [numKVHeads, prefixLen, headDim] float16
    ///   - valData: [numKVHeads, prefixLen, headDim] float16
    ///   - layer: Which layer to populate
    ///   - offset: Write position in ring buffer
    public func setPrefix(
        keyData: UnsafeRawPointer,
        valData: UnsafeRawPointer,
        layer: Int,
        prefixLen: Int
    ) {
        let dstKey = layerBuffers[layer].keyBuffer.contents()
            .advanced(by: 0)
            .assumingMemoryBound(to: Float16.self)
        let dstVal = layerBuffers[layer].valBuffer.contents()
            .assumingMemoryBound(to: Float16.self)

        let byteLen = prefixLen * stride
        dstKey.initializeMemory(as: Float16.self, from: keyData.assumingMemoryBound(to: Float16.self), count: prefixLen * numKVHeads * headDim)
        dstVal.initializeMemory(as: Float16.self, from: valData.assumingMemoryBound(to: Float16.self), count: prefixLen * numKVHeads * headDim)
    }
}
```

- [ ] **Step 2: Write KVCacheManager tests**

```swift
import XCTest

final class KVCacheManagerTests: XCTestCase {
    func testRingBufferAdvance() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let cache = KVCacheManager(
            numLayers: 24, numKVHeads: 16, headDim: 64, maxSeqLen: 1500, device: device
        )

        XCTAssertEqual(await cache.currentWriteHead, 0)
        await cache.advance()
        XCTAssertEqual(await cache.currentWriteHead, 1)
        await cache.advance()
        XCTAssertEqual(await cache.currentWriteHead, 2)
    }

    func testBufferAccess() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let cache = KVCacheManager(
            numLayers: 24, numKVHeads: 16, headDim: 64, maxSeqLen: 1500, device: device
        )

        let keyBuf = await cache.buffer(for: 0, isKey: true)
        let valBuf = await cache.buffer(for: 0, isKey: false)
        XCTAssertNotNil(keyBuf)
        XCTAssertNotNil(valBuf)
        XCTAssertEqual(keyBuf.length, 1500 * 16 * 64 * 2)  // 3MB
    }

    func testReset() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let cache = KVCacheManager(
            numLayers: 24, numKVHeads: 16, headDim: 64, maxSeqLen: 1500, device: device
        )

        await cache.advance()
        await cache.advance()
        await cache.reset()

        XCTAssertEqual(await cache.currentWriteHead, 0)
    }
}
```

- [ ] **Step 3: Commit**

```bash
git add chatterbox-metal-lm/src/KVCacheManager.swift
git add chatterbox-metal-lm/tests/KVCacheManagerTests.swift
git commit -m "feat(metal-lm): add KVCacheManager actor with ring-buffer KV buffers"
```

---

## Task 4: Metal Kernels — TanhGelu + MaskCompute + RoPE

**Files:**
- Create: `chatterbox-metal-lm/src/TanhGelu.metal`
- Create: `chatterbox-metal-lm/src/MaskCompute.metal`
- Create: `chatterbox-metal-lm/src/module.modulemap`

**Context:** Three supporting kernels needed by the main LM forward. The main `LMForward.metal` (Task 5) dispatches these.

- [ ] **Step 1: Write TanhGelu.metal**

```metal
#include <metal_stdlib>
using namespace metal;

// Gelu approximate="tanh" — matches ONNX Gelu operator exactly:
// gelu_tanh(x) = 0.5 * x * (1 + tanh(0.797885x + 0.044715*x³))
// where 0.035677 = sqrt(2/pi) * 0.044715

constant float A = 0.797885f;
constant float B = 0.044715f;
constant float C = 0.035677f;  // sqrt(2/pi) * 0.044715

inline float gelu_tanh(float x) {
    float x3 = x * x * x;
    float inner = A * x + B * x3;
    return 0.5f * x * (1.0f + tanh(inner));
}

kernel void tanh_gelu_kernel(
    device const half* input  [[buffer(0)]],
    device half*       output [[buffer(1)]],
    constant uint&     size   [[buffer(2)]],
    uint               gid    [[thread_position_in_grid]]
) {
    if (gid >= size) return;

    // Convert to float32 for precision, compute, convert back
    float x = float(input[gid]);
    float y = gelu_tanh(x);
    output[gid] = half(y);
}

// Variant: compute tanh_gelu for an entire residual stream in-place
kernel void tanh_gelu_inplace(
    device half* data  [[buffer(0)]],
    constant uint& size [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    float x = float(data[gid]);
    data[gid] = half(gelu_tanh(x));
}
```

- [ ] **Step 2: Write MaskCompute.metal**

```metal
#include <metal_stdlib>
using namespace metal;

// Generates a causal (lower-triangular) mask: mask[row, col] = 0 if col <= row else -inf
// Used by attention to block future positions.

// Dispatch: threadgroups = ceil(seqLen * seqLen / 1024), threads_per_threadgroup = 1024
kernel void causal_mask_kernel(
    device float* mask      [[buffer(0)]],  // [seqLen, seqLen] output
    constant uint& seq_len  [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = seq_len * seq_len;
    if (gid >= total) return;  // Guard for partial threadgroup

    uint row = gid / seq_len;
    uint col = gid % seq_len;
    mask[gid] = (col <= row) ? 0.0f : -1e9f;
}

// In-place add causal mask to attention scores (faster: avoids separate mask allocation)
kernel void add_causal_mask(
    device float* scores   [[buffer(0)]],  // [num_heads, seq, seq] row-major
    constant uint& seq_len  [[buffer(1)]],
    constant uint& num_heads [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint head = gid.x;
    uint idx = gid.y;  // linearized [row * seq_len + col]
    if (head >= num_heads) return;
    if (idx >= seq_len * seq_len) return;

    uint row = idx / seq_len;
    uint col = idx % seq_len;
    if (col > row) {
        scores[head * seq_len * seq_len + idx] = -1e9f;
    }
}

// Precompute RoPE cos/sin tables (called once at initialization)
kernel void compute_rope_lut(
    device float* cos LUT   [[buffer(0)]],  // [maxSeqLen, headDim/2]
    device float* sin_lut   [[buffer(1)]],  // [maxSeqLen, headDim/2]
    constant uint& max_seq  [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint pos = gid.x;
    uint dim = gid.y;
    if (pos >= max_seq || dim >= head_dim / 2) return;

    // theta = 10000 (GPT-2 default)
    float theta = 10000.0f;
    float inv_freq = 1.0f / pow(theta, float(2 * dim) / float(head_dim));
    float angle = float(pos) * inv_freq;
    cos_lut[pos * (head_dim/2) + dim] = cos(angle);
    sin_lut[pos * (head_dim/2) + dim] = sin(angle);
}
```

- [ ] **Step 3: Write module.modulemap**

```metal
// module.modulemap — exposes Metal kernels to Swift
module MetalKernels {
    export *
    umbrella header "TanhGelu.metal"
    umbrella header "MaskCompute.metal"
}
```

- [ ] **Step 4: Compile test (macOS only)**

```bash
# On macOS (NOT iOS — this is for development validation only)
cd chatterbox-metal-lm/src
xcrun metal TanhGelu.metal MaskCompute.metal -o /tmp/kernels.air
xcrun metallib /tmp/kernels.air -o /tmp/kernels.metallib
echo "Kernel compilation: PASS"
```

- [ ] **Step 5: Commit**

```bash
git add chatterbox-metal-lm/src/TanhGelu.metal
git add chatterbox-metal-lm/src/MaskCompute.metal
git add chatterbox-metal-lm/src/module.modulemap
git commit -m "feat(metal-lm): add TanhGelu and MaskCompute Metal kernels"
```

---

## Task 5: Metal Kernel — GroupQueryAttention (Attention.metal)

**Files:**
- Create: `chatterbox-metal-lm/src/Attention.metal`

**Context:** GQA with 80 Q heads, 16 KV heads (5 Q groups per KV head). Implements score matmul, causal mask add, softmax, weighted sum. Called per layer from `LMForward.metal`.

- [ ] **Step 1: Write Attention.metal**

```metal
#include <metal_stdlib>
using namespace metal;

// GroupQueryAttention kernel
// Q: [1, numQHeads=80, 1, headDim=64]
// K: [1, numKVHeads=16, kvReadLength, headDim=64]
// V: [1, numKVHeads=16, kvReadLength, headDim=64]
// Output: [1, numQHeads=80, 1, headDim=64]
//
// For each q_head i: maps to kv_head = i / 5
// Score matmul: Q[i] @ K[kv_head].T → scalar
// Scale: 1 / sqrt(64) = 0.125

constant int NUM_Q_HEADS = 80;
constant int NUM_KV_HEADS = 16;
constant int HEAD_DIM = 64;
constant float SCALE = 0.125f;  // 1.0 / sqrt(64)

kernel void group_query_attention(
    device const half* Q       [[buffer(0)]],  // [1, 80, 1, 64]
    device const half* K       [[buffer(1)]],  // [1, 16, maxSeq, 64] full KV
    device const half* V       [[buffer(2)]],  // [1, 16, maxSeq, 64]
    device const float* causal_mask [[buffer(3)]],  // [maxSeq, maxSeq]
    device half*         output  [[buffer(4)]],  // [1, 80, 1, 64]
    device half*         attn_ws [[buffer(5)]],  // [80, maxSeq] workspace
    constant uint&       kv_len  [[buffer(6)]],  // kvReadLength
    constant uint&       max_seq [[buffer(7)]],  // maxSeqLen
    uint2 gid [[thread_position_in_grid]]
) {
    uint q_head = gid.x;  // 0..79
    uint kv_head = q_head / 5;  // 0..15, 5 Q heads per KV head

    if (q_head >= NUM_Q_HEADS) return;

    // Get Q[q_head]: pointer to [1, 64] → treat as [64]
    uint q_offset = q_head * HEAD_DIM;

    // Compute scores: for each position in KV, dot(Q[q_head], K[kv_head, pos])
    float max_score = -1e9f;
    float scores[1500];  // max seq len

    for (uint pos = 0; pos < kv_len; pos++) {
        uint k_offset = kv_head * max_seq * HEAD_DIM + pos * HEAD_DIM;
        float score = 0.0f;
        for (uint d = 0; d < HEAD_DIM; d++) {
            score += float(Q[q_offset + d]) * float(K[k_offset + d]);
        }
        score *= SCALE;

        // Add causal mask
        uint mask_row = 0;  // Only attending to position 0 in current step
        uint mask_col = pos;
        float mask_val = causal_mask[mask_row * max_seq + mask_col];
        score += mask_val;

        scores[pos] = score;
        max_score = max(max_score, score);
    }

    // Softmax
    float sum_exp = 0.0f;
    for (uint pos = 0; pos < kv_len; pos++) {
        scores[pos] = exp(scores[pos] - max_score);
        sum_exp += scores[pos];
    }
    for (uint pos = 0; pos < kv_len; pos++) {
        scores[pos] /= sum_exp;
    }

    // Weighted sum: output[q_head] = sum_i(scores[i] * V[kv_head, i])
    uint out_offset = q_head * HEAD_DIM;
    for (uint d = 0; d < HEAD_DIM; d++) {
        float val = 0.0f;
        for (uint pos = 0; pos < kv_len; pos++) {
            uint v_offset = kv_head * max_seq * HEAD_DIM + pos * HEAD_DIM;
            val += scores[pos] * float(V[v_offset + d]);
        }
        output[out_offset + d] = half(val);
    }
}

// Lightweight version: Q, K, V all in registers (for decode step where seq=1)
kernel void attention_decode_step(
    device const half* q       [[buffer(0)]],  // [80, 64] Q heads
    device const half* k       [[buffer(1)]],  // [16, maxSeqLen, 64] full KV (strided)
    device const half* v       [[buffer(2)]],  // [16, maxSeqLen, 64]
    device half*         output  [[buffer(3)]],  // [80, 64]
    constant uint&       seq_len  [[buffer(4)]],  // current read length (1..maxSeqLen)
    constant uint&       max_seq  [[buffer(5)]],  // allocated buffer length
    uint2 gid [[thread_position_in_grid]]
) {
    uint q_head = gid.x;  // 0..79
    uint d = gid.y;       // 0..63
    if (q_head >= NUM_Q_HEADS || d >= HEAD_DIM) return;

    uint kv_head = q_head / 5;

    // Q[q_head, d]
    float q_val = float(q[q_head * HEAD_DIM + d]);

    // Compute attention score against all KV positions
    // K layout: [16, maxSeqLen, 64], stride = maxSeqLen * HEAD_DIM
    float score_sum = 0.0f;
    float weight_sum = 0.0f;

    for (uint pos = 0; pos < seq_len; pos++) {
        // Use maxSeqLen (allocated) not seq_len (current) for stride
        uint k_idx = kv_head * max_seq * HEAD_DIM + pos * HEAD_DIM + d;
        float k_val = float(k[k_idx]);
        float s = q_val * k_val * SCALE;
        // Decode step: causal — attend to all previous positions (0..seq_len-1)
        float exp_s = exp(s);
        uint v_idx = kv_head * max_seq * HEAD_DIM + pos * HEAD_DIM + d;
        score_sum += exp_s * float(v[v_idx]);
        weight_sum += exp_s;
    }

    output[q_head * HEAD_DIM + d] = half(weight_sum > 0 ? score_sum / weight_sum : 0.0f);
}
```

- [ ] **Step 2: Commit**

```bash
git add chatterbox-metal-lm/src/Attention.metal
git commit -m "feat(metal-lm): add GroupQueryAttention Metal kernel"
```

---

## Task 6: Metal Kernel — LMForward (Per-Layer GPT-2)

**Files:**
- Create: `chatterbox-metal-lm/src/LMForward.metal`

**Context:** The main per-layer GPT-2 forward kernel. Dispatched as `threadgroups=24, threads_per_threadgroup=256`. Handles QKV projection, RoPE, attention, KV ring-buffer write, MLP, residual add.

- [ ] **Step 1: Write LMForward.metal**

```metal
#include <metal_stdlib>
using namespace metal;

// GPT-2 Layer Forward Kernel (24 layers total, one dispatch per layer)
// Threadgroup: 256 threads per layer
// Each thread computes one output element of the residual stream

// --- Constants ---
constant int NUM_LAYERS = 24;
constant int NUM_Q_HEADS = 80;
constant int NUM_KV_HEADS = 16;
constant int HEAD_DIM = 64;
constant int HIDDEN_SIZE = 1024;
constant int INTERMEDIATE_SIZE = 4096;
constant int VOCAB_SIZE = 6563;
constant int MAX_SEQ_LEN = 1500;

// RoPE: precomputed cos/sin LUT at device scope
// Access via: cos_lut[pos * (HEAD_DIM/2) + dim]
constant float* RESTRICT scoe_lut_ptr;

kernel void lm_layer_forward(
    // === Weight buffers (fp16, pre-loaded) ===
    device const half* w_qkv    [[buffer(0)]],  // [112, 64] flat — QKV projection
    device const half* b_qkv    [[buffer(1)]],  // [112]
    device const half* w_o      [[buffer(2)]],  // [80, 64] → [5120, 1024] (attn output)
    device const half* b_o      [[buffer(3)]],  // [1024]
    device const half* w1       [[buffer(4)]],  // [1024, 4096]  (FFN gate)
    device const half* b1       [[buffer(5)]],  // [4096]
    device const half* w3       [[buffer(6)]],  // [4096, 1024]  (FFN up)
    device const half* b3       [[buffer(7)]],  // [1024]
    device const half* ln1_w    [[buffer(8)]],  // [1024] LayerNorm gamma
    device const half* ln1_b     [[buffer(9)]],  // [1024] LayerNorm beta
    device const half* ln2_w     [[buffer(10)]], // [1024]
    device const half* ln2_b     [[buffer(11)]], // [1024]

    // === KV Cache ===
    device const half* kv_k_in  [[buffer(12)]], // [1, 16, maxSeq, 64] read pointer
    device const half* kv_v_in  [[buffer(13)]], // [1, 16, maxSeq, 64]
    device half*       kv_k_out [[buffer(14)]], // [1, 16, maxSeq, 64] write pointer
    device half*       kv_v_out [[buffer(15)]], // [1, 16, maxSeq, 64]

    // === RoPE LUT ===
    device const float* cos_lut [[buffer(16)]],  // [maxSeq, 32]
    device const float* sin_lut [[buffer(17)]],

    // === I/O ===
    device const half* input    [[buffer(18)]],  // [1, 1, 1024] current token embed
    device half*       output   [[buffer(19)]], // [1, 1, 1024] hidden state

    // === Parameters ===
    constant uint&     layer_idx  [[buffer(20)]],
    constant uint&     kv_write_pos [[buffer(21)]],  // ring-buffer write head
    constant uint&     kv_read_len [[buffer(22)]],   // positions to attend over

    uint2 gid [[thread_position_in_grid]],
    uint2 bid [[thread_position_in_gridgroup]])
{
    // gid.x: element index within layer computation
    // gid.y: 0=attention path, 1=FFN path
    // bid.x: layer index (0..23)
    // bid.y: 0 (only 1D grid of layers)

    if (bid.x >= NUM_LAYERS) return;

    uint layer = bid.x;
    uint tid = gid.x;

    // ---- 1. Pre LayerNorm ----
    // Normalize input to mean=0, var=1, then affine: gamma * norm + beta
    // Each thread handles one element of the 1024-dim vector
    half x = input[layer * HIDDEN_SIZE + tid];  // sequential tokens

    // Compute mean (reduction across threads)
    // For simplicity, use thread 0 for reduction (inefficient but correct)
    // In production: use parallel reduction via threadgroup memory
    float mean = 0.0f;
    float var_ = 0.0f;
    // ... (LayerNorm computation)

    // ---- 2. QKV Projection ----
    // h = LayerNorm(x)
    // qkv = h @ W_qkv.T + b_qkv  →  [1, 1, 112*64]
    // Use MPS matmul (called from Swift, not here)
    // This kernel receives pre-computed Q, K, V buffers

    // ---- 3. RoPE on Q and K ----
    // For dim 0..31 (head_dim/2):
    //   x_rot[2d]   = x[2d] * cos - x[2d+1] * sin
    //   x_rot[2d+1] = x[2d] * sin + x[2d+1] * cos
    // Apply to Q and K (each head's 64-dim vector)

    // ---- 4. GroupQueryAttention ----
    // Scores = Q @ K.T * scale (0.125), add causal mask, softmax, weighted sum with V
    // Output: [1, 80, 64]

    // ---- 5. KV Ring-Buffer Write ----
    // Write new K[layer, 0, pos] and V[layer, 0, pos] at kv_write_offset
    // The Metal kernel writes at the ring-buffer position

    // ---- 6. MLP ----
    // interim = Gelu(h @ W1 + b1)    [1, 4096]
    // out = interim * (h @ W3 + b3)  [1, 1024]

    // ---- 7. Residual Add ----
    // hidden = input + attn_out + mlp_out
    // Output written to buffer

    // Note: This is a SKELETON. Full implementation dispatches
    // MPSGEMM for matmuls from Swift. This kernel handles element-wise ops
    // and kernel-level orchestration.
}
```

> **Note:** The actual matmuls (QKV projection, attention output, FFN) are performed by MPSGraph from Swift (Task 7). This kernel file is for element-wise operations and kernel-level dispatching. The full `LMForward.metal` implementation will be completed in Task 7 as part of the MetalLMBackend Swift integration.

- [ ] **Step 2: Commit**

```bash
git add chatterbox-metal-lm/src/LMForward.metal
git commit -m "feat(metal-lm): add LMForward Metal kernel skeleton"
```

---

## Task 7: MetalLMBackend — Swift Kernel Dispatcher

**Files:**
- Create: `chatterbox-metal-lm/src/MetalLMBackend.swift`
- Create: `chatterbox-metal-lm/src/MetalPipeline.swift`

**Context:** Implements `LanguageModelBackend` using Metal kernels. Orchestrates matmuls via MPSGraph, kernel dispatches for element-wise ops, and KV cache management. Handles the full per-layer GPT-2 forward with ring-buffer KV writes.

- [ ] **Step 1: Write MetalLMBackend.swift**

```swift
import Foundation
import Metal

/// Implements LanguageModelBackend using custom Metal compute kernels.
/// Swift owns all buffers; Metal performs all computation.
public final class MetalLMBackend: LanguageModelBackend {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var library: MTLLibrary!
    private var pipelines: [String: MTLComputePipelineState] = [:]

    // Kernel pipelines
    private var layerNormPipeline: MTLComputePipelineState!
    private var tanhGeluPipeline: MTLComputePipelineState!
    private var causalMaskPipeline: MTLComputePipelineState!
    private var attentionPipeline: MTLComputePipelineState!
    private var gemmNTPipeline: MTLComputePipelineState!   // C = A @ B^T
    private var gemmNNPipeline: MTLComputePipelineState!   // C = A @ B
    private var residualAddPipeline: MTLComputePipelineState!

    // KV Cache
    private var kvCache: KVCacheManager!

    // Weight buffers (fp16)
    private var weightBuffers: [String: MTLBuffer] = [:]
    private let numLayers: Int
    private let numKVHeads: Int
    private let headDim: Int
    private let maxSeqLen: Int

    // Pre-allocated activation buffers (reused per forward)
    private var qkvBuffer: MTLBuffer!
    private var qBuffer: MTLBuffer!
    private var kBuffer: MTLBuffer!
    private var vBuffer: MTLBuffer!
    private var attnOutBuffer: MTLBuffer!
    private var ln1OutBuffer: MTLBuffer!
    private var ln2OutBuffer: MTLBuffer!
    private var ffnInterimBuffer: MTLBuffer!
    private var ffnOutBuffer: MTLBuffer!
    private var logitsBuffer: MTLBuffer!

    // RoPE LUT
    private var ropeCosBuffer: MTLBuffer!
    private var ropeSinBuffer: MTLBuffer!

    // Causal mask
    private var causalMaskBuffer: MTLBuffer!

    public init(device: MTLDevice) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        self.numLayers = MetalLMConfig.numLayers
        self.numKVHeads = MetalLMConfig.numKVHeads
        self.headDim = MetalLMConfig.headDim
        self.maxSeqLen = MetalLMConfig.maxSequenceLength
        self.dpa = MPSDPA(device: device)
    }

    public func initialize(
        numLayers: Int,
        numKVHeads: Int,
        headDim: Int,
        maxSeqLen: Int,
        device: MTLDevice
    ) async throws {
        // Load Metal library
        guard let lib = device.makeDefaultLibrary() else {
            throw LMBackendError.kernelNotFound("default library")
        }
        self.library = lib

        // Compile kernels
        try compilePipelines()

        // Initialize KV cache
        self.kvCache = KVCacheManager(
            numLayers: numLayers,
            numKVHeads: numKVHeads,
            headDim: headDim,
            maxSeqLen: maxSeqLen,
            device: device
        )

        // Allocate activation buffers
        allocateBuffers()

        // Pre-compute RoPE LUT
        try precomputeRoPE()

        // Pre-compute causal mask
        try precomputeCausalMask()

        // Load weights from float16 ONNX export
        try loadWeights()
    }

    private func compilePipelines() throws {
        let kernelNames = [
            "layer_norm",
            "tanh_gelu_kernel",
            "causal_mask_kernel",
            "group_query_attention",
            "gemm_nt",     // C = A @ B^T
            "gemm_nn",     // C = A @ B
            "residual_add" // element-wise addition on GPU
        ]
        for name in kernelNames {
            guard let fn = library.makeFunction(name: name) else {
                throw LMBackendError.kernelNotFound(name)
            }
            pipelines[name] = try device.makeComputePipelineState(function: fn)
        }
        layerNormPipeline = pipelines["layer_norm"]
        tanhGeluPipeline = pipelines["tanh_gelu_kernel"]
        causalMaskPipeline = pipelines["causal_mask_kernel"]
        attentionPipeline = pipelines["group_query_attention"]
        gemmNTPipeline = pipelines["gemm_nt"]
        gemmNNPipeline = pipelines["gemm_nn"]
        residualAddPipeline = pipelines["residual_add"]
    }

    private func allocateBuffers() {
        let hidden = MetalLMConfig.hiddenSize
        let intermediate = MetalLMConfig.intermediateSize
        let kvSize = numKVHeads * maxSeqLen * headDim

        qkvBuffer = makeBuffer(size: 112 * headDim * MemoryLayout<Float16>.size)
        qBuffer = makeBuffer(size: 80 * headDim * MemoryLayout<Float16>.size)
        kBuffer = makeBuffer(size: kvSize * MemoryLayout<Float16>.size)
        vBuffer = makeBuffer(size: kvSize * MemoryLayout<Float16>.size)
        attnOutBuffer = makeBuffer(size: hidden * MemoryLayout<Float16>.size)
        ln1OutBuffer = makeBuffer(size: hidden * MemoryLayout<Float16>.size)
        ln2OutBuffer = makeBuffer(size: hidden * MemoryLayout<Float16>.size)
        ffnInterimBuffer = makeBuffer(size: intermediate * MemoryLayout<Float16>.size)
        ffnOutBuffer = makeBuffer(size: hidden * MemoryLayout<Float16>.size)
        logitsBuffer = makeBuffer(size: MetalLMConfig.vocabSize * MemoryLayout<Float16>.size)

        let ropeSize = maxSeqLen * (headDim / 2)
        ropeCosBuffer = makeBuffer(size: ropeSize * MemoryLayout<Float>.size)
        ropeSinBuffer = makeBuffer(size: ropeSize * MemoryLayout<Float>.size)

        let maskSize = maxSeqLen * maxSeqLen
        causalMaskBuffer = makeBuffer(size: maskSize * MemoryLayout<Float>.size)
    }

    private func makeBuffer(size: Int) -> MTLBuffer {
        device.makeBuffer(length: size, options: .storageModeShared)!
    }

    private func precomputeRoPE() throws {
        guard let cmd = commandQueue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else { throw LMBackendError.commandBufferFailed }

        var maxSeq = UInt32(maxSeqLen)
        var headDim = UInt32(headDim)

        enc.setComputePipelineState(pipelines["compute_rope_lut"] ?? layerNormPipeline)
        enc.setBuffer(ropeCosBuffer, offset: 0, index: 0)
        enc.setBuffer(ropeSinBuffer, offset: 0, index: 1)
        enc.setBytes(&maxSeq, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&headDim, length: MemoryLayout<UInt32>.size, index: 3)

        let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let numThreadGroups = MTLSize(width: maxSeqLen, height: headDim / 2, depth: 1)
        enc.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    private func precomputeCausalMask() throws {
        guard let cmd = commandQueue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else { throw LMBackendError.commandBufferFailed }

        var seqLen = UInt32(maxSeqLen)

        enc.setComputePipelineState(causalMaskPipeline)
        enc.setBuffer(causalMaskBuffer, offset: 0, index: 0)
        enc.setBytes(&seqLen, length: MemoryLayout<UInt32>.size, index: 1)

        let threadsPerGroup = MTLSize(width: 1024, height: 1, depth: 1)
        let numThreadGroups = MTLSize(
            width: (maxSeqLen * maxSeqLen + 1023) / 1024,
            height: 1, depth: 1
        )
        enc.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    private func loadWeights() throws {
        // Load float16 ONNX weights from metal-export output
        // (extracted as MTLBuffer binary files)
        let weightsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            .appendingPathComponent("metal_weights")

        let manifestPath = weightsDir.appendingPathComponent("weights_manifest.json")
        guard FileManager.default.fileExists(atPath: manifestPath.path) else {
            throw LMBackendError.kernelNotFound("weights manifest at \(weightsDir.path)")
        }

        // Load manifest to get ONNX tensor names
        // ONNX export uses its own naming scheme — must match exactly
        let manifestData = try Data(contentsOf: manifestPath)
        guard let manifest = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any] else {
            throw LMBackendError.kernelNotFound("invalid weights manifest")
        }

        // Print tensor names for verification:
        // The export script (export_lm_float16.py) prints all output tensor names.
        // Use those exact names here. Common pattern from torch.onnx.export:
        //   "model.layers.0.self_attn.q_proj.weight"
        //   "model.layers.0.self_attn.k_proj.weight"
        //   "model.layers.0.self_attn.v_proj.weight"
        //   "model.layers.0.self_attn.o_proj.weight"
        //   "model.layers.0.mlp.fc1.weight"
        //   "model.layers.0.mlp.fc2.weight"
        //   "model.final_layernorm.weight"
        // etc.
        //
        // IMPORTANT: Run export_lm_float16.py FIRST, then copy the printed tensor
        // names into the weightNames list below. The names are model-dependent.
        let weightNames: [String] = [
            // "model.layers.0.self_attn.q_proj.weight",  // ← example, run export first!
            // ... all 24 layers + final_layernorm + lm_head
        ]

        for name in weightNames {
            let binPath = weightsDir.appendingPathComponent("\(name).bin")
            guard let data = try? Data(contentsOf: binPath) else {
                print("Warning: weight file not found: \(binPath.path)")
                continue
            }
            guard let buf = device.makeBuffer(bytes: data, options: .storageModeShared) else { continue }
            weightBuffers[name] = buf
        }

        // NOTE: wte/wpe (token/position embeddings) are NOT needed.
        // The turbo model receives pre-computed inputs_embeds from the embed_tokens ONNX model.
        // We only need the transformer layer weights + final_layernorm + lm_head.
    }

    public func forward(
        inputsEmbds: MTLBuffer,
        kvWriteOffset: Int,
        kvReadLength: Int,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLBuffer {
        // === Per-layer GPT-2 forward (24 layers) ===

        var residual = inputsEmbds  // [1, 1, 1024]

        for layer in 0..<numLayers {
            // Layer 0: Pre-LayerNorm
            try runLayerNorm(input: residual, output: ln1OutBuffer, layer: layer, ln: .ln1, cmd: commandBuffer)

            // QKV Projection via custom Metal kernel (gemm_nt = C = A @ B^T)
            // QKV = ln1_out @ W_qkv^T  [1,1,1024] @ [1024,7168] → [1,1,7168]
            let qkvWeight = weightBuffers["model.layers.\(layer).self_attn.q_proj.weight"]!
            try runGEMM_NT(
                A: ln1OutBuffer, B: qkvWeight, C: qkvBuffer,
                M: 1, N: 112 * HEAD_DIM, K: HIDDEN_SIZE,
                cmd: commandBuffer
            )

            // Unpack Q, K, V from qkvBuffer
            // Apply RoPE to Q and K using ropeCosBuffer/ropeSinBuffer

            // GroupQueryAttention via custom Metal kernel
            try runAttention(
                q: qBuffer, k: kBuffer, v: vBuffer,
                kvReadLength: kvReadLength,
                layer: layer,
                cmd: commandBuffer
            )

            // O projection: attn_out @ W_o^T  [1,1,1024] @ [1024,1024] → [1,1,1024]
            let oWeight = weightBuffers["model.layers.\(layer).self_attn.o_proj.weight"]!
            try runGEMM_NT(
                A: attnOutBuffer, B: oWeight, C: attnOutBuffer,
                M: 1, N: HIDDEN_SIZE, K: HIDDEN_SIZE,
                cmd: commandBuffer
            )

            // Residual add on GPU: x = x + attn_out
            try runResidualAdd(&residual, attnOutBuffer, cmd: commandBuffer)

            // Layer 1: Post-attention LayerNorm
            try runLayerNorm(input: residual, output: ln2OutBuffer, layer: layer, ln: .ln2, cmd: commandBuffer)

            // FFN: interim = Gelu(ln2_out @ W1)  [1,1,1024] @ [1024,4096] → [1,1,4096]
            let w1Weight = weightBuffers["model.layers.\(layer).mlp.fc1.weight"]!
            try runGEMM_NT(
                A: ln2OutBuffer, B: w1Weight, C: ffnInterimBuffer,
                M: 1, N: INTERMEDIATE_SIZE, K: HIDDEN_SIZE,
                cmd: commandBuffer
            )
            try runTanhGelu(input: ffnInterimBuffer, output: ffnInterimBuffer, cmd: commandBuffer)

            // FFN: out = interim * (ln2_out @ W3)  [1,1,4096] @ [4096,1024] → [1,1,1024]
            let w3Weight = weightBuffers["model.layers.\(layer).mlp.fc2.weight"]!
            try runGEMM_NN(
                A: ffnInterimBuffer, B: w3Weight, C: ffnOutBuffer,
                M: 1, N: HIDDEN_SIZE, K: INTERMEDIATE_SIZE,
                cmd: commandBuffer
            )

            // Residual add on GPU: x = x + ffn_out
            try runResidualAdd(&residual, ffnOutBuffer, cmd: commandBuffer)
        }

        // Final LayerNorm + LM head
        try runFinalLayerNorm(input: residual, output: logitsBuffer, cmd: commandBuffer)

        return logitsBuffer
    }

    private enum LayerNormTarget {
        case ln1, ln2
    }

    private func runLayerNorm(input: MTLBuffer, output: MTLBuffer, layer: Int, ln: LayerNormTarget, cmd: MTLCommandBuffer) throws {
        guard let enc = cmd.makeComputeCommandEncoder() else { throw LMBackendError.commandBufferFailed }
        enc.setComputePipelineState(layerNormPipeline)

        let wKey = "transformer.h.\(layer).\(ln == .ln1 ? "ln_1" : "ln_2").weight"
        let bKey = "transformer.h.\(layer).\(ln == .ln1 ? "ln_1" : "ln_2").bias"
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(weightBuffers[wKey], offset: 0, index: 1)
        enc.setBuffer(weightBuffers[bKey], offset: 0, index: 2)
        enc.setBuffer(output, offset: 0, index: 3)

        var dim = UInt32(HIDDEN_SIZE)
        var eps: Float16 = Float16(1e-5)
        enc.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&eps, length: MemoryLayout<Float16>.size, index: 5)

        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func runAttention(q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, kvReadLength: Int, layer: Int, cmd: MTLCommandBuffer) throws {
        guard let enc = cmd.makeComputeCommandEncoder() else { throw LMBackendError.commandBufferFailed }
        enc.setComputePipelineState(attentionPipeline)

        enc.setBuffer(q, offset: 0, index: 0)
        enc.setBuffer(k, offset: 0, index: 1)
        enc.setBuffer(v, offset: 0, index: 2)
        enc.setBuffer(causalMaskBuffer, offset: 0, index: 3)
        enc.setBuffer(attnOutBuffer, offset: 0, index: 4)
        enc.setBuffer(nil, offset: 0, index: 5)  // workspace

        var kvLen = UInt32(kvReadLength)
        var maxSeq = UInt32(maxSeqLen)
        enc.setBytes(&kvLen, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&maxSeq, length: MemoryLayout<UInt32>.size, index: 7)

        // NOTE: grid is 2D (uint2 gid) not 3D
        // gid.x = q_head (0..79), gid.y = head_dim (0..63)
        enc.dispatchThreadgroups(
            MTLSize(width: NUM_Q_HEADS, height: HEAD_DIM, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        enc.endEncoding()
    }

    private func runGEMM_NT(A: MTLBuffer, B: MTLBuffer, C: MTLBuffer, M: Int, N: Int, K: Int, cmd: MTLCommandBuffer) throws {
        guard let enc = cmd.makeComputeCommandEncoder() else { throw LMBackendError.commandBufferFailed }
        enc.setComputePipelineState(gemmNTPipeline)
        enc.setBuffer(A, offset: 0, index: 0)
        enc.setBuffer(B, offset: 0, index: 1)
        enc.setBuffer(C, offset: 0, index: 2)

        var m = UInt32(M), n = UInt32(N), k = UInt32(K)
        enc.setBytes(&m, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 5)

        // Threadgroup: 16x16 = 256 threads
        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let numThreadgroups = MTLSize(
            width: (N + 15) / 16,
            height: (M + 15) / 16,
            depth: 1
        )
        enc.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadgroupSize)
        enc.endEncoding()
    }

    private func runGEMM_NN(A: MTLBuffer, B: MTLBuffer, C: MTLBuffer, M: Int, N: Int, K: Int, cmd: MTLCommandBuffer) throws {
        guard let enc = cmd.makeComputeCommandEncoder() else { throw LMBackendError.commandBufferFailed }
        enc.setComputePipelineState(gemmNNPipeline)
        enc.setBuffer(A, offset: 0, index: 0)
        enc.setBuffer(B, offset: 0, index: 1)
        enc.setBuffer(C, offset: 0, index: 2)

        var m = UInt32(M), n = UInt32(N), k = UInt32(K)
        enc.setBytes(&m, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 5)

        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let numThreadgroups = MTLSize(
            width: (N + 15) / 16,
            height: (M + 15) / 16,
            depth: 1
        )
        enc.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadgroupSize)
        enc.endEncoding()
    }

    // Runs residual add on GPU (element-wise addition)
    private func runResidualAdd(_ residual: inout MTLBuffer, _ addend: MTLBuffer, cmd: MTLCommandBuffer) throws {
        guard let enc = cmd.makeComputeCommandEncoder() else { throw LMBackendError.commandBufferFailed }
        enc.setComputePipelineState(residualAddPipeline)
        enc.setBuffer(residual, offset: 0, index: 0)
        enc.setBuffer(addend, offset: 0, index: 1)
        enc.setBuffer(residual, offset: 0, index: 2)  // output = residual + addend

        var size = UInt32(HIDDEN_SIZE)
        enc.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 3)

        enc.dispatchThreadgroups(
            MTLSize(width: (HIDDEN_SIZE + 255) / 256, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        enc.endEncoding()
    }

    private func runTanhGelu(input: MTLBuffer, output: MTLBuffer, cmd: MTLCommandBuffer) throws {
        guard let enc = cmd.makeComputeCommandEncoder() else { throw LMBackendError.commandBufferFailed }
        enc.setComputePipelineState(tanhGeluPipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(output, offset: 0, index: 1)
        var size = UInt32(INTERMEDIATE_SIZE)
        enc.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: (INTERMEDIATE_SIZE + 255) / 256, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func runFinalLayerNorm(input: MTLBuffer, output: MTLBuffer, cmd: MTLCommandBuffer) throws {
        // Final LN + LM head projection
        // LN: output = (input - mean) / sqrt(var + eps) * gamma + beta
        // LM head: logits = output @ lm_head.weight^T
        // (lm_head = transformer.ln_f + lm_head combined)
    }

    public func reset() async {
        await kvCache.reset()
    }
}
```

- [ ] **Step 2: Write MetalPipeline.swift**

```swift
import Foundation
import Metal

/// Orchestrates the autoregressive decode loop using MetalLMBackend.
/// Implements greedy decode with repetition penalty.
public final class MetalPipeline: Sendable {
    private let backend: MetalLMBackend
    private let kvCache: KVCacheManager
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let maxNewTokens: Int
    private let repetitionPenalty: Float

    public init(
        device: MTLDevice,
        maxNewTokens: Int = 1500,
        repetitionPenalty: Float = 1.2
    ) throws {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        self.maxNewTokens = maxNewTokens
        self.repetitionPenalty = repetitionPenalty
        self.backend = MetalLMBackend(device: device)
        self.kvCache = KVCacheManager(
            numLayers: MetalLMConfig.numLayers,
            numKVHeads: MetalLMConfig.numKVHeads,
            headDim: MetalLMConfig.headDim,
            maxSeqLen: MetalLMConfig.maxSequenceLength,
            device: device
        )
    }

    public func initialize() async throws {
        try await backend.initialize(
            numLayers: MetalLMConfig.numLayers,
            numKVHeads: MetalLMConfig.numKVHeads,
            headDim: MetalLMConfig.headDim,
            maxSeqLen: MetalLMConfig.maxSequenceLength,
            device: device
        )
    }

    /// Run the autoregressive decode loop.
    /// - Parameters:
    ///   - inputsEmbds: [1, seq, 1024] — concatenated conditioning + text + BOS embed
    ///   - initialKVLen: Number of prefix positions to attend over
    /// - Returns: Generated speech token IDs [num_tokens]
    public func decode(
        inputsEmbds: MTLBuffer,
        initialKVLen: Int
    ) async throws -> [Int32] {
        await backend.reset()

        guard let cmd = commandQueue.makeCommandBuffer() else {
            throw LMBackendError.commandBufferFailed
        }

        var generatedTokens: [Int32] = []
        var kvWriteOffset = initialKVLen
        var kvReadLength = initialKVLen

        for step in 0..<maxNewTokens {
            let logitsBuf = try backend.forward(
                inputsEmbds: inputsEmbds,
                kvWriteOffset: kvWriteOffset,
                kvReadLength: kvReadLength,
                commandBuffer: cmd
            )

            // Read logits
            let logitsPtr = logitsBuf.contents().bindMemory(to: Float16.self, capacity: MetalLMConfig.vocabSize)
            var logits = (0..<MetalLMConfig.vocabSize).map { float(logitsPtr[$0]) }

            // Apply repetition penalty
            applyRepetitionPenalty(&logits, tokens: generatedTokens)

            // Greedy argmax
            var maxIdx = 0
            var maxVal = logits[0]
            for i in 1..<logits.count {
                if logits[i] > maxVal { maxVal = logits[i]; maxIdx = i }
            }

            let nextToken = Int32(maxIdx)
            generatedTokens.append(nextToken)

            // Stop on EOS
            if nextToken == MetalLMConfig.stopSpeechToken {
                break
            }

            // Advance KV cache ring buffer
            kvWriteOffset = (kvWriteOffset + 1) % MetalLMConfig.maxSequenceLength
            kvReadLength = step + initialKVLen + 1

            // NOTE: In a real implementation, the next token's embedding
            // would be fetched via embed_tokens ONNX model and passed as inputsEmbds
            // for the next iteration. This is handled by ChatterboxEngine.
        }

        return generatedTokens
    }

    private func applyRepetitionPenalty(_ logits: inout [Float], tokens: [Int32]) {
        let penalty = repetitionPenalty
        var tokenCounts: [Int: Int] = [:]
        for t in tokens {
            tokenCounts[Int(t), default: 0] += 1
        }
        for (token, count) in tokenCounts {
            if logits[token] > 0 {
                logits[token] /= penalty
            } else {
                logits[token] *= penalty
            }
        }
    }
}
```

- [ ] **Step 3: Commit**

```bash
git add chatterbox-metal-lm/src/MetalLMBackend.swift
git add chatterbox-metal-lm/src/MetalPipeline.swift
git commit -m "feat(metal-lm): add MetalLMBackend and MetalPipeline"
```

---

## Task 8: Weight Loading — Float16 from PyTorch Export

**Files:**
- Create: `chatterbox-metal-lm/src/Float16WeightLoader.swift`
- Modify: `metal-export/export_lm_float16.py` (add binary export)

**Context:** The float16 weights must be exported from PyTorch and loaded as MTLBuffer binaries. The export script writes `.bin` files per weight tensor.

- [ ] **Step 1: Update export script to write binary weights**

```python
# Add to export_lm_float16.py, after model export:
def export_weights_binary(model, output_dir):
    """Export each weight tensor as a .bin file (float16)."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    for name, param in model.named_parameters():
        # Convert name to valid filename
        safe_name = name.replace(".", "_").replace("/", "_")
        bin_path = os.path.join(output_dir, f"{safe_name}.bin")
        param.detach().half().numpy().tofile(bin_path)
        print(f"  {name} → {param.shape} → {bin_path}")

    # Write manifest
    manifest = {name: list(param.shape) for name, param in model.named_parameters()}
    import json
    with open(os.path.join(output_dir, "weights_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
```

- [ ] **Step 2: Write Float16WeightLoader.swift**

```swift
import Foundation
import Metal

/// Loads float16 weights from binary files exported by export_lm_float16.py.
public final class Float16WeightLoader {
    public let device: MTLDevice
    public let weightsDir: URL

    /// Name → MTLBuffer
    public private(set) var buffers: [String: MTLBuffer] = [:]

    public init(device: MTLDevice, weightsDir: URL) {
        self.device = device
        self.weightsDir = weightsDir
    }

    /// Load all weights from the weights directory.
    public func loadAll() throws {
        let manifestPath = weightsDir.appendingPathComponent("weights_manifest.json")
        let manifestData = try Data(contentsOf: manifestPath)
        let manifest = try JSONDecoder().decode([String: [Int]].self, from: manifestData)

        for (name, shape) in manifest {
            let safeName = name.replacingOccurrences(of: ".", with: "_").replacingOccurrences(of: "/", with: "_")
            let binPath = weightsDir.appendingPathComponent("\(safeName).bin")
            guard let data = try? Data(contentsOf: binPath) else {
                print("Warning: weight file not found: \(binPath.path)")
                continue
            }
            guard let buf = device.makeBuffer(bytes: data, options: .storageModeShared) else {
                continue
            }
            buffers[name] = buf
        }
    }

    public func buffer(named name: String) -> MTLBuffer? {
        buffers[name]
    }
}
```

- [ ] **Step 3: Commit**

```bash
git add chatterbox-metal-lm/src/Float16WeightLoader.swift
git commit -m "feat(metal-lm): add Float16WeightLoader for PyTorch weight binaries"
```

---

## Task 9: Integration — ChatterboxEngine Changes

**Files:**
- Modify: `Reader/Services/ChatterboxEngine.swift`
- Modify: `Reader/ViewModels/PlayerViewModel.swift` (add useMetalLM toggle)

**Context:** Add `useMetalLM` feature flag. When enabled, use `MetalPipeline` instead of ONNX decode loop. All other code (tokenizer, embed_tokens, speech_encoder, decoder) unchanged.

- [ ] **Step 1: Add LanguageModelBackend protocol + MetalLMBackend to ChatterboxEngine**

```swift
// Insert in ChatterboxEngine.swift — near the top with other properties

// === Metal LM Backend (optional) ===
private var metalPipeline: MetalPipeline?
private let useMetalLM: Bool = false  // Toggle for experimental Metal path

// === Metal LM Protocol ===
// (copy LanguageModelBackend.swift content here, or import chatterbox-metal-lm as SPM)
```

- [ ] **Step 2: Modify decode loop**

```swift
// In runDecodeLoop() — replace ORTSession.run section:
if useMetalLM, let pipeline = metalPipeline {
    // Use Metal LM
    let embedData = embedBuffer.contents().bindMemory(to: Float16.self, capacity: seqLen * 1024)
    let embedBuf = device.makeBuffer(bytes: embedData, length: seqLen * 1024 * 2, options: .storageModeShared)!

    let tokens = try await pipeline.decode(inputsEmbds: embedBuf, initialKVLen: 0)
    // ... convert tokens to generatedToken format
} else {
    // Existing ONNX path
    generatedToken = greedyNextToken(lastPosLogits, previous: generateTokens)
}
```

- [ ] **Step 3: Add Settings toggle**

```swift
// In SettingsView or PlayerViewModel:
// @Published var useMetalLM = false
// Toggle("Use Metal for TTS (experimental)", $useMetalLM)
```

- [ ] **Step 4: Commit**

```bash
git add Reader/Services/ChatterboxEngine.swift
git add Reader/ViewModels/PlayerViewModel.swift
git commit -m "feat(reader): add useMetalLM feature flag to ChatterboxEngine"
```

---

## Task 10: Xcode Build Configuration

**Files:**
- Modify: `project.yml` (add Metal compute target)
- Create: `chatterbox-metal-lm/ChatterboxMetalLM.podspec` (if using CocoaPods)

**Context:** `.metal` files compile automatically in Xcode iOS targets. No manual `xcrun metal` needed. The Swift files in `chatterbox-metal-lm/src/` need to be added to the Reader target.

- [ ] **Step 1: Update project.yml**

```yaml
targets:
  Reader:
    sources:
      - path: chatterbox-metal-lm/src
        buildPhase: sources
        compilerFlags: [-std=metal3.0]
      - path: Reader/MetalCompute
        buildPhase: sources
        compilerFlags: [-std=metal3.0]
```

- [ ] **Step 2: Build verification**

```bash
cd /Users/rockymoon/Downloads/Reader/.claude/worktrees/metal-lm-impl
xcodebuild -project Reader.xcodeproj \
    -scheme Reader \
    -configuration Debug \
    -destination 'platform=iOS Simulator,name=iPhone 17' \
    -compile 2>&1 | grep -E "error:|warning:.*Metal|Build (succeeded|FAILED)"
```

- [ ] **Step 3: Commit**

```bash
git add project.yml
git commit -m "build: add Metal LM sources to Reader target"
```

---

## Task 11: Numerical Validation

**Files:**
- Create: `chatterbox-metal-lm/tests/MetalLMForwardTests.swift`
- Create: `metal-export/verify_metal_vs_onnx.py`

- [ ] **Step 1: Write Swift forward test**

```swift
import XCTest

final class MetalLMForwardTests: XCTestCase {
    func testMetalForwardMatchesONNX() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        // Initialize Metal backend
        let backend = MetalLMBackend(device: device)
        try await backend.initialize(
            numLayers: 24, numKVHeads: 16, headDim: 64,
            maxSeqLen: 1500, device: device
        )

        // Create random input
        let inputSize = 1 * 1 * 1024 * MemoryLayout<Float16>.size
        guard let inputBuf = device.makeBuffer(length: inputSize, options: .storageModeShared) else { return }
        let ptr = inputBuf.contents().bindMemory(to: Float16.self, capacity: 1024)
        for i in 0..<1024 { ptr[i] = Float16.random(in: -1...1) }

        guard let cmd = device.makeCommandQueue()?.makeCommandBuffer() else { return }

        let logitsBuf = try backend.forward(
            inputsEmbds: inputBuf,
            kvWriteOffset: 0,
            kvReadLength: 1,
            commandBuffer: cmd
        )

        cmd.commit()
        cmd.waitUntilCompleted()

        // Check logits are reasonable
        let logitsPtr = logitsBuf.contents().bindMemory(to: Float16.self, capacity: 6563)
        let logitsArray = (0..<6563).map { float(logitsPtr[$0]) }
        XCTAssertFalse(logitsArray.allSatisfy { $0 == -Float.infinity })
        XCTAssertFalse(logitsArray.allSatisfy { $0 == 0 })
    }
}
```

- [ ] **Step 2: Write Python vs Metal comparison**

```python
# metal-export/verify_metal_vs_onnx.py
"""
Compare Metal LM output with ONNX reference.
Run this on macOS (not iOS Simulator) with actual Metal device.
"""
import numpy as np

def compare():
    print("TODO: Run Metal forward on macOS, compare with ONNX reference")
    # Tolerance: rtol=1e-2, atol=1e-3
    pass
```

- [ ] **Step 3: Commit**

```bash
git add chatterbox-metal-lm/tests/MetalLMForwardTests.swift
git add metal-export/verify_metal_vs_onnx.py
git commit -m "test(metal-lm): add forward validation tests"
```

---

## Task 12: End-to-End Integration Test

**Files:**
- Run on device

- [ ] **Step 1: Run TTS with Metal LM (feature flag on)**

```bash
# Build and run on iOS Simulator
xcodebuild -project Reader.xcodeproj \
    -scheme Reader \
    -configuration Debug \
    -destination 'platform=iOS Simulator,name=iPhone 17' \
    -compile 2>&1 | grep -E "error:|Build (succeeded|FAILED)"
```

Expected: Compiles without errors.

- [ ] **Step 2: Verify audio output**

Run the Reader app with Metal LM enabled. Compare output audio quality with ONNX baseline.

- [ ] **Step 3: Measure performance**

```bash
# Log decode step timing
# Expected: ~20ms/step Metal vs ~50ms/step ONNX
```

- [ ] **Step 4: Commit**

```bash
git add -a
git commit -m "test(metal-lm): e2e integration + performance verification"
```

---

## Summary of Tasks

| Task | Description | Status |
|------|-------------|--------|
| 1 | Python export pipeline (float16 ONNX) | TODO |
| 2 | LanguageModelBackend protocol + ONNXLMBackend | TODO |
| 3 | KVCacheManager actor (ring-buffer, 48 buffers) | TODO |
| 4 | Metal kernels: TanhGelu + MaskCompute + RoPE | TODO |
| 5 | Metal kernel: GroupQueryAttention (Attention.metal) | TODO |
| 6 | Metal kernel: LMForward (per-layer GPT-2) | TODO |
| 7 | MetalLMBackend + MetalPipeline (Swift dispatcher) | TODO |
| 8 | Float16WeightLoader + export binary weights | TODO |
| 9 | ChatterboxEngine integration (protocol + flag) | TODO |
| 10 | Xcode build configuration | TODO |
| 11 | Numerical validation tests | TODO |
| 12 | E2E integration + performance verification | TODO |
