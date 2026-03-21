# Chatterbox Turbo Metal: Design Specification

## Overview

**Status: IN DESIGN**

Replace the ONNX Runtime language model with a native Metal compute shader pipeline targeting the Apple Neural Engine (ANE) and GPU. The ONNX Runtime implementation remains as a user-toggleable fallback.

**Repository:** `spitfire02020-collab/reader-metal` (forked from `reader`)

**Source model:** `ResembleAI/chatterbox-turbo` (PyTorch, float16 export)

**Target:** iOS 17+ (ANE + GPU)

---

## Scope

**In scope (this design):**
- Language model (`t3` — GPT-2, 24 layers, 350M params)
- `LanguageModelBackend` protocol + `MetalLMBackend` implementation
- `KVCacheManager` — Swift-owned KV cache buffers
- Metal compute kernels: LM forward, TanhGelu activation, GQA attention, causal mask
- Python export pipeline: float16 ONNX from PyTorch source

**Out of scope (deferred):**
- Speech encoder (`speech_encoder_q4f16`) — stays on ONNX Runtime
- Embed tokens (`embed_tokens_q4f16`) — stays on ONNX Runtime (or CoreML option)
- Conditional decoder (`conditional_decoder_q4f16`) — stays on ONNX Runtime

---

## Architecture

```
Reader iOS App
├── ONNX Runtime (fallback, unchanged)
│   ├── speech_encoder_q4f16
│   ├── embed_tokens_q4f16
│   ├── language_model_q4f16  ← will be replaced
│   └── conditional_decoder_q4f16
│
└── Metal LM Pipeline (primary)
    ├── MetalCompute/
    │   ├── LMForward.metal         ← 24-layer GPT-2 forward
    │   ├── TanhGelu.metal          ← activation (Gelu approximate="tanh")
    │   ├── MaskCompute.metal       ← causal mask generation
    │   └── Attention.metal         ← GroupQueryAttention (16 KV, 80 Q)
    ├── Swift/
    │   ├── LanguageModelBackend.swift     ← protocol definition
    │   ├── ONNXLMBackend.swift            ← existing ORTSession wrapper
    │   ├── MetalLMBackend.swift           ← Metal kernel dispatcher
    │   ├── KVCacheManager.swift            ← Swift-owns 48 KV buffers
    │   └── MetalPipeline.swift             ← decode loop orchestration
    └── metal-export/
        ├── requirements.txt
        ├── export_lm_float16.py      ← patch aten::diff + export float16 ONNX
        ├── verify_output.py           ← numerical validation
        └── README.md                  ← step-by-step export guide
```

---

## LanguageModelBackend Protocol

```swift
/// Backend-agnostic interface for the language model forward pass.
protocol LanguageModelBackend: Sendable {
    /// Allocate KV cache buffers. Called once at model load.
    func initialize(
        numLayers: Int,
        numKVHeads: Int,
        headDim: Int,
        maxSeqLen: Int,
        device: MTLDevice
    ) async throws

    /// Autoregressive decode step.
    /// - inputsEmbds: [1, 1, hidden=1024] float16 — single token embedding
    /// - kvWriteOffset: ring-buffer write position for new KV entries
    /// - kvReadLength: how many positions to attend over (grows per step)
    /// Returns logits as MTLBuffer: shape [1, 1, vocab=6563], dtype float16.
    /// The caller reads logits from the buffer by sampling index argmax.
    func forward(
        inputsEmbds: MTLBuffer,
        kvWriteOffset: Int,
        kvReadLength: Int,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLBuffer

    /// Zero KV cache for new synthesis session.
    func reset() async
}
```

### ONNXLMBackend

Wraps the existing `ORTSession` decode loop from `ChatterboxEngine.swift`. No changes to existing ONNX code.

### MetalLMBackend

Implements `LanguageModelBackend` using Metal compute kernels. Swift owns and allocates all buffers; Metal performs all computation.

---

## KV Cache Manager

```swift
struct KVCacheBufferSet: Sendable {
    let keyBuffer: MTLBuffer   // [1, numKVHeads=16, maxSeqLen=1500, headDim=64] float16
    let valBuffer: MTLBuffer   // [1, numKVHeads=16, maxSeqLen=1500, headDim=64] float16
}

actor KVCacheManager {
    let numLayers: Int          // 24
    let numKVHeads: Int         // 16
    let headDim: Int            // 64
    let maxSeqLen: Int          // 1500

    // 24 layers × 2 (key + value) = 48 buffers total
    private var layerBuffers: [KVCacheBufferSet]

    // Ring-buffer write head — advances each decode step
    private var writeHead: Int = 0

    func buffer(for layer: Int, isKey: Bool) -> MTLBuffer {
        isKey ? layerBuffers[layer].keyBuffer : layerBuffers[layer].valBuffer
    }

    func advance() { writeHead = (writeHead + 1) % maxSeqLen }
    func reset() { writeHead = 0; zeroAllBuffers() }
}
```

- **Buffer sizes:** Each KV buffer is `1 × 16 × 1500 × 64 × 2 bytes = 3,072,000 bytes ≈ 3MB`. Total KV cache: `24 × 2 × 3MB ≈ 147MB`.
- **Ring-buffer:** `writeHead` advances modulo `maxSeqLen`. Metal kernel writes new KV at `writeHead`, reads across `kvReadLength`.
- **Allocation:** All 48 buffers allocated once in `initialize()`, never reallocated during synthesis.
- **Reset:** Zero-filled between synthesis sessions.

---

## Metal Kernel Design

### LMForward.kernel

One threadgroup per layer. `grid = [24]` (one dispatch per layer).

**Dispatch:** `threadgroups = 24`, `threads_per_threadgroup = 256` (work per layer: matmuls + attention + FFN).

**Per-layer computation:**

1. **QKV Projection** — fused matmul:
   `W_qkv ∈ ℝ^(1024×(80+16+16)×64)` → `input[1,1,1024] → output[1,1,112×64]`
   Single matmul producing Q (80 heads), K (16 heads), V (16 heads) concatenated.

2. **Unpack Q, K, V** — split the 112×64 output into three tensors.

3. **RoPE** (Rotary Position Embeddings):
   - Precompute `cos/sin` LUT once per session (constant MTLBuffer)
   - Apply: `x_rot[::2] = x[::2] × cos − x[1::2] × sin`
   - `sin` precomputed from `θ = 10000` (GPT-2 default)
   - Applied in-place on Q and K before attention

4. **GroupQueryAttention** — K repeated 5× across Q head groups:
   - For each Q head `i`: maps to KV head `i / 5`
   - Score matmul: `Q[1,80,1,64] × K[1,16,seq,64]^T → scores[1,80,1,seq]`
   - Scale: `1 / √64 = 0.125`
   - Causal mask: add `−∞` for future positions (from `MaskCompute.kernel`)
   - Softmax → weighted sum with V → output[1,80,1,64]

5. **KV Ring-buffer Write** — at `kvWriteOffset` position in K/V buffers

6. **MLP** — two matmuls + TanhGelu activation:
   - `h = gelu(W1 × x + b1)`, `out = W3 × h + b3`
   - `W1 ∈ ℝ^(1024×4096)`, `W3 ∈ ℝ^(4096×1024)`

7. **Residual Add:** `x = x + attn_out + mlp_out`

### TanhGelu Activation

Matches `Gelu approximate="tanh"` from ONNX exactly:

```
gelu_tanh(x) = 0.5 × x × (1 + tanh(0.797885x + 0.044715x³))
             = 0.5 × x × (1 + tanh(0.797885x + 0.035677x³))
```

Where `0.035677 = sqrt(2/π) × 0.044715`.

Implemented as `float32` precision in kernel (converted from float16 input), converted back to float16 output.

### Causal Mask Kernel

Replaces the `aten::diff` operator that blocked PyTorch→ONNX export. Generates causal (lower-triangular) mask on-the-fly:

```metal
kernel void causal_mask_kernel(
    device float* mask      [[buffer(0)]],  // [seqLen, seqLen] output
    constant int& seq_len   [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    int row = gid / seq_len;
    int col = gid % seq_len;
    mask[gid] = (col <= row) ? 0.0f : -1e9f;
}
```

Dispatch: `threadgroups = ceil(seqLen × seqLen / 1024)`, `threads_per_threadgroup = 1024`.
For seqLen=1500: `ceil(2,250,000 / 1024) = 2199` threadgroups. The final threadgroup may have inactive threads (`gid >= seqLen × seqLen`) — the kernel guards out-of-bounds writes with `if (gid >= seqLen * seqLen) return;`.

### Attention Softmax

Standard float32 softmax over seq dimension:

```metal
// Find max for numerical stability
float32 max_val = -INFINITY;
for (int j = 0; j < seq_len; j++) max_val = max(max_val, scores[i][j]);
// Softmax
float32 sum = 0.0f;
for (int j = 0; j < seq_len; j++) {
    float32 e = exp(scores[i][j] - max_val);
    sum += e;
    exp_scores[i][j] = e;
}
for (int j = 0; j < seq_len; j++) scores[i][j] = exp_scores[i][j] / sum;
```

---

## Model Configuration

| Parameter | Value |
|-----------|-------|
| num_hidden_layers | 24 |
| hidden_size | 1024 |
| num_attention_heads (Q) | 80 |
| num_kv_heads | 16 |
| head_dim | 64 |
| intermediate_size | 4096 |
| max_position_embeddings | 1500 |
| vocab_size | 6563 |
| activation | Gelu (tanh approximation) |
| dtype | float16 |

**Weight import from ONNX:**
- All weight matrices imported as `MTLBuffer` (float16) at initialization
- No runtime weight conversion — Metal reads directly from buffer

---

## ChatterboxEngine Integration

### Changes to existing code (minimal):

```swift
// ChatterboxEngine.swift — additions only
protocol LanguageModelBackend: Sendable {
    func initialize(numLayers: Int, numKVHeads: Int, headDim: Int,
                    maxSeqLen: Int, device: MTLDevice) async throws
    func forward(inputsEmbds: MTLBuffer, kvWriteOffset: Int, kvReadLength: Int,
                 commandBuffer: MTLCommandBuffer) throws -> MTLBuffer
    func reset() async
}

// In loadModels():
if useMetal {
    lmBackend = MetalLMBackend()
    try await lmBackend.initialize(
        numLayers: 24, numKVHeads: 16, headDim: 64,
        maxSeqLen: config.maxNewTokens,
        device: MTLCreateSystemDefaultDevice()!
    )
} else {
    lmBackend = ONNXLMBackend(languageModelSession)
}

// In runDecodeLoop(): replace ORTSession.run with lmBackend.forward call
// All other code (tokenizer, embed_tokens, speech_encoder, decoder) unchanged
```

### Feature Flag

```swift
// In ChatterboxConfig:
var useMetalLM: Bool = false  // default to ONNX

// In SettingsView: toggle for "Use Metal for TTS (experimental)"
```

---

## Export Pipeline

### metal-export/export_lm_float16.py

1. **Install dependencies:** `transformers`, `torch`, `onnx`
2. **Patch `aten::diff`:**
   - File: `transformers/src/transformers/masking_utils.py`
   - Change: Replace `torch.diff(position_ids, prepend=..., dim=-1)` with an equivalent using `atleast_1d` + `cat` (traceable)
3. **Load model:** `ChatterboxTurboTTS.from_pretrained()`, extract `t3` (GPT-2)
4. **Export:**
   ```python
   torch.onnx.export(
       t3.tfmr,
       (inputs_embeds, attention_mask, position_ids, past_key_values),
       "language_model_float16.onnx",
       input_names=[...], output_names=[...],
       opset_version=17,
       dynamic_axes={...}
   )
   ```
5. **Verify:** Compare ONNX output with PyTorch reference, `rtol=1e-2, atol=1e-3`

### metal-export/verify_output.py

- Run ONNX model with known input → reference logits
- Run Metal model with same input → test logits
- Compare: `rtol=1e-2, atol=1e-3` (relaxed for float16)

### Requirements

```
torch>=2.0
transformers
onnx
numpy
```

---

## Error Handling

| Failure Mode | Handling |
|-------------|----------|
| Metal device not available | Fall back to ONNX Runtime |
| Kernel compilation fails | Fall back to ONNX Runtime + log error |
| MTLBuffer allocation fails (OOM) | Fall back to ONNX Runtime |
| Kernel hangs (infinite loop / uninitialized pointer) | 10-second watchdog timer on `commandBuffer`; if not completed → kill buffer, fall back to ONNX Runtime + log error |
| Numerical mismatch vs ONNX | Log warning; continue with Metal output |
| Export script fails | Print clear error + patch instructions |

---

## Testing & Verification

### Unit tests
- `test_kvcache_manager.swift` — ring-buffer advance, reset, buffer access
- `test_tanh_gelu.swift` — verify vs reference GELU output
- `test_gqa_kernel.swift` — single-head attention vs PyTorch reference

### Integration tests
- `test_metal_decode_loop.swift` — full autoregressive decode, compare tokens with ONNX
- `test_pipeline_e2e.swift` — text → tokens → LM → audio, compare with ONNX pipeline

### Numerical tolerance
- Attention logits: `rtol=1e-2, atol=1e-3`
- Final waveform: `rtol=1e-2, atol=1e-3`

---

## Files to Create/Modify

### New files (Metal path)
```
Reader/
├── MetalCompute/
│   ├── LMForward.metal
│   ├── TanhGelu.metal
│   ├── MaskCompute.metal
│   ├── Attention.metal
│   └── module.modulemap
├── Swift/
│   ├── LanguageModelBackend.swift      ← protocol
│   ├── ONNXLMBackend.swift             ← wrapper
│   ├── MetalLMBackend.swift            ← Metal impl
│   ├── KVCacheManager.swift
│   └── MetalPipeline.swift
└── Reader/
    └── Services/
        └── ChatterboxEngine.swift      ← minimal changes (add protocol + flag)

metal-export/
├── requirements.txt
├── export_lm_float16.py
├── verify_output.py
└── README.md
```

### Modified files
- `project.yml` — add Metal backend target, MTLCompileSettings
- `ChatterboxEngine.swift` — add `LanguageModelBackend` protocol + `useMetalLM` flag
- `ChatterboxConfig` — add `useMetalLM: Bool`

---

## Performance Targets

| Metric | ONNX Runtime | Metal Target |
|--------|-------------|--------------|
| Decode step latency | ~50ms/step | ~20ms/step |
| Total synthesis (1500 tokens) | ~75s | ~30s |
| Memory (LM buffers) | ~300MB | ~200MB |
| ANE utilization | N/A | >80% |
| GPU utilization | <30% | >60% |

---

## Out of Scope (Future Work)

- Speech encoder Metal implementation (defer to later; use ONNX Runtime)
- Embed tokens Metal implementation (defer to later; use ONNX Runtime)
- Full model Metal pipeline (LM + encoder + decoder all in Metal)
- Float16 model export from ResembleAI (if they provide directly)

---

## Dependencies

- Xcode 15+
- iOS 17+ deployment target
- Metal API validation enabled in Debug
- Python 3.11+ for export pipeline
- `onnxruntime` Python package for verification
- `transformers` Python package (with patch applied)
