# chatterbox-turbo-coreml: Design Specification

## Overview

**Status: ON HOLD — Critical blockers identified (see §Blockers)**

Convert the 4 ONNX models from `ResembleAI/chatterbox-turbo-ONNX` to Apple's CoreML format. The resulting `.mlmodel` files would replace ONNX Runtime in the Reader iOS app, enabling native Apple Silicon GPU acceleration and eliminating the ONNX Runtime dependency for inference.

**Repository:** New standalone repo (TBD — `chatterbox-turbo-coreml`)

**Source models:** `ResembleAI/chatterbox-turbo-ONNX` on HuggingFace — 4 quantized ONNX models (Q4F16)

**Target:** iOS 18+ (required for stateful CoreML KV cache support)

---

## Blockers (2026-03-21)

### Blocker 1: ONNX Converter Removed from coremltools

coremltools v6+ **removed the ONNX converter entirely**. No version of coremltools currently supports direct ONNX→CoreML conversion.

- coremltools 9.0 (latest): only PyTorch, TensorFlow, sklearn, xgboost, libsvm converters
- coremltools 7.1: same — no ONNX converter
- coremltools 5.2: last version with ONNX converter, but NumPy 2.x incompatibility prevents use

**Impact:** Direct ONNX→CoreML path is closed permanently in current coremltools.

### Blocker 2: PyTorch GPT2Model Blocked by `aten::diff`

The PyTorch GPT2Model uses `torch.diff` internally (from `transformers.masking_utils`) for causal mask computation. Neither ONNX nor CoreML converters support this operator:

- ONNX export: `Exporting the operator 'aten::diff' to ONNX opset version 17 is not supported`
- coremltools 7.1 conversion: `PyTorch convert function for op 'diff' not implemented`

This cannot be worked around without modifying the transformers library source.

### Blocker 3: Architecture Mismatch — MHA vs GroupQueryAttention

The ONNX `language_model_q4f16.onnx` uses `GroupQueryAttention` (GQA) with 16 KV heads and MatMulNBits quantization (145 quantized matmul ops). The PyTorch `ChatterboxTurboTTS` uses standard HuggingFace GPT2Model with regular Multi-Head Attention.

These are **mathematically different attention mechanisms** — they produce equivalent results but with different intermediate values. Even if conversion were possible, outputs would not match the ONNX reference, producing different speech.

### Partial Successes

The following components CAN be converted:

| Component | Status | Notes |
|-----------|--------|-------|
| `text_emb` (Embedding 50276×1024) | ✅ WORKS | `torch.jit.trace` + coremltools 7.1 |
| `speech_emb` (Embedding 6563×1024) | ✅ WORKS | `torch.jit.trace` + coremltools 7.1 |
| GPT2Block (single layer) | ✅ WORKS | Traces with explicit KV tuple; returns hidden_states only (KV updated in-place, not returned) |
| S3Gen tokenizer | ❌ BLOCKED | Uses `complex64` STFT — not supported in CoreML |
| S3Gen flow (decoder) | ❌ BLOCKED | Uses Python `timedelta` — not traceable |
| VoiceEncoder | ❌ BLOCKED | `embeds_from_wavs` takes `(Tensor, int)` tuple — not traceable |

---

## Revised Approach

Given the blockers above, the following paths are under consideration:

### Option A: Partial Conversion (embed_tokens only)

Convert only the `text_emb` and `speech_emb` embedding tables via `torch.jit.trace` + coremltools. These are the simplest components — just lookup tables.

**Pros:** Guaranteed to work, minimal complexity
**Cons:** Only replaces 2 of 4 ONNX models; LM and decoder still need ONNX Runtime

**Implementation:** Trivially exportable — single `torch.nn.Embedding` layer.

### Option B: Hybrid ONNX + CoreML

Use ONNX Runtime for LM and decoder (the heavy inference components), CoreML for embedding tables. This reduces ONNX Runtime overhead for the embedding lookup but leaves the heavy inference on ORT.

**Pros:** Addresses the primary CPU bottleneck (embedding) while keeping proven ONNX pipeline
**Cons:** Doesn't eliminate ONNX Runtime dependency; hybrid complexity

### Option C: Metal Compute Shaders

Replace ONNX Runtime entirely with custom Metal compute shaders for the speech pipeline. This is the most complex but eliminates all third-party inference dependencies.

**Pros:** Full control, no ORT dependency, optimized for Apple Silicon
**Cons:** Significant engineering effort; would need to reimplement GPT-2 attention, GQA, quantization in Metal

### Recommended: Option A (Partial Conversion)

Start with the embedding tables — they are proven to convert successfully and will exercise the full conversion pipeline without blocker risk.

---

## Repository Layout (Option A focus)

```
chatterbox-turbo-coreml/
├── scripts/
│   ├── export_embed_tokens.py       # Text + speech embeddings → .mlmodel
│   ├── export_embed_tokens_test.py  # Verify against ONNX reference
│   └── verify_conversion.py          # Numerical comparison helper
├── tests/
│   ├── test_embed_tokens.py
│   └── test_pipeline_hybrid.py       # Hybrid ONNX + CoreML pipeline
├── requirements.txt
├── README.md
└── models/                          # Output .mlmodel files
```

---

## Model Components & Conversion Status

### 1. Embed Tokens — CONVERTABLE ✅

**Status:** Verified working — both `text_emb` and `speech_emb` convert successfully with `torch.jit.trace` + coremltools 7.1.

**PyTorch source:** `ChatterboxTurboTTS.t3.text_emb` and `ChatterboxTurboTTS.t3.speech_emb`

**Conversion:**
```python
text_emb = t3.text_emb  # Embedding(50276, 1024)
speech_emb = t3.speech_emb  # Embedding(6563, 1024)

class EmbedWrapper(torch.nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.emb = emb
    def forward(self, ids):
        return self.emb(ids)

# Traced → converted
mlmodel = ct.convert(traced, source='pytorch',
    inputs=[ct.TensorType(shape=(1, ct.RangeDim(1, 2048)))])
```

**Output shapes:**
- `text_emb`: input_ids(int64) → inputs_embeds(float32) [B, seq, 1024]
- `speech_emb`: speech_ids(int64) → inputs_embeds(float32) [B, seq, 1024]

**Note:** The two embedding tables are separate modules. Both must be converted and used in the synthesis pipeline.

### 2. Speech Encoder — NOT TESTED YET

**ONNX source:** `speech_encoder_q4f16.onnx`

**Expected issues:** Uses complex-valued STFT operations. CoreML does not support complex dtypes (`complex64`, `complex128`).

**Status:** Not yet tested with `torch.jit.trace`.

### 3. Language Model — BLOCKED ❌

**ONNX source:** `language_model_q4f16.onnx`

**Blockers:**
1. coremltools has no ONNX converter (removed in v6)
2. PyTorch GPT2Model blocked by `aten::diff` from `transformers.masking_utils`
3. Architecture mismatch: PyTorch MHA ≠ ONNX GroupQueryAttention

**Status:** Not convertible with current tools.

### 4. Conditional Decoder — BLOCKED ❌

**ONNX source:** `conditional_decoder_q4f16.onnx`

**Blockers:**
1. coremltools has no ONNX converter
2. PyTorch S3Gen flow blocked by Python `timedelta` usage (from `tqdm`)

**Status:** Not convertible with current tools.

---

## Error Handling

| Failure Mode | Handling |
|-------------|----------|
| ONNX file not found / download fails | Exit with clear error + instructions to run `download_models.py` |
| coremltools conversion fails | Catch exception, print layer context, offer `--verbose` debug flag |
| Output shape mismatch | Validate shapes post-conversion, fail if unexpected |
| Model dtype mismatch | Explicitly specify dtypes in conversion inputs |
| iOS version too old | Document iOS 18+ requirement in README |

Each script:
- Checks prerequisites before starting
- Provides `--verbose` flag for detailed debugging
- Validates output before marking success

---

## Testing & Verification

Each export script runs a **numerical verification pass** after conversion:

1. Run ONNX model with known inputs → reference output
2. Run converted `.mlmodel` with same inputs → CoreML output
3. Compare numerically: `rtol=1e-2, atol=1e-3` (relaxed for quantized models)

**Specific tests:**
- `test_speech_encoder.py` — 1s silence + real audio, compare all 4 outputs
- `test_embed_tokens.py` — known token sequence, compare embeddings (text IDs + speech IDs)
- `test_language_model.py` — 3-step decode loop using stateful `predict(state)` API, compare logits at each step against ONNX reference
- `test_decoder.py` — known speech tokens + speaker context, compare waveform

**Integration test (`test_pipeline.py`):**
- Full E2E: text → tokens → LM → decoder → WAV
- Compare output with ONNX reference pipeline

---

## Implementation Notes

### coremltools Version
- Requires `coremltools >= 7.0` for iOS 18 stateful model support
- Use `ct.target.iOS18` explicitly

### Quantized Models (Q4F16)
- Source models are Q4F16 quantized — conversion preserves quantization
- Numerical tolerance is relaxed (`rtol=1e-2`) to account for quantization error

### Stateful LM State Management
- CoreML state is opaque — cannot be inspected from Swift
- State is maintained internally between `predict()` calls
- State must be re-initialized per synthesis session (new `MLModel` instance)
- **Performance note:** Creating a new `MLModel` instance per synthesis session has non-trivial loading cost (~1-3s). For best performance, keep a single `MLModel` instance alive across multiple synthesis calls within the same app session.

### Proof-of-Concept First
Before building the full pipeline, create a minimal PoC to verify:
1. coremltools 7+ can handle a stateful model with **48 state tensors + regular inputs** (`attention_mask`, `position_ids`) simultaneously
2. Q4F16 quantization is preserved through conversion
3. Numerical output matches ONNX reference within tolerance (`rtol=1e-2, atol=1e-3`)
This is the **critical path** — if it fails, the entire stateful approach must be reconsidered (fall back to stateless LM with manual KV cache management).

### Integration with Reader App
- Replace ONNX session creation with CoreML `MLModel` loading
- Decode loop changes: instead of manually managing KV cache tensors, call `predict()` and let CoreML handle state internally
- See: `ChatterboxEngine.swift` lines 841–980 (current ONNX decode loop)

---

## Files to Modify (Reader App)

After CoreML models are generated locally, they are placed in `Reader/Resources/ChatterboxModels/` alongside the existing ONNX files. The app uses ONNX by default; CoreML is enabled via a feature flag.

| File | Change |
|------|--------|
| `ChatterboxEngine.swift` | Add `MLModel` loading alongside `ORTSession`; feature flag selects ONNX vs CoreML path; CoreML decode loop uses `mlmodel.predict(state:)` as shown above |
| `project.yml` | No change — `.mlmodel` files go in the same `ChatterboxModels/` resource directory |
| `ModelDownloadService.swift` | CoreML models are built locally (not downloaded). For future: could upload to HF and download like ONNX models |

**CoreML model build artifact path:** `chatterbox-turbo-coreml/models/` → copy to `Reader/Resources/ChatterboxModels/`

---

## Out of Scope

- Training or fine-tuning — conversion only
- Non-iOS Apple platforms (macOS, visionOS) — not targeted but architecture is portable
- Original PyTorch source conversion — using ONNX intermediate (already public)
- ONNX Runtime removal from Reader app — can coexist or replace later
