# chatterbox-turbo-coreml: Design Specification

## Overview

Convert the 4 ONNX models from `ResembleAI/chatterbox-turbo-ONNX` to Apple's CoreML format. The resulting `.mlmodel` files will replace ONNX Runtime in the Reader iOS app, enabling native Apple Silicon GPU acceleration and eliminating the ONNX Runtime dependency for inference.

**Repository:** New standalone repo (TBD — `chatterbox-turbo-coreml`)

**Source models:** `ResembleAI/chatterbox-turbo-ONNX` on HuggingFace — 4 quantized ONNX models (Q4F16)

**Target:** iOS 18+ (required for stateful CoreML KV cache support)

---

## Repository Layout

```
chatterbox-turbo-coreml/
├── scripts/
│   ├── download_models.py          # Download from HF ONNX repo
│   ├── export_speech_encoder.py    # Audio → .mlmodel
│   ├── export_embed_tokens.py       # Tokens → embeddings
│   ├── export_language_model.py    # LM with KV cache states
│   ├── export_decoder.py           # Tokens → waveform
│   └── verify_conversion.py        # Numerical comparison helper
├── tests/
│   ├── test_speech_encoder.py
│   ├── test_embed_tokens.py
│   ├── test_language_model.py
│   ├── test_decoder.py
│   └── test_pipeline.py            # Full E2E integration
├── requirements.txt
├── README.md
└── models/                          # Output .mlmodel files
```

Each export script is self-contained: downloads its ONNX source if not present, converts, and verifies in one shot.

---

## Model Components & Conversion Strategy

### 1. Speech Encoder

**ONNX source:** `speech_encoder_q4f16.onnx`
**Size:** ~25 MB (quantized)

**I/O Contract (from ONNX metadata):**
- Input:  `audio_values: [batch_size, num_samples]` — float32, fully dynamic length
- Outputs:
  - `audio_features: [batch_size, sequence_length, 1024]` — float16
  - `audio_tokens: [batch_size, audio_sequence_length]` — int64
  - `speaker_embeddings: [batch_size, 192]` — float16
  - `speaker_features: [batch_size, feature_dim, 80]` — float16

**Conversion approach:** Stateless, `RangeDim` for audio length
- Use `ct.RangeDim(lower_bound=24000, upper_bound=240000)` for `num_samples` (1s–10s audio at 24kHz)
- All outputs are bounded via `ct.RangeDim`
- `minimum_deployment_target=iOS18`, `compute_units=CPU_AND_GPU`

**Conversion path:** `torch.jit.trace` or `ct.converters.onnx.convert`

---

### 2. Embed Tokens

**ONNX source:** `embed_tokens_q4f16.onnx`
**Size:** ~60 MB (quantized)

**I/O Contract:**
- Input:  `input_ids: [batch_size, sequence_length]` — int64, dynamic
- Output: `inputs_embeds: [batch_size, sequence_length, 1024]` — float16

**Conversion approach:** Stateless, `RangeDim` for sequence length
- Use `ct.RangeDim(lower_bound=1, upper_bound=2048)` for `sequence_length`
- Simple embedding lookup — no special state handling needed
- `minimum_deployment_target=iOS18`, `compute_units=CPU_AND_GPU`

**Note on token ID ranges:** `embed_tokens` accepts two types of token IDs:
- **Text tokens:** GPT-2 byte-level BPE IDs (0–50256) during text embedding phase
- **Speech tokens:** Generated speech IDs (0–6562) during the decode loop (to embed the LM's own output for next-token prediction)
- The single embedding table internally handles both ranges. CoreML conversion preserves this routing — the same `embed_tokens.mlmodel` is called twice per synthesis: once with text IDs, once per decode step with speech IDs.

---

### 3. Language Model (Complex — Stateful)

**ONNX source:** `language_model_q4f16.onnx`
**Size:** ~280 MB (quantized)
**Architecture:** GPT-2 based, 24 layers, 16 KV heads, 64 head_dim, 1024 hidden_size

**I/O Contract (48 KV cache tensors):**
- Inputs:
  - `inputs_embeds: [batch_size, sequence_length, 1024]` — float16
  - `attention_mask: [batch_size, total_sequence_length]` — int64
  - `position_ids: [batch_size, sequence_length]` — int64
  - 24 × `past_key_values.N.key: [batch_size, 16, past_sequence_length, 64]` — float16
  - 24 × `past_key_values.N.value: [batch_size, 16, past_sequence_length, 64]` — float16
- Outputs:
  - `logits: [batch_size, sequence_length, 6563]` — float16
  - 24 × `present.N.key` + 24 × `present.N.value` — same shapes as inputs

**Conversion approach:** Stateful model with `ct.StateType`
- coremltools 7+ supports **regular inputs alongside state inputs** (confirmed via official docs — `ToyModelWithKVCache` example passes `input_ids` + `causal_mask` as regular inputs with `k_cache`/`v_cache` as states)
- `inputs_embeds`, `attention_mask`, and `position_ids` remain as **regular inputs** passed every `predict()` call
- Each of the 24 KV layers becomes a `ct.StateType` with name matching the **actual ONNX input name** (e.g., `ct.StateType(name="past_key_values.0.key")`)
- State shape: `[batch_size=1, num_kv_heads=16, seq_len=RangeDim(1, 8192), head_dim=64]` — **upper bound 8192** to handle long audio prefixes (audio tokens can be 750+, text tokens 500+, generated speech 1500+, totaling ~2750+)
- `minimum_deployment_target=ct.target.iOS18` (required for stateful model support)
- Swift decode loop: call `predict(inputs_dict, kv_state)` each step; CoreML internally updates KV cache

**Stateful decode loop (Swift):**
```swift
kv_state = mlmodel.makeState()
var positionId = totalSeqLen  // start from end of prefix
var maskLen = totalSeqLen     // attention mask length grows each step
for step in 0..<maxTokens {
    let outputs = try mlmodel.predict(inputs: [
        "inputs_embeds": nextEmbed,              // regular input
        "attention_mask": [Int64](repeating: 1, count: maskLen),  // grows per step
        "position_ids": [Int64](repeating: positionId, count: 1)  // single position
    ], state: kv_state)
    let logits = outputs["logits"]
    let nextToken = greedyDecode(logits)
    if nextToken == STOP_SPEECH { break }
    positionId += 1   // increment per decode step
    maskLen += 1      // mask grows with new token
    // prepare next embed for next step...
}
```

**State mapping from ONNX → CoreML:**
```python
states = [
    ct.StateType(
        wrapped_type=ct.TensorType(shape=[1, 16, ct.RangeDim(1, 8192), 64], dtype=np.float16),
        name="past_key_values.0.key",   # must match actual ONNX input name
    ),
    ct.StateType(
        wrapped_type=ct.TensorType(shape=[1, 16, ct.RangeDim(1, 8192), 64], dtype=np.float16),
        name="past_key_values.0.value",
    ),
    # ... repeat for layers 1-23 (48 total StateType entries)
]
```

**Key challenge:** coremltools must correctly map 48 ONNX KV tensors → 48 CoreML states using explicit `ct.StateType(name=...)` that matches ONNX input names.

---

### 4. Conditional Decoder

**ONNX source:** `conditional_decoder_q4f16.onnx`
**Size:** ~210 MB (quantized)

**I/O Contract:**
- Inputs:
  - `speech_tokens: [batch_size, num_speech_tokens]` — int64, dynamic
  - `speaker_embeddings: [batch_size, 192]` — float16
  - `speaker_features: [batch_size, feature_dim, 80]` — float16
- Output: `waveform: [batch_size, num_samples]` — float32

**Conversion approach:** Stateless, `RangeDim` for speech token length
- NOT autoregressive — called once with all speech tokens to produce full waveform
- Use `ct.RangeDim(lower_bound=1, upper_bound=8192)` for `num_speech_tokens`
- `minimum_deployment_target=iOS18`, `compute_units=CPU_AND_GPU`

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
