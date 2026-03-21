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
- Each of the 24 KV layers becomes a `ct.StateType` with name `k_cache_N` and `v_cache_N`
- State shape: `[batch_size=1, num_kv_heads=16, seq_len=RangeDim(1, 4096), head_dim=64]`
- `minimum_deployment_target=ct.target.iOS18` (required for stateful model support)
- Swift side calls `predict()` repeatedly; CoreML manages cache internally

**Key challenge:** coremltools must correctly map 48 ONNX KV tensors → 48 CoreML states

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
- `test_embed_tokens.py` — known token sequence, compare embeddings
- `test_language_model.py` — 3-step decode with known KV state, compare logits
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

### Integration with Reader App
- Replace ONNX session creation with CoreML `MLModel` loading
- Decode loop changes: instead of manually managing KV cache tensors, call `predict()` and let CoreML handle state internally
- See: `ChatterboxEngine.swift` lines 841–980 (current ONNX decode loop)

---

## Files to Modify (Reader App)

After CoreML models are generated, these files in Reader need updates:

| File | Change |
|------|--------|
| `ChatterboxEngine.swift` | Replace `ORTSession` with `MLModel` for each component; simplify decode loop for stateful LM |
| `project.yml` | Add `.mlmodel` files to bundle resources |
| `ModelDownloadService.swift` | Point to CoreML models instead of ONNX (or parallel support) |

---

## Out of Scope

- Training or fine-tuning — conversion only
- Non-iOS Apple platforms (macOS, visionOS) — not targeted but architecture is portable
- Original PyTorch source conversion — using ONNX intermediate (already public)
- ONNX Runtime removal from Reader app — can coexist or replace later
