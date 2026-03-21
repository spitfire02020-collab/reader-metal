# Chatterbox Metal Language Model — Design Specification

## Overview

**Status: DESIGN — Under Review**

Replace the ONNX Runtime language model (LM) inference path with a custom Metal compute shader implementation, keeping the other 3 ONNX models (embed_tokens, speech_encoder, conditional_decoder) on ONNX Runtime.

**Repository:** `chatterbox-metal-lm` (new, inside Reader repo)

**Source models:** Existing Q4F16 ONNX weight tensors extracted from `language_model_q4f16_merged.onnx`

**Target:** iOS 18+ (Metal 3, ANE access via MPS fallback where needed)

---

## Architecture: Hybrid Metal + ONNX

```
┌─────────────────────────────────────────────────────┐
│                   ChatterboxEngine                  │
│                                                     │
│  [text_emb ONNX] ──────────────────────────────────┼──► text_embeddings
│                                                     │
│  [speech_encoder ONNX] ─────────────────────────────┼──► audio_features + speaker_context
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │        Metal LM (NEW)                       │   │ ◄── inputs_embeds [B,S,1024]
│  │  ┌────────────────────────────────────────┐ │   │     + speaker_context
│  │  │  Q4F16 Dequantization Kernels         │ │   │     + KV cache state
│  │  │  (INT4 → fp16, per-block scales/zp)   │ │   │ ──► logits [B,S,6563]
│  │  └────────────────────────────────────────┘ │   │     + updated KV cache
│  │  ┌────────────────────────────────────────┐ │   │
│  │  │  24× GPT2Block Forward Pass            │ │   │
│  │  │  • LayerNorm → QKV proj (Q4F16)       │ │   │
│  │  │  • SDPA (16 heads, no KV grouping)    │ │   │
│  │  │  • O proj (Q4F16) → residual-add      │ │   │
│  │  │  • FFN: Gelu(FC1) → FC2               │ │   │
│  │  └────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  [conditional_decoder ONNX] ────────────────────────┼──► waveform
└─────────────────────────────────────────────────────┘
```

### Why Hybrid over Full Metal?

| Component | ONNX Runtime | Metal LM | Reason |
|-----------|-------------|----------|--------|
| embed_tokens | ✅ | N/A | Trivial embedding lookup, low compute |
| speech_encoder | ✅ | N/A | Complex audio processing, not ANE-bound |
| **language_model** | ❌ (Q4F16 + GQA blocked) | **✅ This spec** | Primary ANE compute target |
| conditional_decoder | ✅ | N/A | Small model (~2MB), runs fine on GPU |

---

## Model: GPT2-Medium (Chatterbox Turbo)

**Source weight extraction:** Q4F16 quantized weights from `language_model_q4f16_merged.onnx`

### Config (from `llama_configs.py` GPT2_medium):
```
vocab_size:          6563  (speech codec tokens 0-6560, START=6561, STOP=6562)
hidden_size:         1024
num_hidden_layers:   24
num_attention_heads: 16
head_dim:            64
intermediate_size:   4096  (FFN fc1 output)
activation:          Gelu (new)
attention_bias:      False
layer_norm_eps:      1e-05
max_position_embeddings: 8196
```

### Attention: Standard MHA (NOT Grouped)

The ONNX model's `GroupQueryAttention` uses 16 query heads + 16 KV heads (no grouping) — mathematically equivalent to standard multi-head attention. Metal can use a standard MHA kernel rather than a GQA-specific one.

### KV Cache Format (from ONNX model):
```
past_key_values.N.key:   [B, 16, past_seq, 64]
past_key_values.N.value: [B, 16, past_seq, 64]
  (N = 0..23 for each layer)

present.N.key:   [B, 16, total_seq, 64]  ← full KV after update
present.N.value: [B, 16, total_seq, 64]
```

---

## Weight Format: Q4F16 Block-wise INT4

### Per-Layer Weight Structure

Each quantized weight has 3 tensors:

| Tensor | Shape | dtype | Example |
|--------|-------|-------|---------|
| `*_quant` | `[out_dim, block_idx, 16]` | uint8 (packed INT4) | 1024×32×16 |
| `*_scales` | `[out_dim, block_idx]` | fp16 | 1024×32 |
| `*_zp` | `[out_dim, block_idx/16]` | fp16 | 1024×2 |

**Block structure:** 32 rows per block, 16 elements per INT4 pack.
**Zero-points:** 1 per 16 blocks (shape `[out_dim, block_idx/16]`).

**Dequantization formula:**
```
fp16_weight[row, col] = (int4_val - zero_point) * scale
```

Where `int4_val` is extracted from packed uint8 as `(packed[row, block, col/16] >> (4 * (col % 16))) & 0xF`.

### Weight Count Summary

| Layer | Q_proj | K_proj | V_proj | O_proj | FC1 | FC2 | Total |
|-------|--------|--------|--------|--------|-----|-----|-------|
| 24× Attention | 1024 | 1024 | 1024 | 1024 | — | — | |
| 24× MLP | — | — | — | — | 4096 | 1024 | |
| LM Head | — | — | — | 6563 | — | — | |
| **Total** | | | | | | | **146 quantized matmuls** |

---

## Compute Pipeline

### Forward Pass: Single Step

```
Input: inputs_embeds [B, S, 1024]
       past_key_values [24][2][B, 16, past_seq, 64]
       past_sequence_length: Int

1. Prepend speaker conditioning to inputs_embeds
   → x = concat([conditioning, inputs_embeds], axis=1)

2. For each layer i = 0..23:
   a. x_norm = LayerNorm(x)
   b. Q = x_norm @ W_q (Q4F16 → fp16)
      K = x_norm @ W_k (Q4F16 → fp16)
      V = x_norm @ W_v (Q4F16 → fp16)

   c. Q_reshaped = reshape(Q, [B, S, 16, 64])
      K_reshaped = reshape(K, [B, S, 16, 64])
      V_reshaped = reshape(V, [B, S, 16, 64])

   d. [K_cache, V_cache] = past_key_values[i]

   e. K_full = concat([K_cache, K_reshaped], axis=2)
      V_full = concat([V_cache, V_reshaped], axis=2)

   f. attn = SDPATest(Q_reshaped, K_full, V_full)  ← causal mask applied
      (standard MHA: Q @ K^T / sqrt(64) → softmax → V)

   g. attn = reshape(attn, [B, S, 1024])
      attn = attn @ W_o (Q4F16 → fp16)
      x = x + attn  (residual)

   h. x_norm2 = LayerNorm(x)
   i. gate = gelu(x_norm2 @ W_fc1 (Q4F16 → fp16))
   j. ffn_out = gate * (x_norm2 @ W_fc2 (Q4F16 → fp16))
   k. x = x + ffn_out  (residual)

   l. Update past_key_values[i] = [K_full, V_full]

3. Final LayerNorm(x) → hidden [B, S, 1024]
4. logits = hidden @ W_lm_head (Q4F16 → fp16)  → [B, S, 6563]

Output: logits [B, S, 6563]
        updated past_key_values
```

### Decode Loop (Swift side)

```
generateTokens = [START_SPEECH=6561]
past_kv = null
past_seq = 0

for step in 0..maxNewTokens:
    1. embed = speech_emb(generateTokens[-1:])  ← ONNX embed_tokens
    2. concat_embed = concat([conditioning, text_emb, embed], axis=1)

    3. Run Metal forward(inputs_embeds=concat_embed,
                          past_kv=past_kv,
                          past_seq=past_seq)
       → logits, new_past_kv

    4. logits_step = logits[0, -1, :]  ← last token logits
    5. apply repetition_penalty(logits_step, generateTokens)
    6. next_token = argmax(logits_step)

    7. if next_token == STOP_SPEECH: break
    8. generateTokens.append(next_token)
    9. past_kv = new_past_kv
    10. past_seq += 1
```

**Greedy decode only** — no temperature, no top-k, no top-p (matches iOS reference implementation).

---

## Metal Shader Architecture

### Compute Pipeline

```
chatterbox_lm_forward
    ├── dequantize_weight_kernel  (INT4 → fp16, per-block)
    ├── layer_norm_kernel         (inline, no GPU primitive)
    ├── qkv_proj_kernel           (single matmul: A @ W^T)
    ├── sdpa_kernel               (causal mask + softmax)
    ├── o_proj_kernel
    ├── ffn_kernel                (gelu + elementwise multiply + fc2)
    └── final_lm_head_kernel
```

### Memory Layout

| Tensor | Format | Notes |
|--------|--------|-------|
| Q/K/V | `[B, H, S, D]` (row-major) | H=16, D=64 |
| Attention scores | `[B, H, S, S]` (fused in kernel) | Not materialized separately |
| KV cache | `[B, H, max_seq, D]` | Pre-allocated, updated in-place |
| Weights | Q4F16 packed | Dequantized on-the-fly in kernel |

### KV Cache Management

- Metal buffers pre-allocated for `max_seq = 8196`
- Each decode step: `K_cache = concat(K_cache, K_new, axis=2)` — implemented as memcpy + append
- `past_sequence_length` tracks current KV size

### Data Types

| Compute | dtype | Notes |
|---------|-------|-------|
| Activations | fp16 | Matches ANE preference |
| Q/K/V/Attn | fp16 | All matmuls in fp16 |
| Quantized weights | uint8 (INT4 packed) | Scales/zero-points in fp16 |
| KV cache | fp16 | Matches weight format |
| logits output | fp32 | Final output to Swift |

---

## File Structure

```
chatterbox-metal-lm/
├── src/
│   ├── Metal/
│   │   ├── LanguageModel.metal          # All compute shaders
│   │   ├── MetalLMEncoder.swift         # MPS encoding + shader dispatch
│   │   ├── Q4F16Dequant.swift           # Weight loading + dequantization
│   │   └── MetalLMConfig.swift           # Model config constants
│   ├── KVCache/
│   │   ├── KVCacheBuffer.swift          # Pre-allocated KV Metal buffers
│   │   └── KVCacheManager.swift        # Cache lifecycle + concatenation
│   ├── Inference/
│   │   ├── MetalLMForward.swift         # Single-step forward pass
│   │   └── MetalLMDecode.swift          # Decode loop orchestration
│   └── Export/
│       ├── ExtractWeights.swift         # Extract Q4F16 weights from ONNX
│       └── WeightLoader.swift           # Load weights into Metal buffers
├── tests/
│   ├── test_forward.py                  # Numerical validation vs ONNX reference
│   ├── test_decode_loop.py              # E2E decode validation
│   └── test_q4f16_dequant.py            # Dequantization correctness
├── scripts/
│   ├── build_metal_lib.sh               # xcrun metal + metallib
│   └── extract_weights.py               # ONNX → Metal weight binary
├── ChatterboxMetalLM.swift              # Public API (single class)
├── ChatterboxMetalLM.h                  # ObjC bridging header
└── ChatterboxMetalLMModule.swift        # Swift package / module
```

**Integration with Reader app:**
- `ChatterboxEngine.swift` adds `useMetalLM: Bool` flag
- When `true`, replaces ORT LM session with `ChatterboxMetalLM`
- Other 3 ONNX models unchanged

---

## Error Handling

| Failure Mode | Handling |
|-------------|----------|
| Metal device unavailable | Fall back to ONNX Runtime |
| Weight file not found | Return error, ONNX Runtime fallback |
| Dequantization mismatch | Compare with ONNX reference at tolerance `rtol=1e-2, atol=1e-3` |
| Decode timeout | Max tokens guard, return partial result |
| KV cache overflow | Pre-allocate max_seq=8192, assert on overflow |

---

## Testing & Verification

### Unit Tests

1. **`test_q4f16_dequant.py`** — Extract first layer's q_proj weights from ONNX, dequantize in Python, compare against Metal kernel output

2. **`test_forward.py`** — Run single-step Metal forward with known inputs, compare logits against ONNX reference
   - Tolerance: `rtol=1e-2, atol=1e-3`
   - Test both prefix (no KV cache) and decode (with KV cache) paths

3. **`test_decode_loop.py`** — Full decode loop for known text, compare generated speech tokens against ONNX reference
   - Metric: token agreement at each step
   - Final audio quality: subjective (human listen test)

### Integration Tests

4. **Reader app E2E test** — Run TTS with Metal LM, compare output WAV with ONNX pipeline WAV
   - Waveform similarity: use `scipy.signal.correlate` or similar
   - Mel-spectrogram similarity: `np.corrcoef`

---

## Implementation Phases

### Phase 1: Weight Extraction + Metal Shader Skeleton
- Extract Q4F16 weights from ONNX → binary `.metalweight` files
- Write Metal shaders for: LayerNorm, QKV proj, MHA, O proj, FFN, LM head
- Verify single forward pass numerical match

### Phase 2: KV Cache Management
- Pre-allocate KV Metal buffers
- Implement KV cache concat kernel
- Verify autoregressive decode loop correctness

### Phase 3: Integration
- Replace ONNX LM session with MetalLM in `ChatterboxEngine.swift`
- Feature flag: `useMetalLM: Bool`
- Full E2E TTS validation

---

## Out of Scope

- S3Gen decoder Metal implementation
- Speech encoder Metal implementation
- Training or fine-tuning
- Non-iOS Apple platforms
- FP16 unquantized variant (Q4F16 is the working format)
