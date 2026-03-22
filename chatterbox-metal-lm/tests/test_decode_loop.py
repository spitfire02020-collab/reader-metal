#!/usr/bin/env python3
"""
test_decode_loop.py — Full E2E decode loop validation for MetalLM.

Validates the complete MetalLM pipeline:
  1. Loads all 24 GPT2NoEmbed layer weights from metal-export
  2. Runs full multi-head attention forward pass (all 24 layers)
  3. Runs autoregressive decode loop (greedy + rep_penalty)
  4. Compares logits against ResembleAI ONNX reference (if available)

Run on Mac: python3 test_decode_loop.py
"""

import os, sys, json, math
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKTREE_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
METAL_EXPORT_DIR = os.path.join(WORKTREE_ROOT, "metal-export/onnx_output")
WEIGHTS_DIR = METAL_EXPORT_DIR + "/weights"
MANIFEST_PATH = WEIGHTS_DIR + "/weights_manifest.json"
RESEMBLE_LM_ONNX = "/Users/rockymoon/Downloads/Reader/Reader/Resources/ChatterboxModels/language_model_q4f16_merged.onnx"

# Model config (GPT2NoEmbed / ChatterboxTurbo)
HIDDEN = 1024
INTERMEDIATE = 4096
NUM_HEADS = 16
HEAD_DIM = 64
NUM_LAYERS = 24
VOCAB = 6563
EOS_TOKEN = 6562
START_TOKEN = 6561
REP_PENALTY = 1.2


def load_manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def load_fp16(name, manifest, dtype=np.float32):
    """Load a raw FP16 weight and convert to dtype for computation."""
    entry = manifest.get(name)
    if entry is None:
        raise ValueError(f"Weight '{name}' not found in manifest")
    fpath = os.path.join(WEIGHTS_DIR, entry["fp16"])
    data = np.fromfile(fpath, dtype=np.float16).astype(dtype)
    return data


def layer_norm(x, gamma, beta, eps=1e-5):
    """RMSNorm-style LayerNorm: x * (1 + gamma) + beta, then normalize."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return x_norm * gamma + beta


def gelu(x):
    return x * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def causal_mask(seq_len):
    """Create causal mask for autoregressive decoding."""
    mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=1)
    return np.where(mask == 0, 0.0, -1e9)


def multi_head_attention(q, k, v, past_k, past_v, mask):
    """
    Multi-head attention with optional KV cache.

    q/k/v: [B, H, S, D]  (B=1, H=16, S=1, D=64)
    past_k/past_v: [B, H, past_seq, D]
    mask: [S, past_seq + S]

    Returns: attn_out [B, H, S, D], new_past_k [B, H, past+S, D], new_past_v
    """
    B, H, S, D = q.shape
    past_seq = 0 if past_k is None else past_k.shape[2]

    # Combine past K/V with current
    if past_k is not None:
        k_full = np.concatenate([past_k, k], axis=2)  # [B, H, past+S, D]
        v_full = np.concatenate([past_v, v], axis=2)  # [B, H, past+S, D]
    else:
        k_full, v_full = k, v

    # Attention scores: Q @ K^T / sqrt(D)
    # q: [B, H, S, D], k_full: [B, H, past+S, D]
    # scores: [B, H, S, past+S]
    scale = 1.0 / math.sqrt(D)
    scores = np.einsum('bhsv,bhkv->bhsk', q, k_full) * scale

    # Apply causal mask (for current tokens)
    if mask is not None:
        scores = scores + mask[None, None, :, :]

    # Softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    weights = scores_exp / scores_exp.sum(axis=-1, keepdims=True)

    # Apply to values
    attn = np.einsum('bhsk,bhkv->bhsv', weights, v_full)

    return attn, k_full, v_full


class GPT2Block:
    """Single GPT2 transformer block (GPT2NoEmbed)."""

    def __init__(self, c_attn_w, c_attn_b, c_proj_w, c_proj_b,
                 ln1_g, ln1_b, ln2_g, ln2_b,
                 c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b):
        # c_attn: binary [3072, 1024] → reshape to [1024, 3072] → ONNX matmul gives [S, 3072]
        self.c_attn_w = c_attn_w.reshape(HIDDEN, 3 * HIDDEN)  # [1024, 3072]
        self.c_attn_b = c_attn_b  # [3072]
        self.c_proj_w = c_proj_w.reshape(HIDDEN, HIDDEN)       # [1024, 1024]
        self.c_proj_b = c_proj_b
        self.ln1_g = ln1_g
        self.ln1_b = ln1_b
        self.ln2_g = ln2_g
        self.ln2_b = ln2_b
        # c_fc: binary [4096, 1024] → reshape to [1024, 4096] → ONNX matmul gives [S, 4096]
        self.c_fc_w = c_fc_w.reshape(HIDDEN, INTERMEDIATE)     # [1024, 4096]
        self.c_fc_b = c_fc_b
        # c_proj_mlp: binary [4096, 1024] → keep as [4096, 1024] → matmul gives [S, 1024]
        self.c_proj_mlp_w = c_proj_mlp_w.reshape(INTERMEDIATE, HIDDEN)  # [4096, 1024]
        self.c_proj_mlp_b = c_proj_mlp_b

    def forward(self, x, past_k, past_v):
        """
        x: [B=1, S, H]  fp32
        past_k/past_v: cached K/V for this layer [B=1, H, past_seq, D]

        Returns: output [B, S, H], new_past_k, new_past_v
        """
        S = x.shape[1]

        # Pre-attention LayerNorm
        x_norm = layer_norm(x, self.ln1_g, self.ln1_b)

        # QKV: c_attn: [S, 1024] @ [1024, 3072] = [S, 3072] (ONNX transB=0)
        c_attn_out = x_norm @ self.c_attn_w + self.c_attn_b  # [1, S, 3072]
        q = c_attn_out[:, :, :HIDDEN]      # [1, S, 1024]
        k = c_attn_out[:, :, HIDDEN:2*HIDDEN]
        v = c_attn_out[:, :, 2*HIDDEN:3*HIDDEN]

        # Reshape to multi-head: [B, H, S, D]
        q = q.reshape(1, NUM_HEADS, S, HEAD_DIM)
        k = k.reshape(1, NUM_HEADS, S, HEAD_DIM)
        v = v.reshape(1, NUM_HEADS, S, HEAD_DIM)

        # Causal mask for this step
        total_seq = S + (past_k.shape[2] if past_k is not None else 0)
        mask = causal_mask(total_seq)
        if past_k is not None:
            mask = mask[-S:, :]  # Take last S rows (current tokens' causal rows)

        # Multi-head attention
        attn_out, new_past_k, new_past_v = multi_head_attention(q, k, v, past_k, past_v, mask)

        # Reshape attention output: [B, H, S, D] → [B, S, H]
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(1, S, HIDDEN)

        # O projection: [S, 1024] @ [1024, 1024] = [S, 1024]
        o_out = attn_out @ self.c_proj_w + self.c_proj_b

        # Residual 1
        x = x + o_out

        # Pre-MLP LayerNorm
        x_norm2 = layer_norm(x, self.ln2_g, self.ln2_b)

        # MLP: c_fc expand [S, 1024] @ [1024, 4096] = [S, 4096]
        fc1 = x_norm2 @ self.c_fc_w + self.c_fc_b
        fc1_gelu = gelu(fc1)
        # c_proj contract: [S, 4096] @ [4096, 1024] = [S, 1024]
        fc2 = fc1_gelu @ self.c_proj_mlp_w + self.c_proj_mlp_b

        # Residual 2
        x = x + fc2

        return x, new_past_k, new_past_v


def build_model(manifest):
    """Build all 24 GPT2Blocks with weights from manifest."""
    blocks = []

    for layer in range(NUM_LAYERS):
        p = f"h.{layer}"

        # Binary weights: stored as [out_dim, in_dim] row-major
        c_attn_w = load_fp16(f"{p}.attn.c_attn.weight", manifest)    # flat [3072, 1024]
        c_attn_b = load_fp16(f"{p}.attn.c_attn.bias", manifest)      # [3072]
        c_proj_w = load_fp16(f"{p}.attn.c_proj.weight", manifest)    # flat [1024, 1024]
        c_proj_b = load_fp16(f"{p}.attn.c_proj.bias", manifest)      # [1024]
        ln1_g = load_fp16(f"{p}.ln_1.weight", manifest)
        ln1_b = load_fp16(f"{p}.ln_1.bias", manifest)
        ln2_g = load_fp16(f"{p}.ln_2.weight", manifest)
        ln2_b = load_fp16(f"{p}.ln_2.bias", manifest)
        # Binary: c_fc [4096, 1024], c_proj_mlp [4096, 1024]
        c_fc_w = load_fp16(f"{p}.mlp.c_fc.weight", manifest)        # flat [4096, 1024]
        c_fc_b = load_fp16(f"{p}.mlp.c_fc.bias", manifest)          # [4096]
        c_proj_mlp_w = load_fp16(f"{p}.mlp.c_proj.weight", manifest) # flat [4096, 1024]
        c_proj_mlp_b = load_fp16(f"{p}.mlp.c_proj.bias", manifest)  # [4096]

        block = GPT2Block(
            c_attn_w, c_attn_b,
            c_proj_w, c_proj_b,
            ln1_g, ln1_b, ln2_g, ln2_b,
            c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b
        )
        blocks.append(block)

    return blocks


def full_forward(blocks, embeddings, past_kv=None):
    """
    Run full 24-layer forward pass.

    embeddings: [B=1, S, H] fp32 input
    past_kv: list of (past_k, past_v) per layer, or None

    Returns: output [B, S, H], list of new past_kv per layer
    """
    x = embeddings
    new_past_kv = []

    for i, block in enumerate(blocks):
        past_k = None
        past_v = None
        if past_kv is not None:
            past_k, past_v = past_kv[i]

        x, new_past_k, new_past_v = block.forward(x, past_k, past_v)
        new_past_kv.append((new_past_k, new_past_v))

    return x, new_past_kv


def apply_rep_penalty(logits, tokens, penalty=1.2):
    """Apply repetition penalty to logits in-place."""
    for t in tokens:
        idx = int(t)
        if idx < len(logits):
            if logits[idx] > 0:
                logits[idx] /= penalty
            else:
                logits[idx] *= penalty


def greedy_decode(logits):
    """Return argmax token from logits."""
    return int(np.argmax(logits))


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_model_loading():
    """Verify all model weights load and have correct shapes."""
    print("=== Test: Model Loading ===")
    manifest = load_manifest()

    missing = []
    for layer in range(NUM_LAYERS):
        p = f"h.{layer}"
        for comp in ["attn.c_attn.weight", "attn.c_attn.bias", "attn.c_proj.weight",
                     "ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias",
                     "mlp.c_fc.weight", "mlp.c_fc.bias",
                     "mlp.c_proj.weight", "mlp.c_proj.bias"]:
            name = f"{p}.{comp}"
            if name not in manifest:
                missing.append(name)

    for root in ["ln_f.weight", "ln_f.bias", "lm_head.weight", "lm_head.bias"]:
        if root not in manifest:
            missing.append(root)

    if missing:
        print(f"  MISSING {len(missing)} weights:")
        for m in missing:
            print(f"    {m}")
    else:
        print(f"  All {NUM_LAYERS * 11 + 4} weights present ✓")

    # Load and verify shapes (weights are flat 1D, need reshaping)
    layer = 0
    p = f"h.{layer}"
    w = load_fp16(f"{p}.attn.c_attn.weight", manifest)
    expected = HIDDEN * 3 * HIDDEN  # 1024 * 3072 = 3145728
    if len(w) == expected:
        print(f"  {p}.attn.c_attn.weight: {len(w)} ✓ (flat, reshape to [1024, 3072])")
    else:
        print(f"  {p}.attn.c_attn.weight: got {len(w)}, expected {expected}")

    w = load_fp16(f"{p}.mlp.c_fc.weight", manifest)
    expected = INTERMEDIATE * HIDDEN  # 4096 * 1024 = 4194304
    if len(w) == expected:
        print(f"  {p}.mlp.c_fc.weight: {len(w)} ✓ (flat, reshape to [4096, 1024] or [1024, 4096])")
    else:
        print(f"  {p}.mlp.c_fc.weight: got {len(w)}, expected {expected}")

    w = load_fp16(f"{p}.mlp.c_proj.weight", manifest)
    if len(w) == expected:
        print(f"  {p}.mlp.c_proj.weight: {len(w)} ✓ (flat, reshape to [4096, 1024])")
    else:
        print(f"  {p}.mlp.c_proj.weight: got {len(w)}, expected {expected}")

    lm_head = load_fp16("lm_head.weight", manifest)
    expected_lm = HIDDEN * VOCAB  # 1024 * 6563 = 6720512
    if len(lm_head) == expected_lm:
        print(f"  lm_head.weight: {len(lm_head)} ✓ (flat, reshape to [1024, 6563])")
    else:
        print(f"  lm_head.weight: got {len(lm_head)}, expected {expected_lm}")

    print()


def test_single_step():
    """Run a single forward pass with a fixed input."""
    print("=== Test: Single Forward Step ===")
    manifest = load_manifest()
    blocks = build_model(manifest)

    # Load final weights
    ln_f_g = load_fp16("ln_f.weight", manifest)
    ln_f_b = load_fp16("ln_f.bias", manifest)
    lm_head_w = load_fp16("lm_head.weight", manifest).reshape(HIDDEN, VOCAB)  # [1024, 6563]
    lm_head_b = load_fp16("lm_head.bias", manifest)  # [6563]

    # Fixed seed for reproducibility
    np.random.seed(123)
    x = np.random.randn(1, 1, HIDDEN).astype(np.float32)

    # Full forward
    x_out, past_kv = full_forward(blocks, x, past_kv=None)

    # Final LN
    final = layer_norm(x_out[0, -1:], ln_f_g, ln_f_b)  # [1, 1024]

    # LM head
    logits = final @ lm_head_w + lm_head_b  # [1, 6563]
    logits = logits[0]

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {x_out.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"  Top 5 tokens: {np.argsort(logits)[-5:][::-1]}")
    print()


def test_decode_loop():
    """Run the autoregressive decode loop with a dummy prefix."""
    print("=== Test: Decode Loop ===")
    manifest = load_manifest()
    blocks = build_model(manifest)

    # Load final weights
    ln_f_g = load_fp16("ln_f.weight", manifest)
    ln_f_b = load_fp16("ln_f.bias", manifest)
    lm_head_w = load_fp16("lm_head.weight", manifest).reshape(HIDDEN, VOCAB)  # [1024, 6563]
    lm_head_b = load_fp16("lm_head.bias", manifest)  # [6563]

    # Dummy prefix: 5 random embeddings
    np.random.seed(42)
    prefix = np.random.randn(1, 5, HIDDEN).astype(np.float32)

    # Prime the model with prefix (full forward pass)
    x_out, past_kv = full_forward(blocks, prefix, past_kv=None)
    last_hidden = x_out[0, -1:]  # [1, 1024]

    # Final LN + LM head
    final = layer_norm(last_hidden, ln_f_g, ln_f_b)
    logits = (final @ lm_head_w + lm_head_b)[0]  # [6563]

    print(f"  Prefix shape: {prefix.shape}")
    print(f"  After prefix — logits range: [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"  Top 3 after prefix: {np.argsort(logits)[-3:][::-1]}")

    # Greedy decode steps
    generated = []
    max_new = 20

    for step in range(max_new):
        # Apply repetition penalty
        apply_rep_penalty(logits, generated, penalty=REP_PENALTY)

        # Greedy decode
        token = greedy_decode(logits)
        generated.append(token)

        if token == EOS_TOKEN:
            print(f"  Stopped at step {step} (EOS token {EOS_TOKEN})")
            break

        # Get next token embedding (use last hidden as approximation — in real impl would use embed table)
        next_hidden = last_hidden  # placeholder: in real impl, would look up embed

        # Forward step with KV cache
        next_hidden_3d = next_hidden.reshape(1, 1, HIDDEN)
        x_out, past_kv = full_forward(blocks, next_hidden_3d, past_kv=past_kv)
        last_hidden = x_out[0, -1:]

        # Final LN + LM head
        final = layer_norm(last_hidden, ln_f_g, ln_f_b)
        logits = (final @ lm_head_w + lm_head_b)[0]

        if step < 5 or step == max_new - 1:
            print(f"  Step {step}: token={token}, logits range=[{logits.min():.4f}, {logits.max():.4f}]")

    print(f"  Generated {len(generated)} tokens: {generated[:10]}{'...' if len(generated) > 10 else ''}")

    # Sanity checks
    assert all(0 <= t < VOCAB for t in generated), "Tokens out of range!"
    speech_tokens = [t for t in generated if t < VOCAB - 3]  # tokens before special
    print(f"  Speech tokens: {len(speech_tokens)}/{len(generated)}")
    print()


def test_onnx_reference():
    """Compare MetalLM Python forward against ONNX reference (single step)."""
    print("=== Test: ONNX Reference Comparison ===")

    try:
        import onnxruntime as ort
    except ImportError:
        print("  SKIP: onnxruntime not available")
        return

    if not os.path.exists(RESEMBLE_LM_ONNX):
        print(f"  SKIP: {RESEMBLE_LM_ONNX} not found")
        return

    manifest = load_manifest()
    blocks = build_model(manifest)
    ln_f_g = load_fp16("ln_f.weight", manifest)
    ln_f_b = load_fp16("ln_f.bias", manifest)
    lm_head_w = load_fp16("lm_head.weight", manifest).reshape(HIDDEN, VOCAB)  # [1024, 6563]
    lm_head_b = load_fp16("lm_head.bias", manifest)  # [6563]

    # Load ONNX model (may fail if ORT version lacks GatherBlockQuantized custom op)
    try:
        sess = ort.InferenceSession(RESEMBLE_LM_ONNX, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"  SKIP: Could not load ONNX model ({e.__class__.__name__}): {e}")
        return

    # Fixed seed
    np.random.seed(42)
    x = np.random.randn(1, 1, HIDDEN).astype(np.float32)

    # MetalLM Python forward
    x_out, _ = full_forward(blocks, x, past_kv=None)
    final = layer_norm(x_out[0, -1:], ln_f_g, ln_f_b)
    metal_logits = (final @ lm_head_w + lm_head_b)[0].astype(np.float32)

    # ONNX reference
    feeds = {
        "inputs_embeds": x,
        "attention_mask": np.ones([1, 1], dtype=np.int64),
        "position_ids": np.arange(1, dtype=np.int64)[None, :],
    }
    for i in range(25):  # ResembleAI has 25 layers
        feeds[f"past_key_values.{i}.key"] = np.zeros([1, 16, 0, 64], dtype=np.float32)
        feeds[f"past_key_values.{i}.value"] = np.zeros([1, 16, 0, 64], dtype=np.float32)

    try:
        onnx_out = sess.run(None, feeds)
        onnx_logits = onnx_out[0][0, 0]  # [B, S, vocab] → [vocab]

        print(f"  MetalLM logits range: [{metal_logits.min():.4f}, {metal_logits.max():.4f}]")
        print(f"  ONNX logits range:    [{onnx_logits.min():.4f}, {onnx_logits.max():.4f}]")

        # Correlation
        corr = np.corrcoef(metal_logits, onnx_logits)[0, 1]
        print(f"  Pearson correlation:  {corr:.4f}")

        # Top-k overlap
        metal_top5 = set(np.argsort(metal_logits)[-5:][::-1])
        onnx_top5 = set(np.argsort(onnx_logits)[-5:][::-1])
        overlap = len(metal_top5 & onnx_top5)
        print(f"  Top-5 overlap: {overlap}/5")
        print(f"  Metal top-5: {sorted(metal_top5)}")
        print(f"  ONNX top-5:  {sorted(onnx_top5)}")
    except Exception as e:
        print(f"  ONNX inference failed: {e}")

    print()


if __name__ == "__main__":
    print(f"Weights dir: {WEIGHTS_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Model: GPT2NoEmbed {NUM_LAYERS} layers, hidden={HIDDEN}, vocab={VOCAB}\n")

    test_model_loading()
    test_single_step()
    test_decode_loop()
    test_onnx_reference()

    print("Done.")
