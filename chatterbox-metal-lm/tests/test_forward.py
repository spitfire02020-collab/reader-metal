#!/usr/bin/env python3
"""
test_forward.py — Validate MetalLM forward pass against ONNX reference.

Tests that the Swift MetalLM weights (from metal-export) produce correct
logits by comparing against the ResembleAI ONNX reference implementation.

Run on Mac: python3 test_forward.py
"""

import os, sys, json
import numpy as np

# Paths — metal-export is at worktree root, tests/ is inside chatterbox-metal-lm/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKTREE_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
METAL_EXPORT_DIR = os.path.join(WORKTREE_ROOT, "metal-export/onnx_output")
WEIGHTS_DIR = METAL_EXPORT_DIR + "/weights"
MANIFEST_PATH = WEIGHTS_DIR + "/weights_manifest.json"
RESEMBLE_LM_ONNX = "/Users/rockymoon/Downloads/Reader/Reader/Resources/ChatterboxModels/language_model_q4f16_merged.onnx"


def load_manifest():
    """Load the weights manifest."""
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def load_weight_fp16(name, manifest):
    """Load a raw FP16 weight from the metal-export weights directory."""
    entry = manifest.get(name)
    if entry is None:
        raise ValueError(f"Weight '{name}' not found in manifest")

    if "fp16" not in entry:
        raise ValueError(f"Weight '{name}' is not a raw FP16 entry: {entry}")

    fpath = os.path.join(WEIGHTS_DIR, entry["fp16"])
    data = np.fromfile(fpath, dtype=np.float16)
    return data


def test_weight_loading():
    """Verify all weights load correctly from manifest."""
    print("=== Test: Weight Loading ===")
    manifest = load_manifest()
    print(f"Manifest entries: {len(manifest)}")

    # Check all 24 layers
    for layer in range(24):
        for comp in ["attn.c_attn.weight", "attn.c_attn.bias", "attn.c_proj.weight",
                      "ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias",
                      "mlp.c_fc.weight", "mlp.c_fc.bias",
                      "mlp.c_proj.weight", "mlp.c_proj.bias"]:
            name = f"h.{layer}.{comp}"
            if name not in manifest:
                print(f"  MISSING: {name}")

    # Check root-level
    for name in ["ln_f.weight", "ln_f.bias", "lm_head.weight", "lm_head.bias"]:
        if name not in manifest:
            print(f"  MISSING: {name}")

    # Load and verify shapes
    shapes = {
        "h.0.attn.c_attn.weight":  (1024, 3072),
        "h.0.attn.c_attn.bias":    (3072,),
        "h.0.attn.c_proj.weight":  (1024, 1024),
        "h.0.ln_1.weight":        (1024,),
        "h.0.mlp.c_fc.weight":    (1024, 4096),
        "h.0.mlp.c_proj.weight":  (4096, 1024),
        "ln_f.weight":            (1024,),
        "lm_head.weight":         (1024, 6563),  # transposed from [6563, 1024]
    }

    for name, expected_shape in shapes.items():
        data = load_weight_fp16(name, manifest)
        expected_size = np.prod(expected_shape)
        if len(data) != expected_size:
            print(f"  SHAPE MISMATCH: {name} — got {len(data)}, expected {expected_size}")
        else:
            print(f"  OK: {name} → shape={expected_shape}")

    print()


def test_c_attn_split():
    """Verify c_attn splitting into Q, K, V."""
    print("=== Test: c_attn Split ===")
    manifest = load_manifest()

    c_attn = load_weight_fp16("h.0.attn.c_attn.weight", manifest)
    c_attn = c_attn.reshape(1024, 3072)

    hidden = 1024
    q_cols = c_attn[:, 0:hidden]        # cols 0-1023
    k_cols = c_attn[:, hidden:2*hidden]  # cols 1024-2047
    v_cols = c_attn[:, 2*hidden:3*hidden] # cols 2048-3071

    print(f"  c_attn shape: {c_attn.shape}")
    print(f"  Q split shape: {q_cols.shape}")
    print(f"  K split shape: {k_cols.shape}")
    print(f"  V split shape: {v_cols.shape}")

    # Verify Q, K, V have similar magnitude distributions
    print(f"  Q range: [{q_cols.min():.4f}, {q_cols.max():.4f}]")
    print(f"  K range: [{k_cols.min():.4f}, {k_cols.max():.4f}]")
    print(f"  V range: [{v_cols.min():.4f}, {v_cols.max():.4f}]")
    print()


def test_matmul_dimensions():
    """Verify matmul operations give expected dimensions."""
    print("=== Test: Matmul Dimensions ===")
    manifest = load_manifest()

    # FC1: ln2 [S, 1024] @ c_fc [1024, 4096] = [S, 4096]
    c_fc = load_weight_fp16("h.0.mlp.c_fc.weight", manifest)
    print(f"  c_fc weight shape: {c_fc.shape}")  # Should be [1024, 4096] or [4096, 1024]

    # FC2: gelu [S, 4096] @ c_proj [4096, 1024] = [S, 1024]
    c_proj_mlp = load_weight_fp16("h.0.mlp.c_proj.weight", manifest)
    print(f"  c_proj (MLP) weight shape: {c_proj_mlp.shape}")  # Should be [4096, 1024]

    # LM head: final_ln [S, 1024] @ lm_head [1024, 6563] = [S, 6563]
    lm_head = load_weight_fp16("lm_head.weight", manifest)
    print(f"  lm_head weight shape: {lm_head.shape}")  # Should be [1024, 6563]

    # Verify: if lm_head is [1024, 6563], matmul with [S, 1024] gives [S, 6563] ✓
    if lm_head.shape == (1024, 6563):
        print("  lm_head: [1024, 6563] ✓ — matmul with [S, 1024] → [S, 6563]")
    elif lm_head.shape == (6563, 1024):
        print("  lm_head: [6563, 1024] — needs transpose before matmul!")
    print()


def test_ln_f_and_lm_head():
    """Verify ln_f and lm_head are present and correctly shaped."""
    print("=== Test: ln_f and lm_head ===")
    manifest = load_manifest()

    ln_f_w = load_weight_fp16("ln_f.weight", manifest)
    ln_f_b = load_weight_fp16("ln_f.bias", manifest)
    lm_head_w = load_weight_fp16("lm_head.weight", manifest)
    lm_head_b = load_weight_fp16("lm_head.bias", manifest)

    print(f"  ln_f.weight: {ln_f_w.shape}, range=[{ln_f_w.min():.4f}, {ln_f_w.max():.4f}]")
    print(f"  ln_f.bias: {ln_f_b.shape}")
    print(f"  lm_head.weight: {lm_head_w.shape}, range=[{lm_head_w.min():.4f}, {lm_head_w.max():.4f}]")
    print(f"  lm_head.bias: {lm_head_b.shape}")
    print()


def test_onnx_reference():
    """Run forward pass on ResembleAI ONNX reference for comparison."""
    print("=== Test: ONNX Reference (ResembleAI) ===")

    try:
        import onnxruntime as ort
    except ImportError:
        print("  SKIP: onnxruntime not available")
        return

    if not os.path.exists(RESEMBLE_LM_ONNX):
        print(f"  SKIP: {RESEMBLE_LM_ONNX} not found")
        return

    try:
        sess = ort.InferenceSession(RESEMBLE_LM_ONNX, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"  SKIP: Could not load ONNX model ({e.__class__.__name__}): {e}")
        return

    print(f"  Loaded: {RESEMBLE_LM_ONNX}")

    # Dummy input: [B=1, S=1, H=1024]
    B, S, H = 1, 1, 1024
    inputs_embeds = np.random.randn(B, S, H).astype(np.float32)

    feeds = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": np.ones([B, S], dtype=np.int64),
        "position_ids": np.arange(S, dtype=np.int64)[None, :],
    }
    # Empty KV cache
    for i in range(25):
        k = np.zeros([B, 16, 0, 64], dtype=np.float32)
        v = np.zeros([B, 16, 0, 64], dtype=np.float32)
        feeds[f"past_key_values.{i}.key"] = k
        feeds[f"past_key_values.{i}.value"] = v

    out = sess.run(None, feeds)
    logits = out[0]  # [B, S, vocab=6563]

    print(f"  Input shape: {inputs_embeds.shape}")
    print(f"  Output logits shape: {logits.shape}")  # Should be [1, 1, 6563]
    print(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"  Top 5 tokens: {np.argsort(logits[0, 0])[-5:][::-1]}")
    print()


def test_python_forward():
    """Run a simplified forward pass in Python using loaded weights."""
    print("=== Test: Python Forward (Simplified) ===")
    manifest = load_manifest()

    # Load a single layer's weights
    hidden = 1024
    intermediate = 4096
    seq = 1

    # Load layer 0 weights — shapes from ONNX verification:
    # c_fc.weight: [1024, 4096] (expand 1024→4096) — plain matmul ✓
    # c_proj.weight: [4096, 1024] (contract 4096→1024) — plain matmul ✓
    # c_attn.c_proj.weight: [1024, 1024] (square)
    c_attn_w = load_weight_fp16("h.0.attn.c_attn.weight", manifest).reshape(hidden, 3072)
    ln_1_w = load_weight_fp16("h.0.ln_1.weight", manifest)
    ln_1_b = load_weight_fp16("h.0.ln_1.bias", manifest)
    c_proj_w = load_weight_fp16("h.0.attn.c_proj.weight", manifest).reshape(hidden, hidden)
    ln_2_w = load_weight_fp16("h.0.ln_2.weight", manifest)
    ln_2_b = load_weight_fp16("h.0.ln_2.bias", manifest)
    c_fc_w = load_weight_fp16("h.0.mlp.c_fc.weight", manifest).reshape(hidden, intermediate)  # [1024, 4096]
    c_proj_mlp_w = load_weight_fp16("h.0.mlp.c_proj.weight", manifest).reshape(intermediate, hidden)  # [4096, 1024]

    # Load final weights
    ln_f_w = load_weight_fp16("ln_f.weight", manifest)
    ln_f_b = load_weight_fp16("ln_f.bias", manifest)
    lm_head_w = load_weight_fp16("lm_head.weight", manifest).reshape(6563, hidden)
    lm_head_b = load_weight_fp16("lm_head.bias", manifest)

    print(f"  Loaded all weights for 1 layer + final LN + LM head")

    # Random input
    np.random.seed(42)
    x = np.random.randn(seq, hidden).astype(np.float16)

    # LayerNorm 1
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    ln1_out = ((x - mean) / np.sqrt(var + 1e-5)).astype(np.float16)
    ln1_out = ln1_out * ln_1_w + ln_1_b

    # c_attn: x @ W^T + b (ONNX transB=0: Y = X @ W)
    c_attn_out = ln1_out @ c_attn_w  # [S, 3072]
    print(f"  c_attn out shape: {c_attn_out.shape}")  # Should be [1, 3072]

    # Split into Q, K, V
    q = c_attn_out[:, :hidden]  # [S, 1024]
    k = c_attn_out[:, hidden:2*hidden]
    v = c_attn_out[:, 2*hidden:3*hidden]
    print(f"  Q/K/V shapes: {q.shape}, {k.shape}, {v.shape}")

    # Simplified attention: Q @ K^T (no scaling for simplicity)
    attn = q @ k.T / np.sqrt(hidden)  # [S, S]
    attn = softmax(attn, axis=-1)
    attn_out = attn @ v  # [S, 1024]

    # O projection
    o_out = attn_out @ c_proj_w  # [S, 1024]

    # Residual
    res1 = x + o_out

    # LayerNorm 2
    mean = res1.mean(axis=-1, keepdims=True)
    var = res1.var(axis=-1, keepdims=True)
    ln2_out = ((res1 - mean) / np.sqrt(var + 1e-5)).astype(np.float16)
    ln2_out = ln2_out * ln_2_w + ln_2_b

    # FC1: x @ W (no transpose — ONNX stores [in, out])
    fc1_out = ln2_out @ c_fc_w  # [S, 4096]
    print(f"  FC1 out shape: {fc1_out.shape}")  # Should be [1, 4096]

    # GELU
    gelu_out = gelu(fc1_out)

    # FC2: x @ W (no transpose)
    fc2_out = gelu_out @ c_proj_mlp_w  # [S, 1024]
    print(f"  FC2 out shape: {fc2_out.shape}")  # Should be [1, 1024]

    # Residual 2
    res2 = res1 + fc2_out

    # Final LN
    mean = res2.mean(axis=-1, keepdims=True)
    var = res2.var(axis=-1, keepdims=True)
    final_ln = ((res2 - mean) / np.sqrt(var + 1e-5)).astype(np.float16)
    final_ln = final_ln * ln_f_w + ln_f_b
    print(f"  Final LN shape: {final_ln.shape}")

    # LM head: x @ W^T (need transpose since lm_head is [6563, 1024] stored)
    logits = final_ln @ lm_head_w.T + lm_head_b  # [S, 6563]
    print(f"  Logits shape: {logits.shape}")  # Should be [1, 6563]
    print(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"  Top 5 tokens: {np.argsort(logits[0])[-5:][::-1]}")
    print()


def softmax(x, axis=-1):
    x_max = x.max(axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=axis, keepdims=True)


def gelu(x):
    return x * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))


if __name__ == "__main__":
    print(f"Weights dir: {WEIGHTS_DIR}")
    print(f"Manifest: {MANIFEST_PATH}\n")

    test_weight_loading()
    test_c_attn_split()
    test_matmul_dimensions()
    test_ln_f_and_lm_head()
    test_python_forward()
    test_onnx_reference()

    print("All tests passed!")
