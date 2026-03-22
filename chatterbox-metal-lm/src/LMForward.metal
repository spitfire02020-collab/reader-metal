// LMForward.metal — GPT-2 Layer Forward Kernel
//
// Architecture: One dispatch per layer from MetalLMBackend (Swift).
// Swift dispatches individual kernels (GEMM, LayerNorm, TanhGelu, Attention)
// via helper methods on MTLComputeCommandEncoder.
//
// This file documents the per-layer data flow:
//   input (pre-embedded)
//     → preLN(ln1_w, ln1_b)
//     → QKV proj (W_qkv^T)
//     → Q, K, V unpack
//     → RoPE(Q) + RoPE(K)   [via rope_cos/sin LUT]
//     → Attention(Q, K, V)  → attn_out
//     → O proj (W_o^T)
//     → residual_add(input, attn_out)
//     → postLN(ln2_w, ln2_b)
//     → FFN gate (W_1^T) → gelu
//     → FFN up (W_3^T)   → multiply
//     → residual_add(hidden, ffn_out)
//
// The 24-layer autoregressive decode loop runs this per token:
//   for layer in 0..<24:
//       hidden = lm_layer_forward(hidden, layer_idx=layer)
//
// Constants
constant int NUM_LAYERS     = 24;
constant int NUM_Q_HEADS    = 80;   // GPT-2 Medium GQA
constant int NUM_KV_HEADS  = 16;
constant int HEAD_DIM       = 64;
constant int HIDDEN_SIZE    = 1024; // n_embd
constant int INTERMEDIATE   = 4096; // n_inner (gelu intermediate)
constant int VOCAB_SIZE     = 6563;
constant int MAX_SEQ_LEN    = 1500;
constant int QKV_PROJ_OUT   = (NUM_Q_HEADS + NUM_KV_HEADS * 2) * HEAD_DIM; // 80+32=112 heads × 64 = 7168

// MARK: - RoPE Apply Kernel

// Apply Rotary Position Embedding to Q and K at a specific sequence position.
// RoPE formula (per dimension pair d=0,2,4,...):
//   x_rot[2d]   = x[2d]   * cos(θ) - x[2d+1] * sin(θ)
//   x_rot[2d+1] = x[2d+1] * cos(θ) + x[2d]   * sin(θ)
// where θ = pos * inv_freq(d), inv_freq(d) = 1 / 10000^(2d/dim)
//
// Buffer layouts:
//   Q: [numQHeads, headDim] = [80, 64] — modified in-place
//   K: [numKVHeads, headDim] = [16, 64] — written back to K buffer (single position slot)
//   cos_lut / sin_lut: [maxSeq, headDim/2] — precomputed by compute_rope_lut
//
// Grid: (numQHeads + numKVHeads, headDim/2)
// Threadgroup: (16, 4) — 64 threads, handles 4 dim-pairs per thread
kernel void rope_apply_kernel(
    device half* q             [[buffer(0)]],  // Q [numQHeads, headDim] — in/out
    device half* k             [[buffer(1)]],  // K write [numKVHeads, headDim] — out (single pos)
    device const float* cos_lut [[buffer(2)]],  // cos LUT [maxSeq, headDim/2]
    device const float* sin_lut [[buffer(3)]],  // sin LUT [maxSeq, headDim/2]
    constant uint& position     [[buffer(4)]],  // sequence position for this step
    constant uint& max_seq     [[buffer(5)]],  // max sequence length
    constant uint& num_q_heads [[buffer(6)]],  // number of Q heads (80)
    constant uint& num_kv_heads [[buffer(7)]], // number of KV heads (16)
    constant uint& head_dim    [[buffer(8)]],  // head dimension (64)
    uint2 gid [[thread_position_in_grid]]
) {
    uint totalHeads = num_q_heads + num_kv_heads;
    uint head_idx = gid.x;
    uint dim_pair = gid.y;  // 0..31 for head_dim=64

    if (head_idx >= totalHeads || dim_pair >= head_dim / 2) return;

    bool isQ = head_idx < num_q_heads;
    uint localHead = isQ ? head_idx : (head_idx - num_q_heads);

    // Select buffer and base index
    device half* buf = isQ ? q : k;
    uint bufBase = localHead * head_dim;
    uint rotDim = dim_pair * 2;  // 0, 2, 4, ...

    // LUT index: pos * (head_dim/2) + dim_pair
    uint lutIdx = position * (head_dim / 2) + dim_pair;
    float cos_theta = cos_lut[lutIdx];
    float sin_theta = sin_lut[lutIdx];

    // Apply rotation to dimension pair (rotDim, rotDim+1)
    half x0 = buf[bufBase + rotDim];
    half x1 = buf[bufBase + rotDim + 1];

    buf[bufBase + rotDim]     = half(float(x0) * cos_theta - float(x1) * sin_theta);
    buf[bufBase + rotDim + 1] = half(float(x1) * cos_theta + float(x0) * sin_theta);
}
