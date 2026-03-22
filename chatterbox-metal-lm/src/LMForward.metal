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
