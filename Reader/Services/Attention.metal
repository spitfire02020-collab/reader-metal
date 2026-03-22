#include <metal_stdlib>
using namespace metal;

// Standard Multi-Head Attention kernels for GPT2NoEmbed
// NUM_Q_HEADS=16, NUM_KV_HEADS=16, HEAD_DIM=64
// Each Q head attends to its corresponding KV head (ratio=1, no sharing)

constant int NUM_Q_HEADS = 16;
constant int HEAD_DIM = 64;
constant float ATTN_SCALE = 0.125f;  // 1.0 / sqrt(64)

// Flat buffer layouts (row-major):
// Q:       [NUM_Q_HEADS * HEAD_DIM]       -> q_head * HEAD_DIM + d
// K:       [NUM_KV_HEADS * maxSeqLen * HEAD_DIM] -> kv_head * maxSeqLen * HEAD_DIM + pos * HEAD_DIM + d
// V:       [NUM_KV_HEADS * maxSeqLen * HEAD_DIM] -> same as K
// Output:  [NUM_Q_HEADS * HEAD_DIM]       -> q_head * HEAD_DIM + d

kernel void attention_decode_step(
    device const half* q       [[buffer(0)]],  // [NUM_Q_HEADS, HEAD_DIM] = [16, 64]
    device const half* k       [[buffer(1)]],  // [NUM_KV_HEADS, maxSeqLen, HEAD_DIM] = [16, maxSeqLen, 64]
    device const half* v       [[buffer(2)]],  // [NUM_KV_HEADS, maxSeqLen, HEAD_DIM] = [16, maxSeqLen, 64]
    device half*         output  [[buffer(3)]],  // [NUM_Q_HEADS, HEAD_DIM] = [16, 64]
    constant uint&       kv_len  [[buffer(4)]],  // current read length (positions in KV cache)
    constant uint&       max_seq [[buffer(5)]],  // allocated buffer length (maxSeqLen)
    uint2 gid [[thread_position_in_grid]]
)
{
    // gid.x = q_head index (0 .. 15)
    // gid.y = dimension index (0 .. 63)

    if (gid.x >= NUM_Q_HEADS || gid.y >= HEAD_DIM) {
        return;
    }

    uint q_head = gid.x;
    uint d = gid.y;

    // MHA: each Q head maps 1:1 to its KV head
    uint kv_head = q_head;

    // Get query vector for this head and dimension
    half q_val = q[q_head * HEAD_DIM + d];

    float weight_sum = 0.0f;
    float normalizer = 0.0f;
    float max_score = -INFINITY;

    // First pass: find max score for numerical stability
    for (uint pos = 0; pos < kv_len; ++pos) {
        // K layout: kv_head * max_seq * HEAD_DIM + pos * HEAD_DIM + d
        uint k_idx = kv_head * max_seq * HEAD_DIM + pos * HEAD_DIM + d;

        half k_val = k[k_idx];
        half score_half = q_val * k_val;
        float score = float(score_half) * ATTN_SCALE;

        if (score > max_score) {
            max_score = score;
        }
    }

    // Second pass: compute weighted sum with exp
    for (uint pos = 0; pos < kv_len; ++pos) {
        // K layout: kv_head * max_seq * HEAD_DIM + pos * HEAD_DIM + d
        uint k_idx = kv_head * max_seq * HEAD_DIM + pos * HEAD_DIM + d;

        half k_val = k[k_idx];
        half score_half = q_val * k_val;
        float score = float(score_half) * ATTN_SCALE;

        float exp_score = exp(score - max_score);
        normalizer += exp_score;

        // V layout: same as K
        uint v_idx = kv_head * max_seq * HEAD_DIM + pos * HEAD_DIM + d;
        half v_val = v[v_idx];

        weight_sum += exp_score * float(v_val);
    }

    // Normalize
    float output_val = (normalizer > 0.0f) ? (weight_sum / normalizer) : 0.0f;

    // Write output: q_head * HEAD_DIM + d
    output[q_head * HEAD_DIM + d] = half(output_val);
}

// Prefill kernel: full softmax over entire sequence
// For prefill with longer sequences. Each thread handles one (q_head, seq_pos) pair.
kernel void group_query_attention(
    device const half* Q       [[buffer(0)]],  // [1, NUM_Q_HEADS, seq, HEAD_DIM] = [1, 16, seq, 64]
    device const half* K       [[buffer(1)]],  // [1, NUM_KV_HEADS, seq, HEAD_DIM] = [1, 16, seq, 64]
    device const half* V       [[buffer(2)]],  // [1, NUM_KV_HEADS, seq, HEAD_DIM] = [1, 16, seq, 64]
    device float*       output  [[buffer(3)]],  // [1, NUM_Q_HEADS, seq, HEAD_DIM] = [1, 16, seq, 64]
    constant uint&       seq_len [[buffer(4)]], // sequence length
    uint2 gid [[thread_position_in_grid]]
)
{
    // gid.x = q_head index (0 .. 15)
    // gid.y = seq_pos index (0 .. seq_len-1)

    if (gid.x >= NUM_Q_HEADS || gid.y >= seq_len) {
        return;
    }

    uint q_head = gid.x;
    uint seq_pos = gid.y;

    // MHA: each Q head maps 1:1 to its KV head
    uint kv_head = q_head;

    // Q layout: 1 * NUM_Q_HEADS * seq * HEAD_DIM + 0 * ... + q_head * seq * HEAD_DIM + seq_pos * HEAD_DIM + d
    // Simplified: (q_head * seq_len + seq_pos) * HEAD_DIM + d
    uint q_base = (q_head * seq_len + seq_pos) * HEAD_DIM;

    float max_score = -INFINITY;

    // First pass: find max score for numerical stability
    for (uint pos = 0; pos < seq_len; ++pos) {
        // K layout: 1 * NUM_KV_HEADS * seq * HEAD_DIM + 0 * ... + kv_head * seq * HEAD_DIM + pos * HEAD_DIM + d
        // Simplified: (kv_head * seq_len + pos) * HEAD_DIM + d
        uint k_base = (kv_head * seq_len + pos) * HEAD_DIM;

        // Compute dot product for this position
        float dot_prod = 0.0f;
        for (uint d = 0; d < HEAD_DIM; ++d) {
            half q_val = Q[q_base + d];
            half k_val = K[k_base + d];
            dot_prod += float(q_val) * float(k_val);
        }

        float score = dot_prod * ATTN_SCALE;
        if (score > max_score) {
            max_score = score;
        }
    }

    // Second pass: compute weighted sum with exp
    float weight_sum[HEAD_DIM];
    float normalizer = 0.0f;

    for (uint d = 0; d < HEAD_DIM; ++d) {
        weight_sum[d] = 0.0f;
    }

    for (uint pos = 0; pos < seq_len; ++pos) {
        uint k_base = (kv_head * seq_len + pos) * HEAD_DIM;

        // Compute dot product for this position
        float dot_prod = 0.0f;
        for (uint d = 0; d < HEAD_DIM; ++d) {
            half q_val = Q[q_base + d];
            half k_val = K[k_base + d];
            dot_prod += float(q_val) * float(k_val);
        }

        float score = dot_prod * ATTN_SCALE;
        float exp_score = exp(score - max_score);
        normalizer += exp_score;

        // V layout: same as K
        uint v_base = (kv_head * seq_len + pos) * HEAD_DIM;
        for (uint d = 0; d < HEAD_DIM; ++d) {
            half v_val = V[v_base + d];
            weight_sum[d] += exp_score * float(v_val);
        }
    }

    // Normalize and write output
    // Output layout: (q_head * seq_len + seq_pos) * HEAD_DIM + d
    uint out_base = (q_head * seq_len + seq_pos) * HEAD_DIM;
    for (uint d = 0; d < HEAD_DIM; ++d) {
        float output_val = (normalizer > 0.0f) ? (weight_sum[d] / normalizer) : 0.0f;
        output[out_base + d] = output_val;
    }
}
