// Causal mask kernel: mask[row, col] = 0 if col <= row else -inf
// Dispatch: threadgroups = ceil(seqLen * seqLen / 1024), threads_per_tg = 1024
kernel void causal_mask_kernel(
    device float* mask     [[buffer(0)]],
    constant uint& seq_len [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = seq_len * seq_len;
    if (gid >= total) return;
    uint row = gid / seq_len;
    uint col = gid % seq_len;
    mask[gid] = (col <= row) ? 0.0f : -1e9f;
}

// Precompute RoPE cos/sin tables at init
kernel void compute_rope_lut(
    device float* cos_lut  [[buffer(0)]],
    device float* sin_lut  [[buffer(1)]],
    constant uint& max_seq [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint pos = gid.x;
    uint dim = gid.y;
    if (pos >= max_seq || dim >= head_dim / 2) return;
    float theta = 10000.0f;
    float inv_freq = 1.0f / pow(theta, float(2 * dim) / float(head_dim));
    float angle = float(pos) * inv_freq;
    cos_lut[pos * (head_dim/2) + dim] = cos(angle);
    sin_lut[pos * (head_dim/2) + dim] = sin(angle);
}