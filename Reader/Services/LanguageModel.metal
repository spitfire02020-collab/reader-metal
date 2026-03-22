#include <metal_stdlib>
using namespace metal;

// Q4F16 Block-wise Dequantization Kernel
// ACTUAL format (confirmed from ONNX inspection):
//   quant:      uint8  [out_dim, block_count, 16]  — 1 byte per element
//   scales:     float16 [out_dim, block_count]
//   zero_points: uint8   [out_dim, block_count/16]  — 1 zp per 16 blocks, NOT float16!
// Out: fp16 output [out_dim, block_count*16]
//
// Dequantization: out[row, block, col] = (quant[row,block,col] - zp[row,block/16]) * scale[row,block]

kernel void dequant_q4f16(
    device const uint8_t* quant    [[buffer(0)]],
    device const half*     scales   [[buffer(1)]],
    device const uint8_t* zp       [[buffer(2)]],  // uint8 NOT half!
    device half*           output   [[buffer(3)]],
    constant uint&         out_dim  [[buffer(4)]],
    constant uint&         block_count [[buffer(5)]],
    uint2                 gid [[thread_position_in_grid]]) {
    uint row = gid.y;
    uint block_idx = gid.x;
    if (row >= out_dim || block_idx >= block_count) return;

    uint quant_base = (row * block_count + block_idx) * 16;
    uint scale_idx = row * block_count + block_idx;
    uint zp_idx = row * (block_count / 16) + block_idx / 16;
    half scale = scales[scale_idx];
    half zero = half(zp[zp_idx]);  // uint8 → float16

    uint out_base = row * block_count * 16 + block_idx * 16;
    for (uint col = 0; col < 16; col++) {
        uint quant_idx = quant_base + col;
        half val = half(quant[quant_idx]) - zero;
        output[out_base + col] = val * scale;
    }
}

// LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
kernel void layer_norm(
    device const half*  input  [[buffer(0)]],
    device const half*  gamma  [[buffer(1)]],  // weight [dim]
    device const half*  beta   [[buffer(2)]],  // bias [dim]
    device half*         output [[buffer(3)]],
    constant uint&       dim    [[buffer(4)]],
    constant half&       eps    [[buffer(5)]],
    uint2               gid [[thread_position_in_grid]]) {
    uint row = gid.y;
    uint batch_size = gid.x;  // batch = grid y
    if (row >= batch_size) return;

    uint offset = row * dim;

    // Compute mean
    half sum = 0;
    for (uint i = 0; i < dim; i++) sum += input[offset + i];
    half mean = sum / half(dim);

    // Compute variance
    half sq_sum = 0;
    for (uint i = 0; i < dim; i++) {
        half diff = input[offset + i] - mean;
        sq_sum += diff * diff;
    }
    half var = sq_sum / half(dim);
    half inv_std = metal::rsqrt(var + eps);

    // Normalize + affine
    for (uint i = 0; i < dim; i++) {
        half x = input[offset + i];
        half norm = (x - mean) * inv_std;
        output[offset + i] = norm * gamma[i] + beta[i];
    }
}