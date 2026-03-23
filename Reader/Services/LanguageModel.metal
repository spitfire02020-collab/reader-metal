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
// Dispatch: numThreadGroups=(1,1,1), threadsPerThreadgroup=(256,1,1)
// Each thread processes dim/256 elements in a strided loop.
// All threads ALWAYS participate in barriers (no early-exit before barrier).
//
// CRITICAL: Accumulation uses float (float32) to avoid float16 overflow.
// Per-thread x² sums (e.g. 1²+257²+513²+769² ≈ 943,000) exceed float16's
// max (~65,504), causing infinity overflow → NaN. float32 handles up to 3.4e38.
kernel void layer_norm(
    device const half*  input  [[buffer(0)]],
    device const half*  gamma  [[buffer(1)]],  // weight [dim]
    device const half*  beta   [[buffer(2)]],  // bias [dim]
    device half*         output [[buffer(3)]],
    constant uint&       dim    [[buffer(4)]],
    constant half&       eps    [[buffer(5)]],
    uint                tid [[thread_position_in_threadgroup]]) {
    uint numThreads = 256;
    uint elemsPerThread = dim / numThreads;  // 4 for dim=1024

    // Per-thread partial sum in float to prevent float16 overflow
    // (max x² sum ≈ 4*1024² ≈ 4.2M, well within float32 range)
    float threadSum = 0;
    float threadSqSum = 0;
    for (uint k = 0; k < elemsPerThread; k++) {
        uint idx = tid + k * numThreads;
        float val = float(input[idx]);
        threadSum += val;
        threadSqSum += val * val;
    }

    // Reduction tree in threadgroup shared memory (float32)
    threadgroup float sbuf[256];
    threadgroup float sqbuf[256];
    sbuf[tid] = threadSum;
    sqbuf[tid] = threadSqSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree-reduction: 256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1
    // Threads 128-255 skip the if() body but reach every barrier.
    if (tid < 128) { sbuf[tid] += sbuf[tid + 128]; sqbuf[tid] += sqbuf[tid + 128]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 64) { sbuf[tid] += sbuf[tid + 64]; sqbuf[tid] += sqbuf[tid + 64]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) { sbuf[tid] += sbuf[tid + 32]; sqbuf[tid] += sqbuf[tid + 32]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16) { sbuf[tid] += sbuf[tid + 16]; sqbuf[tid] += sqbuf[tid + 16]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8) { sbuf[tid] += sbuf[tid + 8]; sqbuf[tid] += sqbuf[tid + 8]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 4) { sbuf[tid] += sbuf[tid + 4]; sqbuf[tid] += sqbuf[tid + 4]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 2) { sbuf[tid] += sbuf[tid + 2]; sqbuf[tid] += sqbuf[tid + 2]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 1) { sbuf[tid] += sbuf[tid + 1]; sqbuf[tid] += sqbuf[tid + 1]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // sbuf[0] = sum, sqbuf[0] = sum of squares
    float sum = sbuf[0];
    float sq_sum = sqbuf[0];
    float mean = sum / float(dim);
    float var = sq_sum / float(dim) - mean * mean;
    float inv_std = metal::rsqrt(var + float(eps));

    // Each thread normalizes and writes its slice (convert back to half)
    for (uint k = 0; k < elemsPerThread; k++) {
        uint idx = tid + k * numThreads;
        float x = float(input[idx]);
        float norm = (x - mean) * inv_std;
        output[idx] = half(norm * float(gamma[idx]) + float(beta[idx]));
    }
}