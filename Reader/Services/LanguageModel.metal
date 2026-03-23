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
// Each thread processes (dim / 256) elements in a strided loop.
// Threadgroup barrier + parallel reduction for mean and variance.
kernel void layer_norm(
    device const half*  input  [[buffer(0)]],
    device const half*  gamma  [[buffer(1)]],  // weight [dim]
    device const half*  beta   [[buffer(2)]],  // bias [dim]
    device half*         output [[buffer(3)]],
    constant uint&       dim    [[buffer(4)]],
    constant half&       eps    [[buffer(5)]],
    uint                gid [[thread_position_in_grid]],
    uint                tid [[thread_position_in_threadgroup]]) {
    // Total threads = 256, dim = 1024, each thread handles 4 elements (stride 256)
    uint numThreads = 256;
    uint elemsPerThread = dim / numThreads;  // 4

    // Per-thread partial sum of input values
    half threadSum = 0;
    half threadSqSum = 0;
    uint base = tid;
    for (uint k = 0; k < elemsPerThread; k++) {
        uint idx = base + k * numThreads;
        half val = input[idx];
        threadSum += val;
        half d = val * val;
        threadSqSum += d;
    }

    // Parallel reduction in threadgroup shared memory
    threadgroup half sdata[256];
    threadgroup half sq_sdata[256];
    sdata[tid] = threadSum;
    sq_sdata[tid] = threadSqSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // In-place reduction: accumulate into sdata[0]
    if (tid >= 128) { sdata[0] += sdata[tid]; sq_sdata[0] += sq_sdata[tid]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 128) { sdata[0] += sdata[tid + 128]; sq_sdata[0] += sq_sdata[tid + 128]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 64) { sdata[0] += sdata[tid + 64]; sq_sdata[0] += sq_sdata[tid + 64]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) { sdata[0] += sdata[tid + 32]; sq_sdata[0] += sq_sdata[tid + 32]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16) { sdata[0] += sdata[tid + 16]; sq_sdata[0] += sq_sdata[tid + 16]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8) { sdata[0] += sdata[tid + 8]; sq_sdata[0] += sq_sdata[tid + 8]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 4) { sdata[0] += sdata[tid + 4]; sq_sdata[0] += sq_sdata[tid + 4]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 2) { sdata[0] += sdata[tid + 2]; sq_sdata[0] += sq_sdata[tid + 2]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 1) { sdata[0] += sdata[tid + 1]; sq_sdata[0] += sq_sdata[tid + 1]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 broadcasts mean and inv_std to all threads
    half mean = sdata[0] / half(dim);
    half var = sq_sdata[0] / half(dim) - mean * mean;
    half inv_std = metal::rsqrt(var + eps);

    // Each thread writes its portion: normalize + affine transform
    for (uint k = 0; k < elemsPerThread; k++) {
        uint outIdx = base + k * numThreads;
        half x = input[outIdx];
        half norm = (x - mean) * inv_std;
        output[outIdx] = norm * gamma[outIdx] + beta[outIdx];
    }
}