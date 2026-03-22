#include <metal_stdlib>
using namespace metal;

constant float A = 0.797885f;
constant float B = 0.044715f;

inline float gelu_tanh(float x) {
    float x3 = x * x * x;
    float inner = A * x + B * x3;
    return 0.5f * x * (1.0f + tanh(inner));
}

kernel void tanh_gelu_kernel(
    device const half* input  [[buffer(0)]],
    device half*       output [[buffer(1)]],
    constant uint&     size   [[buffer(2)]],
    uint               gid    [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    float x = float(input[gid]);
    output[gid] = half(gelu_tanh(x));
}

kernel void tanh_gelu_inplace(
    device half* data  [[buffer(0)]],
    constant uint& size [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    float x = float(data[gid]);
    data[gid] = half(gelu_tanh(x));
}