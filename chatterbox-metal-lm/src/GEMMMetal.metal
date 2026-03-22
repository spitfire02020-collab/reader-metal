// C = A @ B (A[M,K] * B[K,N] → C[M,N])
kernel void gemm_nn(
    device const half* A    [[buffer(0)]],
    device const half* B    [[buffer(1)]],
    device half*       C    [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= N || gid.y >= M) return;
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += float(A[gid.y * K + k]) * float(B[k * N + gid.x]);
    }
    C[gid.y * N + gid.x] = half(sum);
}

// C = A @ B^T (A[M,K] * B[K,N] → C[M,N])
// B is stored in row-major [K,N] layout in memory (ONNX convention).
// B^T[k, n] = B[n, k] = B[n*K + k] for row-major B.
kernel void gemm_nt(
    device const half* A    [[buffer(0)]],
    device const half* B    [[buffer(1)]],
    device half*       C    [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= N || gid.y >= M) return;
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        // B^T[k, gid.x] = B[gid.x, k] = B[gid.x * K + k]
        sum += float(A[gid.y * K + k]) * float(B[gid.x * K + k]);
    }
    C[gid.y * N + gid.x] = half(sum);
}

// out = a + b (element-wise, GPU kernel to avoid CPU loop)
kernel void residual_add(
    device const half* a    [[buffer(0)]],
    device const half* b    [[buffer(1)]],
    device half*       out  [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    out[gid] = half(float(a[gid]) + float(b[gid]));
}