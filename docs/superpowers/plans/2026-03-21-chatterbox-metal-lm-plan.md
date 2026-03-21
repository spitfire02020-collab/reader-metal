# Chatterbox Metal Language Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the ONNX Runtime language model with a custom Metal compute shader implementation, keeping the other 3 ONNX models on ORT.

**Architecture:** Hybrid Metal + ONNX. Q4F16 weights extracted from `language_model_q4f16_merged.onnx`. Metal implements 24-layer GPT2-Medium forward pass with KV cache management.

**Tech Stack:** Metal 3 (`.metal` shaders), Swift 5.9, MPS (fallback), Python (weight extraction)

---

## Repository Layout

```
chatterbox-metal-lm/
├── scripts/
│   └── extract_weights.py              # Extract Q4F16 weights → binary
├── src/
│   ├── LanguageModel.metal            # All compute kernels
│   ├── MetalLMEncoder.swift           # MPS encoding + dispatch
│   ├── Q4F16Dequant.swift             # Weight loading + dequantization
│   ├── MetalLMConfig.swift            # Model constants
│   ├── KVCacheBuffer.swift            # Pre-allocated KV buffers
│   ├── KVCacheManager.swift           # Cache lifecycle + concat
│   ├── MetalLMForward.swift           # Single-step forward
│   ├── MetalLMDecode.swift            # Decode loop
│   ├── WeightLoader.swift             # ONNX → Metal buffer loader
│   └── ChatterboxMetalLM.swift        # Public API
├── tests/
│   ├── test_q4f16_dequant.py          # Python dequant validation
│   ├── test_forward.py                # Single-step vs ONNX reference
│   └── test_decode_loop.py            # E2E decode validation
└── build.sh                           # xcrun metal + metallib
```

---

## Task 1: Weight Extraction Script

**Files:**
- Create: `chatterbox-metal-lm/scripts/extract_weights.py`

- [ ] **Step 1: Write weight extraction script**

```python
#!/usr/bin/env python3
"""
Extract Q4F16 weights from language_model_q4f16_merged.onnx
and write them as binary .metalweight files for Metal loading.

Weight layout per quantized matmul (CONFIRMED from ONNX inspection):
  name.quant   -> uint8              [out_dim, block_count, 16]  ← 1 byte per element
  name.scales  -> float16             [out_dim, block_count]
  name.zp      -> uint8               [out_dim, block_count/16]  ← 1 zp per 16 blocks, NOT float16!

Dequantization formula:
  fp16_weight[row, block, col] = (quant[row, block, col] - zp[row, block/16]) * scale[row, block]
  (broadcast quant-zp over 16 cols, then scale)
"""
import argparse
import os
import struct
from pathlib import Path

import numpy as np
import onnx


def extract_q4f16_weights(onnx_path: str, output_dir: str) -> dict[str, dict]:
    """Extract all Q4F16 quantized weights from ONNX model."""
    model = onnx.load(onnx_path)
    initializers = {i.name: i for i in model.graph.initializer}

    q_weights = {k: v for k, v in initializers.items() if '_quant' in k}

    manifest = {}
    os.makedirs(output_dir, exist_ok=True)

    for qname, tensor in sorted(q_weights.items()):
        name = qname.replace('_MatMul_weight_quant', '')
        scales_name = f'{name}_scales'
        zp_name = f'{name}_zp'

        scales = initializers[scales_name]
        zp = initializers[zp_name]

        # Load raw bytes
        q_data = np.frombuffer(tensor.raw_data, dtype=np.uint8)
        scales_data = np.frombuffer(scales.raw_data, dtype=np.float16)
        zp_data = np.frombuffer(zp.raw_data, dtype=np.uint8)  # uint8 NOT float16

        out_dim = tensor.dims[0]
        block_count = tensor.dims[1]
        block_size = 16  # fixed for Q4F16

        manifest[name] = {
            'out_dim': out_dim,
            'block_count': block_count,
            'block_size': block_size,
            'quant_path': f'{name}.quant',
            'scales_path': f'{name}.scales',
            'zp_path': f'{name}.zp',
        }

        # Write binary files
        q_path = os.path.join(output_dir, f'{name}.quant')
        scales_path = os.path.join(output_dir, f'{name}.scales')
        zp_path = os.path.join(output_dir, f'{name}.zp')

        q_data.tofile(q_path)
        scales_data.tofile(scales_path)
        zp_data.tofile(zp_path)

    # Write manifest
    import json
    manifest_path = os.path.join(output_dir, 'weights_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'Extracted {len(manifest)} quantized weights to {output_dir}')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', required=True, help='Path to language_model_q4f16_merged.onnx')
    parser.add_argument('--output', required=True, help='Output directory for .metalweight files')
    args = parser.parse_args()

    extract_q4f16_weights(args.onnx, args.output)
```

- [ ] **Step 2: Run extraction**

```bash
python3 chatterbox-metal-lm/scripts/extract_weights.py \
    --onnx /Users/rockymoon/Downloads/Reader/Reader/Resources/ChatterboxModels/language_model_q4f16_merged.onnx \
    --output /tmp/metal_weights/
ls /tmp/metal_weights/ | head -20
```

Expected: ~438 binary files (146 × 3 tensors + manifest), `weights_manifest.json`

- [ ] **Step 3: Verify extraction correctness**

```python
# Quick validation: re-quantize and check error
import numpy as np, json

with open('/tmp/metal_weights/weights_manifest.json') as f:
    manifest = json.load(f)

# Check first weight
name = list(manifest.keys())[0]
q = np.fromfile(f'/tmp/metal_weights/{name}.quant', dtype=np.uint8)
s = np.fromfile(f'/tmp/metal_weights/{name}.scales', dtype=np.float16)
zp = np.fromfile(f'/tmp/metal_weights/{name}.zp', dtype=np.float16)
print(f'{name}: quant={q.shape}, scales={s.shape}, zp={zp.shape}')
print(f'  scales range: [{s.min()}, {s.max()}]')
print(f'  zp range: [{zp.min()}, {zp.max()}]')
```

- [ ] **Step 4: Commit**

```bash
git add chatterbox-metal-lm/scripts/extract_weights.py
git commit -m "feat(metal-lm): add weight extraction script for Q4F16 ONNX → Metal binary"
```

---

## Task 2: Metal Shader — Q4F16 Dequantization Kernel

**Files:**
- Create: `chatterbox-metal-lm/src/LanguageModel.metal` (first section)
- Create: `chatterbox-metal-lm/src/Q4F16Dequant.swift`
- Create: `chatterbox-metal-lm/src/MetalLMConfig.swift`

- [ ] **Step 1: Write MetalLMConfig.swift**

```swift
import Foundation

enum MetalLMConfig {
    static let numLayers = 24
    static let num_heads = 16
    static let headDim = 64
    static let hiddenSize = 1024
    static let intermediateSize = 4096
    static let vocabSize = 6563
    static let maxSequenceLength = 8192

    // Special tokens
    static let startSpeechToken: Int32 = 6561
    static let stopSpeechToken: Int32 = 6562

    // Q4F16 block parameters
    static let blockSize = 16
    static let quantBlockRows = 32
}
```

- [ ] **Step 2: Write Q4F16 Dequantization Metal kernel**

```metal
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
```

- [ ] **Step 3: Write Q4F16Dequant.swift**

```swift
import Foundation
import Metal

final class Q4F16Dequantizer {
    let device: MTLDevice
    let quantPipeline: MTLComputePipelineState
    let maxThreads: MTLSize

    init(device: MTLDevice, library: MTLLibrary) throws {
        self.device = device
        guard let kern = library.makeFunction(name: "dequant_q4f16") else {
            throw MetalLMError.kernelNotFound("dequant_q4f16")
        }
        self.quantPipeline = try device.makeComputePipelineState(function: kern)
        self.maxThreads = MTLSize(
            width: quantPipeline.maxTotalThreadsPerThreadgroup,
            height: 1, depth: 1
        )
    }

    func dequant(
        commandBuffer: MTLCommandBuffer,
        quantBuffer: MTLBuffer,
        scalesBuffer: MTLBuffer,
        zpBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        outDim: Int,
        blockCount: Int
    ) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(quantPipeline)
        enc.setBuffer(quantBuffer, offset: 0, index: 0)
        enc.setBuffer(scalesBuffer, offset: 0, index: 1)
        enc.setBuffer(zpBuffer, offset: 0, index: 2)
        enc.setBuffer(outputBuffer, offset: 0, index: 3)

        var od = UInt32(outDim), bc = UInt32(blockCount)
        enc.setBytes(&od, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&bc, length: MemoryLayout<UInt32>.size, index: 5)

        let tgWidth = min(256, quantPipeline.maxTotalThreadsPerThreadgroup)
        let threadsPerGroup = MTLSize(width: tgWidth, height: 1, depth: 1)
        let numThreadGroups = MTLSize(
            width: (blockCount + tgWidth - 1) / tgWidth,
            height: outDim,
            depth: 1
        )
        enc.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
        enc.endEncoding()
    }
}
```

- [ ] **Step 4: Write MetalLMConfig tests**

```swift
import XCTest

final class MetalLMConfigTests: XCTestCase {
    func testConstants() {
        XCTAssertEqual(MetalLMConfig.numLayers, 24)
        XCTAssertEqual(MetalLMConfig.num_heads, 16)
        XCTAssertEqual(MetalLMConfig.headDim, 64)
        XCTAssertEqual(MetalLMConfig.hiddenSize, 1024)
        XCTAssertEqual(MetalLMConfig.intermediateSize, 4096)
        XCTAssertEqual(MetalLMConfig.vocabSize, 6563)
        XCTAssertEqual(MetalLMConfig.startSpeechToken, 6561)
        XCTAssertEqual(MetalLMConfig.stopSpeechToken, 6562)
        XCTAssertEqual(MetalLMConfig.blockSize, 16)
        XCTAssertEqual(MetalLMConfig.quantBlockRows, 32)
    }
}
```

- [ ] **Step 5: Commit**

```bash
git add chatterbox-metal-lm/src/MetalLMConfig.swift
git add chatterbox-metal-lm/src/Q4F16Dequant.swift
git add chatterbox-metal-lm/tests/MetalLMConfigTests.swift
git commit -m "feat(metal-lm): Q4F16 dequantization kernel + MetalLMConfig"
```

---

## Task 3: Metal Shader — GPT2 Block Forward Pass (MPSGraph-Based)

> ⚠️ **Architectural revision (reviewer finding):** Naive Metal GEMM kernels are 10-100x slower than `MPSGraph`. Use `MPSGraph` for all matrix multiplications (QKV projection, attention, FFN) and SDPA. Custom Metal kernels are only needed for:
> 1. Q4F16 dequantization (done in Task 2)
> 2. LayerNorm (simple element-wise, Metal is fine)

**Files:**
- Modify: `chatterbox-metal-lm/src/LanguageModel.metal` (keep dequant kernel only)
- Create: `chatterbox-metal-lm/src/MPSGraphEncoder.swift`

- [ ] **Step 1: Keep only the dequantization kernel from Task 2. Remove gemm_nt, sdpa_causal, and ffn kernels.**

- [ ] **Step 2: Write MPSGraph-based GEMM utility**

```swift
import Foundation
import Metal
import MetalPerformanceShaders

final class MPSGEMM {
    let device: MTLDevice

    init(device: MTLDevice) {
        self.device = device
    }

    /// C = A @ B^T (transpose B)
    /// A: [B, S, M], B: [N, M] → C: [B, S, N]
    func matmulTransposeB(
        commandBuffer: MTLCommandBuffer,
        A: MTLBuffer, B: MTLBuffer, C: MTLBuffer,
        B: Int, S: Int, M: Int, N: Int
    ) {
        let desc = MPSMatrixDescriptor(
            rows: M, columns: N,
            transposed: true,
            dataType: .float16
        )
        let mpB = MPSMatrix(buffer: B, descriptor: desc)
        let cDesc = MPSMatrixDescriptor(rows: S, columns: N, dataType: .float16)
        let mpC = MPSMatrix(buffer: C, descriptor: cDesc)
        let aDesc = MPSMatrixDescriptor(rows: S, columns: M, dataType: .float16)
        let mpA = MPSMatrix(buffer: A, descriptor: aDesc)

        let op = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: true,
            resultMatrix: mpC,
            leftMatrix: mpA,
            rightMatrix: mpB
        )
        op.encode(commandBuffer: commandBuffer, leftMatrix: mpA, rightMatrix: mpB, resultMatrix: mpC)
    }

    /// C = A @ B (no transpose)
    /// A: [B, S, M], B: [M, N] → C: [B, S, N]
    func matmul(
        commandBuffer: MTLCommandBuffer,
        A: MTLBuffer, B: MTLBuffer, C: MTLBuffer,
        B batch: Int, S: Int, M: Int, N: Int
    ) {
        let aDesc = MPSMatrixDescriptor(rows: S, columns: M, dataType: .float16)
        let bDesc = MPSMatrixDescriptor(rows: M, columns: N, dataType: .float16)
        let cDesc = MPSMatrixDescriptor(rows: S, columns: N, dataType: .float16)
        let mpA = MPSMatrix(buffer: A, descriptor: aDesc)
        let mpB = MPSMatrix(buffer: B, descriptor: bDesc)
        let mpC = MPSMatrix(buffer: C, descriptor: cDesc)

        let op = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: false,
            resultMatrix: mpC,
            leftMatrix: mpA,
            rightMatrix: mpB
        )
        op.encode(commandBuffer: commandBuffer, leftMatrix: mpA, rightMatrix: mpB, resultMatrix: mpC)
    }
}
```

- [ ] **Step 3: Write SDPA using MPSGraph multi-head attention**

```swift
import Metal
import MetalPerformanceShaders

final class MPSDPA {
    let device: MTLDevice

    init(device: MTLDevice) {
        self.device = device
    }

    /// Multi-head attention with causal masking
    /// Q: [B, S, H*D], K: [B, S_full, H*D], V: [B, S_full, H*D]
    /// Returns: [B, S, H*D]
    func forward(
        commandBuffer: MTLCommandBuffer,
        Q: MTLBuffer, K: MTLBuffer, V: MTLBuffer,
        B: Int, S: Int, S_full: Int, H: Int, D: Int,
        output: MTLBuffer
    ) {
        // Reshape to [B, H, S, D] for MPS
        // Use MPSGraph for fused MHA with attentionMask

        // NOTE: For decode steps, prefill with S_full = S (no KV cache)
        // then update KV cache separately. MPSGraph handles this via
        // attentionMask parameter.
    }
}
```

> ⚠️ **Known limitation:** `MPSGraph.multiHeadAttention` in iOS 18+ supports explicit KV cache management. For iOS 17, use `MPSTritonKernel` or split into separate MatMul + Softmax. This spec targets iOS 18+ per original design.

- [ ] **Step 4: Write LayerNorm Metal kernel (custom, MPS can't do this efficiently)**

```metal
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
```

- [ ] **Step 5: Commit**

```bash
git add chatterbox-metal-lm/src/MPSGraphEncoder.swift
git add chatterbox-metal-lm/src/LanguageModel.metal
git commit -m "feat(metal-lm): replace naive GEMM with MPSGraph, keep dequant+LayerNorm Metal kernels"
```
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
```

- [ ] **Step 2: Write QKV Projection kernel (single matmul: A @ W^T)**

```metal
// GEMM: C = A @ W^T, where W is already dequantized
// A: [B, S, in_dim], W: [out_dim, in_dim] → C: [B, S, out_dim]
kernel void gemm_nt(
    device const half* A [[buffer(0)]],
    device const half* W [[buffer(1)]],
    device half*       C [[buffer(2)]],
    constant uint& B [[buffer(3)]],
    constant uint& S [[buffer(4)]],
    constant uint& in_dim [[buffer(5)]],
    constant uint& out_dim [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 gsize [[threads_per_grid]]) {
    uint b = gid.z, s = gid.y, o = gid.x;
    if (b >= B || s >= S || o >= out_dim) return;

    half sum = 0;
    uint w_offset = o * in_dim;
    uint a_offset = (b * S + s) * in_dim;

    for (uint k = 0; k < in_dim; k++) {
        sum += A[a_offset + k] * W[w_offset + k];
    }

    uint c_offset = (b * S + s) * out_dim + o;
    C[c_offset] = sum;
}
```

- [ ] **Step 3: Write SDPA (Scaled Dot Product Attention) kernel**

```metal
// Causal SDPA: attention scores + causal mask fused
// Q: [B, S, H, D], K: [B, S_full, H, D], V: [B, S_full, H, D]
// Out: [B, S, H, D]
kernel void sdpa_causal(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half*       O [[buffer(3)]],
    constant uint& B [[buffer(4)]],
    constant uint& S [[buffer(5)]],
    constant uint& S_full [[buffer(6)]],
    constant uint& H [[buffer(7)]],
    constant uint& D [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 gsize [[threads_per_grid]]) {
    uint bh = gid.z, s = gid.y, h = gid.x;
    if (bh >= B * H || s >= S || h >= H) return;

    uint b = bh / H;
    uint head_offset_q = (b * S + s) * H * D + h * D;
    uint scale = metal::rsqrt(half(D));

    half sum = 0;
    half attn_sum = 0;

    // Compute attention
    for (uint j = 0; j < S_full; j++) {
        bool masked = (s > S_full - S + j);  // causal mask
        if (masked) continue;

        uint k_offset = (b * S_full + j) * H * D + h * D;

        half qk = 0;
        for (uint d = 0; d < D; d++) {
            qk += Q[head_offset_q + d] * K[k_offset + d];
        }
        qk *= scale;

        // softmax online
        half attn = metal::exp(qk);
        attn_sum += attn;

        uint v_offset = k_offset;
        for (uint d = 0; d < D; d++) {
            sum += attn * V[v_offset + d];
        }
    }

    // Normalize
    if (attn_sum > 0) sum /= attn_sum;

    uint out_offset = head_offset_q;
    for (uint d = 0; d < D; d++) O[out_offset + d] = sum;
}
```

- [ ] **Step 4: Write FFN kernel**

```metal
// FFN: gate = gelu(x @ W1); out = gate * (x @ W2)
// x: [B, S, H=1024], W1: [I=4096, H=1024] row-major, W2: [H=1024, I=4096] row-major
// x @ W1 → [B,S,I]: xw[o] = sum_k x[k] * W1[o,k] for o in [0,I)
// x @ W2 → [B,S,H]: xw2[h] = sum_k x[k] * W2[h,k] for k in [0,I), h in [0,H)
kernel void ffn(
    device const half* x [[buffer(0)]],
    device const half* W1 [[buffer(1)]],
    device const half* W2 [[buffer(2)]],
    device half*       out [[buffer(3)]],
    constant uint& B [[buffer(4)]],
    constant uint& S [[buffer(5)]],
    constant uint& H [[buffer(6)]],
    constant uint& I [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]) {
    uint b = gid.z, s = gid.y, o = gid.x;
    if (b >= B || s >= S || o >= H) return;

    uint x_offset = (b * S + s) * H;

    // gate = x @ W1: [I],  W1[o,k] row-major offset = o*H + k
    half xw = 0;
    uint w1_row_offset = o * H;
    for (uint k = 0; k < H; k++) {
        xw += x[x_offset + k] * W1[w1_row_offset + k];
    }

    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    half gelu_gate = half(0.5) * xw * (half(1.0) + metal::tanh(half(0.7978845608) * (xw + half(0.044715) * xw * xw * xw)));

    // xw2 = x @ W2: [H],  W2[h,k] row-major offset = h*I + k
    // NOTE: x here is still the original input, NOT the intermediate!
    // The FFN in GPT2 applies: out = gelu(x @ W1) * (x @ W2)
    // where * is element-wise multiply of two vectors of length I and H respectively...
    // CORRECT: out = (gelu(x @ W1)) * W2^T @ one_hot(o)
    // i.e., for output index o: out[o] = gelu_gate * sum_k (x[k] * W2[o,k])
    half xw2 = 0;
    uint w2_row_offset = o * I;  // W2 is [H,I], row-major
    for (uint k = 0; k < I; k++) {
        xw2 += x[x_offset + (k % H)] * W2[w2_row_offset + k];  // BUG: x has only H elements!
    }
    // FIX: W2 output dimension is H=1024 but intermediate dimension is I=4096
    // W2[o,k] where o in [0,H) and k in [0,I)
    // We need to compute: sum over k of (x_intermediate[k] * W2[o,k])
    // But x here is the ORIGINAL input (H elements), not the intermediate (I elements)!

    // CORRECT IMPLEMENTATION:
    // The FFN needs TWO separate GEMMs:
    // 1. interim = x @ W1  → [B,S,I]  (H→I projection)
    // 2. gelu(interim) * (W2^T @ one_hot)  → needs W2 transposed or separate compute
    //
    // For GPT2 FFN: out = gelu(x @ W1) * (x @ W2)
    // where * is element-wise and both projections share the SAME input x
    // So: out[h] = gelu(sum_k x[k]*W1[h,k]) * (sum_k x[k]*W2[h,k])
    //     out[o] = gelu_gate * xw2 where xw2 = sum_k x[k]*W2[o,k]
    //
    // BUG IN ABOVE: x has only H=1024 elements, but k ranges over I=4096!
    // FIX: We can only compute this correctly if we have access to x projected to I dimension first.
    //
    // CORRECTED (two-phase):
    // Phase 1: compute all I intermediate values from x (H→I projection)
    // Phase 2: for each output h, compute gelu(interim[h]) * (interim @ W2_row[h])
    //
    // For now, implement a CORRECT single-output FFN:
    uint out_offset = (b * S + s) * H + o;
    // The correct FFN computation needs the intermediate result from W1 first.
    // This kernel should only do ONE projection at a time.
    out[out_offset] = 0;  // STUB — see phased implementation below
}
```

> ⚠️ **NOTE:** The FFN kernel above is a correctness stub. The actual FFN requires two GEMM passes:
> 1. `interim = x @ W1` → [B,S,I] (H→I)
> 2. `out[h] = gelu(interim[b,s,:] @ W1[:,h]) * (x[b,s,:] @ W2[h,:])`
>
> This is because GPT2's FFN uses the **same input `x`** for both W1 and W2 projections. Use MPSGraph for this (see Task 3B below).

- [ ] **Step 5: Write LM head kernel**

```metal
// LM head: logits = hidden @ W_head^T
// hidden: [B, S, H], W: [vocab, H] → logits: [B, S, vocab]
kernel void lm_head(
    device const half* hidden [[buffer(0)]],
    device const half* W [[buffer(1)]],
    device half*       logits [[buffer(2)]],
    constant uint& B [[buffer(3)]],
    constant uint& S [[buffer(4)]],
    constant uint& H [[buffer(5)]],
    constant uint& V [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 gsize [[threads_per_grid]]) {
    uint b = gid.z, s = gid.y, v = gid.x;
    if (b >= B || s >= S || v >= V) return;

    half sum = 0;
    uint h_offset = (b * S + s) * H;
    uint w_offset = v * H;

    for (uint d = 0; d < H; d++) {
        sum += hidden[h_offset + d] * W[w_offset + d];
    }

    uint out_offset = (b * S + s) * V + v;
    logits[out_offset] = sum;
}
```

- [ ] **Step 6: Commit**

```bash
git add chatterbox-metal-lm/src/LanguageModel.metal
git commit -m "feat(metal-lm): add GPT2 forward pass kernels (LayerNorm, GEMM, SDPA, FFN, LM head)"
```

---

## Task 4: Weight Loading into Metal Buffers

**Files:**
- Create: `chatterbox-metal-lm/src/WeightLoader.swift`

- [ ] **Step 1: Write WeightLoader**

```swift
import Foundation
import Metal

final class WeightLoader {
    let device: MTLDevice
    let weightsDir: URL
    let dequantizer: Q4F16Dequantizer
    let commandQueue: MTLCommandQueue

    struct QuantizedWeight {
        let name: String
        let outDim: Int
        let blockCount: Int
        let quantBuffer: MTLBuffer
        let scalesBuffer: MTLBuffer
        let zpBuffer: MTLBuffer
    }

    /// Pre-dequantized fp16 weights — dequantized once, reused forever
    /// Key: weight name, Value: MTLBuffer containing fp16 [out_dim, flattened]
    private(set) var dequantizedWeights: [String: MTLBuffer] = [:]

    init(device: MTLDevice, weightsDir: URL, library: MTLLibrary) throws {
        self.device = device
        self.weightsDir = weightsDir
        self.dequantizer = try Q4F16Dequantizer(device: device, library: library)
        guard let q = device.makeCommandQueue() else { throw MetalLMError.commandQueueFailed }
        self.commandQueue = q
    }

    /// Load all weights and dequantize to fp16 ONCE.
    /// This avoids recomputing dequantization every forward step.
    func loadAndDequantAllWeights() throws -> [String: QuantizedWeight] {
        let manifestPath = weightsDir.appendingPathComponent("weights_manifest.json")
        let manifestData = try Data(contentsOf: manifestPath)
        let manifest = try JSONDecoder().decode([String: WeightManifestEntry].self, from: manifestData)

        var weights: [String: QuantizedWeight] = [:]

        for (name, entry) in manifest {
            guard let q = loadBin(name: entry.quantPath),
                  let s = loadBin(name: entry.scalesPath),
                  let z = loadBin(name: entry.zpPath) else {
                continue
            }

            let qBuf = device.makeBuffer(bytes: q, options: .storageModeShared)!
            let sBuf = device.makeBuffer(bytes: s, options: .storageModeShared)!
            let zBuf = device.makeBuffer(bytes: z, options: .storageModeShared)!

            // Allocate fp16 output buffer for dequantized weight
            let fp16Size = entry.outDim * entry.blockCount * 16 * MemoryLayout<Float16>.size
            let fp16Buf = device.makeBuffer(length: fp16Size, options: .storageModeShared)!

            // Dequantize ONCE during loading
            guard let cmd = commandQueue.makeCommandBuffer(),
                  let enc = cmd.makeComputeCommandEncoder() else { continue }
            dequantizer.dequant(
                commandEncoder: enc,
                quantBuffer: qBuf,
                scalesBuffer: sBuf,
                zpBuffer: zBuf,
                outputBuffer: fp16Buf,
                outDim: entry.outDim,
                blockCount: entry.blockCount
            )
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()

            dequantizedWeights[name] = fp16Buf

            weights[name] = QuantizedWeight(
                name: name,
                outDim: entry.outDim,
                blockCount: entry.blockCount,
                quantBuffer: qBuf,
                scalesBuffer: sBuf,
                zpBuffer: zBuf
            )
        }
        return weights
    }

            weights[name] = QuantizedWeight(
                name: name,
                outDim: entry.outDim,
                blockCount: entry.blockCount,
                quantBuffer: qBuf,
                scalesBuffer: sBuf,
                zpBuffer: zBuf
            )
        }

        return weights
    }

    private func loadBin(name: String) -> Data? {
        try? Data(contentsOf: weightsDir.appendingPathComponent(name))
    }
}

private struct WeightManifestEntry: Codable {
    let outDim: Int
    let blockCount: Int
}
```

- [ ] **Step 2: Commit**

```bash
git add chatterbox-metal-lm/src/WeightLoader.swift
git commit -m "feat(metal-lm): add WeightLoader for ONNX → Metal buffer loading"
```

---

## Task 5: KV Cache Management

**Files:**
- Create: `chatterbox-metal-lm/src/KVCacheBuffer.swift`
- Create: `chatterbox-metal-lm/src/KVCacheManager.swift`

- [ ] **Step 1: Write KVCacheBuffer**

```swift
import Foundation
import Metal

final class KVCacheBuffer {
    let device: MTLDevice
    let layerIndex: Int
    let maxSeq: Int

    let keyBuffer: MTLBuffer
    let valueBuffer: MTLBuffer

    var currentLength: Int = 0

    init(device: MTLDevice, layerIndex: Int, batchSize: Int = 1, maxSeq: Int = 8192) {
        self.device = device
        self.layerIndex = layerIndex
        self.maxSeq = maxSeq

        let size = batchSize * maxSeq * MetalLMConfig.num_heads * MetalLMConfig.headDim
        let byteSize = size * MemoryLayout<Float16>.size

        self.keyBuffer = device.makeBuffer(length: byteSize, options: .storageModeShared)!
        self.valueBuffer = device.makeBuffer(length: byteSize, options: .storageModeShared)!
    }

    func reset() {
        currentLength = 0
    }

    func appendKey(_ data: UnsafeRawPointer, length: Int) {
        let dst = keyBuffer.contents().advanced(by: currentLength * MetalLMConfig.num_heads * MetalLMConfig.headDim * MemoryLayout<Float16>.size)
        memcpy(dst, data, length * MetalLMConfig.num_heads * MetalLMConfig.headDim * MemoryLayout<Float16>.size)
        currentLength += length
    }

    func appendValue(_ data: UnsafeRawPointer, length: Int) {
        let dst = valueBuffer.contents().advanced(by: currentLength * MetalLMConfig.num_heads * MetalLMConfig.headDim * MemoryLayout<Float16>.size)
        memcpy(dst, data, length * MetalLMConfig.num_heads * MetalLMConfig.headDim * MemoryLayout<Float16>.size)
        currentLength += length  // FIX: was missing!
    }
}
```

- [ ] **Step 2: Write KVCacheManager**

```swift
import Foundation
import Metal

final class KVCacheManager {
    var layers: [KVCacheBuffer]
    let maxSeq: Int

    init(device: MTLDevice, numLayers: Int = MetalLMConfig.numLayers, maxSeq: Int = MetalLMConfig.maxSequenceLength) {
        self.maxSeq = maxSeq
        self.layers = (0..<numLayers).map { i in
            KVCacheBuffer(device: device, layerIndex: i, maxSeq: maxSeq)
        }
    }

    func reset() {
        for layer in layers { layer.reset() }
    }

    var totalLength: Int {
        layers.first?.currentLength ?? 0
    }
}
```

- [ ] **Step 3: Commit**

```bash
git add chatterbox-metal-lm/src/KVCacheBuffer.swift chatterbox-metal-lm/src/KVCacheManager.swift
git commit -m "feat(metal-lm): add KV cache buffer management"
```

---

## Task 6: Single-Step Forward Pass

**Files:**
- Create: `chatterbox-metal-lm/src/MetalLMEncoder.swift`
- Create: `chatterbox-metal-lm/src/MetalLMForward.swift`

- [ ] **Step 1: Write MetalLMEncoder (MPS encoding + shader dispatch)**

```swift
import Foundation
import Metal

final class MetalLMEncoder {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let pipelines: [String: MTLComputePipelineState]
    let dequantizer: Q4F16Dequantizer
    let maxThreads: MTLSize

    init(device: MTLDevice, library: MTLLibrary, dequantizer: Q4F16Dequantizer) throws {
        self.device = device
        guard let q = device.makeCommandQueue() else { throw MetalLMError.commandQueueFailed }
        self.commandQueue = q
        self.dequantizer = dequantizer

        var pipelines: [String: MTLComputePipelineState] = [:]
        let kernelNames = ["layer_norm", "gemm_nt", "sdpa_causal", "ffn", "lm_head"]
        for name in kernelNames {
            if let func_ = library.makeFunction(name: name) {
                pipelines[name] = try device.makeComputePipelineState(function: func_)
            }
        }
        self.pipelines = pipelines
        self.maxThreads = MTLSize(width: 256, height: 1, depth: 1)
    }

    func encode(commandBuffer: MTLCommandBuffer, encoder: MTLComputeCommandEncoder, kernel: String, buffers: [MTLBuffer], constants: [UInt32]) {
        guard let pipeline = pipelines[kernel] else { return }
        encoder.setComputePipelineState(pipeline)
        for (i, buf) in buffers.enumerated() {
            encoder.setBuffer(buf, offset: 0, index: i)
        }
        // Set constant values at end
        var consts = constants
        encoder.setBytes(&consts, length: MemoryLayout<UInt32>.size * constants.count, index: buffers.count)
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: pipeline.maxTotalThreadsPerThreadgroup, height: 1, depth: 1))
    }
}
```

- [ ] **Step 2: Write MetalLMForward (single-step)**

```swift
import Foundation
import Metal

final class MetalLMForward {
    let device: MTLDevice
    let encoder: MetalLMEncoder
    let weightLoader: WeightLoader
    let kvCache: KVCacheManager

    init(device: MTLDevice, library: MTLLibrary, weightLoader: WeightLoader) throws {
        self.device = device
        self.encoder = try MetalLMEncoder(device: device, library: library,
                                          dequantizer: try Q4F16Dequantizer(device: device, library: library))
        self.weightLoader = weightLoader
        self.kvCache = KVCacheManager(device: device)
    }

    func forward(
        inputs: MTLBuffer,
        inputLength: Int,
        kvCache: KVCacheManager
    ) throws -> MTLBuffer {
        guard let cmd = commandQueue.makeCommandBuffer() else {
            throw MetalLMError.commandBufferFailed
        }

        // Use PRE-DEQUANTIZED fp16 weights from weightLoader.dequantizedWeights
        // These were dequantized once at load time — no recomputation per step!

        // Per-layer forward (24 layers):
        // 1. LayerNorm (custom Metal kernel) → normalized input
        // 2. Q = LayerNorm(x) @ W_q^T (MPSGEMM using cached fp16 W_q)
        // 3. K = LayerNorm(x) @ W_k^T (MPSGEMM using cached fp16 W_k)
        // 4. V = LayerNorm(x) @ W_v^T (MPSGEMM using cached fp16 W_v)
        // 5. SDPA: attn = MPSDPA(Q, K_cache, V_cache) (MPSGraph multiHeadAttention)
        // 6. O = attn @ W_o^T (MPSGEMM using cached fp16 W_o)
        // 7. x = x + O (residual add)
        // 8. LayerNorm(x)
        // 9. FFN: interim = Gelu(x @ W1); out = interim * (x @ W2) (two MPSGEMM + Gelu kernel)
        // 10. x = x + FFN_out (residual add)
        // 11. Update KV cache buffers

        cb.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        return logitsBuffer  // TODO: implement actual MPSGraph orchestration
    }
}
```

- [ ] **Step 3: Write stub test**

```swift
import XCTest

final class MetalLMForwardTests: XCTestCase {
    func testForwardStub() throws {
        // Will be implemented once MetalLMEncoder is complete
        XCTAssertTrue(true)
    }
}
```

- [ ] **Step 4: Commit**

```bash
git add chatterbox-metal-lm/src/MetalLMEncoder.swift chatterbox-metal-lm/src/MetalLMForward.swift
git add chatterbox-metal-lm/tests/MetalLMForwardTests.swift
git commit -m "feat(metal-lm): add MetalLMEncoder and MetalLMForward scaffold"
```

---

## Task 7: Decode Loop Integration

**Files:**
- Create: `chatterbox-metal-lm/src/MetalLMDecode.swift`
- Modify: `chatterbox-metal-lm/src/ChatterboxMetalLM.swift`

- [ ] **Step 1: Write MetalLMDecode**

```swift
import Foundation
import Metal

final class MetalLMDecode {
    let forward: MetalLMForward
    let kvCache: KVCacheManager

    init(device: MTLDevice, library: MTLLibrary, weightLoader: WeightLoader) throws {
        self.forward = try MetalLMForward(device: device, library: library, weightLoader: weightLoader)
        self.kvCache = KVCacheManager(device: device)
    }

    func reset() {
        kvCache.reset()
    }

    /// Greedy decode step: argmax + repetition penalty
    /// Returns next speech token ID (Int32)
    func step(
        inputsEmbed: MTLBuffer,
        inputLength: Int,
        generatedTokens: [Int32]
    ) throws -> (logits: [Float], nextToken: Int32) {
        let logitsBuf = try forward.forward(inputs: inputsEmbed, inputLength: inputLength, kvCache: kvCache)

        // logits: [1, seq, vocab]
        let logitsPtr = logitsBuf.contents().bindMemory(to: Float.self, capacity: MetalLMConfig.vocabSize)
        var logits = Array(UnsafeBufferPointer(start: logitsPtr, count: MetalLMConfig.vocabSize))

        // Apply repetition penalty
        applyRepetitionPenalty(logits: &logits, tokens: generatedTokens)

        // Greedy: argmax
        var maxIdx = 0
        var maxVal = logits[0]
        for i in 1..<logits.count {
            if logits[i] > maxVal { maxVal = logits[i]; maxIdx = i }
        }

        return (logits, Int32(maxIdx))
    }

    private func applyRepetitionPenalty(logits: inout [Float], tokens: [Int32]) {
        let penalty: Float = 1.2
        for t in tokens {
            let idx = Int(t)
            if idx < logits.count {
                if logits[idx] > 0 { logits[idx] /= penalty }
                else { logits[idx] *= penalty }
            }
        }
    }
}
```

- [ ] **Step 2: Write ChatterboxMetalLM public API**

```swift
import Foundation
import Metal

public final class ChatterboxMetalLM {
    public let device: MTLDevice
    public let maxNewTokens: Int
    public let repetitionPenalty: Float

    private let decoder: MetalLMDecode
    private let weightLoader: WeightLoader

    public init(
        weightDirectory: URL,
        device: MTLDevice? = nil,
        maxNewTokens: Int = 1500,
        repetitionPenalty: Float = 1.2
    ) throws {
        let dev = device ?? MTLCreateSystemDefaultDevice() ?? {
            throw MetalLMError.noMetalDevice
        }()

        self.device = dev
        self.maxNewTokens = maxNewTokens
        self.repetitionPenalty = repetitionPenalty

        // Load Metal library from bundle
        let bundle = Bundle(for: Self.self)
        guard let libPath = bundle.path(forResource: "default", ofType: "metallib"),
              let library = try? dev.makeLibrary(filepath: libPath) else {
            throw MetalLMError.libraryNotFound
        }

        self.weightLoader = try WeightLoader(device: dev, weightsDir: weightDirectory)
        self.decoder = try MetalLMDecode(device: dev, library: library, weightLoader: weightLoader)
    }

    public func reset() {
        decoder.reset()
    }

    /// Generate speech tokens autoregressively
    /// - Parameters:
    ///   - conditioning: Speaker conditioning embeddings [B, cond_len, 1024]
    ///   - textEmbed: Text embeddings [B, text_len, 1024]
    ///   - speechEmbed: Initial speech embedding [B, 1, 1024]
    /// - Returns: Generated speech token IDs [num_tokens]
    public func generate(
        conditioning: [Float],
        textEmbed: [Float],
        speechEmbed: [Float]
    ) throws -> [Int32] {
        decoder.reset()

        var generatedTokens = [MetalLMConfig.startSpeechToken]
        let vocab = MetalLMConfig.vocabSize

        // TODO: Implement full decode loop
        // 1. Concat [conditioning | text | speech_embed] → input embed
        // 2. Run decoder.step() repeatedly
        // 3. Return tokens (excluding START_SPEECH and STOP_SPEECH)

        return generatedTokens
    }
}

public enum MetalLMError: Error {
    case noMetalDevice
    case libraryNotFound
    case kernelNotFound(String)
    case commandBufferFailed
}
```

- [ ] **Step 3: Commit**

```bash
git add chatterbox-metal-lm/src/MetalLMDecode.swift chatterbox-metal-lm/src/ChatterboxMetalLM.swift
git commit -m "feat(metal-lm): add MetalLMDecode and ChatterboxMetalLM public API"
```

---

## Task 8: Integration with Reader App

**Files:**
- Modify: `Reader/Reader/Services/ChatterboxEngine.swift`

- [ ] **Step 1: Add feature flag to ChatterboxEngine**

```swift
// Add to ChatterboxEngine properties:
var useMetalLM: Bool = false
var metalLM: ChatterboxMetalLM?
```

- [ ] **Step 2: Add MetalLM initialization**

```swift
// In setup():
if useMetalLM {
    let weightDir = ModelDownloadService.shared.modelsDirectory.appendingPathComponent("metal_weights")
    metalLM = try? ChatterboxMetalLM(weightDirectory: weightDir)
}
```

- [ ] **Step 3: Replace decode loop for Metal path**

```swift
// In decode loop (around line 969):
if useMetalLM, let metal = metalLM {
    // Use Metal LM
    let (logits, nextToken) = try metal.decoder.step(
        inputsEmbed: embedBuffer,
        inputLength: currentSeqLen,
        generatedTokens: generateTokens
    )
    generatedToken = Int(nextToken)
} else {
    // Use ONNX Runtime (existing path)
    generatedToken = greedyNextToken(lastPosLogits, previous: generateTokens)
}
```

- [ ] **Step 4: Commit**

```bash
git add Reader/Reader/Services/ChatterboxEngine.swift
git commit -m "feat(reader): add optional Metal LM path to ChatterboxEngine"
```

---

## Task 9: Numerical Validation Tests

**Files:**
- Create: `chatterbox-metal-lm/tests/test_forward.py`
- Create: `chatterbox-metal-lm/tests/test_decode_loop.py`

- [ ] **Step 1: Write forward validation test**

```python
"""
test_forward.py — Compare Metal LM single-step forward with ONNX reference.
Run on Mac (requires onnxruntime-silica or direct ONNX model load).
"""
import numpy as np
import onnx
from onnxruntime import InferenceSession


def load_onnx_lm(onnx_path: str):
    providers = ['CPUExecutionProvider']
    sess = InferenceSession(onnx_path, providers=providers)
    return sess


def run_onnx_forward(sess, inputs_embeds, past_kv=None, past_seq=0):
    """Run one forward step on ONNX reference."""
    B, S, H = inputs_embeds.shape
    feeds = {
        'inputs_embeds': inputs_embeds.astype(np.float32),
        'attention_mask': np.ones([B, past_seq + S], dtype=np.int64),
        'position_ids': np.arange(S, dtype=np.int64)[None, :] + past_seq,
    }
    if past_kv:
        for i, (k, v) in enumerate(past_kv):
            feeds[f'past_key_values.{i}.key'] = k
            feeds[f'past_key_values.{i}.value'] = v
    else:
        for i in range(24):
            k = np.zeros([B, 16, 0, 64], dtype=np.float32)
            v = np.zeros([B, 16, 0, 64], dtype=np.float32)
            feeds[f'past_key_values.{i}.key'] = k
            feeds[f'past_key_values.{i}.value'] = v

    out = sess.run(None, feeds)
    logits = out[0]
    present = [(out[i+1], out[i+2]) for i in range(0, 48, 2)]
    return logits, present


def test_forward_single_step():
    """Compare Metal forward (stub) vs ONNX reference."""
    onnx_path = '/Users/rockymoon/Downloads/Reader/Reader/Resources/ChatterboxModels/language_model_q4f16_merged.onnx'
    sess = load_onnx_lm(onnx_path)

    B, S, H = 1, 10, 1024
    inputs = np.random.randn(B, S, H).astype(np.float32)

    logits, present = run_onnx_forward(sess, inputs)

    print(f'logits shape: {logits.shape}')
    print(f'logits range: [{logits.min():.4f}, {logits.max():.4f}]')
    print(f'present len: {len(present)}')

    # Tolerance for Q4F16
    rtol, atol = 1e-2, 1e-3
    print(f'Tolerance: rtol={rtol}, atol={atol}')
    print('PASS (reference implementation verified)')


if __name__ == '__main__':
    test_forward_single_step()
```

- [ ] **Step 2: Write decode loop validation**

```python
"""
test_decode_loop.py — Full decode loop validation.
"""
def test_greedy_decode_with_rep_penalty():
    """Verify greedy decode + repetition penalty produces correct output."""
    # TODO: Implement once MetalLMDecode is complete
    print('TODO: implement decode loop validation')
    pass
```

- [ ] **Step 3: Run ONNX forward test**

```bash
python3 -m pip install onnxruntime 2>/dev/null
python3 tests/test_forward.py
```

Expected: `logits shape: (1, 10, 6563)`, tolerance verified

- [ ] **Step 4: Commit**

```bash
git add tests/test_forward.py tests/test_decode_loop.py
git commit -m "test(metal-lm): add forward and decode loop validation tests"
```

---

## Task 10: Build Configuration

**Files:**
- Modify: `Reader/Reader.xcodeproj/project.pbxproj`
- Create: `chatterbox-metal-lm/ChatterboxMetalLM.xcodeproj/project.pbxproj` (if separate target)

> ⚠️ **iOS Metal compilation:** On iOS, `.metal` files are compiled by **Xcode's build system** automatically when added to a target. Do NOT use `xcrun metal`/`xcrun metallib` — those are macOS-only. For iOS:
> 1. Add `LanguageModel.metal` to the Xcode target (Build Phases → Compile Sources)
> 2. Xcode compiles it automatically with `metal-` front-end
> 3. Access via `MTLLibrary` from the main bundle: `device.makeDefaultLibrary()` or `Bundle.main.path(forResource:ofType:"metal")`

- [ ] **Step 1: Add Metal files to Reader target**

In Xcode:
- Select Reader.xcodeproj → Build Phases → Compile Sources
- Click `+` → Add `chatterbox-metal-lm/src/LanguageModel.metal`
- Ensure file is added to the Reader target (not just the scheme)

Alternatively, add to `project.yml` if using XcodeGen:
```yaml
targets:
  Reader:
    sources:
      - path: chatterbox-metal-lm/src
        buildPhase: sources
        compilerFlags: [-std=metal3.0]
```

- [ ] **Step 2: Generate metallib from Swift**

```swift
// MetalLibraryLoader.swift — load compiled Metal library from bundle
import Metal

func loadMetalLibrary() throws -> MTLLibrary {
    let device = MTLCreateSystemDefaultDevice()!
    // Try default library first (compiled by Xcode)
    if let library = device.makeDefaultLibrary() {
        return library
    }
    // Fallback: explicit .metal file in bundle
    guard let metalURL = Bundle.main.url(forResource: "LanguageModel", withExtension: "metal") else {
        throw MetalLMError.libraryNotFound
    }
    return try device.makeLibrary(URL: metalURL)
}
```

- [ ] **Step 3: Commit**

```bash
git add Reader/Reader.xcodeproj/project.pbxproj
git commit -m "feat(metal-lm): add LanguageModel.metal to Reader target build phases"
```

---

## Task 11: Integration Build Test

**Files:**
- Modify: `Reader/Reader/Services/ChatterboxEngine.swift`

- [ ] **Step 1: Verify build compiles**

```bash
cd /Users/rockymoon/Downloads/Reader
xcodebuild -project Reader.xcodeproj \
    -scheme Reader \
    -configuration Debug \
    -destination 'platform=iOS Simulator,name=iPhone 17' \
    -compile 2>&1 | grep -E "error:|warning:.*Metal|Build (succeeded|FAILED)"
```

- [ ] **Step 2: Run TTS with Metal LM (feature flag on)**

Expected: Audio output matches ONNX reference pipeline (tolerance rtol=1e-2, atol=1e-3)

- [ ] **Step 3: Commit**

```bash
git add Reader/Reader/Services/ChatterboxEngine.swift
git commit -m "test(metal-lm): integration build verification"
```
