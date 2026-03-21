import Foundation
import Metal
import MetalPerformanceShaders

// MARK: - GEMM Utilities

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
        batch: Int, S: Int, M: Int, N: Int
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
        batch: Int, S: Int, M: Int, N: Int
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

// MARK: - SDPA (Multi-head Attention with Causal Mask)

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
        // MPSGraph multi-head attention with causal mask
        // For decode steps: S_full = past_seq + S (full KV length)
        // For prefix: S_full = S (no KV cache yet)
        //
        // IMPORTANT: iOS 18+ required for MPSGraph multiHeadAttention
        // with proper attentionMask support. The attentionMask should
        // be a causal lower-triangular mask where positions that should
        // attend have 0 and masked positions have -inf.
    }
}

// MARK: - LayerNorm (Metal compute pipeline)

final class LayerNormPipeline {
    let device: MTLDevice
    let pipeline: MTLComputePipelineState
    let maxThreads: MTLSize

    init(device: MTLDevice, library: MTLLibrary) throws {
        self.device = device
        guard let kern = library.makeFunction(name: "layer_norm") else {
            throw MetalLMError.kernelNotFound("layer_norm")
        }
        self.pipeline = try device.makeComputePipelineState(function: kern)
        self.maxThreads = MTLSize(
            width: pipeline.maxTotalThreadsPerThreadgroup,
            height: 1, depth: 1
        )
    }

    func normalize(
        commandBuffer: MTLCommandBuffer,
        input: MTLBuffer, gamma: MTLBuffer, beta: MTLBuffer, output: MTLBuffer,
        batch: Int, dim: Int
    ) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(gamma, offset: 0, index: 1)
        enc.setBuffer(beta, offset: 0, index: 2)
        enc.setBuffer(output, offset: 0, index: 3)

        var d = UInt32(dim)
        var e: Float16 = Float16(1e-5)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&e, length: MemoryLayout<Float16>.size, index: 5)

        let tgWidth = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let threadsPerGroup = MTLSize(width: tgWidth, height: 1, depth: 1)
        let numThreadGroups = MTLSize(
            width: (batch + tgWidth - 1) / tgWidth,
            height: batch,
            depth: 1
        )
        enc.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
        enc.endEncoding()
    }
}

// MARK: - GPT2 Block Forward Pass

final class GPT2BlockForward {
    let device: MTLDevice
    let gemm: MPSGEMM
    let dpa: MPSDPA
    let ln1: LayerNormPipeline
    let ln2: LayerNormPipeline

    // Pre-allocated temporary buffers (reused across calls)
    let qBuf: MTLBuffer
    let kBuf: MTLBuffer
    let vBuf: MTLBuffer
    let attnBuf: MTLBuffer
    let qkBuf: MTLBuffer
    let attnWeights: MTLBuffer
    let fc1Out: MTLBuffer
    let fc2Out: MTLBuffer

    init(device: MTLDevice, library: MTLLibrary) throws {
        self.device = device
        self.gemm = MPSGEMM(device: device)
        self.dpa = MPSDPA(device: device)
        self.ln1 = try LayerNormPipeline(device: device, library: library)
        self.ln2 = try LayerNormPipeline(device: device, library: library)

        // Pre-allocate temp buffers
        let hidden = MetalLMConfig.hiddenSize
        let intermediate = MetalLMConfig.intermediateSize
        let maxSeq = MetalLMConfig.maxSequenceLength
        let heads = MetalLMConfig.num_heads
        let headDim = MetalLMConfig.headDim

        qBuf = device.makeBuffer(length: maxSeq * heads * headDim * MemoryLayout<Float16>.size, options: .storageModeShared)!
        kBuf = device.makeBuffer(length: maxSeq * heads * headDim * MemoryLayout<Float16>.size, options: .storageModeShared)!
        vBuf = device.makeBuffer(length: maxSeq * heads * headDim * MemoryLayout<Float16>.size, options: .storageModeShared)!
        attnBuf = device.makeBuffer(length: maxSeq * heads * headDim * MemoryLayout<Float16>.size, options: .storageModeShared)!
        qkBuf = device.makeBuffer(length: maxSeq * maxSeq * MemoryLayout<Float16>.size, options: .storageModeShared)!
        attnWeights = device.makeBuffer(length: maxSeq * hidden * MemoryLayout<Float16>.size, options: .storageModeShared)!
        fc1Out = device.makeBuffer(length: maxSeq * intermediate * MemoryLayout<Float16>.size, options: .storageModeShared)!
        fc2Out = device.makeBuffer(length: maxSeq * hidden * MemoryLayout<Float16>.size, options: .storageModeShared)!
    }

    // TODO: Implement full GPT2 block forward pass
    // The actual implementation should:
    // 1. LayerNorm on input
    // 2. QKV projections (3x matmulTransposeB with dequantized weights)
    // 3. SDPA with causal mask
    // 4. O projection (matmul with dequantized weights)
    // 5. Residual add
    // 6. LayerNorm
    // 7. FFN: FC1 (matmulTransposeB) -> GELU -> FC2 (matmulTransposeB) -> residual add
}