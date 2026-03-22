import Foundation
import Metal

final class MetalLMEncoder {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let layerNormPipeline: LayerNormPipeline
    let gemm: MPSGEMM
    let dpa: MPSDPA

    // Pre-allocated intermediate buffers (reused across calls)
    private let ln1Out: MTLBuffer
    private let qBuf: MTLBuffer
    private let qkvOut: MTLBuffer  // fused QKV result before split
    private let attnOut: MTLBuffer
    private let oOut: MTLBuffer
    private let residual1: MTLBuffer
    private let ln2Out: MTLBuffer
    private let fc1Out: MTLBuffer
    private let geluOut: MTLBuffer
    private let fc2Out: MTLBuffer

    private let hidden: Int
    private let intermediate: Int
    private let heads: Int
    private let headDim: Int

    init(device: MTLDevice, library: MTLLibrary) throws {
        self.device = device
        guard let q = device.makeCommandQueue() else {
            throw MetalLMError.commandQueueFailed
        }
        self.commandQueue = q

        self.layerNormPipeline = try LayerNormPipeline(device: device, library: library)
        self.gemm = MPSGEMM(device: device)
        self.dpa = MPSDPA(device: device)

        self.hidden = MetalLMConfig.hiddenSize
        self.intermediate = MetalLMConfig.intermediateSize
        self.heads = MetalLMConfig.num_heads
        self.headDim = MetalLMConfig.headDim

        let maxSeq = MetalLMConfig.maxSequenceLength
        let fp16 = MemoryLayout<Float16>.size
        ln1Out     = device.makeBuffer(length: maxSeq * hidden * fp16, options: .storageModeShared)!
        qBuf       = device.makeBuffer(length: maxSeq * hidden * fp16, options: .storageModeShared)!
        qkvOut     = device.makeBuffer(length: maxSeq * hidden * 3 * fp16, options: .storageModeShared)!
        attnOut    = device.makeBuffer(length: maxSeq * hidden * fp16, options: .storageModeShared)!
        oOut       = device.makeBuffer(length: maxSeq * hidden * fp16, options: .storageModeShared)!
        residual1  = device.makeBuffer(length: maxSeq * hidden * fp16, options: .storageModeShared)!
        ln2Out     = device.makeBuffer(length: maxSeq * hidden * fp16, options: .storageModeShared)!
        fc1Out     = device.makeBuffer(length: maxSeq * intermediate * fp16, options: .storageModeShared)!
        geluOut    = device.makeBuffer(length: maxSeq * intermediate * fp16, options: .storageModeShared)!
        fc2Out     = device.makeBuffer(length: maxSeq * hidden * fp16, options: .storageModeShared)!
    }

    /// Single GPT2 block forward pass.
    /// Input/output layout: [1, seq, hidden] = [1, seq, 1024] row-major Float16.
    ///
    /// qWeight/kWeight/vWeight are the SPLIT QKV weight buffers [hidden, hidden] each.
    /// kActBuf/vActBuf are the full K/V activation buffers (past + new, [1, totalSeq, hidden]).
    /// These bypass the internal kBuf/vBuf for K/V to avoid re-copying.
    /// cAttnBias: fused QKV bias [3*hidden=3072] — bias is added after matmul, before split.
    func forward(
        input: MTLBuffer,
        inputLength: Int,
        qWeight: MTLBuffer,
        kWeight: MTLBuffer,
        vWeight: MTLBuffer,
        cAttnBias: MTLBuffer,
        kActBuf: MTLBuffer,
        vActBuf: MTLBuffer,
        totalSeq: Int,
        oWeight: MTLBuffer,
        fc1Weight: MTLBuffer,
        fc2Weight: MTLBuffer,
        ln1Gamma: MTLBuffer,
        ln1Beta: MTLBuffer,
        ln2Gamma: MTLBuffer,
        ln2Beta: MTLBuffer,
        output: MTLBuffer,
        commandBuffer: MTLCommandBuffer
    ) {
        // ----- 1. LayerNorm 1 -----
        layerNormPipeline.normalize(
            commandBuffer: commandBuffer,
            input: input,
            gamma: ln1Gamma,
            beta: ln1Beta,
            output: ln1Out,
            batch: 1 * inputLength,
            dim: hidden
        )

        // ----- 2. Q projection with bias -----
        // ONNX Gemm: Y = X @ W^T + B (transB=0 for all weights)
        // Q = ln1_out @ qWeight^T + b_q[1024]
        gemm.matmulTransposeB(
            commandBuffer: commandBuffer,
            A: ln1Out, B: qWeight, C: qBuf,
            batch: 1, S: inputLength, M: hidden, N: hidden
        )
        // Add Q bias
        addBias(qBuf, cAttnBias, offset: 0, count: inputLength * hidden)

        // ----- 3. K projection with bias -----
        gemm.matmulTransposeB(
            commandBuffer: commandBuffer,
            A: ln1Out, B: kWeight, C: kActBuf,  // reuse kActBuf for K result
            batch: 1, S: inputLength, M: hidden, N: hidden
        )
        addBias(kActBuf, cAttnBias, offset: hidden, count: inputLength * hidden)

        // ----- 4. V projection with bias -----
        gemm.matmulTransposeB(
            commandBuffer: commandBuffer,
            A: ln1Out, B: vWeight, C: vActBuf,  // reuse vActBuf for V result
            batch: 1, S: inputLength, M: hidden, N: hidden
        )
        addBias(vActBuf, cAttnBias, offset: 2 * hidden, count: inputLength * hidden)

        // ----- 5. SDPA with causal mask -----
        // Q: [1, inputLength, 1024]
        // K: [1, totalSeq, 1024] (past + new)
        // V: [1, totalSeq, 1024] (past + new)
        dpa.forward(
            commandBuffer: commandBuffer,
            Q: qBuf,
            K: kActBuf,
            V: vActBuf,
            B: 1,
            S: inputLength,
            S_full: totalSeq,
            H: heads,
            D: headDim,
            output: attnOut
        )

        // ----- 6. O projection -----
        gemm.matmulTransposeB(
            commandBuffer: commandBuffer,
            A: attnOut, B: oWeight, C: oOut,
            batch: 1, S: inputLength, M: hidden, N: hidden
        )

        // ----- 7. Residual add 1 -----
        addBuffers(input, oOut, residual1, count: inputLength * hidden)

        // ----- 8. LayerNorm 2 -----
        layerNormPipeline.normalize(
            commandBuffer: commandBuffer,
            input: residual1,
            gamma: ln2Gamma,
            beta: ln2Beta,
            output: ln2Out,
            batch: 1 * inputLength,
            dim: hidden
        )

        // ----- 9. FC1 (intermediate size) -----
        // ONNX: c_fc.weight is stored [4096, 1024] (out_dim=4096, in_dim=1024)
        // ONNX Gemm transB=0: Y = X @ W, so W is used AS-IS (no transpose)
        // matmul: A [S,1024] @ B [1024,4096] = [S,4096] ✓
        gemm.matmul(
            commandBuffer: commandBuffer,
            A: ln2Out, B: fc1Weight, C: fc1Out,
            batch: 1, S: inputLength, M: hidden, N: intermediate
        )

        // ----- 10. GELU activation -----
        applyGELU(commandBuffer: commandBuffer, input: fc1Out, output: geluOut, count: inputLength * intermediate)

        // ----- 11. FC2 -----
        // ONNX: c_proj.weight [4096, 1024], transB=0: Y = [S,4096] @ [4096,1024] = [S,1024]
        // Plain matmul (NO transpose) — W is already [in=4096, out=1024]
        gemm.matmul(
            commandBuffer: commandBuffer,
            A: geluOut, B: fc2Weight, C: fc2Out,
            batch: 1, S: inputLength, M: intermediate, N: hidden
        )

        // ----- 12. Residual add 2 -----
        addBuffers(residual1, fc2Out, output, count: inputLength * hidden)
    }

    // MARK: - Helper methods

    /// element-wise add: output = a + b
    private func addBuffers(_ a: MTLBuffer, _ b: MTLBuffer, _ output: MTLBuffer, count: Int) {
        let aPtr = a.contents().bindMemory(to: Float16.self, capacity: count)
        let bPtr = b.contents().bindMemory(to: Float16.self, capacity: count)
        let outPtr = output.contents().bindMemory(to: Float16.self, capacity: count)
        for i in 0..<count {
            outPtr[i] = aPtr[i] + bPtr[i]
        }
    }

    /// Add bias vector to buffer in-place (row-major).
    /// bias[offset..offset+count] is added to each row of buffer (which has width=hidden).
    /// offset: starting position in bias (0 for Q, hidden for K, 2*hidden for V).
    private func addBias(_ buffer: MTLBuffer, _ bias: MTLBuffer, offset: Int, count: Int) {
        let bufPtr = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        let biasPtr = bias.contents().bindMemory(to: Float16.self, capacity: offset + count)
        for i in 0..<count {
            bufPtr[i] = bufPtr[i] + biasPtr[offset + i]
        }
    }

    /// GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    /// Applied element-wise to the buffer.
    private func applyGELU(commandBuffer: MTLCommandBuffer, input: MTLBuffer, output: MTLBuffer, count: Int) {
        // CPU fallback — GELU kernel in Metal would be faster for large tensors
        let inp = input.contents().bindMemory(to: Float16.self, capacity: count)
        let out = output.contents().bindMemory(to: Float16.self, capacity: count)
        let sqrt2OverPi: Float = 0.7978845608028654  // sqrt(2/pi)
        let k: Float = 0.044715

        for i in 0..<count {
            let x = Float(inp[i])
            let x3 = x * x * x
            let tanhArg = sqrt2OverPi * (x + k * x3)
            let tanhVal = tanhf(tanhArg)  // Use tanhf for Float
            let gelu = x * 0.5 * (1.0 + tanhVal)
            out[i] = Float16(gelu)
        }
    }
}