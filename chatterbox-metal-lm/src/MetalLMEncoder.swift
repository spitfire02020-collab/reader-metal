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
    /// qWeight/kWeight/vWeight are the QKV projection weight matrices (fp16).
    /// kActBuf/vActBuf are the full K/V activation buffers (past + new, [1, totalSeq, hidden]).
    /// These bypass the internal kBuf/vBuf for K/V to avoid re-copying.
    func forward(
        input: MTLBuffer,
        inputLength: Int,
        qWeight: MTLBuffer,
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

        // ----- 2. Q projection (matmulTransposeB) -----
        // Q = ln1_out @ W_q^T → qBuf [1, inputLength, hidden]
        gemm.matmulTransposeB(
            commandBuffer: commandBuffer,
            A: ln1Out, B: qWeight, C: qBuf,
            batch: 1, S: inputLength, M: hidden, N: hidden
        )

        // ----- 3. SDPA with causal mask -----
        // Q: [1, inputLength, 1024]
        // K: [1, totalSeq, 1024] (past + new, pre-concatenated)
        // V: [1, totalSeq, 1024] (past + new, pre-concatenated)
        // Attention output: [1, inputLength, 1024]
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

        // ----- 4. O projection -----
        // attn_out [1, inputLength, 1024] @ W_o^T → o_out [1, inputLength, 1024]
        gemm.matmulTransposeB(
            commandBuffer: commandBuffer,
            A: attnOut, B: oWeight, C: oOut,
            batch: 1, S: inputLength, M: hidden, N: hidden
        )

        // ----- 5. Residual add 1 -----
        addBuffers(input, oOut, residual1, count: inputLength * hidden)

        // ----- 6. LayerNorm 2 -----
        layerNormPipeline.normalize(
            commandBuffer: commandBuffer,
            input: residual1,
            gamma: ln2Gamma,
            beta: ln2Beta,
            output: ln2Out,
            batch: 1 * inputLength,
            dim: hidden
        )

        // ----- 7. FC1 (intermediate size) -----
        // ln2_out [1, inputLength, 1024] @ W_fc1^T → fc1Out [1, inputLength, 4096]
        gemm.matmulTransposeB(
            commandBuffer: commandBuffer,
            A: ln2Out, B: fc1Weight, C: fc1Out,
            batch: 1, S: inputLength, M: hidden, N: intermediate
        )

        // ----- 8. GELU activation -----
        applyGELU(commandBuffer: commandBuffer, input: fc1Out, output: geluOut, count: inputLength * intermediate)

        // ----- 9. FC2 -----
        // gelu_out [1, inputLength, 4096] @ W_fc2^T → fc2_out [1, inputLength, 1024]
        gemm.matmulTransposeB(
            commandBuffer: commandBuffer,
            A: geluOut, B: fc2Weight, C: fc2Out,
            batch: 1, S: inputLength, M: intermediate, N: hidden
        )

        // ----- 10. Residual add 2 -----
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