import Foundation
import Metal

final class MetalLMForward {
    let device: MTLDevice
    let encoder: MetalLMEncoder
    let weightLoader: WeightLoader
    let kvCache: KVCacheManager
    let commandQueue: MTLCommandQueue

    // Pre-allocated intermediate buffer (reused across layers)
    private let hiddenBuf: MTLBuffer

    // For LM head logits computation
    private let logitsBuf: MTLBuffer

    // Weight names per layer (built from GPT2 ONNX manifest)
    private var layerWeightNames: [[String]] = []

    init(device: MTLDevice, library: MTLLibrary, weightLoader: WeightLoader) throws {
        self.device = device
        self.weightLoader = weightLoader
        self.encoder = try MetalLMEncoder(device: device, library: library)

        guard let q = device.makeCommandQueue() else {
            throw MetalLMError.commandQueueFailed
        }
        self.commandQueue = q

        self.kvCache = KVCacheManager(device: device)

        let hidden = MetalLMConfig.hiddenSize
        let maxSeq = MetalLMConfig.maxSequenceLength
        let vocab = MetalLMConfig.vocabSize
        let fp16 = MemoryLayout<Float16>.size

        hiddenBuf   = device.makeBuffer(length: maxSeq * hidden * fp16, options: .storageModeShared)!
        logitsBuf   = device.makeBuffer(length: vocab * MemoryLayout<Float32>.size, options: .storageModeShared)!

        loadManifest()
    }

    /// Single-step forward pass.
    /// Processes the full input sequence (prefix) at each step.
    /// For efficiency, implement KV cache append in a follow-up.
    ///
    /// inputs: [1, seq, 1024] fp16 row-major
    /// pastSequenceLength: unused in this simplified implementation
    /// Returns: logits buffer [vocab] as Float32 for the LAST position
    func forward(
        inputs: MTLBuffer,
        inputLength: Int,
        pastSequenceLength: Int  // unused — each step processes full prefix
    ) throws -> MTLBuffer {
        guard let cmd = commandQueue.makeCommandBuffer() else {
            throw MetalLMError.commandBufferFailed
        }

        var prevOutput = inputs

        for layer in 0..<MetalLMConfig.numLayers {
            let names = layerWeightNames[layer]

            // c_attn: [1024, 3072] → split into q, k, v each [1024, 1024]
            guard let cAttnWeight = weightLoader.dequantizedWeights[names[0]] else {
                throw MetalLMError.kernelNotFound("c_attn weight for layer \(layer)")
            }

            // Get remaining weights
            let ln1Gamma  = weightLoader.dequantizedWeights[names[1]]!
            let ln1Beta   = weightLoader.dequantizedWeights[names[2]]!
            let ln2Gamma  = weightLoader.dequantizedWeights[names[3]]!
            let ln2Beta   = weightLoader.dequantizedWeights[names[4]]!
            let cProj     = weightLoader.dequantizedWeights[names[5]]!
            let cFc1      = weightLoader.dequantizedWeights[names[6]]!
            let cFc2      = weightLoader.dequantizedWeights[names[7]]!

            // Reusable buffers for QKV projections
            let qBuf = device.makeBuffer(length: inputLength * MetalLMConfig.hiddenSize * MemoryLayout<Float16>.size, options: .storageModeShared)!
            let kBuf = device.makeBuffer(length: inputLength * MetalLMConfig.hiddenSize * MemoryLayout<Float16>.size, options: .storageModeShared)!
            let vBuf = device.makeBuffer(length: inputLength * MetalLMConfig.hiddenSize * MemoryLayout<Float16>.size, options: .storageModeShared)!

            // Split c_attn into Q, K, V weight buffers
            // Each is [hidden, hidden] = [1024, 1024]
            splitCAttn(cAttnWeight, q: qBuf, k: kBuf, v: vBuf, hidden: MetalLMConfig.hiddenSize, seq: inputLength)

            let outputBuf: MTLBuffer
            if layer < MetalLMConfig.numLayers - 1 {
                outputBuf = hiddenBuf
            } else {
                outputBuf = device.makeBuffer(
                    length: inputLength * MetalLMConfig.hiddenSize * MemoryLayout<Float16>.size,
                    options: .storageModeShared
                )!
            }

            encoder.forward(
                input: prevOutput,
                inputLength: inputLength,
                qWeight: qBuf,
                kActBuf: kBuf,
                vActBuf: vBuf,
                totalSeq: inputLength,
                oWeight: cProj,
                fc1Weight: cFc1,
                fc2Weight: cFc2,
                ln1Gamma: ln1Gamma,
                ln1Beta: ln1Beta,
                ln2Gamma: ln2Gamma,
                ln2Beta: ln2Beta,
                output: outputBuf,
                commandBuffer: cmd
            )

            if layer < MetalLMConfig.numLayers - 1 {
                memcpy(prevOutput.contents(), outputBuf.contents(),
                       inputLength * MetalLMConfig.hiddenSize * MemoryLayout<Float16>.size)
            } else {
                memcpy(hiddenBuf.contents(), outputBuf.contents(),
                       inputLength * MetalLMConfig.hiddenSize * MemoryLayout<Float16>.size)
            }
        }

        // ----- Final LayerNorm -----
        let lnFGamma = weightLoader.dequantizedWeights["transformer.ln_f.gamma"]!
        let lnFBeta  = weightLoader.dequantizedWeights["transformer.ln_f.beta"]!
        let finalLnBuf = device.makeBuffer(
            length: inputLength * MetalLMConfig.hiddenSize * MemoryLayout<Float16>.size,
            options: .storageModeShared
        )!

        encoder.layerNormPipeline.normalize(
            commandBuffer: cmd,
            input: hiddenBuf,
            gamma: lnFGamma,
            beta: lnFBeta,
            output: finalLnBuf,
            batch: 1 * inputLength,
            dim: MetalLMConfig.hiddenSize
        )

        // ----- LM Head -----
        // lm_head: [vocab=6563, hidden=1024]
        // final_ln_out: [inputLength, 1024] (row-major)
        // logits = final_ln_out @ lm_head^T = [inputLength, 1024] @ [1024, 6563] = [inputLength, vocab]
        let lmHead = weightLoader.dequantizedWeights["lm_head"]!

        let allLogitsBuf = device.makeBuffer(
            length: inputLength * MetalLMConfig.vocabSize * MemoryLayout<Float16>.size,
            options: .storageModeShared
        )!

        // [inputLength, hidden] @ [hidden, vocab] → [inputLength, vocab]
        encoder.gemm.matmul(
            commandBuffer: cmd,
            A: finalLnBuf,
            B: lmHead,
            C: allLogitsBuf,
            batch: 1,
            S: inputLength,
            M: MetalLMConfig.hiddenSize,
            N: MetalLMConfig.vocabSize
        )

        // Extract logits for last position, convert fp16 → fp32
        let lastPos = inputLength - 1
        let logitsPtr = allLogitsBuf.contents().bindMemory(to: Float16.self, capacity: MetalLMConfig.vocabSize)
        let outPtr = logitsBuf.contents().bindMemory(to: Float32.self, capacity: MetalLMConfig.vocabSize)
        let baseOffset = lastPos * MetalLMConfig.vocabSize
        for i in 0..<MetalLMConfig.vocabSize {
            outPtr[i] = Float32(logitsPtr[baseOffset + i])
        }

        cmd.commit()
        cmd.waitUntilCompleted()

        return logitsBuf
    }

    // MARK: - Helpers

    /// Split fused c_attn [hidden, 3*hidden] into q, k, v weight buffers.
    /// Each is [hidden, hidden] = [1024, 1024].
    /// Column layout of c_attn [1024, 3072]: [q_cols(1024) | k_cols(1024) | v_cols(1024)].
    private func splitCAttn(_ cAttn: MTLBuffer, q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, hidden: Int, seq: Int) {
        let hiddenBytes = hidden * MemoryLayout<Float16>.size
        let cAttnPtr = cAttn.contents().bindMemory(to: Float16.self, capacity: hidden * 3 * hidden)
        let qPtr = q.contents().bindMemory(to: Float16.self, capacity: hidden * hidden)
        let kPtr = k.contents().bindMemory(to: Float16.self, capacity: hidden * hidden)
        let vPtr = v.contents().bindMemory(to: Float16.self, capacity: hidden * hidden)

        // Copy q weight: rows 0..1023, columns 0..1023
        for row in 0..<hidden {
            let srcOffset = (row * 3 * hidden) * MemoryLayout<Float16>.size
            let dstOffset = row * hidden * MemoryLayout<Float16>.size
            memcpy(qPtr.advanced(by: dstOffset), cAttnPtr.advanced(by: srcOffset), hiddenBytes)
        }

        // Copy k weight: rows 0..1023, columns 1024..2047
        for row in 0..<hidden {
            let srcOffset = (row * 3 * hidden + hidden) * MemoryLayout<Float16>.size
            let dstOffset = row * hidden * MemoryLayout<Float16>.size
            memcpy(kPtr.advanced(by: dstOffset), cAttnPtr.advanced(by: srcOffset), hiddenBytes)
        }

        // Copy v weight: rows 0..1023, columns 2048..3071
        for row in 0..<hidden {
            let srcOffset = (row * 3 * hidden + 2 * hidden) * MemoryLayout<Float16>.size
            let dstOffset = row * hidden * MemoryLayout<Float16>.size
            memcpy(vPtr.advanced(by: dstOffset), cAttnPtr.advanced(by: srcOffset), hiddenBytes)
        }
    }

    private func loadManifest() {
        // GPT2 ONNX manifest weight names (base names):
        //   h.N.attn.c_attn     — fused qkv [1024, 3072]
        //   h.N.attn.ln1.gamma / .beta
        //   h.N.mlp.ln2.gamma / .beta
        //   h.N.attn.c_proj    — [1024, 1024]
        //   h.N.mlp.c_fc1     — [4096, 1024]
        //   h.N.mlp.c_fc2     — [1024, 4096]
        for layer in 0..<MetalLMConfig.numLayers {
            let p = "h.\(layer)"
            layerWeightNames.append([
                "\(p).attn.c_attn",
                "\(p).attn.ln1.gamma",
                "\(p).attn.ln1.beta",
                "\(p).mlp.ln2.gamma",
                "\(p).mlp.ln2.beta",
                "\(p).attn.c_proj",
                "\(p).mlp.c_fc1",
                "\(p).mlp.c_fc2",
            ])
        }
    }
}
