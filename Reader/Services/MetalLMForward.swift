import Foundation
import Metal

final class MetalLMForward {
    let device: MTLDevice
    let encoder: MetalLMEncoder
    let weightLoader: WeightLoader
    let commandQueue: MTLCommandQueue

    // Pre-allocated intermediate buffer (reused across layers)
    private let hiddenBuf: MTLBuffer

    // For LM head logits computation
    private let logitsBuf: MTLBuffer

    // Per-layer weight ONNX names (GPT2NoEmbed convention)
    // Each layer has 12 entries: c_attn, c_attn_bias, c_proj, c_proj_bias, ln_1_w, ln_1_b, ln_2_w, ln_2_b, c_fc_w, c_fc_b, c_proj_w, c_proj_b
    private var layerWeightNames: [[String]] = []

    init(device: MTLDevice, library: MTLLibrary, weightLoader: WeightLoader) throws {
        self.device = device
        self.weightLoader = weightLoader
        self.encoder = try MetalLMEncoder(device: device, library: library)

        guard let q = device.makeCommandQueue() else {
            throw MetalLMError.commandQueueFailed
        }
        self.commandQueue = q

        let hidden = MetalLMConfig.hiddenSize
        let maxSeq = MetalLMConfig.maxSequenceLength
        let vocab = MetalLMConfig.vocabSize

        hiddenBuf   = device.makeBuffer(length: maxSeq * hidden * MemoryLayout<Float16>.size, options: .storageModeShared)!
        logitsBuf   = device.makeBuffer(length: vocab * MemoryLayout<Float32>.size, options: .storageModeShared)!

        loadManifest()
    }

    /// Single-step forward pass.
    /// Processes the full input sequence (prefix) at each step.
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

            // Load weights by ONNX name
            guard let cAttnWeight = weightLoader.dequantizedWeights[names[0]],
                  let cAttnBias = weightLoader.dequantizedWeights[names[1]] else {
                throw MetalLMError.kernelNotFound("c_attn weights for layer \(layer)")
            }
            let ln1Gamma  = weightLoader.dequantizedWeights[names[4]]!
            let ln1Beta   = weightLoader.dequantizedWeights[names[5]]!
            let ln2Gamma  = weightLoader.dequantizedWeights[names[6]]!
            let ln2Beta   = weightLoader.dequantizedWeights[names[7]]!
            let cProjWeight = weightLoader.dequantizedWeights[names[10]]!
            let cFcWeight   = weightLoader.dequantizedWeights[names[8]]!

            // Allocate QKV weight buffers — split from fused c_attn [1024, 3072]
            let hidden = MetalLMConfig.hiddenSize
            let hiddenBytes = hidden * MemoryLayout<Float16>.size
            let qWeightBuf = device.makeBuffer(length: hiddenBytes, options: .storageModeShared)!
            let kWeightBuf = device.makeBuffer(length: hiddenBytes, options: .storageModeShared)!
            let vWeightBuf = device.makeBuffer(length: hiddenBytes, options: .storageModeShared)!

            // Split fused c_attn [1024, 3072] → Q/K/V each [1024, 1024]
            splitCAttn(cAttnWeight, q: qWeightBuf, k: kWeightBuf, v: vWeightBuf, hidden: hidden)

            // Allocate K/V activation buffers (used for attention)
            let kActBuf = device.makeBuffer(length: inputLength * hidden * MemoryLayout<Float16>.size, options: .storageModeShared)!
            let vActBuf = device.makeBuffer(length: inputLength * hidden * MemoryLayout<Float16>.size, options: .storageModeShared)!

            let outputBuf: MTLBuffer
            if layer < MetalLMConfig.numLayers - 1 {
                outputBuf = hiddenBuf
            } else {
                outputBuf = device.makeBuffer(
                    length: inputLength * hidden * MemoryLayout<Float16>.size,
                    options: .storageModeShared
                )!
            }

            encoder.forward(
                input: prevOutput,
                inputLength: inputLength,
                qWeight: qWeightBuf,
                kWeight: kWeightBuf,
                vWeight: vWeightBuf,
                cAttnBias: cAttnBias,
                kActBuf: kActBuf,
                vActBuf: vActBuf,
                totalSeq: inputLength,
                oWeight: cProjWeight,
                fc1Weight: cFcWeight,
                ln1Gamma: ln1Gamma,
                ln1Beta: ln1Beta,
                ln2Gamma: ln2Gamma,
                ln2Beta: ln2Beta,
                output: outputBuf,
                commandBuffer: cmd
            )

            if layer < MetalLMConfig.numLayers - 1 {
                memcpy(prevOutput.contents(), outputBuf.contents(),
                       inputLength * hidden * MemoryLayout<Float16>.size)
            } else {
                memcpy(hiddenBuf.contents(), outputBuf.contents(),
                       inputLength * hidden * MemoryLayout<Float16>.size)
            }
        }

        // ----- Final LayerNorm -----
        guard let lnFGamma = weightLoader.dequantizedWeights["ln_f.weight"],
              let lnFBeta  = weightLoader.dequantizedWeights["ln_f.bias"] else {
            throw MetalLMError.kernelNotFound("ln_f weights")
        }
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
        // final_ln_out [inputLength, 1024] @ lm_head [6563, 1024] = [inputLength, 6563]
        // ONNX weight: [out_dim=6563, in_dim=1024], matmul expects [M,N] @ [N,K] = [M,K]
        // So: matmul(A [S,1024], B [1024,6563]) → matmul (NOT transposeB)
        guard let lmHeadWeight = weightLoader.dequantizedWeights["lm_head.weight"] else {
            throw MetalLMError.kernelNotFound("lm_head.weight")
        }

        let allLogitsBuf = device.makeBuffer(
            length: inputLength * MetalLMConfig.vocabSize * MemoryLayout<Float16>.size,
            options: .storageModeShared
        )!

        encoder.gemm.matmul(
            commandBuffer: cmd,
            A: finalLnBuf,
            B: lmHeadWeight,
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
    /// Column layout of c_attn [1024, 3072]: [q_cols(1024) | k_cols(1024) | v_cols(1024)].
    private func splitCAttn(_ cAttn: MTLBuffer, q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, hidden: Int) {
        let hiddenBytes = hidden * MemoryLayout<Float16>.size
        let cAttnPtr = cAttn.contents().bindMemory(to: Float16.self, capacity: hidden * 3 * hidden)
        let qPtr = q.contents().bindMemory(to: Float16.self, capacity: hidden * hidden)
        let kPtr = k.contents().bindMemory(to: Float16.self, capacity: hidden * hidden)
        let vPtr = v.contents().bindMemory(to: Float16.self, capacity: hidden * hidden)

        // Q: rows 0..1023, columns 0..1023
        for row in 0..<hidden {
            let srcOffset = (row * 3 * hidden) * MemoryLayout<Float16>.size
            let dstOffset = row * hidden * MemoryLayout<Float16>.size
            memcpy(qPtr.advanced(by: dstOffset), cAttnPtr.advanced(by: srcOffset), hiddenBytes)
        }
        // K: rows 0..1023, columns 1024..2047
        for row in 0..<hidden {
            let srcOffset = (row * 3 * hidden + hidden) * MemoryLayout<Float16>.size
            let dstOffset = row * hidden * MemoryLayout<Float16>.size
            memcpy(kPtr.advanced(by: dstOffset), cAttnPtr.advanced(by: srcOffset), hiddenBytes)
        }
        // V: rows 0..1023, columns 2048..3071
        for row in 0..<hidden {
            let srcOffset = (row * 3 * hidden + 2 * hidden) * MemoryLayout<Float16>.size
            let dstOffset = row * hidden * MemoryLayout<Float16>.size
            memcpy(vPtr.advanced(by: dstOffset), cAttnPtr.advanced(by: srcOffset), hiddenBytes)
        }
    }

    private func loadManifest() {
        // GPT2NoEmbed ONNX weight names (verified from metal-export/language_model_fp16.onnx)
        //
        // Per-layer (h.{N}.*):
        //   attn.c_attn.weight  — fused QKV [1024, 3072]
        //   attn.c_attn.bias    — QKV bias [3072]
        //   attn.c_proj.weight  — output projection [1024, 1024]
        //   attn.c_proj.bias    — output bias [1024]
        //   ln_1.weight / .bias — attention LayerNorm
        //   ln_2.weight / .bias — MLP LayerNorm
        //   mlp.c_fc.weight     — expand: [4096, 1024]
        //   mlp.c_fc.bias       — expand bias: [4096]
        //   mlp.c_proj.weight   — contract: [4096, 1024]
        //   mlp.c_proj.bias     — contract bias: [4096]
        //
        // Root-level:
        //   ln_f.weight / .bias — final LayerNorm
        //   lm_head.weight / .bias — [6563, 1024]
        for layer in 0..<MetalLMConfig.numLayers {
            let p = "h.\(layer)"
            layerWeightNames.append([
                "\(p).attn.c_attn.weight",   // [1024, 3072]
                "\(p).attn.c_attn.bias",     // [3072]
                "\(p).attn.c_proj.weight",   // [1024, 1024]
                "\(p).attn.c_proj.bias",     // [1024]
                "\(p).ln_1.weight",           // [1024]
                "\(p).ln_1.bias",            // [1024]
                "\(p).ln_2.weight",          // [1024]
                "\(p).ln_2.bias",            // [1024]
                "\(p).mlp.c_fc.weight",     // [4096, 1024]
                "\(p).mlp.c_fc.bias",       // [4096]
                "\(p).mlp.c_proj.weight",   // [4096, 1024]
                "\(p).mlp.c_proj.bias",     // [4096]
            ])
        }
    }
}
