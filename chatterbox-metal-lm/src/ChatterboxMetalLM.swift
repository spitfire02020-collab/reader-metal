import Foundation
import Metal

public final class ChatterboxMetalLM {
    public let device: MTLDevice
    public let maxNewTokens: Int
    public let repetitionPenalty: Float

    private let decoder: MetalLMDecode
    private let weightLoader: WeightLoader

    // Pre-allocated embedding buffers
    private let conditioningBuf: MTLBuffer
    private let textBuf: MTLBuffer
    private let speechEmbedBuf: MTLBuffer
    private let concatBuf: MTLBuffer

    private let hidden: Int
    private let maxSeq: Int

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
        self.hidden = MetalLMConfig.hiddenSize
        self.maxSeq = MetalLMConfig.maxSequenceLength

        // Load Metal library from bundle
        let bundle = Bundle(for: Self.self)
        guard let libPath = bundle.path(forResource: "default", ofType: "metallib"),
              let library = try? dev.makeLibrary(filepath: libPath) else {
            throw MetalLMError.libraryNotFound
        }

        self.weightLoader = try WeightLoader(device: dev, weightsDir: weightDirectory, library: library)
        self.decoder = try MetalLMDecode(device: dev, library: library, weightLoader: weightLoader, repetitionPenalty: repetitionPenalty)

        // Pre-allocate buffers for embeddings
        // Max sizes: conditioning=8192, text=8192, speech=1
        let fp16 = MemoryLayout<Float16>.size
        conditioningBuf  = dev.makeBuffer(length: maxSeq * hidden * fp16, options: .storageModeShared)!
        textBuf          = dev.makeBuffer(length: maxSeq * hidden * fp16, options: .storageModeShared)!
        speechEmbedBuf   = dev.makeBuffer(length: hidden * fp16, options: .storageModeShared)!
        concatBuf        = dev.makeBuffer(length: (maxSeq * 2 + 1) * hidden * fp16, options: .storageModeShared)!

        // Load and dequantize all weights ONCE
        _ = try weightLoader.loadAndDequantAllWeights()
    }

    public func reset() {
        decoder.reset()
    }

    /// Generate speech tokens autoregressively.
    ///
    /// Reference-style decode loop: each step processes the growing prefix
    /// (conditioning + text + all generated speech tokens) with a growing
    /// KV cache. This matches the reference ONNX implementation pattern.
    ///
    /// - Parameters:
    ///   - conditioning: Speaker conditioning embeddings [cond_len, 1024] as Float32
    ///   - textEmbed: Text embeddings [text_len, 1024] as Float32
    ///   - speechEmbed: Initial speech embedding [1, 1024] as Float32 (START_SPEECH embed)
    /// - Returns: Generated speech token IDs [num_tokens], excluding START_SPEECH and STOP_SPEECH
    public func generate(
        conditioning: [Float],
        textEmbed: [Float],
        speechEmbed: [Float]
    ) throws -> [Int32] {
        decoder.reset()

        let condLen = conditioning.count / hidden
        let textLen = textEmbed.count / hidden

        // Convert Float32 → Float16 and copy to Metal buffers
        float32ToFloat16(in: conditioning, out: conditioningBuf, count: conditioning.count)
        float32ToFloat16(in: textEmbed, out: textBuf, count: textEmbed.count)
        float32ToFloat16(in: speechEmbed, out: speechEmbedBuf, count: speechEmbed.count)

        var generatedTokens = [Int32]()
        generatedTokens.reserveCapacity(maxNewTokens)

        // Concatenate: [conditioning | text | speech_embed]
        // Each embedding is [seq_len, 1024] row-major fp16
        let prefixLen = condLen + textLen + 1
        concatEmbeddings(
            conditioning: conditioningBuf, condLen: condLen,
            text: textBuf, textLen: textLen,
            speech: speechEmbedBuf,
            output: concatBuf
        )

        for step in 0..<maxNewTokens {
            // Run decoder step: processes concatBuf (growing prefix) with KV cache
            let result = try decoder.step(
                inputsEmbed: concatBuf,
                inputLength: prefixLen + step,  // prefix grows by 1 each step
                generatedTokens: generatedTokens
            )

            let nextToken = result.nextToken

            if nextToken == MetalLMConfig.stopSpeechToken {
                break
            }

            // Skip START_SPEECH token in output
            if nextToken != MetalLMConfig.startSpeechToken {
                generatedTokens.append(nextToken)

                // Extend concatBuf with the new token's embedding for next step
                // speechEmbedBuf contains the new token's embedding
                // We need to append it to concatBuf for the next forward pass
                let newTokenOffset = (prefixLen + step) * hidden * MemoryLayout<Float16>.size
                let embPtr = speechEmbedBuf.contents()
                memcpy(concatBuf.contents().advanced(by: newTokenOffset), embPtr, hidden * MemoryLayout<Float16>.size)
            }
        }

        return generatedTokens
    }

    // MARK: - Helpers

    private func float32ToFloat16(in input: [Float], out output: MTLBuffer, count: Int) {
        let inPtr = input.withUnsafeBufferPointer { $0.baseAddress! }
        let outPtr = output.contents().bindMemory(to: Float16.self, capacity: count)
        for i in 0..<count {
            outPtr[i] = Float16(inPtr[i])
        }
    }

    /// Concatenate three embeddings along the sequence dimension.
    /// output = [conditioning | text | speech], each [seq, hidden]
    private func concatEmbeddings(
        conditioning: MTLBuffer,
        condLen: Int,
        text: MTLBuffer,
        textLen: Int,
        speech: MTLBuffer,
        output: MTLBuffer
    ) {
        let hiddenBytes = hidden * MemoryLayout<Float16>.size

        // Copy conditioning
        memcpy(output.contents(), conditioning.contents(), condLen * hiddenBytes)

        // Copy text after conditioning
        let textOffset = condLen * hiddenBytes
        memcpy(output.contents().advanced(by: textOffset), text.contents(), textLen * hiddenBytes)

        // Copy speech at the end
        let speechOffset = (condLen + textLen) * hiddenBytes
        memcpy(output.contents().advanced(by: speechOffset), speech.contents(), hiddenBytes)
    }
}

public enum MetalLMError: Error {
    case noMetalDevice
    case libraryNotFound
    case kernelNotFound(String)
    case commandBufferFailed
    case commandQueueFailed
}
