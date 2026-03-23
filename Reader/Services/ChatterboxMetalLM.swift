import Foundation
import Metal

/// High-level Metal language model wrapper — conforms to LanguageModelBackend.
///
/// This type wraps a MetalPipeline and implements the full autoregressive generate()
/// loop that ChatterboxEngine expects. It is NOT the same as the broken older
/// MetalLMDecode-based class that was excluded from the build.
///
/// The generate() loop follows the reference ONNX pattern:
///   1. Concatenate [conditioning | text | speech_embed] into a flat float16 buffer
///   2. Extend buffer with START_SPEECH tokens for the max decode length
///   3. Run prefill step to populate KV cache at position 0
///   4. Decode loop: greedy argmax → STOP_SPEECH → return generated tokens
public final class ChatterboxMetalLM: LanguageModelBackend, Sendable {

    // MARK: - Properties

    private let pipeline: MetalPipeline
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    /// Pre-allocated concatenated embeddings buffer [maxSeqLen, hidden] float16.
    /// Reused across generate() calls to avoid per-call allocation.
    private let concatBuf: MTLBuffer

    /// Pre-allocated hidden-state buffer [1, 1, hidden] float16.
    private let hiddenBuf: MTLBuffer!

    /// Scratch buffer for logits [vocabSize] float32.
    private let logitsScratch: UnsafeMutablePointer<Float>

    private let hidden: Int
    private let maxSeq: Int
    private let vocabSize: Int

    private let maxNewTokens: Int

    /// Current decode state (protected by actor-like serialization via generate())
    private var generatedTokens: [Int32] = []
    private var currentSeqLen: Int = 0

    // MARK: - Initialization

    /// - Parameters:
    ///   - device: Metal device
    ///   - weightsDir: Directory containing float16 weight binaries from metal-export
    ///   - maxNewTokens: Maximum number of new tokens to generate (default 1500)
    ///   - repetitionPenalty: Repetition penalty factor (default 1.2)
    public init(
        device: MTLDevice,
        weightsDir: URL,
        maxNewTokens: Int = 1500,
        repetitionPenalty: Float = 1.2
    ) throws {
        NSLog("[ChatterboxMetalLM] init START, device=%@", device.name)
        self.device = device
        self.hidden = MetalLMConfig.hiddenSize
        self.maxSeq = MetalLMConfig.maxSequenceLength
        self.vocabSize = MetalLMConfig.vocabSize
        self.maxNewTokens = maxNewTokens

        guard let q = device.makeCommandQueue() else {
            NSLog("[ChatterboxMetalLM] init FAIL: makeCommandQueue returned nil")
            throw MetalLMError.commandQueueFailed
        }
        self.commandQueue = q
        NSLog("[ChatterboxMetalLM] init: commandQueue created OK")

        NSLog("[ChatterboxMetalLM] init: creating MetalPipeline...")
        self.pipeline = try MetalPipeline(
            device: device,
            weightsDir: weightsDir,
            maxNewTokens: maxNewTokens,
            repetitionPenalty: repetitionPenalty
        )
        NSLog("[ChatterboxMetalLM] init: MetalPipeline created OK")

        // Pre-allocate buffers
        NSLog("[ChatterboxMetalLM] init: allocating concatBuf (\((maxSeq + 1) * hidden * MemoryLayout<Float16>.size) bytes)...")
        let fp16 = MemoryLayout<Float16>.size
        self.concatBuf = device.makeBuffer(
            length: (maxSeq + 1) * hidden * fp16,
            options: .storageModeShared
        )!
        NSLog("[ChatterboxMetalLM] init: concatBuf OK")

        NSLog("[ChatterboxMetalLM] init: allocating hiddenBuf (\(1 * 1 * hidden * fp16) bytes)...")
        self.hiddenBuf = device.makeBuffer(
            length: 1 * 1 * hidden * fp16,
            options: .storageModeShared
        )!
        NSLog("[ChatterboxMetalLM] init: hiddenBuf OK")

        NSLog("[ChatterboxMetalLM] init: allocating logitsScratch (\(vocabSize * MemoryLayout<Float>.size) bytes)...")
        self.logitsScratch = UnsafeMutablePointer<Float>.allocate(capacity: vocabSize)
        NSLog("[ChatterboxMetalLM] init: logitsScratch OK")

        NSLog("[ChatterboxMetalLM] init DONE")
    }

    deinit {
        logitsScratch.deallocate()
    }

    // MARK: - LanguageModelBackend

    /// Initialize the Metal pipeline with model configuration.
    /// Calls MetalPipeline.initialize() which compiles Metal kernels and loads weights.
    public func initialize(
        numLayers: Int,
        numKVHeads: Int,
        headDim: Int,
        maxSeqLen: Int,
        device: MTLDevice
    ) async throws {
        // MetalPipeline.initialize() uses MetalLMConfig internally — the
        // numLayers/numKVHeads/headDim/maxSeqLen params are accepted for
        // LanguageModelBackend protocol conformance but ignored here.
        try await pipeline.initialize()
    }

    /// Run one forward step of the language model.
    ///
    /// For prefill (kvReadLength == 0): processes full inputsEmbds with no KV cache.
    /// For decode steps: processes inputsEmbds[*, seqLen-1:*] at kvWriteOffset,
    ///   attending over kvReadLength positions from the KV cache.
    ///
    /// Note: inputsEmbds is expected to be the FULL growing sequence buffer
    /// [1, currentSeqLen, hidden]. The backend reads the LAST position.
    public func forward(
        inputsEmbds: MTLBuffer,
        kvWriteOffset: Int,
        kvReadLength: Int,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLBuffer {
        try pipeline.forward(
            inputsEmbds: inputsEmbds,
            kvWriteOffset: kvWriteOffset,
            kvReadLength: kvReadLength,
            commandBuffer: commandBuffer
        )
    }

    /// Reset KV cache and generated-token state.
    public func reset() async {
        await pipeline.reset()
        generatedTokens = []
        currentSeqLen = 0
    }

    /// Expose repetition penalty for use by external decode loops (e.g. metalDecodeLoop).
    public var repetitionPenalty: Float {
        pipeline.repetitionPenalty
    }

    // MARK: - Generate (convenience, not in protocol)

    /// Token embedding callback used during decode steps.
    /// The caller (metalDecodeLoop) uses ONNX embed_tokens to embed each generated token.
    public typealias EmbedTokenCallback = (Int32) throws -> [Float]

    /// Generate speech tokens autoregressively.
    ///
    /// Architecture matches ONNX reference decode path:
    ///   1. Prefill: forward on [conditioning | text] → KV cache populated at pos 0
    ///   2. First decode step: embed START_SPEECH (6561), forward, extract logits
    ///   3. Subsequent steps: embed each newly generated token, forward, extract logits
    ///   4. STOP_SPEECH (6562) terminates generation
    ///
    /// - Parameters:
    ///   - conditioning: Speaker conditioning embeddings [cond_len, 1024] Float32
    ///   - textEmbed: Text embeddings [text_len, 1024] Float32
    ///   - embedToken: Callback to embed a token ID → [1, 1024] Float32.
    ///                  metalDecodeLoop uses ONNX embed_tokens here.
    /// - Returns: Generated speech token IDs excluding START_SPEECH and STOP_SPEECH
    public func generate(
        conditioning: [Float],
        textEmbed: [Float],
        embedToken: @escaping EmbedTokenCallback
    ) throws -> [Int32] {
        generatedTokens = []
        currentSeqLen = 0

        let condLen = conditioning.count / hidden
        let textLen = textEmbed.count / hidden
        let fp16 = MemoryLayout<Float16>.size

        // Pointer to concatBuf as float16
        let concatPtr16 = concatBuf.contents().bindMemory(to: Float16.self, capacity: (maxSeq + 1) * hidden)

        // Copy conditioning [cond_len, hidden] at position 0
        conditioning.withUnsafeBufferPointer { src in
            let srcPtr = src.baseAddress!
            for i in 0..<(condLen * hidden) {
                concatPtr16[i] = Float16(srcPtr[i])
            }
        }

        // Copy text [text_len, hidden] after conditioning
        textEmbed.withUnsafeBufferPointer { src in
            let srcPtr = src.baseAddress!
            let dstOffset = condLen * hidden
            for i in 0..<(textLen * hidden) {
                concatPtr16[dstOffset + i] = Float16(srcPtr[i])
            }
        }

        // Prefix length = [conditioning | text], NO speech embed (matches ONNX)
        let totalPrefixLen = condLen + textLen

        // Prefill step: process full [conditioning | text] to populate KV cache at position 0.
        // kvWriteOffset=0, kvReadLength=0 (no past to attend to).
        // Each step needs its own command buffer — Metal buffers can't be reused after commit+wait.
        currentSeqLen = totalPrefixLen
        guard let prefillCmd = commandQueue.makeCommandBuffer() else {
            throw MetalLMError.commandBufferFailed
        }
        let prefillLogitsBuf = try forward(
            inputsEmbds: concatBuf,
            kvWriteOffset: 0,
            kvReadLength: 0,
            commandBuffer: prefillCmd
        )
        prefillCmd.commit()
        prefillCmd.waitUntilCompleted()

        // Inspect GPU state after prefill (hidden state and logits after commit+wait)
        let hiddenAbxmax = pipeline.getLastHiddenStateAbsmax()
        NSLog("[ChatterboxMetalLM] generate: post-prefill hidden absmax=%.6f", hiddenAbxmax)

        // Log prefill logits
        let prefillLogitsPtr = prefillLogitsBuf.contents().bindMemory(to: Float.self, capacity: vocabSize)
        var prefillMaxVal = prefillLogitsPtr[0]
        var prefillMaxIdx = 0
        for i in 1..<vocabSize {
            if prefillLogitsPtr[i] > prefillMaxVal {
                prefillMaxVal = prefillLogitsPtr[i]
                prefillMaxIdx = i
            }
        }
        NSLog("[ChatterboxMetalLM] generate: prefill logits max=\(prefillMaxVal), argmax=\(prefillMaxIdx), logits[0]=\(prefillLogitsPtr[0]), logits[6561]=\(prefillLogitsPtr[6561]), logits[6562]=\(prefillLogitsPtr[6562])")
        NSLog("[ChatterboxMetalLM] generate: prefill done, currentSeqLen=\(currentSeqLen)")

        // Decode loop
        for step in 0..<maxNewTokens {
            // Each decode step gets a fresh command buffer.
            // The same buffer cannot be committed twice; waitUntilCompleted() retires it.
            guard let cmd = commandQueue.makeCommandBuffer() else {
                throw MetalLMError.commandBufferFailed
            }

            // Forward step: reads concatBuf up to currentSeqLen, writes K/V at currentSeqLen-1,
            // attends over currentSeqLen positions.
            let logitsBuf = try forward(
                inputsEmbds: concatBuf,
                kvWriteOffset: currentSeqLen - 1,
                kvReadLength: currentSeqLen,
                commandBuffer: cmd
            )

            // Extract logits (wait for GPU to finish)
            cmd.commit()
            cmd.waitUntilCompleted()

            // Inspect hidden state after this decode step
            if step < 2 {
                let hAbsmax = pipeline.getLastHiddenStateAbsmax()
                NSLog("[ChatterboxMetalLM] generate: step=\(step) post-forward hidden absmax=%.6f", hAbsmax)
            }

            // Copy logits to scratch
            let logitsPtr = logitsBuf.contents().bindMemory(to: Float.self, capacity: vocabSize)
            if step < 3 {
                var maxVal = logitsPtr[0]
                var maxIdx = 0
                for i in 1..<vocabSize {
                    if logitsPtr[i] > maxVal {
                        maxVal = logitsPtr[i]
                        maxIdx = i
                    }
                }
                // Also check raw bytes to detect NaN/inf
                let raw0 = logitsPtr[0]
                NSLog("[ChatterboxMetalLM] generate: step=\(step) logits max=\(maxVal), argmax=\(maxIdx), logits[0]=\(raw0), isNaN=\(raw0.isNaN), isInf=\(raw0.isInfinite), logits[6561]=\(logitsPtr[6561])")
            }
            for i in 0..<vocabSize {
                logitsScratch[i] = logitsPtr[i]
            }

            // Apply repetition penalty
            let penalty = pipeline.repetitionPenalty
            for token in generatedTokens {
                let idx = Int(token)
                guard idx >= 0 && idx < vocabSize else { continue }
                if logitsScratch[idx] > 0 {
                    logitsScratch[idx] /= penalty
                } else {
                    logitsScratch[idx] *= penalty
                }
            }

            // Greedy argmax
            var bestIdx = 0
            var bestVal = logitsScratch[0]
            for i in 1..<vocabSize {
                if logitsScratch[i] > bestVal {
                    bestVal = logitsScratch[i]
                    bestIdx = i
                }
            }
            let nextToken = Int32(bestIdx)

            if step < 3 || step % 50 == 0 {
                NSLog("[ChatterboxMetalLM] generate: step=\(step), nextToken=\(nextToken), currentSeqLen=\(currentSeqLen)")
            }

            if nextToken == 6562 {  // STOP_SPEECH
                NSLog("[ChatterboxMetalLM] generate: STOP_SPEECH at step=\(step)")
                break
            }

            if nextToken != 6561 {  // skip START_SPEECH in output
                generatedTokens.append(nextToken)
            }

            // Embed the newly generated token and extend concatBuf for the next step.
            // This is the critical fix: use the REAL token embedding, not START_SPEECH.
            let nextEmbedding = try embedToken(nextToken)
            let newPos = currentSeqLen
            nextEmbedding.withUnsafeBufferPointer { src in
                let srcPtr = src.baseAddress!
                for i in 0..<hidden {
                    concatPtr16[newPos * hidden + i] = Float16(srcPtr[i])
                }
            }
            currentSeqLen += 1
        }

        NSLog("[ChatterboxMetalLM] generate: done, \(generatedTokens.count) tokens generated")

        return generatedTokens
    }
}

// MARK: - MetalLMError (canonical — shared with Q4F16Dequant)

public enum MetalLMError: Error, LocalizedError {
    case noMetalDevice
    case libraryNotFound
    case kernelNotFound(String)
    case commandBufferFailed
    case commandQueueFailed
    case metalDeviceUnavailable
    case weightNotLoaded(String)

    public var errorDescription: String? {
        switch self {
        case .noMetalDevice:
            return "No Metal-capable GPU device found"
        case .libraryNotFound:
            return "Metal shader library not found"
        case .kernelNotFound(let name):
            return "Metal kernel not found: \(name)"
        case .commandBufferFailed:
            return "Metal command buffer failed"
        case .commandQueueFailed:
            return "Metal command queue creation failed"
        case .metalDeviceUnavailable:
            return "Metal device unavailable"
        case .weightNotLoaded(let name):
            return "Metal weight not loaded: \(name)"
        }
    }
}
