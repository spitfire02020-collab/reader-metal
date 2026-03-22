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
        let prefixLen = condLen + textLen

        guard let cmd = commandQueue.makeCommandBuffer() else {
            throw MetalLMError.commandBufferFailed
        }

        // Prefill step: process full [conditioning | text] to populate KV cache at position 0.
        // kvWriteOffset=0, kvReadLength=0 (no past to attend to).
        currentSeqLen = prefixLen
        _ = try forward(
            inputsEmbds: concatBuf,
            kvWriteOffset: 0,
            kvReadLength: 0,
            commandBuffer: cmd
        )
        cmd.commit()
        cmd.waitUntilCompleted()

        // Decode loop
        for _ in 0..<maxNewTokens {
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

            // Copy logits to scratch
            let logitsPtr = logitsBuf.contents().bindMemory(to: Float.self, capacity: vocabSize)
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

            if nextToken == MetalLMConfig.stopSpeechToken {
                break
            }

            if nextToken != MetalLMConfig.startSpeechToken {
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
