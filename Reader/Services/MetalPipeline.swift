import Foundation
import Metal

// MARK: - MetalPipeline

/// High-level Metal language model pipeline.
/// Handles the autoregressive decode loop with greedy sampling + repetition penalty.
/// Wraps MetalLMBackend for per-layer computation.
///
/// Reference-style decode loop (matches Python/ONNX reference):
///   - Each step processes the growing full prefix [conditioning | text | speech_tokens_so_far]
///   - KV cache accumulates across steps for efficient attention
///   - Greedy decoding: argmax + repetition penalty (penalty > 1.0 discourages repeats)
public final class MetalPipeline: LanguageModelBackend, @unchecked Sendable {

    // MARK: - Properties

    private let backend: MetalLMBackend
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let maxNewTokens: Int
    let repetitionPenalty: Float

    // Current decode state
    private var generatedTokens: [Int32] = []
    private var currentSeqLen: Int = 0  // total sequence length (prefix + generated)

    // Buffers for current step's hidden state
    private var hiddenBuffer: MTLBuffer!   // [1, 1, 1024] half — current input embedding
    private var logitsBuffer: MTLBuffer!   // [vocabSize] float32 — logits output

    // Preallocated scratch for logits extraction
    private var logitsScratch: [Float] = []

    // MARK: - Initialization

    /// - Parameters:
    ///   - device: Metal device
    ///   - weightsDir: Directory containing float16 weight binaries from metal-export
    ///   - maxNewTokens: Maximum number of new tokens to generate (default 1500)
    ///   - repetitionPenalty: Penalty for repeated tokens (default 1.2, > 1.0 penalizes repeats)
    public init(
        device: MTLDevice,
        weightsDir: URL,
        maxNewTokens: Int = 1500,
        repetitionPenalty: Float = 1.2
    ) throws {
        NSLog("[MetalPipeline] init START")
        self.device = device
        self.maxNewTokens = maxNewTokens
        self.repetitionPenalty = repetitionPenalty
        NSLog("[MetalPipeline] init: creating MetalLMBackend...")
        self.backend = try MetalLMBackend(device: device, weightsDir: weightsDir)
        NSLog("[MetalPipeline] init: MetalLMBackend created OK")

        guard let q = device.makeCommandQueue() else {
            throw MetalPipelineError.commandQueueFailed
        }
        self.commandQueue = q

        // Pre-allocate scratch buffers
        self.hiddenBuffer = device.makeBuffer(
            length: 1 * 1 * MetalLMConfig.hiddenSize * MemoryLayout<Float16>.size,
            options: .storageModeShared
        )!
        self.logitsBuffer = device.makeBuffer(
            length: MetalLMConfig.vocabSize * MemoryLayout<Float32>.size,
            options: .storageModeShared
        )!
        self.logitsScratch = [Float](repeating: 0, count: MetalLMConfig.vocabSize)
        NSLog("[MetalPipeline] init DONE")
    }

    // MARK: - LanguageModelBackend conformance

    /// Initialize the backend with model configuration.
    /// LanguageModelBackend protocol requires this specific signature.
    public func initialize(
        numLayers: Int,
        numKVHeads: Int,
        headDim: Int,
        maxSeqLen: Int,
        device: MTLDevice
    ) async throws {
        // MetalLMConfig values are used internally; the passed parameters
        // are accepted for protocol conformance but ignored here.
        try await backend.initialize(
            numLayers: MetalLMConfig.numLayers,
            numKVHeads: MetalLMConfig.numKVHeads,
            headDim: MetalLMConfig.headDim,
            maxSeqLen: MetalLMConfig.maxSequenceLength,
            device: device
        )
    }

    /// Convenience initialize — uses stored device and MetalLMConfig values.
    /// This matches the worktree signature for compatibility with ChatterboxMetalLM.
    public func initialize() async throws {
        try await backend.initialize(
            numLayers: MetalLMConfig.numLayers,
            numKVHeads: MetalLMConfig.numKVHeads,
            headDim: MetalLMConfig.headDim,
            maxSeqLen: MetalLMConfig.maxSequenceLength,
            device: device
        )
    }

    /// Synchronous forward pass (GPU dispatch only, no waiting).
    public func forward(
        inputsEmbds: MTLBuffer,
        kvWriteOffset: Int,
        kvReadLength: Int,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLBuffer {
        try backend.forward(
            inputsEmbds: inputsEmbds,
            kvWriteOffset: kvWriteOffset,
            kvReadLength: kvReadLength,
            commandBuffer: commandBuffer
        )
    }

    /// Reset KV cache state.
    public func reset() async {
        await backend.reset()
        generatedTokens = []
        currentSeqLen = 0
    }

    /// Return the absmax of the last hidden state from MetalLMBackend's residual buffer.
    /// Safe to call after commit+wait.
    public func getLastHiddenStateAbsmax() -> Float {
        backend.getLastHiddenStateAbsmax()
    }

    /// Inspect the final hidden state and logits from the last forward pass.
    /// Must be called after commit+wait on the command buffer used for the forward pass.
    public func inspectFinalState(hidden: MTLBuffer, logitsBuf: MTLBuffer) {
        backend.inspectFinalState(hidden: hidden, logitsBuf: logitsBuf)
    }

    // MARK: - Decode Loop

    /// Main decode entrypoint — generates speech tokens autoregressively.
    ///
    /// Reference-style loop:
    ///   1. Prefill: process full prefix (conditioning + text + START_SPEECH) → KV cache at pos 0
    ///   2. Decode loop (maxNewTokens iterations):
    ///      a. Forward pass with current sequence → logits at last position
    ///      b. Apply repetition penalty to logits
    ///      c. Greedy argmax → next token
    ///      d. STOP_SPEECH (6562) → break
    ///      e. Append token, advance sequence
    ///
    /// - Parameters:
    ///   - inputsEmbds: Full input embeddings [1, prefillLen, 1024] half.
    ///                  Includes conditioning + text + START_SPEECH (6561) initial token embed.
    ///   - initialKVLen: Number of prefill positions to attend over (typically prefillLen).
    ///                  KV cache write starts at offset 0.
    /// - Returns: Generated speech token IDs [N], excluding START_SPEECH and STOP_SPEECH
    public func decode(
        inputsEmbds: MTLBuffer,
        initialKVLen: Int
    ) async throws -> [Int32] {
        guard let cmd = commandQueue.makeCommandBuffer() else {
            throw MetalPipelineError.commandBufferFailed
        }

        generatedTokens = []
        currentSeqLen = initialKVLen

        // Prefill step: forward full prefix to populate KV cache at position 0
        // kvWriteOffset = 0 (first position), kvReadLength = 0 (no past to attend to)
        var logitsBuf = try backend.forward(
            inputsEmbds: inputsEmbds,
            kvWriteOffset: 0,
            kvReadLength: 0,
            commandBuffer: cmd
        )
        cmd.commit()
        _ = await cmd.completed()

        // Decode loop
        for _ in 0..<maxNewTokens {
            // Extract logits for the last position (from previous forward pass)
            extractLogits(from: logitsBuf, to: &logitsScratch)

            // Apply repetition penalty based on generated tokens so far
            applyRepetitionPenalty(&logitsScratch, tokens: generatedTokens)

            // Greedy argmax
            let nextToken = greedyArgmax(logitsScratch)

            // Check for stop
            if nextToken == MetalLMConfig.stopSpeechToken {
                break
            }

            // Skip START_SPEECH in output (it's the initial tracking token)
            if nextToken != MetalLMConfig.startSpeechToken {
                generatedTokens.append(nextToken)
            }

            // Advance: the caller (ChatterboxEngine) extends inputsEmbds by
            // appending the embedding for nextToken. We track the new length.
            // kvWriteOffset = currentSeqLen (where to write this step's K/V)
            // kvReadLength = currentSeqLen + 1 (all positions including new one)
            currentSeqLen += 1

            logitsBuf = try backend.forward(
                inputsEmbds: inputsEmbds,
                kvWriteOffset: currentSeqLen - 1,
                kvReadLength: currentSeqLen,
                commandBuffer: cmd
            )
        }

        // Final commit — last forward() was already submitted above
        cmd.commit()

        return generatedTokens
    }

    /// Decode with a single decode step (after prefill).
    /// This is a pure forward-only operation — no waiting, no extraction, no argmax.
    /// The caller (generate() or metalDecodeLoop) manages the full loop:
    ///   1. Extract logits from the logitsBuf returned by the PREVIOUS decodeStep/prefill
    ///   2. Apply repetition penalty + greedy argmax → next token
    ///   3. Embed next token → extend sequence buffer
    ///   4. Call decodeStep() for the next forward pass
    ///
    /// - Parameters:
    ///   - currentEmbedding: Current input embedding [1, 1, 1024] half at the new position
    ///   - kvWriteOffset: Position in KV cache to write this step's keys/values
    ///   - kvReadLength: Total positions to attend over (kvWriteOffset + 1)
    ///   - commandBuffer: Command buffer to encode this step's forward pass
    /// - Returns: MTLBuffer containing logits [vocabSize] Float32 from this step's forward pass.
    ///            Caller must cmd.commit() + cmd.waitUntilCompleted() before extracting logits.
    public func decodeStep(
        currentEmbedding: MTLBuffer,
        kvWriteOffset: Int,
        kvReadLength: Int,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLBuffer {
        return try backend.forward(
            inputsEmbds: currentEmbedding,
            kvWriteOffset: kvWriteOffset,
            kvReadLength: kvReadLength,
            commandBuffer: commandBuffer
        )
    }

    /// Decode with a single decode step including waiting and extraction.
    /// Convenience method for callers that want a self-contained step.
    /// NOTE: For the correct pipelined decode loop, prefer the synchronous decodeStep()
    /// above and manage the loop externally.
    ///
    /// - Parameters:
    ///   - currentEmbedding: Current input embedding [1, 1, 1024] half at the new position
    ///   - kvWriteOffset: Position in KV cache to write this step's keys/values
    ///   - kvReadLength: Total positions to attend over (kvWriteOffset + 1)
    /// - Returns: Next token ID (Int32), or STOP_SPEECH if generation is complete
    public func decodeStepWithExtraction(
        currentEmbedding: MTLBuffer,
        kvWriteOffset: Int,
        kvReadLength: Int
    ) async throws -> Int32 {
        guard let cmd = commandQueue.makeCommandBuffer() else {
            throw MetalPipelineError.commandBufferFailed
        }

        let logitsBuf = try backend.forward(
            inputsEmbds: currentEmbedding,
            kvWriteOffset: kvWriteOffset,
            kvReadLength: kvReadLength,
            commandBuffer: cmd
        )

        cmd.commit()
        _ = await cmd.completed()

        // Extract logits
        let logitsPtr = logitsBuf.contents().bindMemory(
            to: Float.self,
            capacity: MetalLMConfig.vocabSize
        )
        logitsScratch = Array(UnsafeBufferPointer(start: logitsPtr, count: MetalLMConfig.vocabSize))

        // Apply repetition penalty
        applyRepetitionPenalty(&logitsScratch, tokens: generatedTokens)

        // Greedy argmax
        let nextToken = greedyArgmax(logitsScratch)

        if nextToken != MetalLMConfig.startSpeechToken && nextToken != MetalLMConfig.stopSpeechToken {
            generatedTokens.append(nextToken)
        }

        return nextToken
    }

    // MARK: - Repetition Penalty

    /// Apply repetition penalty to logits.
    /// For each previously generated token:
    ///   - If logits[token] > 0: divide by penalty (suppress positive logits)
    ///   - If logits[token] < 0: multiply by penalty (enhance negative logits)
    /// This matches the reference implementation's rep_penalty logic.
    private func applyRepetitionPenalty(_ logits: inout [Float], tokens: [Int32]) {
        let penalty = repetitionPenalty
        for token in tokens {
            let idx = Int(token)
            guard idx >= 0 && idx < logits.count else { continue }
            if logits[idx] > 0 {
                logits[idx] /= penalty
            } else {
                logits[idx] *= penalty
            }
        }
    }

    // MARK: - Greedy Decoding

    /// Pure greedy argmax — returns the token ID with the highest logit.
    private func greedyArgmax(_ logits: [Float]) -> Int32 {
        var maxIdx = 0
        var maxVal = logits[0]
        for i in 1..<logits.count {
            if logits[i] > maxVal {
                maxVal = logits[i]
                maxIdx = i
            }
        }
        return Int32(maxIdx)
    }

    // MARK: - Helpers

    /// Extract logits from a Metal buffer (float32) into a Swift array.
    private func extractLogits(from buffer: MTLBuffer, to scratch: inout [Float]) {
        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: MetalLMConfig.vocabSize)
        for i in 0..<MetalLMConfig.vocabSize {
            scratch[i] = ptr[i]
        }
    }
}

// MARK: - MetalPipelineError

public enum MetalPipelineError: Error, LocalizedError {
    case commandQueueFailed
    case commandBufferFailed
    case backendError(String)

    public var errorDescription: String? {
        switch self {
        case .commandQueueFailed:
            return "Metal pipeline: could not create command queue"
        case .commandBufferFailed:
            return "Metal pipeline: could not create command buffer"
        case .backendError(let msg):
            return "Metal pipeline backend error: \(msg)"
        }
    }
}
