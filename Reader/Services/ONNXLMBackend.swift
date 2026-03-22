import Foundation
import Metal
import OnnxRuntimeBindings

// MARK: - ONNX Language Model Backend

/// ONNX-based language model backend wrapping an existing ORTSession decode loop.
///
/// This backend replicates the ORT decode step behavior from ChatterboxEngine:
///   - Maintains KV cache state as ORTValue tensors internally
///   - Runs one ORTSession.run() per forward() call
///   - Returns logits as MTLBuffer for the caller to apply greedy decoding
///
/// The full decode loop (token generation, repetition penalty, STOP_SPEECH detection)
/// stays in ChatterboxEngine; ONNXLMBackend only handles the single-step inference.
final class ONNXLMBackend: LanguageModelBackend {

    // MARK: - Properties

    /// The ONNX Runtime session for the language model.
    private let ortSession: ORTSession

    /// ONNX input tensor names (e.g. "inputs_embeds", "past_key_values.0.key", ...).
    private var lmInputNames: [String] = []

    /// ONNX output tensor names.
    private var lmOutputNames: [String] = []

    /// Logits output name resolved from lmOutputNames.
    private var logitsOutputName: String = "logits"

    /// Number of transformer layers (inferred from input names).
    private(set) var numLayers: Int = 0

    /// Number of KV attention heads (GQA).
    private(set) var numKVHeads: Int = 16

    /// Head dimension per attention head.
    private(set) var headDim: Int = 64

    /// Maximum sequence length for pre-allocation.
    private(set) var maxSeqLen: Int = 1500

    /// Current KV cache state — carried across forward() calls.
    /// Key: "past_key_values.{layer}.key" / "past_key_values.{layer}.value"
    /// Value: ORTValue tensor [1, numKVHeads, pastLen, headDim] float16
    private var kvCache: [String: ORTValue] = [:]

    /// Reusable token ID buffer for position_ids (avoids per-step heap allocation).
    private let reusableTokenBuffer = UnsafeMutableBufferPointer<Int64>.allocate(capacity: 1)

    /// Reusable position ID buffer for position_ids (avoids per-step heap allocation).
    private let reusablePositionBuffer = UnsafeMutableBufferPointer<Int64>.allocate(capacity: 1)

    /// Reusable mask buffer for attention_mask (pre-allocated for max length).
    private var reusableMaskBuffer: UnsafeMutableBufferPointer<Int64>?

    /// Total number of tokens processed so far (used for position_ids).
    private var totalProcessedLen: Int = 0

    /// Whether initialize() has been called.
    private var isInitialized: Bool = false

    /// Device used for allocating output MTLBuffers.
    private var device: MTLDevice?

    /// Pre-allocated logits output buffer (reused across steps).
    private var logitsOutputBuffer: MTLBuffer?

    // MARK: - Initialization

    /// Create a new ONNX LM backend wrapping an existing ORTSession.
    ///
    /// The ORTSession is typically obtained from ChatterboxEngine.languageModelSession.
    /// Since ORTSession is created and owned by ChatterboxEngine, this backend holds
    /// a reference to it without owning it.
    ///
    /// - Parameter ortSession: The ONNX Runtime session for the language model.
    public init(ortSession: ORTSession) {
        self.ortSession = ortSession
    }

    deinit {
        reusableTokenBuffer.deallocate()
        reusablePositionBuffer.deallocate()
        reusableMaskBuffer?.deallocate()
    }

    // MARK: - LanguageModelBackend

    public func initialize(
        numLayers: Int,
        numKVHeads: Int,
        headDim: Int,
        maxSeqLen: Int,
        device: MTLDevice
    ) async throws {
        self.numLayers = numLayers
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.maxSeqLen = maxSeqLen
        self.device = device
        self.isInitialized = true

        // Retrieve input/output names from the ORT session.
        self.lmInputNames = try ortSession.inputNames() as [String]
        self.lmOutputNames = try ortSession.outputNames() as [String]

        // Resolve logits output name.
        self.logitsOutputName = lmOutputNames.first { $0 == "logits" }
            ?? lmOutputNames.first { $0.contains("logit") }
            ?? lmOutputNames[0]

        // Infer numLayers from input names if not provided.
        if self.numLayers == 0 {
            self.numLayers = lmInputNames.filter {
                $0.hasPrefix("past_key_values.") && $0.hasSuffix(".key")
            }.count
        }

        // Pre-allocate reusable mask buffer for attention_mask.
        let maskLen = self.maxSeqLen + 500  // Extra margin for generated tokens
        self.reusableMaskBuffer = UnsafeMutableBufferPointer<Int64>.allocate(capacity: maskLen)
        for i in 0..<maskLen {
            self.reusableMaskBuffer?[i] = 1
        }

        // Pre-allocate logits output buffer: [vocabSize] Float32.
        let vocabSize = 6563
        guard let dev = self.device,
              let buf = dev.makeBuffer(
                length: vocabSize * MemoryLayout<Float>.size,
                options: .storageModeShared
              ) else {
            throw LMBackendError.metalDeviceUnavailable
        }
        self.logitsOutputBuffer = buf

        // Initialize empty KV cache tensors for all layers.
        try resetKVCache()
    }

    public func forward(
        inputsEmbds: MTLBuffer,
        kvWriteOffset: Int,
        kvReadLength: Int,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLBuffer {
        guard isInitialized else { throw LMBackendError.notInitialized }
        guard let dev = device, let logitsBuf = logitsOutputBuffer else {
            throw LMBackendError.notInitialized
        }

        // ── Step 1: Convert MTLBuffer → ORTValue for inputs_embeds ─────────────
        //
        // inputsEmbds layout: [1, seqLen, hiddenDim] float16 row-major.
        // We read the bytes directly and wrap them in an ORTValue.
        let seqLen = 1  // Always 1 token per decode step after prefill
        let hiddenDim = 1024

        let embedsByteCount = inputsEmbds.length
        let embedsData = NSMutableData(length: embedsByteCount)!
        memcpy(embedsData.mutableBytes, inputsEmbds.contents(), embedsByteCount)

        let embedsShape: [NSNumber] = [1, NSNumber(value: seqLen), NSNumber(value: hiddenDim)]
        let inputsEmbdsORT = try ORTValue(
            tensorData: embedsData,
            elementType: .float16,
            shape: embedsShape
        )

        // ── Step 2: Build LM inputs dictionary ──────────────────────────────────
        var lmInputs: [String: ORTValue] = [
            "inputs_embeds": inputsEmbdsORT
        ]

        // Position IDs: single token at position = totalProcessedLen
        reusablePositionBuffer[0] = Int64(totalProcessedLen)
        let positionORT = try makeInt64ORTValue(from: reusablePositionBuffer, count: 1, shape: [1, 1])
        lmInputs["position_ids"] = positionORT

        // Attention mask: all ones up to current total length.
        let totalLen = totalProcessedLen + 1
        let maskORT = try makeAttentionMaskORTValue(count: totalLen)
        lmInputs["attention_mask"] = maskORT

        // KV cache inputs: carry forward from previous steps.
        for (key, value) in kvCache {
            lmInputs[key] = value
        }

        // ── Step 3: Run ORT inference ────────────────────────────────────────────
        //
        // autoreleasepool ensures ORT's ObjC-autoreleased tensors (logits,
        // intermediate activations) are freed immediately after each step.
        // Without this, ~400 decode steps accumulate ~23 GB of autoreleased objects.
        let outputNameSet = Set(lmOutputNames)
        var lmOutputs: [String: ORTValue] = [:]

        let runError = autoreleasepool { () -> Error? in
            do {
                lmOutputs = try ortSession.run(
                    withInputs: lmInputs,
                    outputNames: outputNameSet,
                    runOptions: nil
                )
                return nil
            } catch {
                return error
            }
        }

        if let err = runError {
            throw LMBackendError.onnxSessionFailed(err.localizedDescription)
        }

        guard let logitsORT = lmOutputs[logitsOutputName] else {
            throw LMBackendError.onnxSessionFailed("No logits output at step \(totalProcessedLen)")
        }

        // ── Step 4: Extract logits, convert to Float32 ───────────────────────────
        //
        // Logits shape: [1, seqLen, vocabSize].
        // seqLen == 1 for decode steps; may be >1 for prefill.
        // We always extract the LAST position's logits (index curSeqLen-1).
        let logitsInfo = try logitsORT.tensorTypeAndShapeInfo()
        guard logitsInfo.shape.count >= 2 else {
            throw LMBackendError.invalidInputShape
        }

        let vocabSize = 6563
        let returnedSeqLen = logitsInfo.shape[1].intValue
        let lastRowStart = (returnedSeqLen - 1) * vocabSize

        let logitsRawData = try logitsORT.tensorData() as Data
        let outPtr32 = logitsBuf.contents().bindMemory(to: Float.self, capacity: vocabSize)

        // Handle both float16 and float32 output dtypes from ORT.
        switch logitsInfo.elementType {
        case .float16:
            // Raw bytes are float16; copy last position converting to float32.
            logitsRawData.withUnsafeBytes { ptr in
                let buf16 = ptr.bindMemory(to: UInt16.self)
                for i in 0..<vocabSize {
                    outPtr32[i] = float16ToFloat32(buf16[lastRowStart + i])
                }
            }

        case .float:
            // Raw bytes are float32; copy directly.
            logitsRawData.withUnsafeBytes { ptr in
                let buf32 = ptr.bindMemory(to: Float.self)
                for i in 0..<vocabSize {
                    outPtr32[i] = buf32[lastRowStart + i]
                }
            }

        default:
            throw LMBackendError.onnxSessionFailed(
                "Unsupported logits dtype: \(logitsInfo.elementType.rawValue)"
            )
        }

        // ── Step 5: Update KV cache state (present → past for next step) ───────
        //
        // ORT LM output: present.{layer}.key/value
        // Next step input: past_key_values.{layer}.key/value
        for layer in 0..<numLayers {
            if let presentKey = lmOutputs["present.\(layer).key"] {
                kvCache["past_key_values.\(layer).key"] = presentKey
            }
            if let presentValue = lmOutputs["present.\(layer).value"] {
                kvCache["past_key_values.\(layer).value"] = presentValue
            }
        }

        totalProcessedLen += seqLen

        return logitsBuf
    }

    public func reset() async {
        kvCache.removeAll()
        totalProcessedLen = 0
        try? resetKVCache()
    }

    // MARK: - KV Cache Reset

    /// Re-initialize empty KV cache tensors.
    private func resetKVCache() throws {
        kvCache.removeAll()
        for layer in 0..<numLayers {
            let shape: [Int] = [1, numKVHeads, 0, headDim]
            let emptyORT = try createEmptyFloat16Tensor(shape: shape)
            kvCache["past_key_values.\(layer).key"] = emptyORT
            kvCache["past_key_values.\(layer).value"] = emptyORT
        }
    }

    // MARK: - ORT Value Factory Helpers

    /// Create an int64 ORTValue from a pre-allocated buffer (zero-copy).
    private func makeInt64ORTValue(
        from buffer: UnsafeMutableBufferPointer<Int64>,
        count: Int,
        shape: [NSNumber]
    ) throws -> ORTValue {
        let mutableData = NSMutableData(
            bytesNoCopy: buffer.baseAddress!,
            length: count * MemoryLayout<Int64>.size,
            freeWhenDone: false
        )
        return try ORTValue(tensorData: mutableData, elementType: .int64, shape: shape)
    }

    /// Create an attention_mask ORTValue (all-ones int64) from reusable buffer.
    private func makeAttentionMaskORTValue(count: Int) throws -> ORTValue {
        guard let maskBuffer = reusableMaskBuffer, count <= maskBuffer.count else {
            // Fallback: allocate new buffer
            let maskData = NSMutableData(length: count * MemoryLayout<Int64>.size)!
            let ptr = maskData.mutableBytes.bindMemory(to: Int64.self, capacity: count)
            for i in 0..<count { ptr[i] = 1 }
            return try ORTValue(tensorData: maskData, elementType: .int64, shape: [1, NSNumber(value: count)])
        }
        return try makeInt64ORTValue(from: maskBuffer, count: count, shape: [1, NSNumber(value: count)])
    }

    /// Create an empty float16 tensor with one dimension set to 0 (for KV cache init).
    private func createEmptyFloat16Tensor(shape: [Int]) throws -> ORTValue {
        let nsShape = shape.map { NSNumber(value: $0) }
        let emptyData = NSMutableData()
        return try ORTValue(tensorData: emptyData, elementType: .float16, shape: nsShape)
    }

    // MARK: - Float16 Conversion

    /// Convert IEEE-754 float16 to float32.
    private func float16ToFloat32(_ value: UInt16) -> Float {
        let bits = UInt32(value)
        let sign = (bits >> 15) & 0x1
        let exp = (bits >> 10) & 0x1F
        let mantissa = bits & 0x3FF

        if exp == 0 {
            if mantissa == 0 {
                return Float(sign == 1 ? -0.0 : 0.0)
            } else {
                // Denormalized
                let result = Float(mantissa) / 1024.0 * pow(Float(2.0), Float(-14.0))
                return sign == 1 ? -result : result
            }
        } else if exp == 31 {
            if mantissa == 0 {
                return sign == 1 ? -.infinity : .infinity
            } else {
                return .nan
            }
        } else {
            let result = (1.0 + Float(mantissa) / 1024.0) * pow(Float(2.0), Float(Int(exp) - 15))
            return sign == 1 ? -result : result
        }
    }

    // MARK: - Greedy Decode (for caller use)

    /// Greedy decode with repetition penalty — matches reference pipeline exactly.
    ///
    /// Called by the decode loop in ChatterboxEngine after forward() returns logits.
    /// This is a static helper that does NOT modify backend state.
    ///
    /// - Parameters:
    ///   - logits: Raw logits [vocabSize] Float32 from forward()
    ///   - previousTokens: All tokens generated so far (including initial START_SPEECH=6561)
    ///   - repetitionPenalty: Penalty factor (1.2 per generation_config.json)
    /// - Returns: Next token index (argmax after repetition penalty)
    public static func greedyNextToken(
        logits: [Float],
        previousTokens: [Int],
        repetitionPenalty: Float = 1.2
    ) -> Int {
        guard !logits.isEmpty else { return 6562 }  // STOP_SPEECH on empty

        // Apply repetition penalty.
        var adjLogits = logits
        for token in previousTokens {
            guard token < adjLogits.count else { continue }
            if adjLogits[token] > 0 {
                adjLogits[token] /= repetitionPenalty
            } else {
                adjLogits[token] *= repetitionPenalty
            }
        }

        // Argmax.
        var bestIdx = 0
        var bestVal = adjLogits[0]
        for i in 1..<adjLogits.count {
            if adjLogits[i] > bestVal {
                bestVal = adjLogits[i]
                bestIdx = i
            }
        }
        return bestIdx
    }
}
