import Foundation
import Metal

// MARK: - Language Model Backend Protocol

/// Unified protocol for language model backends (ORT on GPU, Metal, etc.).
///
/// The backend is responsible for a single autoregressive decode step:
///   1. Forward pass with current inputs_embeds + KV cache state
///   2. Return logits for the current position
///
/// KV cache management:
///   - `kvWriteOffset`: position in the KV cache buffer where new keys/values are written
///   - `kvReadLength`: how many positions to read from the KV cache (includes past + current)
///
/// The caller (decode loop in ChatterboxEngine) handles the full token-generation loop,
/// greedy decoding (argmax + repetition penalty), and STOP_SPEECH detection.
public protocol LanguageModelBackend: Sendable {
    /// Initialize the backend with model configuration.
    ///
    /// - Parameters:
    ///   - numLayers: Number of transformer layers (24 for chatterbox-turbo)
    ///   - numKVHeads: Number of KV attention heads (16 for GQA)
    ///   - headDim: Dimension per attention head (64)
    ///   - maxSeqLen: Maximum sequence length for KV cache allocation
    ///   - device: MTLDevice for Metal-backed backends (ignored by ORT backend)
    func initialize(
        numLayers: Int,
        numKVHeads: Int,
        headDim: Int,
        maxSeqLen: Int,
        device: MTLDevice
    ) async throws

    /// Run one forward pass of the language model.
    ///
    /// For the prefill step: inputsEmbeds = [1, totalPrefixLen, hiddenDim], kvReadLength = 0
    /// For decode steps:   inputsEmbds = [1, 1, hiddenDim],   kvReadLength = pastSeqLen
    ///
    /// - Parameters:
    ///   - inputsEmbds: Current input embeddings [1, seqLen, hiddenDim] as MTLBuffer
    ///   - kvWriteOffset: Write offset for this step's KV entries (starts at 0 for prefill)
    ///   - kvReadLength: Number of positions to read from KV cache (0 for first prefill)
    ///   - commandBuffer: Metal command buffer for GPU synchronization
    /// - Returns: Logits buffer [vocabSize] as Float32 for the LAST position
    func forward(
        inputsEmbds: MTLBuffer,
        kvWriteOffset: Int,
        kvReadLength: Int,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLBuffer

    /// Reset all KV cache state. Called between synthesis chunks.
    func reset() async
}

// MARK: - LM Backend Errors

public enum LMBackendError: Error, LocalizedError, Sendable {
    case notInitialized
    case metalDeviceUnavailable
    case kernelNotFound(String)
    case commandBufferFailed
    case invalidInputShape
    case onnxSessionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "Language model backend not initialized"
        case .metalDeviceUnavailable:
            return "Metal device unavailable"
        case .kernelNotFound(let name):
            return "Metal kernel not found: \(name)"
        case .commandBufferFailed:
            return "Metal command buffer failed"
        case .invalidInputShape:
            return "Invalid input shape"
        case .onnxSessionFailed(let message):
            return "ONNX session failed: \(message)"
        }
    }
}
