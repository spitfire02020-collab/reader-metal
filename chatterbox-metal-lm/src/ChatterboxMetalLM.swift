import Foundation
import Metal

public final class ChatterboxMetalLM {
    public let device: MTLDevice
    public let maxNewTokens: Int
    public let repetitionPenalty: Float

    private let decoder: MetalLMDecode
    private let weightLoader: WeightLoader

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

        // Load Metal library from bundle
        let bundle = Bundle(for: Self.self)
        guard let libPath = bundle.path(forResource: "default", ofType: "metallib"),
              let library = try? dev.makeLibrary(filepath: libPath) else {
            throw MetalLMError.libraryNotFound
        }

        self.weightLoader = try WeightLoader(device: dev, weightsDir: weightDirectory, library: library)
        self.decoder = try MetalLMDecode(device: dev, library: library, weightLoader: weightLoader, repetitionPenalty: repetitionPenalty)
    }

    public func reset() {
        decoder.reset()
    }

    /// Generate speech tokens autoregressively
    /// - Parameters:
    ///   - conditioning: Speaker conditioning embeddings [B, cond_len, 1024]
    ///   - textEmbed: Text embeddings [B, text_len, 1024]
    ///   - speechEmbed: Initial speech embedding [B, 1, 1024]
    /// - Returns: Generated speech token IDs [num_tokens]
    public func generate(
        conditioning: [Float],
        textEmbed: [Float],
        speechEmbed: [Float]
    ) throws -> [Int32] {
        decoder.reset()

        var generatedTokens = [MetalLMConfig.startSpeechToken]

        // TODO: Implement full decode loop
        // 1. Concat [conditioning | text | speech_embed] → input embed
        // 2. Run decoder.step() repeatedly
        // 3. Return tokens (excluding START_SPEECH and STOP_SPEECH)

        return generatedTokens
    }
}

public enum MetalLMError: Error {
    case noMetalDevice
    case libraryNotFound
    case kernelNotFound(String)
    case commandBufferFailed
    case commandQueueFailed
}