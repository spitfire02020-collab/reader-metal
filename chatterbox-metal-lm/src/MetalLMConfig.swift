import Foundation

enum MetalLMConfig {
    static let numLayers = 24
    static let numQueryHeads = 80  // Q heads for GPT-2 Medium GQA
    static let numKVHeads = 16      // KV heads (GQA — all layers share same KV)
    static let headDim = 64
    static let hiddenSize = 1024
    static let intermediateSize = 4096
    static let vocabSize = 6563
    static let maxSequenceLength = 1500  // Per spec

    // Backward compatibility alias — num_heads was ambiguous (could mean Q or KV heads)
    // swiftlint:disable:next deprecated
    static let num_heads = 16

    // Special tokens
    static let startSpeechToken: Int32 = 6561
    static let stopSpeechToken: Int32 = 6562

    // Q4F16 block parameters
    static let blockSize = 16
    static let quantBlockRows = 32
}