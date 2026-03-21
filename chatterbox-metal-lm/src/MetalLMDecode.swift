import Foundation
import Metal

final class MetalLMDecode {
    let forward: MetalLMForward

    init(device: MTLDevice, library: MTLLibrary, weightLoader: WeightLoader) throws {
        self.forward = try MetalLMForward(device: device, library: library, weightLoader: weightLoader)
    }

    func reset() {
        forward.kvCache.reset()
    }

    /// Greedy decode step: argmax + repetition penalty
    /// Returns next speech token ID (Int32)
    func step(
        inputsEmbed: MTLBuffer,
        inputLength: Int,
        generatedTokens: [Int32]
    ) throws -> (logits: [Float], nextToken: Int32) {
        let logitsBuf = try forward.forward(inputs: inputsEmbed, inputLength: inputLength)

        // logits: [1, seq, vocab]
        let logitsPtr = logitsBuf.contents().bindMemory(to: Float.self, capacity: MetalLMConfig.vocabSize)
        var logits = Array(UnsafeBufferPointer(start: logitsPtr, count: MetalLMConfig.vocabSize))

        // Apply repetition penalty
        applyRepetitionPenalty(logits: &logits, tokens: generatedTokens)

        // Greedy: argmax
        var maxIdx = 0
        var maxVal = logits[0]
        for i in 1..<logits.count {
            if logits[i] > maxVal { maxVal = logits[i]; maxIdx = i }
        }

        return (logits, Int32(maxIdx))
    }

    private func applyRepetitionPenalty(logits: inout [Float], tokens: [Int32]) {
        let penalty: Float = 1.2
        for t in tokens {
            let idx = Int(t)
            if idx < logits.count {
                if logits[idx] > 0 { logits[idx] /= penalty }
                else { logits[idx] *= penalty }
            }
        }
    }
}