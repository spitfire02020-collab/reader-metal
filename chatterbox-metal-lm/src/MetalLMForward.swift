import Foundation
import Metal

final class MetalLMForward {
    let device: MTLDevice
    let encoder: MetalLMEncoder
    let weightLoader: WeightLoader
    let kvCache: KVCacheManager

    init(device: MTLDevice, library: MTLLibrary, weightLoader: WeightLoader) throws {
        self.device = device
        self.encoder = try MetalLMEncoder(device: device, library: library)
        self.weightLoader = weightLoader
        self.kvCache = KVCacheManager(device: device)
    }

    /// Single-step forward pass (STUB — actual implementation uses MPSGraph GEMMs)
    func forward(
        inputs: MTLBuffer,
        inputLength: Int
    ) throws -> MTLBuffer {
        // TODO: Implement actual GPT2 block forward using:
        // 1. Pre-dequantized fp16 weights from weightLoader.dequantizedWeights
        // 2. MPSGEMM.matmulTransposeB for QKV projections
        // 3. MPSDPA.forward for attention
        // 4. LayerNormPipeline.normalize for layer norms
        // 5. KVCacheBuffer.appendKey/appendValue for cache updates
        fatalError("MetalLMForward.forward() is not yet implemented")
    }
}