import Foundation
import Metal

final class MetalLMEncoder {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let layerNormPipeline: LayerNormPipeline
    let gemm: MPSGEMM
    let dpa: MPSDPA

    init(device: MTLDevice, library: MTLLibrary) throws {
        self.device = device
        guard let q = device.makeCommandQueue() else {
            throw MetalLMError.commandQueueFailed
        }
        self.commandQueue = q
        self.layerNormPipeline = try LayerNormPipeline(device: device, library: library)
        self.gemm = MPSGEMM(device: device)
        self.dpa = MPSDPA(device: device)
    }

    // TODO: Orchestrate full GPT2 block forward:
    // 1. layerNormPipeline.normalize on input
    // 2. MPSGEMM.matmulTransposeB for Q, K, V projections (3x)
    // 3. MPSDPA.forward for attention
    // 4. MPSGEMM.matmulTransposeB for O projection
    // 5. residual add
    // 6. layerNormPipeline.normalize
    // 7. MPSGEMM.matmulTransposeB for FC1
    // 8. GELU activation
    // 9. MPSGEMM.matmulTransposeB for FC2
    // 10. residual add
    func forward(
        input: MTLBuffer,
        inputLength: Int,
        qWeight: MTLBuffer,
        kWeight: MTLBuffer,
        vWeight: MTLBuffer,
        oWeight: MTLBuffer,
        fc1Weight: MTLBuffer,
        fc2Weight: MTLBuffer,
        ln1Gamma: MTLBuffer,
        ln1Beta: MTLBuffer,
        ln2Gamma: MTLBuffer,
        ln2Beta: MTLBuffer,
        output: MTLBuffer,
        commandBuffer: MTLCommandBuffer
    ) {
        // STUB: TODO implementation
        fatalError("MetalLMEncoder.forward() not yet implemented")
    }
}