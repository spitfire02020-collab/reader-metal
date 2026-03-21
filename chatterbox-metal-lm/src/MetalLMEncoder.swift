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
}