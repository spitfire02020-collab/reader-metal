import Foundation
import Metal

final class Q4F16Dequantizer {
    let device: MTLDevice
    let quantPipeline: MTLComputePipelineState
    let maxThreads: MTLSize

    init(device: MTLDevice, library: MTLLibrary) throws {
        self.device = device
        guard let kern = library.makeFunction(name: "dequant_q4f16") else {
            throw MetalLMError.kernelNotFound("dequant_q4f16")
        }
        self.quantPipeline = try device.makeComputePipelineState(function: kern)
        self.maxThreads = MTLSize(
            width: quantPipeline.maxTotalThreadsPerThreadgroup,
            height: 1, depth: 1
        )
    }

    func dequant(
        commandBuffer: MTLCommandBuffer,
        quantBuffer: MTLBuffer,
        scalesBuffer: MTLBuffer,
        zpBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        outDim: Int,
        blockCount: Int
    ) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(quantPipeline)
        enc.setBuffer(quantBuffer, offset: 0, index: 0)
        enc.setBuffer(scalesBuffer, offset: 0, index: 1)
        enc.setBuffer(zpBuffer, offset: 0, index: 2)
        enc.setBuffer(outputBuffer, offset: 0, index: 3)

        var od = UInt32(outDim), bc = UInt32(blockCount)
        enc.setBytes(&od, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&bc, length: MemoryLayout<UInt32>.size, index: 5)

        let tgWidth = min(256, quantPipeline.maxTotalThreadsPerThreadgroup)
        let threadsPerGroup = MTLSize(width: tgWidth, height: 1, depth: 1)
        let numThreadGroups = MTLSize(
            width: (blockCount + tgWidth - 1) / tgWidth,
            height: outDim,
            depth: 1
        )
        enc.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
        enc.endEncoding()
    }
}

