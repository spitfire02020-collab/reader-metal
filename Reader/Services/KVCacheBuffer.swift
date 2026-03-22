import Foundation
import Metal

final class KVCacheBuffer {
    let device: MTLDevice
    let layerIndex: Int
    let maxSeq: Int

    let keyBuffer: MTLBuffer
    let valueBuffer: MTLBuffer

    var currentLength: Int = 0

    init(device: MTLDevice, layerIndex: Int, batchSize: Int = 1, maxSeq: Int = 8192) {
        self.device = device
        self.layerIndex = layerIndex
        self.maxSeq = maxSeq

        let size = batchSize * maxSeq * MetalLMConfig.num_heads * MetalLMConfig.headDim
        let byteSize = size * MemoryLayout<Float16>.size

        self.keyBuffer = device.makeBuffer(length: byteSize, options: .storageModeShared)!
        self.valueBuffer = device.makeBuffer(length: byteSize, options: .storageModeShared)!
    }

    func reset() {
        currentLength = 0
    }

    func appendKey(_ data: UnsafeRawPointer, length: Int) {
        let dst = keyBuffer.contents().advanced(by: currentLength * MetalLMConfig.num_heads * MetalLMConfig.headDim * MemoryLayout<Float16>.size)
        memcpy(dst, data, length * MetalLMConfig.num_heads * MetalLMConfig.headDim * MemoryLayout<Float16>.size)
        currentLength += length
    }

    func appendValue(_ data: UnsafeRawPointer, length: Int) {
        let dst = valueBuffer.contents().advanced(by: currentLength * MetalLMConfig.num_heads * MetalLMConfig.headDim * MemoryLayout<Float16>.size)
        memcpy(dst, data, length * MetalLMConfig.num_heads * MetalLMConfig.headDim * MemoryLayout<Float16>.size)
        currentLength += length
    }
}