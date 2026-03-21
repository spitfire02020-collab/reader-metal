import Foundation
import Metal

final class KVCacheManager {
    var layers: [KVCacheBuffer]
    let maxSeq: Int

    init(device: MTLDevice, numLayers: Int = MetalLMConfig.numLayers, maxSeq: Int = MetalLMConfig.maxSequenceLength) {
        self.maxSeq = maxSeq
        self.layers = (0..<numLayers).map { i in
            KVCacheBuffer(device: device, layerIndex: i, maxSeq: maxSeq)
        }
    }

    func reset() {
        for layer in layers { layer.reset() }
    }

    var totalLength: Int {
        layers.first?.currentLength ?? 0
    }
}