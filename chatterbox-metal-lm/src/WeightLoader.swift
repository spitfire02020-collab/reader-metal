import Foundation
import Metal

final class WeightLoader {
    let device: MTLDevice
    let weightsDir: URL
    let dequantizer: Q4F16Dequantizer
    let commandQueue: MTLCommandQueue

    struct QuantizedWeight {
        let name: String
        let outDim: Int
        let blockCount: Int
        let quantBuffer: MTLBuffer
        let scalesBuffer: MTLBuffer
        let zpBuffer: MTLBuffer
    }

    /// Pre-dequantized fp16 weights — dequantized once, reused forever
    /// Key: weight name, Value: MTLBuffer containing fp16 [out_dim, flattened]
    private(set) var dequantizedWeights: [String: MTLBuffer] = [:]

    init(device: MTLDevice, weightsDir: URL, library: MTLLibrary) throws {
        self.device = device
        self.weightsDir = weightsDir
        self.dequantizer = try Q4F16Dequantizer(device: device, library: library)
        guard let q = device.makeCommandQueue() else { throw MetalLMError.commandQueueFailed }
        self.commandQueue = q
    }

    /// Load all weights and dequantize to fp16 ONCE.
    /// This avoids recomputing dequantization every forward step.
    func loadAndDequantAllWeights() throws -> [String: QuantizedWeight] {
        let manifestPath = weightsDir.appendingPathComponent("weights_manifest.json")
        let manifestData = try Data(contentsOf: manifestPath)
        let manifest = try JSONDecoder().decode([String: WeightManifestEntry].self, from: manifestData)

        var weights: [String: QuantizedWeight] = [:]

        for (name, entry) in manifest {
            guard let q = loadBin(name: entry.quantPath),
                  let s = loadBin(name: entry.scalesPath),
                  let z = loadBin(name: entry.zpPath) else {
                continue
            }

            let qBuf = device.makeBuffer(bytes: q, options: .storageModeShared)!
            let sBuf = device.makeBuffer(bytes: s, options: .storageModeShared)!
            let zBuf = device.makeBuffer(bytes: z, options: .storageModeShared)!

            // Allocate fp16 output buffer for dequantized weight
            let fp16Size = entry.outDim * entry.blockCount * 16 * MemoryLayout<Float16>.size
            let fp16Buf = device.makeBuffer(length: fp16Size, options: .storageModeShared)!

            // Dequantize ONCE during loading
            guard let cmd = commandQueue.makeCommandBuffer(),
                  let enc = cmd.makeComputeCommandEncoder() else { continue }
            dequantizer.dequant(
                commandBuffer: cmd,
                quantBuffer: qBuf,
                scalesBuffer: sBuf,
                zpBuffer: zBuf,
                outputBuffer: fp16Buf,
                outDim: entry.outDim,
                blockCount: entry.blockCount
            )
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()

            dequantizedWeights[name] = fp16Buf

            weights[name] = QuantizedWeight(
                name: name,
                outDim: entry.outDim,
                blockCount: entry.blockCount,
                quantBuffer: qBuf,
                scalesBuffer: sBuf,
                zpBuffer: zBuf
            )
        }
        return weights
    }

    private func loadBin(name: String) -> Data? {
        try? Data(contentsOf: weightsDir.appendingPathComponent(name))
    }
}

private struct WeightManifestEntry: Codable {
    let outDim: Int
    let blockCount: Int
    let quantPath: String
    let scalesPath: String
    let zpPath: String
}
