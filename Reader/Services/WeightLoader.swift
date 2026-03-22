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

    /// Pre-loaded fp16 weights keyed by ONNX name
    private(set) var dequantizedWeights: [String: MTLBuffer] = [:]

    init(device: MTLDevice, weightsDir: URL, library: MTLLibrary) throws {
        self.device = device
        self.weightsDir = weightsDir
        self.dequantizer = try Q4F16Dequantizer(device: device, library: library)
        guard let q = device.makeCommandQueue() else { throw MetalLMError.commandQueueFailed }
        self.commandQueue = q
    }

    /// Load all weights.
    ///
    /// manifest.json entry types:
    ///   - fp16: "filename.bin" — raw Float16 binary (metal-export weights)
    ///   - outDim/blockCount/quant/scales/zp: Q4F16 quantized (e.g. ResembleAI ONNX weights)
    func loadAndDequantAllWeights() throws -> [String: QuantizedWeight] {
        let manifestPath = weightsDir.appendingPathComponent("weights_manifest.json")
        let manifestData = try Data(contentsOf: manifestPath)
        let manifest = try JSONDecoder().decode([String: WeightManifestEntry].self, from: manifestData)

        var weights: [String: QuantizedWeight] = [:]

        for (name, entry) in manifest {
            if let fp16File = entry.fp16 {
                // Raw FP16 weight — load directly
                let fp16Data = try Data(contentsOf: weightsDir.appendingPathComponent(fp16File))
                let fp16Buf = fp16Data.withUnsafeBytes { ptr in
                    device.makeBuffer(bytes: ptr.baseAddress!, length: fp16Data.count, options: .storageModeShared)!
                }
                dequantizedWeights[name] = fp16Buf
                let outDim = fp16Data.count / MemoryLayout<Float16>.size
                weights[name] = QuantizedWeight(
                    name: name, outDim: outDim, blockCount: 1,
                    quantBuffer: fp16Buf, scalesBuffer: fp16Buf, zpBuffer: fp16Buf
                )
            } else if let quantPath = entry.quant,
                      let outDim = entry.outDim,
                      let blockCount = entry.blockCount {
                // Q4F16 quantized weight
                guard let qData = loadBin(name: quantPath),
                      let sData = loadBin(name: entry.scales ?? ""),
                      let zData = loadBin(name: entry.zp ?? "") else {
                    continue
                }

                let qBuf = qData.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: qData.count, options: .storageModeShared)! }
                let sBuf = sData.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: sData.count, options: .storageModeShared)! }
                let zBuf = zData.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: zData.count, options: .storageModeShared)! }

                let fp16Size = outDim * blockCount * 16 * MemoryLayout<Float16>.size
                let fp16Buf = device.makeBuffer(length: fp16Size, options: .storageModeShared)!

                guard let cmd = commandQueue.makeCommandBuffer(),
                      let enc = cmd.makeComputeCommandEncoder() else { continue }
                dequantizer.dequant(
                    commandBuffer: cmd,
                    quantBuffer: qBuf,
                    scalesBuffer: sBuf,
                    zpBuffer: zBuf,
                    outputBuffer: fp16Buf,
                    outDim: outDim,
                    blockCount: blockCount
                )
                enc.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()

                dequantizedWeights[name] = fp16Buf
                weights[name] = QuantizedWeight(
                    name: name, outDim: outDim, blockCount: blockCount,
                    quantBuffer: qBuf, scalesBuffer: sBuf, zpBuffer: zBuf
                )
            }
        }
        return weights
    }

    private func loadBin(name: String) -> Data? {
        try? Data(contentsOf: weightsDir.appendingPathComponent(name))
    }
}

private struct WeightManifestEntry: Codable {
    // Raw Float16 entry
    let fp16: String?

    // Q4F16 quantized entry
    let outDim: Int?
    let blockCount: Int?
    let quant: String?
    let scales: String?
    let zp: String?
}
