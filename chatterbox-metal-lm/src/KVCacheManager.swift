import Foundation
import Metal

/// A ring-buffer KV cache managed by Swift actor for thread-safe access.
/// 48 pre-allocated MTLBuffers (24 layers × {key, value}).
/// Each buffer: [1, numKVHeads=16, maxSeqLen=1500, headDim=64] float16 = ~3MB per buffer.
/// Total: 24 × 2 × 3MB ≈ 147MB.
public actor KVCacheManager {

    // MARK: - Configuration

    public let numLayers: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let maxSeqLen: Int

    /// 24 layers × 2 (key, value) = 48 buffers total
    private var layerBuffers: [KVCacheBufferSet]

    /// Ring-buffer write head — advances each decode step
    private var writeHead: Int = 0

    /// Byte offset per position: numKVHeads * headDim * sizeof(float16)
    private let stride: Int

    // MARK: - Buffer Set

    /// A pair of key/value buffers for a single layer.
    public struct KVCacheBufferSet: @unchecked Sendable {
        public let keyBuffer: MTLBuffer   // [1, numKVHeads, maxSeqLen, headDim] float16
        public let valBuffer: MTLBuffer   // [1, numKVHeads, maxSeqLen, headDim] float16
    }

    // MARK: - Initialization

    /// Creates a new KVCacheManager with pre-allocated Metal buffers.
    /// - Parameters:
    ///   - numLayers: Number of transformer layers (default 24)
    ///   - numKVHeads: Number of KV heads (default 16, GQA)
    ///   - headDim: Dimension per head (default 64)
    ///   - maxSeqLen: Maximum sequence length for ring buffer (default 1500)
    ///   - device: Metal device for buffer allocation
    public init(
        numLayers: Int,
        numKVHeads: Int,
        headDim: Int,
        maxSeqLen: Int,
        device: MTLDevice
    ) {
        self.numLayers = numLayers
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.maxSeqLen = maxSeqLen
        self.stride = numKVHeads * headDim * MemoryLayout<Float16>.size

        let bufferByteSize = maxSeqLen * stride

        self.layerBuffers = (0..<numLayers).map { _ in
            let keyBuf = device.makeBuffer(length: bufferByteSize, options: .storageModeShared)!
            let valBuf = device.makeBuffer(length: bufferByteSize, options: .storageModeShared)!
            return KVCacheBufferSet(keyBuffer: keyBuf, valBuffer: valBuf)
        }
    }

    // MARK: - Buffer Access

    /// Returns the buffer for a given layer and key/value type.
    /// - Parameters:
    ///   - layer: Layer index (0..<numLayers)
    ///   - isKey: True for key buffer, false for value buffer
    /// - Returns: The requested MTLBuffer
    public func buffer(for layer: Int, isKey: Bool) -> MTLBuffer {
        isKey ? layerBuffers[layer].keyBuffer : layerBuffers[layer].valBuffer
    }

    /// Returns the current write position (in tokens).
    public var currentWriteHead: Int {
        writeHead
    }

    // MARK: - Ring Buffer Operations

    /// Advance write head by 1 position (wraps around at maxSeqLen).
    public func advance() {
        writeHead = (writeHead + 1) % maxSeqLen
    }

    // MARK: - Reset

    /// Reset all buffers to zero and return write head to 0.
    public func reset() {
        writeHead = 0
        let byteLen = maxSeqLen * stride
        for set in layerBuffers {
            // memset operates on raw memory; contents() returns UnsafeMutableRawPointer
            memset(set.keyBuffer.contents(), 0, byteLen)
            memset(set.valBuffer.contents(), 0, byteLen)
        }
    }

    // MARK: - Prefix Caching

    /// Pre-populate KV cache with a prefix (for prefix caching optimization).
    /// - Parameters:
    ///   - keyData: Pointer to [numKVHeads, prefixLen, headDim] float16 data
    ///   - valData: Pointer to [numKVHeads, prefixLen, headDim] float16 data
    ///   - layer: Which layer to populate (0..<numLayers)
    ///   - prefixLen: Number of positions in the prefix
    public func setPrefix(
        keyData: UnsafeRawPointer,
        valData: UnsafeRawPointer,
        layer: Int,
        prefixLen: Int
    ) {
        let count = prefixLen * numKVHeads * headDim

        layerBuffers[layer].keyBuffer.contents().withMemoryRebound(
            to: Float16.self, capacity: count
        ) { dstKey in
            keyData.withMemoryRebound(to: Float16.self, capacity: count) { srcKey in
                dstKey.initialize(from: srcKey, count: count)
            }
        }

        layerBuffers[layer].valBuffer.contents().withMemoryRebound(
            to: Float16.self, capacity: count
        ) { dstVal in
            valData.withMemoryRebound(to: Float16.self, capacity: count) { srcVal in
                dstVal.initialize(from: srcVal, count: count)
            }
        }
    }

    // MARK: - Ring Buffer Write (for generation loop)

    /// Write key/value tensors at the current write head position.
    /// Advances the write head by 1 after writing.
    /// - Parameters:
    ///   - keyData: Pointer to [numKVHeads, headDim] float16 key tensor
    ///   - valData: Pointer to [numKVHeads, headDim] float16 value tensor
    ///   - layer: Which layer to write to (0..<numLayers)
    public func write(
        keyData: UnsafeRawPointer,
        valData: UnsafeRawPointer,
        layer: Int
    ) {
        let offset = writeHead * stride
        let dstKey = layerBuffers[layer].keyBuffer.contents()
            .advanced(by: offset)
            .assumingMemoryBound(to: Float16.self)
        let dstVal = layerBuffers[layer].valBuffer.contents()
            .advanced(by: offset)
            .assumingMemoryBound(to: Float16.self)

        let count = numKVHeads * headDim
        memcpy(dstKey, keyData, count * MemoryLayout<Float16>.size)
        memcpy(dstVal, valData, count * MemoryLayout<Float16>.size)

        advance()
    }

    // MARK: - Debug Info

    /// Total buffer size in bytes (both key and value for all layers)
    public var totalBufferSize: Int {
        maxSeqLen * stride * 2 * numLayers
    }

    /// Buffer size per layer (key + value combined)
    public var layerBufferSize: Int {
        maxSeqLen * stride * 2
    }
}