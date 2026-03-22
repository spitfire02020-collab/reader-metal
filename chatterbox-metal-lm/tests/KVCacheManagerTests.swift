import XCTest
import Metal

final class KVCacheManagerTests: XCTestCase {

    // MARK: - Properties

    var device: MTLDevice!
    var manager: KVCacheManager!

    // MARK: - Lifecycle

    override func setUp() {
        super.setUp()
        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal device available")
            return
        }
        self.device = device
    }

    override func tearDown() {
        manager = nil
        device = nil
        super.tearDown()
    }

    // MARK: - Helpers

    private func createManager(
        numLayers: Int = 24,
        numKVHeads: Int = 16,
        headDim: Int = 64,
        maxSeqLen: Int = 1500
    ) -> KVCacheManager {
        KVCacheManager(
            numLayers: numLayers,
            numKVHeads: numKVHeads,
            headDim: headDim,
            maxSeqLen: maxSeqLen,
            device: device
        )
    }

    // MARK: - Buffer Access Tests

    func testBufferAccess() async throws {
        let sut = createManager(numLayers: 24, numKVHeads: 16, headDim: 64, maxSeqLen: 1500)

        // Verify we have correct number of layers
        XCTAssertEqual(await sut.numLayers, 24)
        XCTAssertEqual(await sut.numKVHeads, 16)
        XCTAssertEqual(await sut.headDim, 64)
        XCTAssertEqual(await sut.maxSeqLen, 1500)

        // Verify buffers are accessible
        for layer in 0..<24 {
            let keyBuffer = await sut.buffer(for: layer, isKey: true)
            let valBuffer = await sut.buffer(for: layer, isKey: false)

            XCTAssertNotNil(keyBuffer, "Key buffer for layer \(layer) should not be nil")
            XCTAssertNotNil(valBuffer, "Value buffer for layer \(layer) should not be nil")

            // Verify buffer sizes: [1, 16, 1500, 64] float16 = 16 * 1500 * 64 * 2 bytes = 3,072,000 bytes
            let expectedSize = 1500 * 16 * 64 * MemoryLayout<Float16>.size
            XCTAssertEqual(keyBuffer.length, expectedSize, "Key buffer size mismatch for layer \(layer)")
            XCTAssertEqual(valBuffer.length, expectedSize, "Value buffer size mismatch for layer \(layer)")
        }
    }

    func testBufferSizeCalculations() async throws {
        let sut = createManager(numLayers: 24, numKVHeads: 16, headDim: 64, maxSeqLen: 1500)

        // Verify stride: numKVHeads * headDim * sizeof(float16)
        let expectedStride = 16 * 64 * MemoryLayout<Float16>.size // 2048 bytes
        let expectedLayerBufferSize = 1500 * expectedStride * 2 // ~147MB per layer (key+val)
        let expectedTotalSize = expectedLayerBufferSize * 24 // ~147MB * 24 layers ≈ 3.5GB

        XCTAssertEqual(await sut.layerBufferSize, expectedLayerBufferSize)
        XCTAssertEqual(await sut.totalBufferSize, expectedTotalSize)
    }

    // MARK: - Ring Buffer Advance Tests

    func testRingBufferAdvance() async throws {
        let maxSeqLen = 100 // Use smaller size for easier wrap-around testing
        let sut = createManager(numLayers: 24, numKVHeads: 16, headDim: 64, maxSeqLen: maxSeqLen)

        // Initial state
        XCTAssertEqual(await sut.currentWriteHead, 0)

        // Advance to middle
        for i in 1..<50 {
            await sut.advance()
            XCTAssertEqual(await sut.currentWriteHead, i, "Write head should be \(i) after \(i) advances")
        }

        // Advance to wrap point
        for _ in 50..<100 {
            await sut.advance()
        }
        XCTAssertEqual(await sut.currentWriteHead, 0, "Write head should wrap to 0")

        // Continue and verify wrap
        await sut.advance()
        XCTAssertEqual(await sut.currentWriteHead, 1)
    }

    func testRingBufferWrapAround() async throws {
        let maxSeqLen = 10
        let sut = createManager(numLayers: 24, numKVHeads: 16, headDim: 64, maxSeqLen: maxSeqLen)

        // Advance 25 times (past 2 full wraps)
        for _ in 0..<25 {
            await sut.advance()
        }

        // 25 % 10 = 5
        XCTAssertEqual(await sut.currentWriteHead, 5)
    }

    // MARK: - Reset Tests

    func testReset() async throws {
        let maxSeqLen = 100
        let sut = createManager(numLayers: 24, numKVHeads: 16, headDim: 64, maxSeqLen: maxSeqLen)

        // Advance write head
        for _ in 0..<50 {
            await sut.advance()
        }
        XCTAssertEqual(await sut.currentWriteHead, 50)

        // Write some data to first buffer
        let keyBuffer = await sut.buffer(for: 0, isKey: true)
        let float16Ptr = keyBuffer.contents().bindMemory(to: Float16.self, capacity: maxSeqLen * 16 * 64)
        float16Ptr[0] = Float16(1.0)

        // Reset
        await sut.reset()

        // Verify write head is 0
        XCTAssertEqual(await sut.currentWriteHead, 0)

        // Verify buffer is cleared (first element should be 0)
        let float16PtrAfter = keyBuffer.contents().bindMemory(to: Float16.self, capacity: maxSeqLen * 16 * 64)
        XCTAssertEqual(float16PtrAfter[0], Float16(0.0), "Buffer should be cleared after reset")
    }

    func testResetClearsAllLayers() async throws {
        let sut = createManager(numLayers: 2, numKVHeads: 16, headDim: 64, maxSeqLen: 100)

        // Write non-zero data to all buffers
        for layer in 0..<2 {
            let keyBuffer = await sut.buffer(for: layer, isKey: true)
            let valBuffer = await sut.buffer(for: layer, isKey: false)

            let keyPtr = keyBuffer.contents().bindMemory(to: Float16.self, capacity: 100 * 16 * 64)
            let valPtr = valBuffer.contents().bindMemory(to: Float16.self, capacity: 100 * 16 * 64)

            keyPtr[0] = Float16(42.0)
            valPtr[0] = Float16(42.0)
        }

        // Reset
        await sut.reset()

        // Verify all buffers are cleared
        for layer in 0..<2 {
            let keyBuffer = await sut.buffer(for: layer, isKey: true)
            let valBuffer = await sut.buffer(for: layer, isKey: false)

            let keyPtr = keyBuffer.contents().bindMemory(to: Float16.self, capacity: 100 * 16 * 64)
            let valPtr = valBuffer.contents().bindMemory(to: Float16.self, capacity: 100 * 16 * 64)

            XCTAssertEqual(keyPtr[0], Float16(0.0), "Key buffer for layer \(layer) should be cleared")
            XCTAssertEqual(valPtr[0], Float16(0.0), "Value buffer for layer \(layer) should be cleared")
        }
    }

    // MARK: - Write Tests

    func testWrite() async throws {
        let maxSeqLen = 100
        let sut = createManager(numLayers: 2, numKVHeads: 16, headDim: 64, maxSeqLen: maxSeqLen)

        // Create test data: [numKVHeads, headDim] = [16, 64] float16
        let testDataSize = 16 * 64
        var testData = [Float16](repeating: Float16(0), count: testDataSize)
        for i in 0..<testDataSize {
            testData[i] = Float16(Float(i))
        }

        // Write at position 0 — allocate heap memory so actor can safely access it
        let byteCount = testDataSize * MemoryLayout<Float16>.size
        let rawPtr = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: 16)
        defer { rawPtr.deallocate() }
        rawPtr.copyMemory(from: testData, count: byteCount)

        await sut.write(keyData: rawPtr, valData: rawPtr, layer: 0)

        // The write head should have advanced
        XCTAssertEqual(await sut.currentWriteHead, 1)

        // Verify data was written
        let keyBuffer = await sut.buffer(for: 0, isKey: true)
        let keyPtr = keyBuffer.contents().bindMemory(to: Float16.self, capacity: testDataSize)

        XCTAssertEqual(keyPtr[0], Float16(0.0))
        XCTAssertEqual(keyPtr[1], Float16(1.0))
    }

    // MARK: - Prefix Caching Tests

    func testSetPrefix() async throws {
        let maxSeqLen = 100
        let prefixLen = 10
        let sut = createManager(numLayers: 2, numKVHeads: 16, headDim: 64, maxSeqLen: maxSeqLen)

        // Create test prefix data: [numKVHeads, prefixLen, headDim] = [16, 10, 64] float16
        let testDataSize = 16 * prefixLen * 64
        var testData = [Float16](repeating: Float16(0), count: testDataSize)
        for i in 0..<testDataSize {
            testData[i] = Float16(Float(i) * 0.1)
        }

        // Allocate heap memory for the actor to safely access
        let byteCount = testDataSize * MemoryLayout<Float16>.size
        let rawPtr = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: 16)
        defer { rawPtr.deallocate() }
        rawPtr.copyMemory(from: testData, count: byteCount)

        await sut.setPrefix(keyData: rawPtr, valData: rawPtr, layer: 0, prefixLen: prefixLen)

        // Verify data was written at start of buffer
        let keyBuffer = await sut.buffer(for: 0, isKey: true)
        let keyPtr = keyBuffer.contents().bindMemory(to: Float16.self, capacity: testDataSize)

        XCTAssertEqual(keyPtr[0], Float16(0.0))
        XCTAssertEqual(keyPtr[1], Float16(0.1))
    }
}