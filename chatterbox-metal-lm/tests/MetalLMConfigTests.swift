import XCTest

final class MetalLMConfigTests: XCTestCase {
    func testConstants() {
        XCTAssertEqual(MetalLMConfig.numLayers, 24)
        XCTAssertEqual(MetalLMConfig.num_heads, 16)
        XCTAssertEqual(MetalLMConfig.headDim, 64)
        XCTAssertEqual(MetalLMConfig.hiddenSize, 1024)
        XCTAssertEqual(MetalLMConfig.intermediateSize, 4096)
        XCTAssertEqual(MetalLMConfig.vocabSize, 6563)
        XCTAssertEqual(MetalLMConfig.startSpeechToken, 6561)
        XCTAssertEqual(MetalLMConfig.stopSpeechToken, 6562)
        XCTAssertEqual(MetalLMConfig.blockSize, 16)
        XCTAssertEqual(MetalLMConfig.quantBlockRows, 32)
    }
}