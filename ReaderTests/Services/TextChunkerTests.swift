import XCTest
@testable import Reader

final class TextChunkerTests: XCTestCase {

    // MARK: - Basic Functionality

    func testSingleSentence() {
        let result = TextChunker.chunkText("This is a single sentence.")
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0], "This is a single sentence.")
    }

    func testMultipleSentences() {
        let result = TextChunker.chunkText("First sentence. Second sentence. Third!")
        XCTAssertEqual(result.count, 3)
    }

    func testQuestionMarks() {
        let result = TextChunker.chunkText("What is this? Is it working?")
        XCTAssertEqual(result.count, 2)
    }

    func testExclamationMarks() {
        let result = TextChunker.chunkText("Wow! That's amazing! Really!")
        XCTAssertEqual(result.count, 3)
    }

    // MARK: - Edge Cases

    func testEmptyString() {
        let result = TextChunker.chunkText("")
        XCTAssertTrue(result.isEmpty)
    }

    func testOnlyWhitespace() {
        let result = TextChunker.chunkText("   \n\t  ")
        XCTAssertTrue(result.isEmpty)
    }

    func testMultipleWhitespace() {
        let result = TextChunker.chunkText("Hello.   \n\n  World!")
        XCTAssertEqual(result.count, 2)
    }

    // MARK: - Non-Verbal Cues

    func testPreservesParalinguisticCues() {
        let result = TextChunker.chunkText("Hello (laughs) world.")
        XCTAssertTrue(result.first?.contains("(laughs)") ?? false)
    }

    // MARK: - Duration Estimation

    func testDurationEstimation() {
        let duration = TextChunker.estimateDuration(for: "One two three four five")
        // 5 words at ~150 WPM = 2 seconds
        XCTAssertEqual(duration, 2.0, accuracy: 0.1)
    }

    func testDurationEstimationEmpty() {
        let duration = TextChunker.estimateDuration(for: "")
        XCTAssertEqual(duration, 0.0, accuracy: 0.01)
    }

    // MARK: - Cache Behavior

    func testCacheReturnsCachedResult() {
        let text = "Cached sentence. Second one."

        let result1 = TextChunker.chunkText(text)
        let result2 = TextChunker.chunkText(text)

        XCTAssertEqual(result1, result2)
    }

    // MARK: - Unicode/Localization

    func testHandlesUnicodeCharacters() {
        let result = TextChunker.chunkText("Hello 世界! こんにちは!")
        XCTAssertEqual(result.count, 3)
    }

    func testHandlesEmoji() {
        let result = TextChunker.chunkText("Hi 👋! Bye 👎!")
        XCTAssertEqual(result.count, 2)
    }

    // MARK: - Quote Handling

    func testHandlesQuotations() {
        let result = TextChunker.chunkText("He said \"Hello world\" and left.")
        // Should handle quotes appropriately
        XCTAssertFalse(result.isEmpty)
    }

    func testHandlesNestedQuotes() {
        let result = TextChunker.chunkText("She said \"He replied 'Hello'\" and smiled.")
        XCTAssertFalse(result.isEmpty)
    }
}
