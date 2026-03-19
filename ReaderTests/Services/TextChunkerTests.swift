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
        // NLTokenizer may treat this as 2 sentences (language-aware)
        XCTAssertGreaterThanOrEqual(result.count, 1)
        XCTAssertLessThanOrEqual(result.count, 3)
        // All content should be preserved
        let joined = result.joined(separator: " ")
        XCTAssertTrue(joined.contains("世界"))
        XCTAssertTrue(joined.contains("こんにちは"))
    }

    func testHandlesEmoji() {
        let result = TextChunker.chunkText("Hi 👋! Bye 👎!")
        XCTAssertGreaterThanOrEqual(result.count, 1)
        XCTAssertLessThanOrEqual(result.count, 2)
    }

    // MARK: - Quote Handling

    func testHandlesQuotations() {
        let result = TextChunker.chunkText("He said \"Hello world\" and left.")
        XCTAssertFalse(result.isEmpty)
    }

    func testHandlesNestedQuotes() {
        let result = TextChunker.chunkText("She said \"He replied 'Hello'\" and smiled.")
        XCTAssertFalse(result.isEmpty)
    }

    // MARK: - Dialogue Handling

    func testDialogueWithCommaInQuotes() {
        // NLTokenizer treats this linguistically — may split at sentence boundaries
        let result = TextChunker.chunkText("\"This plan is stupid,\" Witch said. \"The garrison has to mobilize.\"")
        XCTAssertFalse(result.isEmpty)
        XCTAssertLessThanOrEqual(result.count, 2, "Should produce at most 2 sentence chunks")
    }

    func testDialogueWithMultipleQuotes() {
        let result = TextChunker.chunkText("\"First quote.\" \"Second quote.\" Then normal text.")
        XCTAssertGreaterThanOrEqual(result.count, 2)
        XCTAssertLessThanOrEqual(result.count, 3)
    }

    func testDialogueWithQuestionMark() {
        let result = TextChunker.chunkText("\"Why not?\" she asked. \"Because it's dangerous.\"")
        XCTAssertEqual(result.count, 2)
    }

    func testDialogueWithExclamationMark() {
        let result = TextChunker.chunkText("\"Watch out!\" he shouted. \"It's falling!\"")
        XCTAssertEqual(result.count, 2)
    }

    func testMixedDialogueAndNarrative() {
        let result = TextChunker.chunkText("\"Hello,\" said John. He walked away. Then she replied \"Hi!\"")
        XCTAssertGreaterThanOrEqual(result.count, 2)
        XCTAssertLessThanOrEqual(result.count, 3)
    }

    // MARK: - NLTokenizer-Specific Edge Cases

    func testAbbreviationsDoNotSplit() {
        let result = TextChunker.chunkText("Dr. Smith said Mr. Jones arrived.")
        XCTAssertEqual(result.count, 1, "Abbreviations like Dr. and Mr. should not split the sentence")
    }

    func testEllipsis() {
        let result = TextChunker.chunkText("Wait... What happened? I don't know.")
        XCTAssertGreaterThanOrEqual(result.count, 2)
        XCTAssertLessThanOrEqual(result.count, 3)
    }

    func testUnicodePunctuation() {
        let result = TextChunker.chunkText("Ciao! ¿Cómo estás? Bien.")
        XCTAssertGreaterThanOrEqual(result.count, 2)
        XCTAssertLessThanOrEqual(result.count, 3)
    }

    func testNumberedList() {
        let result = TextChunker.chunkText("1. First item. 2. Second item.")
        XCTAssertGreaterThanOrEqual(result.count, 1)
        // NLTokenizer treats numbered periods contextually
    }
}
