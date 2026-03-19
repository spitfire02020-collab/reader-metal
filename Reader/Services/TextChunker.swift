import Foundation
import NaturalLanguage

// MARK: - Text Chunker
// Splits long text into optimal chunks for TTS synthesis
// Based on Chatterbox-TTS-Server's intelligent sentence-based chunking

final class TextChunker {
    // MARK: - Non-verbal Cue Handling

    /// Pattern for non-verbal cues like (laughs), (sighs), (coughs), etc.
    private static let nonVerbalCuePattern = try! NSRegularExpression(
        pattern: "(\\([\\w\\s'-]+\\))",
        options: []
    )

    /// Tags to preserve (mapped from server's paralinguistic tags)
    private static let paralinguisticTags: Set<String> = [
        "[laugh]", "[chuckle]", "[sigh]", "[gasp]", "[cough]",
        "[clear throat]", "[sniff]", "[groan]", "[shush]",
        "[laughs]", "[sighs]", "[gasps]", "[coughs]"
    ]

    // MARK: - Caching

    /// Maximum number of entries in the cache to prevent unbounded memory growth
    private static let maxCacheEntries = 100

    /// Simple cache for chunkText results
    private static var chunkCache: [String: [String]] = [:]
    
    /// Array to track insertion order for proper FIFO eviction
    private static var cacheKeys: [String] = []
    
    private static let cacheQueue = DispatchQueue(label: "com.reader.textchunker.cache", attributes: .concurrent)

    /// Evict oldest entries when cache exceeds max size (proper FIFO)
    /// Must be called within cacheQueue with barrier
    private static func evictCacheIfNeeded() {
        if chunkCache.count >= maxCacheEntries {
            // Remove oldest 20% of entries using our tracking array
            let removeCount = maxCacheEntries / 5
            let keysToRemove = cacheKeys.prefix(removeCount)
            for key in keysToRemove {
                chunkCache.removeValue(forKey: key)
            }
            cacheKeys.removeFirst(removeCount)
        }
    }

    /// Split text into chunks - one sentence per chunk
    /// Result is cached for performance
    static func chunkText(_ text: String) -> [String] {
        // Check cache first
        if let cached = cacheQueue.sync(execute: { chunkCache[text] }) {
            return cached
        }

        let cleaned = cleanText(text)
        guard !cleaned.isEmpty else { return [] }

        // Preserve non-verbal cues with placeholders
        let (processedText, cuePlaceholders) = preserveNonVerbalCues(cleaned)

        // Split into sentences - each sentence is a chunk
        let sentences = splitIntoSentences(processedText)
        guard !sentences.isEmpty else { return [] }

        // Restore non-verbal cues in sentences
        let sentencesWithCues = restoreNonVerbalCues(sentences, cuePlaceholders)

        // Return each sentence as its own chunk
        let result = sentencesWithCues.filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

        // Cache the result
        // Use barrier for thread-safe write
        cacheQueue.async(flags: .barrier) {
            if chunkCache[text] == nil {
                evictCacheIfNeeded()
                cacheKeys.append(text)
                chunkCache[text] = result
            }
        }

        return result
    }

    // MARK: - Non-verbal Cue Preservation

    /// Preserve non-verbal cues with placeholders to prevent splitting
    private static func preserveNonVerbalCues(_ text: String) -> (String, [String]) {
        var placeholders: [String] = []

        // Find all non-verbal cues
        let range = NSRange(text.startIndex..., in: text)
        let matches = nonVerbalCuePattern.matches(in: text, options: [], range: range)

        // Replace with numbered placeholders
        var result = text
        for match in matches.reversed() {
            if let range = Range(match.range, in: result) {
                let cue = String(result[range])
                let placeholder = "___CUE_\(placeholders.count)___"
                placeholders.append(cue)
                result.replaceSubrange(range, with: placeholder)
            }
        }

        return (result, placeholders)
    }

    /// Restore non-verbal cues from placeholders
    private static func restoreNonVerbalCues(_ sentences: [String], _ placeholders: [String]) -> [String] {
        return sentences.map { sentence in
            var result = sentence
            for (index, cue) in placeholders.enumerated() {
                let placeholder = "___CUE_\(index)___"
                result = result.replacingOccurrences(of: placeholder, with: cue)
            }
            return result
        }
    }

    // MARK: - Text Cleaning

    /// Clean text for display - removes URLs, HTML, normalizes whitespace
    /// Use this text for display so chunks match exactly
    static func cleanTextForDisplay(_ text: String) -> String {
        return cleanText(text)
    }

    private static func cleanText(_ text: String) -> String {
        var cleaned = text

        // Normalize line endings
        cleaned = cleaned.replacingOccurrences(of: "\r\n", with: "\n")
        cleaned = cleaned.replacingOccurrences(of: "\r", with: "\n")

        // Remove excessive whitespace
        cleaned = cleaned.replacingOccurrences(
            of: "[ \\t]+",
            with: " ",
            options: .regularExpression
        )

        // Normalize newlines
        cleaned = cleaned.replacingOccurrences(
            of: "\\n{3,}",
            with: "\n\n",
            options: .regularExpression
        )

        // Remove URLs
        cleaned = cleaned.replacingOccurrences(
            of: "https?://\\S+",
            with: "",
            options: .regularExpression
        )

        // Remove HTML tags
        cleaned = cleaned.replacingOccurrences(
            of: "<[^>]+>",
            with: "",
            options: .regularExpression
        )

        // Fix common OCR artifacts
        cleaned = cleaned.replacingOccurrences(of: "\u{00AD}", with: "")
        cleaned = cleaned.replacingOccurrences(of: "\u{200B}", with: "")

        return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Sentence Splitting (NLTokenizer)

    /// Split text into sentences using Apple's NLTokenizer for robust,
    /// linguistically-aware boundary detection. Handles abbreviations,
    /// Unicode, quotes, bullet points, and ellipsis out of the box.
    private static func splitIntoSentences(_ text: String) -> [String] {
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text

        var sentences: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let sentence = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !sentence.isEmpty {
                sentences.append(sentence)
            }
            return true
        }

        // Fallback: if tokenizer found nothing, return the whole text
        if sentences.isEmpty && !text.isEmpty {
            return [text]
        }

        // Filter out fragments that are just punctuation
        sentences = sentences.filter { sentence in
            let trimmed = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
            guard trimmed.count >= 2 else { return false }
            let isJustPunctuation = trimmed.rangeOfCharacter(from: CharacterSet.letters) == nil
            return !isJustPunctuation
        }

        return sentences
    }

    // MARK: - Duration Estimation

    /// Estimate audio duration for a text chunk (based on average speaking rate)
    static func estimateDuration(for text: String, wordsPerMinute: Double = 150) -> TimeInterval {
        let wordCount = text.split(separator: " ").count
        return Double(wordCount) / wordsPerMinute * 60.0
    }

    /// Estimate total duration for full text
    static func estimateTotalDuration(for text: String, wordsPerMinute: Double = 150) -> TimeInterval {
        let chunks = chunkText(text)
        return chunks.reduce(0) { $0 + estimateDuration(for: $1, wordsPerMinute: wordsPerMinute) }
    }
}
