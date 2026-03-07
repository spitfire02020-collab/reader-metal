import Foundation

// MARK: - Text Chunker
// Splits long text into optimal chunks for TTS synthesis
// Based on Chatterbox-TTS-Server's intelligent sentence-based chunking

final class TextChunker {
    /// Target characters per chunk (0 = no limit, sentences only)
    static let targetChunkSize = 0  // Each sentence is a chunk

    /// Minimum characters to avoid tiny fragments
    static let minChunkSize = 100

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

    /// Simple cache for chunkText results to avoid repeated regex processing
    private static var chunkCache: [String: [String]] = [:]
    private static let cacheQueue = DispatchQueue(label: "com.reader.textchunker.cache")

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
        cacheQueue.async { chunkCache[text] = result }

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

    // MARK: - Sentence Splitting

    /// Abbreviations that should NOT end a sentence
    private static let abbreviations: Set<String> = [
        "mr", "mrs", "ms", "dr", "prof", "rev", "hon", "st",
        "sgt", "capt", "lt", "col", "gen", "etc", "eg", "ie",
        "vs", "approx", "apt", "dept", "fig", "gov", "inc",
        "jr", "sr", "ltd", "no", "p", "pp", "vol", "op",
        "cit", "ca", "cf", "ed", "esp", "et", "al", "ibid",
        "id", "inf", "sup", "viz", "sc", "fl", "d", "b",
        "r", "c", "v", "u", "s", "a", "m", "p", "d",
        "bc", "ad", "rn", "bsn", "md", "do", "dds", "dmd"
    ]

    /// Pattern for bullet points
    private static let bulletPointPattern = try! NSRegularExpression(
        pattern: "(?:^|\\n)\\s*([-•*]|\\d+\\.)\\s+",
        options: []
    )

    private static func splitIntoSentences(_ text: String) -> [String] {
        // Handle bullet points first - treat each bullet as a segment
        let range = NSRange(text.startIndex..., in: text)
        let bulletMatches = bulletPointPattern.matches(in: text, options: [], range: range)

        if !bulletMatches.isEmpty {
            return splitBulletPointText(text, matches: bulletMatches)
        }

        return splitByPunctuation(text)
    }

    /// Split text with bullet points, preserving list formatting
    private static func splitBulletPointText(_ text: String, matches: [NSTextCheckingResult]) -> [String] {
        var sentences: [String] = []
        var currentPosition = text.startIndex

        for (index, match) in matches.enumerated() {
            guard let range = Range(match.range, in: text) else { continue }

            // Process text before this bullet
            let beforeBullet = String(text[currentPosition..<range.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !beforeBullet.isEmpty {
                sentences.append(contentsOf: splitByPunctuation(beforeBullet))
            }

            // Process this bullet item
            let nextStart: String.Index
            if index + 1 < matches.count {
                if let nextRange = Range(matches[index + 1].range, in: text) {
                    nextStart = nextRange.lowerBound
                } else {
                    nextStart = text.endIndex
                }
            } else {
                nextStart = text.endIndex
            }

            let bulletItem = String(text[range.lowerBound..<nextStart]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !bulletItem.isEmpty {
                sentences.append(bulletItem)
            }

            currentPosition = range.upperBound
        }

        // Process remaining text after last bullet
        if currentPosition < text.endIndex {
            let remaining = String(text[currentPosition...]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !remaining.isEmpty {
                sentences.append(contentsOf: splitByPunctuation(remaining))
            }
        }

        return sentences
    }

    /// Split text by punctuation marks (. ! ?)
    private static func splitByPunctuation(_ text: String) -> [String] {
        var sentences: [String] = []
        var currentSentence = ""
        var quoteDepth = 0  // Track nested quotes properly

        let chars = Array(text)
        var i = 0

        while i < chars.count {
            let char = chars[i]

            switch char {
            case "\"":
                currentSentence.append(char)
                // Only toggle if not escaped (not preceded by backslash)
                if i == 0 || chars[i-1] != "\\" {
                    quoteDepth += 1
                    if quoteDepth > 1 {
                        quoteDepth -= 1  // Keep depth at 1 for non-nested quotes
                    }
                }

            case "'":
                // Handle apostrophes separately - only count as quote if:
                // 1. Surrounded by whitespace (opening/closing quote)
                // 2. At start/end of word with whitespace on one side
                let isApostrophe = {
                    let prevIsWord = i > 0 && chars[i-1].isLetter
                    let nextIsWord = i + 1 < chars.count && chars[i+1].isLetter
                    return prevIsWord || nextIsWord
                }()
                currentSentence.append(char)
                if !isApostrophe {
                    quoteDepth += 1
                    if quoteDepth > 1 {
                        quoteDepth -= 1
                    }
                }

            case ".", "!", "?":
                // Check quote context:
                // - nextIsQuote: punctuation before closing quote → don't split
                // - quoteDepth > 0: currently inside quotes → don't split
                // - prevIsQuote but NOT nextIsQuote: punctuation after closing quote → DO split (sentence ended)
                let nextIsQuote = i + 1 < chars.count && chars[i+1] == "\""
                let isInsideQuotes = quoteDepth > 0 || nextIsQuote

                if isInsideQuotes {
                    currentSentence.append(char)
                    i += 1
                    continue
                }

                currentSentence.append(char)

                // Check for abbreviation (e.g., "Dr." or "Mr.")
                if char == "." {
                    // Look at next few chars to check for abbreviation
                    let remaining = String(chars[(i+1)...]).prefix(3).lowercased()
                    let wordEnd = remaining.prefix(while: { $0.isLetter }).lowercased()

                    if abbreviations.contains(wordEnd) || wordEnd.hasSuffix(".") {
                        // Skip - this is an abbreviation
                        i += 1
                        continue
                    }
                }

                // Skip if followed by lowercase (likely abbreviation like "e.g.")
                if i + 1 < chars.count {
                    let nextChar = chars[i + 1]
                    if nextChar.isWhitespace {
                        // Check if sentence ends with period followed by lowercase
                        if let lastWord = currentSentence.split(separator: " ").last?.lowercased(),
                           abbreviations.contains(lastWord.trimmingCharacters(in: CharacterSet(charactersIn: "."))) {
                            i += 1
                            continue
                        }
                    }
                }

                // Skip if followed by lowercase letter (not end of sentence)
                if i + 1 < chars.count {
                    let nextNonSpace = String(chars[(i+1)...]).prefix(while: { $0.isWhitespace }).dropFirst()
                    if let first = nextNonSpace.first, first.isLowercase {
                        i += 1
                        continue
                    }
                }

                // End of sentence - capture whitespace and continue
                i += 1
                while i < chars.count && chars[i].isWhitespace {
                    i += 1
                }
                if !currentSentence.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    sentences.append(currentSentence.trimmingCharacters(in: .whitespacesAndNewlines))
                }
                currentSentence = ""
                continue

            default:
                currentSentence.append(char)
            }

            i += 1
        }

        // Don't forget the last sentence
        if !currentSentence.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            sentences.append(currentSentence.trimmingCharacters(in: .whitespacesAndNewlines))
        }

        // Handle edge case: no sentences found, return original text
        if sentences.isEmpty && !text.isEmpty {
            return [text]
        }

        return sentences
    }

    // MARK: - Long Sentence Splitting

    /// Split a sentence that's too long by clauses (commas, semicolons, etc.)
    private static func splitLongSentence(_ text: String) -> [String] {
        let delimiters = [", ", "; ", " - ", " — ", ": "]
        var chunks: [String] = []
        var current = ""

        let words = text.split(separator: " ", omittingEmptySubsequences: true).map(String.init)

        for word in words {
            let test = current.isEmpty ? word : current + " " + word

            if test.count > targetChunkSize && !current.isEmpty {
                chunks.append(current.trimmingCharacters(in: .whitespaces))
                current = word
            } else {
                current = test
            }

            // Check for natural break points
            for delimiter in delimiters {
                if current.hasSuffix(delimiter.trimmingCharacters(in: .whitespaces)) && current.count > minChunkSize {
                    chunks.append(current.trimmingCharacters(in: .whitespaces))
                    current = ""
                    break
                }
            }
        }

        if !current.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            // If we have a previous chunk and this one is small, merge
            if let last = chunks.last, current.count < minChunkSize {
                chunks[chunks.count - 1] = last + " " + current
            } else {
                chunks.append(current.trimmingCharacters(in: .whitespaces))
            }
        }

        return chunks
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
