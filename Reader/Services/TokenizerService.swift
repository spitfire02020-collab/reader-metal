import Foundation

// MARK: - GPT-2 Byte-Level BPE Tokenizer
//
// The ResembleAI/chatterbox-turbo-ONNX model uses a GPT-2 byte-level BPE tokenizer
// (50,257 base vocab + emotion/paralinguistic added tokens).
//
// Key facts:
//   • embed_tokens accepts full GPT-2 IDs (0-50256) for text tokens
//   • LM logits output vocab = 6563 (speech codec space)
//   • eos_token_id = 6562 (STOP_SPEECH); generation ends when LM predicts this
//   • Input format: [GPT2_text_ids..., 50256, 50256]
//     (the tokenizer's TemplateProcessing post-processor appends 2× endoftext)
//
// Reference: ResembleAI/chatterbox-turbo-ONNX tokenizer.json (GPT2Tokenizer, ByteLevel)

final class TokenizerService {
    // Vocabulary: byte-encoded token string → ID (50,257+ entries)
    private var encoder: [String: Int] = [:]
    // BPE merge ranks: pair → priority rank (lower = applied first)
    private var bpeRanks: [BpePair: Int] = [:]
    // Special added tokens: e.g. "[angry]" → 50257, "[laughter]" → ...
    private var addedTokens: [(text: String, id: Int)] = []  // sorted longest-first
    // Byte → unicode char mapping (GPT-2 standard byte encoder)
    // Static to avoid rebuilding for each instance
    private static let _byteEncoder = TokenizerService.buildByteEncoder()
    private let byteEncoder: [UInt8: Character]

    // Pre-compiled regex for pretokenization - avoid recompiling each call
    private static var pretokenizeRegex: NSRegularExpression? = {
        let pattern = "'(?:s|t|re|ve|m|ll|d)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
        return try? NSRegularExpression(pattern: pattern, options: .caseInsensitive)
    }()

    private(set) var isLoaded = false

    static let endOfTextId   = 50256  // <|endoftext|> — appended ×2 as speech-start signal
    static let startSpeechId = 6561   // [START_SPEECH]
    static let stopSpeechId  = 6562   // [STOP_SPEECH]  — eos_token_id per generation_config.json

    private struct BpePair: Hashable {
        let first: String
        let second: String
    }

    // MARK: - Init

    init() {
        byteEncoder = TokenizerService._byteEncoder
    }

    // MARK: - GPT-2 Byte Encoder
    //
    // Maps bytes 0-255 to unicode characters:
    //   Printable ASCII 33-126 + extended Latin 161-172, 174-255 → themselves
    //   Remaining bytes (0-32, 127-160, 173) → unicode codepoints 256, 257, ...
    //
    // e.g.: ' ' (0x20=32) → U+0120 'Ġ'  (32nd "extra" byte → 256+32 = 288? No:
    //   bytes 0..32 are the first 33 of the 68 "extra" bytes, so 0→256, 1→257, ..., 32→288)
    private static func buildByteEncoder() -> [UInt8: Character] {
        // Bytes that map to themselves (printable ASCII + extended Latin)
        let nativeBytes = Array(33...126) + Array(161...172) + Array(174...255)
        let nativeSet = Set(nativeBytes)

        var result: [UInt8: Character] = [:]
        for b in nativeBytes {
            result[UInt8(b)] = Character(UnicodeScalar(b)!)
        }

        // Map remaining bytes to sequential unicode above 255
        var offset: UInt32 = 256
        for b in 0...255 {
            if !nativeSet.contains(b) {
                result[UInt8(b)] = Character(UnicodeScalar(offset)!)
                offset += 1
            }
        }
        return result
    }

    // MARK: - Load

    func load(from path: URL) throws {
        let data = try Data(contentsOf: path)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw TokenizerError.invalidFormat
        }

        // Vocabulary and merge rules
        guard let model = json["model"] as? [String: Any] else {
            throw TokenizerError.invalidFormat
        }
        if let vocabMap = model["vocab"] as? [String: Any] {
            encoder.reserveCapacity(vocabMap.count)
            for (token, idAny) in vocabMap {
                if let id = idAny as? Int {
                    encoder[token] = id
                }
            }
        }
        if let merges = model["merges"] as? [[String]] {
            bpeRanks.reserveCapacity(merges.count)
            for (rank, merge) in merges.enumerated() {
                guard merge.count == 2 else { continue }
                bpeRanks[BpePair(first: merge[0], second: merge[1])] = rank
            }
        } else if let merges = model["merges"] as? [String] {
            // Fallback for space-separated format
            bpeRanks.reserveCapacity(merges.count)
            for (rank, merge) in merges.enumerated() {
                let parts = merge.split(separator: " ", maxSplits: 1)
                guard parts.count == 2 else { continue }
                bpeRanks[BpePair(first: String(parts[0]), second: String(parts[1]))] = rank
            }
        }

        // Added tokens (special tags like <|endoftext|>, [angry], etc.)
        if let added = json["added_tokens"] as? [[String: Any]] {
            var raw: [(String, Int)] = []
            for token in added {
                if let content = token["content"] as? String,
                   let id = token["id"] as? Int,
                   !content.isEmpty {
                    raw.append((content, id))
                }
            }
            // Sort longest first to avoid partial-match ambiguity
            addedTokens = raw
                .sorted { $0.0.count > $1.0.count }
                .map { (text: $0.0, id: $0.1) }
        }

        isLoaded = true
    }

    // MARK: - Encode

    /// Tokenize text for the turbo model.
    /// Returns: [GPT2_token_ids..., 50256, 50256]
    func encode(_ text: String) -> [Int] {
        guard isLoaded, !text.isEmpty else { return [] }

        let normalized = text
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespaces)
        guard !normalized.isEmpty else { return [] }

        // Process text, substituting special added tokens (e.g. [laughter])
        var result: [Int] = []
        var remaining = normalized

        while !remaining.isEmpty {
            // Find earliest-occurring special token
            var bestRange: Range<String.Index>? = nil
            var bestId = 0

            for (tokenStr, tokenId) in addedTokens {
                // Skip the two-token endoftext — we add those manually at the end
                if tokenStr == "<|endoftext|>" { continue }
                if let range = remaining.range(of: tokenStr) {
                    if bestRange == nil || range.lowerBound < bestRange!.lowerBound {
                        bestRange = range
                        bestId = tokenId
                    }
                }
            }

            if let range = bestRange {
                // Encode plain text before the special token
                let prefix = String(remaining[remaining.startIndex..<range.lowerBound])
                result.append(contentsOf: encodeText(prefix))
                result.append(bestId)
                remaining = String(remaining[range.upperBound...])
            } else {
                result.append(contentsOf: encodeText(remaining))
                break
            }
        }

        // Append 2× endoftext — the TemplateProcessing post-processor from the
        // tokenizer.json appends these to signal the end of text / start of speech.
        result.append(Self.endOfTextId)
        result.append(Self.endOfTextId)
        return result
    }

    // MARK: - Plain Text Encoding (byte-level BPE)

    private func encodeText(_ text: String) -> [Int] {
        guard !text.isEmpty else { return [] }
        let segments = pretokenize(text)
        var ids: [Int] = []
        for segment in segments {
            let byteStr = byteLevelEncode(segment)
            guard !byteStr.isEmpty else { continue }
            let tokens = applyBPE(byteStr)
            for token in tokens {
                if let id = encoder[token] {
                    ids.append(id)
                }
                // Unknown tokens: GPT-2 vocab covers all 256 bytes, so this shouldn't happen
            }
        }
        return ids
    }

    // MARK: - Pre-tokenizer
    //
    // Approximates GPT-2's ByteLevel pre-tokenizer (use_regex=true).
    // The GPT-2 regex splits text into: contractions, words (with optional leading space),
    // numbers (with optional leading space), and punctuation/other.
    //
    // Each matched segment is byte-encoded as a unit, so spaces become 'Ġ' prefixes.
    private func pretokenize(_ text: String) -> [String] {
        guard let regex = TokenizerService.pretokenizeRegex else {
            return [text]
        }
        let nsText = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: nsText.length))
        return matches.map { nsText.substring(with: $0.range) }
    }

    // MARK: - Byte-Level Encoding
    //
    // Converts each UTF-8 byte in the string to its corresponding GPT-2 unicode character.
    // e.g. " Hello" → "ĠHello" (space 0x20 → U+0120 'Ġ')
    private func byteLevelEncode(_ text: String) -> String {
        var result = ""
        result.reserveCapacity(text.utf8.count)
        for byte in text.utf8 {
            if let char = byteEncoder[byte] {
                result.append(char)
            }
        }
        return result
    }

    // MARK: - BPE

    /// Apply BPE merge rules to a byte-encoded token string.
    /// Returns the list of token strings after all applicable merges.
    private func applyBPE(_ token: String) -> [String] {
        guard token.count > 1 else { return [token] }

        var word = token.unicodeScalars.map { String($0) }

        while word.count > 1 {
            // Find the highest-priority adjacent pair (lowest rank)
            var bestRank = Int.max
            var bestIdx = -1

            for i in 0..<(word.count - 1) {
                let pair = BpePair(first: word[i], second: word[i + 1])
                if let rank = bpeRanks[pair], rank < bestRank {
                    bestRank = rank
                    bestIdx = i
                }
            }

            guard bestIdx >= 0 else { break }

            // Merge the pair at bestIdx
            var newWord: [String] = []
            newWord.reserveCapacity(word.count - 1)
            var i = 0
            while i < word.count {
                if i == bestIdx {
                    newWord.append(word[i] + word[i + 1])
                    i += 2
                } else {
                    newWord.append(word[i])
                    i += 1
                }
            }
            word = newWord
        }

        return word
    }
}

// MARK: - Errors

enum TokenizerError: Error {
    case invalidFormat
    case notLoaded
}
