import Foundation

// MARK: - Library Item

enum ContentSource: String, Codable {
    case webpage
    case epub
    case pdf
    case text
}

enum ItemStatus: String, Codable {
    case pending      // Not yet synthesized
    case processing   // Currently being synthesized
    case ready        // Audio ready to play
    case error        // Synthesis failed
}

struct LibraryItem: Identifiable, Codable {
    let id: UUID
    var title: String
    var author: String?
    var source: ContentSource
    var sourceURL: String?
    var coverImageData: Data?
    var dateAdded: Date
    var lastPlayed: Date?
    var duration: TimeInterval?
    var currentPosition: TimeInterval
    var textContent: String
    var chapters: [Chapter]
    var selectedVoiceID: String?
    var status: ItemStatus
    var audioFileURL: String?
    var progress: Double // 0.0 - 1.0
    /// Track which chunks have been generated (chunk index -> file path)
    var generatedChunks: [Int: String] = [:]

    var displayAuthor: String {
        author ?? (source == .webpage ? sourceURL?.hostFromURL ?? "Web" : "Unknown")
    }

    var formattedDuration: String {
        guard let duration else { return "--:--" }
        let minutes = Int(duration) / 60
        let seconds = Int(duration) % 60
        if minutes >= 60 {
            let hours = minutes / 60
            let mins = minutes % 60
            return String(format: "%d:%02d:%02d", hours, mins, seconds)
        }
        return String(format: "%d:%02d", minutes, seconds)
    }

    var formattedProgress: String {
        "\(Int(progress * 100))%"
    }
}

struct Chapter: Identifiable, Codable {
    let id: UUID
    var title: String
    var textContent: String
    var startTime: TimeInterval?
    var endTime: TimeInterval?
    var audioFileURL: String?

    init(id: UUID = UUID(), title: String, textContent: String, startTime: TimeInterval? = nil, endTime: TimeInterval? = nil, audioFileURL: String? = nil) {
        self.id = id
        self.title = title
        self.textContent = textContent
        self.startTime = startTime
        self.endTime = endTime
        self.audioFileURL = audioFileURL
    }
}

// MARK: - Helpers

extension String {
    /// Decode HTML character entities.  Handles named (&amp; &lt; etc.),
    /// decimal (&#39;) and hex (&#x27;) forms.
    var decodedHTMLEntities: String {
        guard contains("&") else { return self }
        var result = self
        let named: [(String, String)] = [
            ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
            ("&quot;", "\""), ("&#39;", "'"), ("&apos;", "'"),
            ("&nbsp;", " "), ("&mdash;", "—"), ("&ndash;", "–"),
            ("&hellip;", "…"), ("&lsquo;", "\u{2018}"), ("&rsquo;", "\u{2019}"),
            ("&ldquo;", "\u{201C}"), ("&rdquo;", "\u{201D}"),
        ]
        for (entity, char) in named {
            result = result.replacingOccurrences(of: entity, with: char, options: .caseInsensitive)
        }
        // Decimal: &#NNN;
        if let rx = try? NSRegularExpression(pattern: "&#(\\d+);") {
            let ns = result as NSString
            let ms = rx.matches(in: result, range: NSRange(location: 0, length: ns.length))
            for m in ms.reversed() {
                if let code = Int(ns.substring(with: m.range(at: 1))),
                   let scalar = Unicode.Scalar(code) {
                    result = (result as NSString).replacingCharacters(in: m.range, with: String(Character(scalar)))
                }
            }
        }
        // Hex: &#xNN;
        if let rx = try? NSRegularExpression(pattern: "&#[xX]([0-9a-fA-F]+);") {
            let ns = result as NSString
            let ms = rx.matches(in: result, range: NSRange(location: 0, length: ns.length))
            for m in ms.reversed() {
                if let code = UInt32(ns.substring(with: m.range(at: 1)), radix: 16),
                   let scalar = Unicode.Scalar(code) {
                    result = (result as NSString).replacingCharacters(in: m.range, with: String(Character(scalar)))
                }
            }
        }
        return result
    }

    var hostFromURL: String? {
        guard let url = URL(string: self),
              let host = url.host else { return nil }
        return host.replacingOccurrences(of: "www.", with: "")
    }
}

// MARK: - Sample Data

extension LibraryItem {
    static let sampleItems: [LibraryItem] = [
        LibraryItem(
            id: UUID(),
            title: "The Future of Artificial Intelligence",
            author: "MIT Technology Review",
            source: .webpage,
            sourceURL: "https://technologyreview.com/ai-future",
            coverImageData: nil,
            dateAdded: Date().addingTimeInterval(-86400),
            lastPlayed: Date().addingTimeInterval(-3600),
            duration: 720,
            currentPosition: 240,
            textContent: "Sample article about AI...",
            chapters: [Chapter(title: "Introduction", textContent: "AI is transforming...")],
            selectedVoiceID: "default",
            status: .ready,
            audioFileURL: nil,
            progress: 0.33
        ),
        LibraryItem(
            id: UUID(),
            title: "Thinking, Fast and Slow",
            author: "Daniel Kahneman",
            source: .epub,
            sourceURL: nil,
            coverImageData: nil,
            dateAdded: Date().addingTimeInterval(-172800),
            lastPlayed: nil,
            duration: 54000,
            currentPosition: 0,
            textContent: "In the 1970s...",
            chapters: [
                Chapter(title: "Chapter 1: The Characters of the Story", textContent: "Two systems..."),
                Chapter(title: "Chapter 2: Attention and Effort", textContent: "Mental effort...")
            ],
            selectedVoiceID: "default",
            status: .ready,
            audioFileURL: nil,
            progress: 0.0
        ),
        LibraryItem(
            id: UUID(),
            title: "Understanding Quantum Computing",
            author: "Nature",
            source: .webpage,
            sourceURL: "https://nature.com/quantum",
            coverImageData: nil,
            dateAdded: Date().addingTimeInterval(-3600),
            lastPlayed: nil,
            duration: nil,
            currentPosition: 0,
            textContent: "Quantum computing is...",
            chapters: [],
            selectedVoiceID: nil,
            status: .processing,
            audioFileURL: nil,
            progress: 0.0
        ),
    ]
}

// MARK: - Notification Names

extension Notification.Name {
    /// Posted when user taps generate button from library view
    /// UserInfo contains "item": LibraryItem
    static let startGenerationFromLibrary = Notification.Name("startGenerationFromLibrary")
}
