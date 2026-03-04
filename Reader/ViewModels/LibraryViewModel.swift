import Foundation
import SwiftUI
import UniformTypeIdentifiers

// MARK: - Library View Model

@MainActor
final class LibraryViewModel: ObservableObject {
    @Published var items: [LibraryItem] = []
    @Published var searchText = ""
    @Published var isLoading = false
    @Published var showAddContent = false
    @Published var selectedFilter: ContentFilter = .all
    @Published var errorMessage: String?
    @Published var showError = false

    init() {
        // Load library items on init
        loadLibrary()
    }

    private let extractor = WebContentExtractor()
    private let engine = ChatterboxEngine()
    private let storageKey = "library_items"

    enum ContentFilter: String, CaseIterable {
        case all = "All"
        case articles = "Articles"
        case books = "Books"
        case inProgress = "In Progress"

        var icon: String {
            switch self {
            case .all: return "square.grid.2x2"
            case .articles: return "doc.text"
            case .books: return "book"
            case .inProgress: return "play.circle"
            }
        }
    }

    var filteredItems: [LibraryItem] {
        var result = items

        switch selectedFilter {
        case .all: break
        case .articles:
            result = result.filter { $0.source == .webpage }
        case .books:
            result = result.filter { $0.source == .epub || $0.source == .pdf }
        case .inProgress:
            result = result.filter { $0.progress > 0 && $0.progress < 1.0 }
        }

        if !searchText.isEmpty {
            result = result.filter {
                $0.title.localizedCaseInsensitiveContains(searchText) ||
                ($0.author?.localizedCaseInsensitiveContains(searchText) ?? false)
            }
        }

        return result.sorted { ($0.lastPlayed ?? $0.dateAdded) > ($1.lastPlayed ?? $1.dateAdded) }
    }

    // MARK: - Persistence

    func loadLibrary() {
        if let data = UserDefaults.standard.data(forKey: storageKey),
           let decoded = try? JSONDecoder().decode([LibraryItem].self, from: data) {
            // Sanitize HTML entities in titles that may have been stored before entity decoding was in place
            // Also reset .processing status to .pending (app was likely killed during generation)
            items = decoded.map { item in
                guard item.title.contains("&") || item.status == .processing else { return item }
                var mutable = item
                mutable.title = item.title.decodedHTMLEntities
                // Reset processing status - generation was interrupted
                if mutable.status == .processing {
                    mutable.status = .pending
                }
                return mutable
            }
        }
    }

    func saveLibrary() {
        if let data = try? JSONEncoder().encode(items) {
            UserDefaults.standard.set(data, forKey: storageKey)
        }
    }

    // MARK: - Add Content from URL

    func addFromURL(_ urlString: String) async {
        let trimmed = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: trimmed), url.scheme != nil else {
            errorMessage = "Please enter a valid URL."
            showError = true
            return
        }

        isLoading = true
        defer {
            isLoading = false
            showAddContent = false
        }

        do {
            let content = try await extractor.extract(from: url)

            let item = LibraryItem(
                id: UUID(),
                title: content.title,
                author: content.author,
                source: .webpage,
                sourceURL: content.sourceURL,
                coverImageData: nil,
                dateAdded: Date(),
                lastPlayed: nil,
                duration: TextChunker.estimateTotalDuration(for: content.text),
                currentPosition: 0,
                textContent: content.text,
                chapters: [Chapter(title: content.title, textContent: content.text)],
                selectedVoiceID: VoiceProfile.defaultVoice.id,
                status: .pending,
                audioFileURL: nil,
                progress: 0
            )

            items.insert(item, at: 0)
            saveLibrary()
        } catch {
            errorMessage = error.localizedDescription
            showError = true
        }
    }

    // MARK: - Add Content from File

    func addFromFile(at url: URL) async {
        isLoading = true
        defer {
            isLoading = false
            showAddContent = false
        }

        do {
            let accessed = url.startAccessingSecurityScopedResource()
            defer { if accessed { url.stopAccessingSecurityScopedResource() } }

            let ext = url.pathExtension.lowercased()
            let parsed: ParsedBook

            switch ext {
            case "epub":
                parsed = try BookParser.parseEPUB(at: url)
            case "pdf":
                parsed = try BookParser.parsePDF(at: url)
            case "txt", "md":
                let text = try String(contentsOf: url, encoding: .utf8)
                let title = url.deletingPathExtension().lastPathComponent
                parsed = BookParser.parsePlainText(text, title: title)
            default:
                errorMessage = "Unsupported file format: .\(ext)"
                showError = true
                return
            }

            let source: ContentSource = ext == "epub" ? .epub : (ext == "pdf" ? .pdf : .text)

            let item = LibraryItem(
                id: UUID(),
                title: parsed.title,
                author: parsed.author,
                source: source,
                sourceURL: nil,
                coverImageData: parsed.coverImageData,
                dateAdded: Date(),
                lastPlayed: nil,
                duration: TextChunker.estimateTotalDuration(for: parsed.fullText),
                currentPosition: 0,
                textContent: parsed.fullText,
                chapters: parsed.chapters,
                selectedVoiceID: VoiceProfile.defaultVoice.id,
                status: .pending,
                audioFileURL: nil,
                progress: 0
            )

            items.insert(item, at: 0)
            saveLibrary()
        } catch {
            errorMessage = error.localizedDescription
            showError = true
        }
    }

    // MARK: - Add from Pasted Text

    func addFromText(_ text: String, title: String) {
        let parsed = BookParser.parsePlainText(text, title: title)

        let item = LibraryItem(
            id: UUID(),
            title: parsed.title,
            author: nil,
            source: .text,
            sourceURL: nil,
            coverImageData: nil,
            dateAdded: Date(),
            lastPlayed: nil,
            duration: TextChunker.estimateTotalDuration(for: text),
            currentPosition: 0,
            textContent: text,
            chapters: parsed.chapters,
            selectedVoiceID: VoiceProfile.defaultVoice.id,
            status: .pending,
            audioFileURL: nil,
            progress: 0
        )

        items.insert(item, at: 0)
        saveLibrary()
        showAddContent = false
    }

    // MARK: - Synthesize Item

    func synthesize(item: LibraryItem) async {
        guard let index = items.firstIndex(where: { $0.id == item.id }) else { return }
        items[index].status = .processing

        do {
            if !engine.isLoaded {
                try await engine.loadModels()
            }

            let outputDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
                .appendingPathComponent("Audio", isDirectory: true)
            try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

            let outputURL = outputDir.appendingPathComponent("\(item.id.uuidString).wav")

            try await engine.synthesize(
                text: item.textContent,
                outputURL: outputURL,
                onProgress: { progress in
                    Task { @MainActor in
                        if let idx = self.items.firstIndex(where: { $0.id == item.id }) {
                            self.items[idx].progress = progress * 0.5 // Synthesis is 50% of total
                        }
                    }
                }
            )

            items[index].audioFileURL = outputURL.path
            items[index].status = .ready
            saveLibrary()
        } catch {
            items[index].status = .error
            errorMessage = "Synthesis failed: \(error.localizedDescription)"
            showError = true
        }
    }

    // MARK: - Delete Item

    func deleteItem(_ item: LibraryItem) {
        // Clean up audio file
        if let audioPath = item.audioFileURL {
            try? FileManager.default.removeItem(atPath: audioPath)
        }
        items.removeAll { $0.id == item.id }
        saveLibrary()
    }

    func deleteItems(at offsets: IndexSet) {
        let filtered = filteredItems
        for index in offsets {
            deleteItem(filtered[index])
        }
    }
}
