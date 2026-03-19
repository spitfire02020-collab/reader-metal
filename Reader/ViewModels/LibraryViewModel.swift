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

    private let synthesisDB = SynthesisDatabase.shared

    init() {
        loadLibrary()
    }

    private let extractor = WebContentExtractor()
    private let engine = ChatterboxEngine.shared

    // Legacy file storage path (for migration only)
    private var legacyLibraryFileURL: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        try? FileManager.default.createDirectory(at: appSupport, withIntermediateDirectories: true)
        return appSupport.appendingPathComponent("library_protected.json")
    }

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

    // MARK: - Persistence (SQLite)

    /// Load library items from SQLite database
    func loadLibrary() {
        do {
            // Check if SQLite has data; if not, try migrating from legacy storage
            let hasData = try synthesisDB.hasLibraryItems()
            if !hasData {
                migrateFromLegacyStorage()
            }

            items = try synthesisDB.getAllLibraryItems()

            // Sanitize HTML entities and reset stale processing status
            var needsUpdate = false
            for i in items.indices {
                if items[i].title.contains("&") {
                    items[i].title = items[i].title.decodedHTMLEntities
                    needsUpdate = true
                }
                if items[i].status == .processing {
                    items[i].status = .pending
                    needsUpdate = true
                }
            }
            if needsUpdate {
                for item in items {
                    try? synthesisDB.updateLibraryItem(item)
                }
            }
        } catch {
            NSLog("[LibraryViewModel] Failed to load from SQLite: \(error)")
            migrateFromLegacyStorage()
            items = (try? synthesisDB.getAllLibraryItems()) ?? []
        }
    }

    /// Save a single library item to SQLite
    func saveItem(_ item: LibraryItem) {
        do {
            try synthesisDB.updateLibraryItem(item)
        } catch {
            NSLog("[LibraryViewModel] Failed to save item: \(error)")
        }
    }

    /// Save the full library (for bulk operations like reorder)
    func saveLibrary() {
        for item in items {
            saveItem(item)
        }
    }

    /// Migrate from legacy JSON file and/or UserDefaults
    private func migrateFromLegacyStorage() {
        var legacyItems: [LibraryItem] = []

        // 1. Try JSON file (library_protected.json)
        if let data = try? Data(contentsOf: legacyLibraryFileURL),
           let decoded = try? JSONDecoder().decode([LibraryItem].self, from: data) {
            legacyItems = decoded
            NSLog("[LibraryViewModel] Found \(decoded.count) items in legacy JSON file")
        }

        // 2. Try UserDefaults (oldest legacy path)
        if legacyItems.isEmpty {
            let legacyKey = "library_items"
            if let data = UserDefaults.standard.data(forKey: legacyKey),
               let decoded = try? JSONDecoder().decode([LibraryItem].self, from: data) {
                legacyItems = decoded
                NSLog("[LibraryViewModel] Found \(decoded.count) items in UserDefaults")
                UserDefaults.standard.removeObject(forKey: legacyKey)
            }
        }

        // 3. Migrate to SQLite
        if !legacyItems.isEmpty {
            do {
                try synthesisDB.migrateLibraryItems(legacyItems)
                // Clean up legacy JSON file after successful migration
                try? FileManager.default.removeItem(at: legacyLibraryFileURL)
                NSLog("[LibraryViewModel] Migration complete: \(legacyItems.count) items → SQLite")
            } catch {
                NSLog("[LibraryViewModel] Migration failed: \(error)")
            }
        }
    }

    // MARK: - Add Content from URL

    func addFromURL(_ urlString: String) async {
        let trimmed = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: trimmed),
              let scheme = url.scheme?.lowercased(),
              (scheme == "http" || scheme == "https") else {
            errorMessage = "Only HTTP and HTTPS URLs are allowed."
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

            try synthesisDB.insertLibraryItem(item)
            items.insert(item, at: 0)
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

            try synthesisDB.insertLibraryItem(item)
            items.insert(item, at: 0)
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

        do {
            try synthesisDB.insertLibraryItem(item)
        } catch {
            NSLog("[LibraryViewModel] Failed to insert item: \(error)")
        }
        items.insert(item, at: 0)
        showAddContent = false
    }

    // MARK: - Synthesize Item

    func synthesize(item: LibraryItem) async {
        guard let index = items.firstIndex(where: { $0.id == item.id }) else { return }
        items[index].status = .processing
        saveItem(items[index])

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
                            self.items[idx].progress = progress * 0.5
                        }
                    }
                }
            )

            items[index].audioFileURL = outputURL.path
            items[index].status = .ready
            saveItem(items[index])
        } catch {
            items[index].status = .error
            saveItem(items[index])
            errorMessage = "Synthesis failed: \(error.localizedDescription)"
            showError = true
        }
    }

    // MARK: - Delete Item

    func deleteItem(_ item: LibraryItem) {
        if let audioPath = item.audioFileURL {
            try? FileManager.default.removeItem(atPath: audioPath)
        }
        do {
            try synthesisDB.deleteLibraryItem(id: item.id.uuidString)
        } catch {
            NSLog("[LibraryViewModel] Failed to delete item: \(error)")
        }
        items.removeAll { $0.id == item.id }
    }

    func deleteItems(at offsets: IndexSet) {
        let filtered = filteredItems
        for index in offsets {
            deleteItem(filtered[index])
        }
    }
}
