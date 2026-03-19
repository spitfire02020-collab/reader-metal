import Foundation
import SQLite

// MARK: - Synthesis Status Enums

enum SynthesisItemStatus: Int {
    case pending = 0
    case synthesizing = 1
    case completed = 2
    case paused = 3
    case cancelled = 4
    case error = 5
}

enum ChunkStatus: Int {
    case pending = 0
    case inProgress = 1
    case completed = 2
    case failed = 3
}

// MARK: - Synthesis Item Row

struct SynthesisItemRow {
    let id: String
    var title: String
    var totalChunks: Int
    var completedChunks: Int
    var failedChunks: Int
    var status: SynthesisItemStatus
    var voiceId: String?
    var seed: Int?
    var exaggeration: Float?
    var cfgWeight: Float?
    var speedFactor: Float?
    var createdAt: Date
    var updatedAt: Date
}

// MARK: - Chunk Row

struct ChunkRow {
    let id: Int64
    let itemId: String
    let chunkIndex: Int
    let textContent: String
    var status: ChunkStatus
    var filePath: String?
    var errorMessage: String?
    var retryCount: Int
    var durationSeconds: Double?
    var createdAt: Date
    var updatedAt: Date
}

// MARK: - Synthesis Database

@MainActor
final class SynthesisDatabase {
    static let shared = SynthesisDatabase()

    private var db: Connection?

    // Tables
    private let synthesisItems = Table("synthesis_items")
    private let chunks = Table("chunks")
    private let libraryItems = Table("library_items")
    private let libraryChapters = Table("library_chapters")
    private let customVoices = Table("custom_voices")

    // Synthesis Item columns
    private let siId = SQLite.Expression<String>("id")
    private let siTitle = SQLite.Expression<String>("title")
    private let siTotalChunks = SQLite.Expression<Int>("total_chunks")
    private let siCompletedChunks = SQLite.Expression<Int>("completed_chunks")
    private let siFailedChunks = SQLite.Expression<Int>("failed_chunks")
    private let siStatus = SQLite.Expression<Int>("status")
    private let siVoiceId = SQLite.Expression<String?>("voice_id")
    private let siSeed = SQLite.Expression<Int?>("seed")
    private let siExaggeration = SQLite.Expression<Double?>("exaggeration")
    private let siCfgWeight = SQLite.Expression<Double?>("cfg_weight")
    private let siSpeedFactor = SQLite.Expression<Double?>("speed_factor")
    private let siCreatedAt = SQLite.Expression<Double>("created_at")
    private let siUpdatedAt = SQLite.Expression<Double>("updated_at")

    // Chunk columns
    private let chId = SQLite.Expression<Int64>("id")
    private let chItemId = SQLite.Expression<String>("item_id")
    private let chChunkIndex = SQLite.Expression<Int>("chunk_index")
    private let chTextContent = SQLite.Expression<String>("text_content")
    private let chStatus = SQLite.Expression<Int>("status")
    private let chFilePath = SQLite.Expression<String?>("file_path")
    private let chErrorMessage = SQLite.Expression<String?>("error_message")
    private let chRetryCount = SQLite.Expression<Int>("retry_count")
    private let chDurationSeconds = SQLite.Expression<Double?>("duration_seconds")
    private let chCreatedAt = SQLite.Expression<Double>("created_at")
    private let chUpdatedAt = SQLite.Expression<Double>("updated_at")

    // Library Item columns
    private let liId = SQLite.Expression<String>("id")
    private let liTitle = SQLite.Expression<String>("title")
    private let liAuthor = SQLite.Expression<String?>("author")
    private let liSource = SQLite.Expression<String>("source")
    private let liSourceURL = SQLite.Expression<String?>("source_url")
    private let liCoverImage = SQLite.Expression<SQLite.Blob?>("cover_image")
    private let liDateAdded = SQLite.Expression<Double>("date_added")
    private let liLastPlayed = SQLite.Expression<Double?>("last_played")
    private let liDuration = SQLite.Expression<Double?>("duration")
    private let liCurrentPosition = SQLite.Expression<Double>("current_position")
    private let liTextContent = SQLite.Expression<String>("text_content")
    private let liSelectedVoiceId = SQLite.Expression<String?>("selected_voice_id")
    private let liItemStatus = SQLite.Expression<String>("status")
    private let liAudioFileURL = SQLite.Expression<String?>("audio_file_url")
    private let liProgress = SQLite.Expression<Double>("progress")
    private let liGeneratedChunks = SQLite.Expression<String?>("generated_chunks")

    // Library Chapter columns
    private let lcId = SQLite.Expression<String>("id")
    private let lcItemId = SQLite.Expression<String>("item_id")
    private let lcChapterIndex = SQLite.Expression<Int>("chapter_index")
    private let lcTitle = SQLite.Expression<String>("title")
    private let lcTextContent = SQLite.Expression<String>("text_content")
    private let lcStartTime = SQLite.Expression<Double?>("start_time")
    private let lcEndTime = SQLite.Expression<Double?>("end_time")
    private let lcAudioFileURL = SQLite.Expression<String?>("audio_file_url")

    // Custom Voice columns
    private let cvId = SQLite.Expression<String>("id")
    private let cvName = SQLite.Expression<String>("name")
    private let cvData = SQLite.Expression<SQLite.Blob>("data")

    init() {
        do {
            let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            let dbPath = documentsPath.appendingPathComponent("synthesis.sqlite3")
            db = try Connection(dbPath.path)
            try createTables()
            NSLog("[SynthesisDB] Database initialized at: \(dbPath.path)")
        } catch {
            NSLog("[SynthesisDB] Failed to initialize database: \(error)")
        }
    }

    private func createTables() throws {
        // Synthesis items table
        try db?.run(synthesisItems.create(ifNotExists: true) { t in
            t.column(siId, primaryKey: true)
            t.column(siTitle)
            t.column(siTotalChunks, defaultValue: 0)
            t.column(siCompletedChunks, defaultValue: 0)
            t.column(siFailedChunks, defaultValue: 0)
            t.column(siStatus, defaultValue: 0)
            t.column(siVoiceId)
            t.column(siSeed)
            t.column(siExaggeration)
            t.column(siCfgWeight)
            t.column(siSpeedFactor)
            t.column(siCreatedAt)
            t.column(siUpdatedAt)
        })

        // Chunks table
        try db?.run(chunks.create(ifNotExists: true) { t in
            t.column(chId, primaryKey: .autoincrement)
            t.column(chItemId)
            t.column(chChunkIndex)
            t.column(chTextContent)
            t.column(chStatus, defaultValue: 0)
            t.column(chFilePath)
            t.column(chErrorMessage)
            t.column(chRetryCount, defaultValue: 0)
            t.column(chDurationSeconds)
            t.column(chCreatedAt)
            t.column(chUpdatedAt)
            t.unique(chItemId, chChunkIndex)
            t.foreignKey(chItemId, references: synthesisItems, siId, delete: .cascade)
        })

        // Indexes - single column
        try db?.run(chunks.createIndex(chItemId, ifNotExists: true))
        try db?.run(chunks.createIndex(chStatus, ifNotExists: true))

        // Composite indexes for common query patterns
        try db?.run(chunks.createIndex(chItemId, chChunkIndex, ifNotExists: true))
        try db?.run(chunks.createIndex(chItemId, chStatus, ifNotExists: true))

        // Library items table
        try db?.run(libraryItems.create(ifNotExists: true) { t in
            t.column(liId, primaryKey: true)
            t.column(liTitle)
            t.column(liAuthor)
            t.column(liSource)
            t.column(liSourceURL)
            t.column(liCoverImage)
            t.column(liDateAdded)
            t.column(liLastPlayed)
            t.column(liDuration)
            t.column(liCurrentPosition, defaultValue: 0)
            t.column(liTextContent)
            t.column(liSelectedVoiceId)
            t.column(liItemStatus, defaultValue: "pending")
            t.column(liAudioFileURL)
            t.column(liProgress, defaultValue: 0)
            t.column(liGeneratedChunks)
        })

        // Library chapters table
        try db?.run(libraryChapters.create(ifNotExists: true) { t in
            t.column(lcId, primaryKey: true)
            t.column(lcItemId)
            t.column(lcChapterIndex)
            t.column(lcTitle)
            t.column(lcTextContent)
            t.column(lcStartTime)
            t.column(lcEndTime)
            t.column(lcAudioFileURL)
            t.unique(lcItemId, lcChapterIndex)
        })

        // Custom voices table
        try db?.run(customVoices.create(ifNotExists: true) { t in
            t.column(cvId, primaryKey: true)
            t.column(cvName)
            t.column(cvData)
        })

        // Library indexes
        try db?.run(libraryItems.createIndex(liDateAdded, ifNotExists: true))
        try db?.run(libraryItems.createIndex(liItemStatus, ifNotExists: true))
        try db?.run(libraryChapters.createIndex(lcItemId, ifNotExists: true))
    }

    // MARK: - Synthesis Item Operations

    /// Create a new synthesis item with chunks from text
    func createItem(
        id: String,
        title: String,
        text: String,
        voiceId: String?,
        settings: SynthesisSettings?
    ) throws {
        guard let db else {
            NSLog("[SynthesisDB] ERROR: Database not initialized in createItem")
            return
        }

        let now = Date().timeIntervalSince1970
        let chunkTexts = TextChunker.chunkText(text)
        let totalChunks = chunkTexts.count

        try db.run(synthesisItems.insert(
            siId <- id,
            siTitle <- title,
            siTotalChunks <- totalChunks,
            siCompletedChunks <- 0,
            siFailedChunks <- 0,
            siStatus <- SynthesisItemStatus.pending.rawValue,
            siVoiceId <- voiceId,
            siSeed <- settings?.seed,
            siExaggeration <- settings.map { Double($0.exaggeration) },
            siCfgWeight <- settings.map { Double($0.cfgWeight) },
            siSpeedFactor <- settings.map { Double($0.speed) },
            siCreatedAt <- now,
            siUpdatedAt <- now
        ))

        // Create chunk rows in a single transaction for performance
        try db.transaction {
            for (index, textContent) in chunkTexts.enumerated() {
                try db.run(chunks.insert(
                    chItemId <- id,
                    chChunkIndex <- index,
                    chTextContent <- textContent,
                    chStatus <- ChunkStatus.pending.rawValue,
                    chRetryCount <- 0,
                    chCreatedAt <- now,
                    chUpdatedAt <- now
                ))
            }
        }

        NSLog("[SynthesisDB] Created item \(id) with \(totalChunks) chunks")
    }

    /// Get synthesis item by ID
    func getItem(id: String) throws -> SynthesisItemRow? {
        guard let db else {
            NSLog("[SynthesisDB] ERROR: Database not initialized in getItem")
            return nil
        }

        let query = synthesisItems.filter(siId == id)
        guard let row = try db.pluck(query) else { return nil }

        return SynthesisItemRow(
            id: row[siId],
            title: row[siTitle],
            totalChunks: row[siTotalChunks],
            completedChunks: row[siCompletedChunks],
            failedChunks: row[siFailedChunks],
            status: SynthesisItemStatus(rawValue: row[siStatus]) ?? .pending,
            voiceId: row[siVoiceId],
            seed: row[siSeed],
            exaggeration: row[siExaggeration].map { Float($0) },
            cfgWeight: row[siCfgWeight].map { Float($0) },
            speedFactor: row[siSpeedFactor].map { Float($0) },
            createdAt: Date(timeIntervalSince1970: row[siCreatedAt]),
            updatedAt: Date(timeIntervalSince1970: row[siUpdatedAt])
        )
    }

    /// Get all synthesis items with optional pagination
    /// - Parameters:
    ///   - limit: Maximum number of items to return (default 50)
    ///   - offset: Number of items to skip (for pagination)
    /// - Returns: Array of synthesis items
    func getAllItems(limit: Int = 50, offset: Int = 0) throws -> [SynthesisItemRow] {
        guard let db else { return [] }

        let query = synthesisItems
            .order(siUpdatedAt.desc)
            .limit(limit, offset: offset)
        return try db.prepare(query).map { row in
            SynthesisItemRow(
                id: row[siId],
                title: row[siTitle],
                totalChunks: row[siTotalChunks],
                completedChunks: row[siCompletedChunks],
                failedChunks: row[siFailedChunks],
                status: SynthesisItemStatus(rawValue: row[siStatus]) ?? .pending,
                voiceId: row[siVoiceId],
                seed: row[siSeed],
                exaggeration: row[siExaggeration].map { Float($0) },
                cfgWeight: row[siCfgWeight].map { Float($0) },
                speedFactor: row[siSpeedFactor].map { Float($0) },
                createdAt: Date(timeIntervalSince1970: row[siCreatedAt]),
                updatedAt: Date(timeIntervalSince1970: row[siUpdatedAt])
            )
        }
    }

    /// Get total count of synthesis items (useful for pagination)
    func getItemCount() throws -> Int {
        guard let db else { return 0 }
        return try db.scalar(synthesisItems.count)
    }

    /// Update synthesis item status
    func updateItemStatus(id: String, status: SynthesisItemStatus) throws {
        let now = Date().timeIntervalSince1970
        let query = synthesisItems.filter(siId == id)
        try db?.run(query.update(
            siStatus <- status.rawValue,
            siUpdatedAt <- now
        ))
        NSLog("[SynthesisDB] Updated item \(id) status to \(status)")
    }

    /// Update synthesis item progress
    func updateItemProgress(id: String) throws {
        guard let db else { return }

        let now = Date().timeIntervalSince1970
        let query = synthesisItems.filter(siId == id)

        // Count completed and failed chunks
        let completedCount = try db.scalar(chunks.filter(chItemId == id && chStatus == ChunkStatus.completed.rawValue).count)
        let failedCount = try db.scalar(chunks.filter(chItemId == id && chStatus == ChunkStatus.failed.rawValue).count)

        try db.run(query.update(
            siCompletedChunks <- completedCount,
            siFailedChunks <- failedCount,
            siUpdatedAt <- now
        ))
    }

    /// Update synthesis item voice
    func updateItemVoice(id: String, voiceId: String) throws {
        let now = Date().timeIntervalSince1970
        let query = synthesisItems.filter(siId == id)
        try db?.run(query.update(
            siVoiceId <- voiceId,
            siUpdatedAt <- now
        ))
        NSLog("[SynthesisDB] Updated item \(id) voice to \(voiceId)")
    }

    /// Reset chunks from a certain index onwards (for re-synthesis with new voice)
    func resetChunksFromIndex(itemId: String, fromIndex: Int) throws {
        guard let db else { return }

        let now = Date().timeIntervalSince1970
        let query = chunks.filter(chItemId == itemId && chChunkIndex >= fromIndex)

        // Reset status to pending and clear file path
        try db.run(query.update(
            chStatus <- ChunkStatus.pending.rawValue,
            chFilePath <- nil as String?,
            chErrorMessage <- nil as String?,
            chRetryCount <- 0,
            chDurationSeconds <- nil as Double?,
            chUpdatedAt <- now
        ))
        NSLog("[SynthesisDB] Reset chunks from index \(fromIndex) for item \(itemId)")
    }

    /// Delete synthesis item and all chunks
    func deleteItem(id: String) throws {
        let query = synthesisItems.filter(siId == id)
        try db?.run(query.delete())
        NSLog("[SynthesisDB] Deleted item \(id)")
    }

    // MARK: - Chunk Operations

    /// Get next pending chunk for synthesis (includes failed chunks with retries left)
    func getNextPendingChunk(itemId: String) throws -> ChunkRow? {
        guard let db else { return nil }

        let query = chunks
            .filter(chItemId == itemId)
            .filter(chStatus == ChunkStatus.pending.rawValue || chStatus == ChunkStatus.failed.rawValue)
            .filter(chRetryCount < 3)
            .order(chChunkIndex.asc)
            .limit(1)

        guard let row = try db.pluck(query) else { return nil }

        return ChunkRow(
            id: row[chId],
            itemId: row[chItemId],
            chunkIndex: row[chChunkIndex],
            textContent: row[chTextContent],
            status: ChunkStatus(rawValue: row[chStatus]) ?? .pending,
            filePath: row[chFilePath],
            errorMessage: row[chErrorMessage],
            retryCount: row[chRetryCount],
            durationSeconds: row[chDurationSeconds],
            createdAt: Date(timeIntervalSince1970: row[chCreatedAt]),
            updatedAt: Date(timeIntervalSince1970: row[chUpdatedAt])
        )
    }

    /// Get all completed chunks for an item (in order)
    func getCompletedChunks(itemId: String) throws -> [ChunkRow] {
        guard let db else {
            NSLog("[SynthesisDB] ERROR: Database not initialized in getCompletedChunks for item: \(itemId)")
            return []
        }

        let query = chunks
            .filter(chItemId == itemId)
            .filter(chStatus == ChunkStatus.completed.rawValue)
            .order(chChunkIndex.asc)

        return try db.prepare(query).map { row in
            ChunkRow(
                id: row[chId],
                itemId: row[chItemId],
                chunkIndex: row[chChunkIndex],
                textContent: row[chTextContent],
                status: .completed,
                filePath: row[chFilePath],
                errorMessage: nil,
                retryCount: row[chRetryCount],
                durationSeconds: row[chDurationSeconds],
                createdAt: Date(timeIntervalSince1970: row[chCreatedAt]),
                updatedAt: Date(timeIntervalSince1970: row[chUpdatedAt])
            )
        }
    }

    /// Mark chunk as in progress
    func markChunkInProgress(id: Int64) throws {
        let now = Date().timeIntervalSince1970
        let query = chunks.filter(chId == id)
        try db?.run(query.update(
            chStatus <- ChunkStatus.inProgress.rawValue,
            chUpdatedAt <- now
        ))
    }

    /// Get chunk ID by item ID and chunk index
    func getChunkId(itemId: String, chunkIndex: Int) throws -> Int64? {
        guard let db else { return nil }

        let query = chunks
            .filter(chItemId == itemId)
            .filter(chChunkIndex == chunkIndex)

        guard let row = try db.pluck(query) else { return nil }
        return row[chId]
    }

    /// Mark chunk as completed with file path (by item ID and chunk index)
    func markChunkCompleted(itemId: String, chunkIndex: Int, filePath: String, duration: Double?) throws {
        guard let chunkId = try getChunkId(itemId: itemId, chunkIndex: chunkIndex) else {
            NSLog("[SynthesisDB] Warning: Could not find chunk for item \(itemId) index \(chunkIndex)")
            return
        }
        try markChunkCompleted(id: chunkId, filePath: filePath, duration: duration)
    }

    /// Mark chunk as completed with file path
    func markChunkCompleted(id: Int64, filePath: String, duration: Double?) throws {
        let now = Date().timeIntervalSince1970
        let query = chunks.filter(chId == id)
        try db?.run(query.update(
            chStatus <- ChunkStatus.completed.rawValue,
            chFilePath <- filePath,
            chDurationSeconds <- duration,
            chUpdatedAt <- now
        ))
        NSLog("[SynthesisDB] Chunk \(id) marked completed: \(filePath)")
    }

    /// Mark chunk as failed with error (by item ID and chunk index)
    func markChunkFailed(itemId: String, chunkIndex: Int, error: String) throws {
        guard let db else { return }
        let query = chunks.filter(chItemId == itemId && chChunkIndex == chunkIndex)
        guard let row = try db.pluck(query) else { return }
        let chunkId = row[chId]
        try markChunkFailed(id: chunkId, error: error)
    }

    /// Mark chunk as failed with error
    func markChunkFailed(id: Int64, error: String) throws {
        guard let db else { return }

        let now = Date().timeIntervalSince1970

        // First get current retry count
        let query = chunks.filter(chId == id)
        guard let row = try db.pluck(query) else { return }
        let currentRetry = row[chRetryCount]

        try db.run(query.update(
            chStatus <- ChunkStatus.failed.rawValue,
            chErrorMessage <- error,
            chRetryCount <- currentRetry + 1,
            chUpdatedAt <- now
        ))
        NSLog("[SynthesisDB] Chunk \(id) marked failed (retry \(currentRetry + 1)): \(error)")
    }

    /// Get overall progress for an item (0.0 - 1.0)
    func getProgress(itemId: String) throws -> Double {
        guard let db else {
            NSLog("[SynthesisDB] ERROR: Database not initialized in getProgress for item: \(itemId)")
            return 0
        }

        let total = try db.scalar(chunks.filter(chItemId == itemId).count)
        guard total > 0 else { return 0 }

        let completed = try db.scalar(chunks.filter(chItemId == itemId && chStatus == ChunkStatus.completed.rawValue).count)
        return Double(completed) / Double(total)
    }

    /// Get pending/in_progress chunk count
    func getRemainingChunks(itemId: String) throws -> Int {
        guard let db else { return 0 }

        return try db.scalar(chunks
            .filter(chItemId == itemId)
            .filter(chStatus == ChunkStatus.pending.rawValue || chStatus == ChunkStatus.inProgress.rawValue)
            .count)
    }

    /// Check if item has any failed chunks (with retries exhausted)
    func hasExhaustedRetries(itemId: String) throws -> Bool {
        guard let db else { return false }

        let count = try db.scalar(chunks
            .filter(chItemId == itemId && chStatus == ChunkStatus.failed.rawValue)
            .filter(chRetryCount >= 3)
            .count)

        return count > 0
    }

    /// Get count of failed chunks with retries left
    func getFailedChunksCount(itemId: String) throws -> Int {
        guard let db else { return 0 }

        return try db.scalar(chunks
            .filter(chItemId == itemId && chStatus == ChunkStatus.failed.rawValue)
            .filter(chRetryCount < 3)
            .count)
    }

    // MARK: - Library Item Operations

    /// Insert a new library item with its chapters
    func insertLibraryItem(_ item: LibraryItem) throws {
        guard let db else { return }

        // Encode generatedChunks as JSON
        let chunksJSON: String?
        if !item.generatedChunks.isEmpty {
            let data = try JSONEncoder().encode(item.generatedChunks)
            chunksJSON = String(data: data, encoding: .utf8)
        } else {
            chunksJSON = nil
        }

        try db.transaction {
            try db.run(libraryItems.insert(or: .replace,
                liId <- item.id.uuidString,
                liTitle <- item.title,
                liAuthor <- item.author,
                liSource <- item.source.rawValue,
                liSourceURL <- item.sourceURL,
                liCoverImage <- item.coverImageData.map { SQLite.Blob(bytes: [UInt8]($0)) },
                liDateAdded <- item.dateAdded.timeIntervalSince1970,
                liLastPlayed <- item.lastPlayed?.timeIntervalSince1970,
                liDuration <- item.duration,
                liCurrentPosition <- item.currentPosition,
                liTextContent <- item.textContent,
                liSelectedVoiceId <- item.selectedVoiceID,
                liItemStatus <- item.status.rawValue,
                liAudioFileURL <- item.audioFileURL,
                liProgress <- item.progress,
                liGeneratedChunks <- chunksJSON
            ))

            // Delete existing chapters for this item
            try db.run(libraryChapters.filter(lcItemId == item.id.uuidString).delete())

            // Insert chapters
            for (index, chapter) in item.chapters.enumerated() {
                try db.run(libraryChapters.insert(
                    lcId <- chapter.id.uuidString,
                    lcItemId <- item.id.uuidString,
                    lcChapterIndex <- index,
                    lcTitle <- chapter.title,
                    lcTextContent <- chapter.textContent,
                    lcStartTime <- chapter.startTime,
                    lcEndTime <- chapter.endTime,
                    lcAudioFileURL <- chapter.audioFileURL
                ))
            }
        }
    }

    /// Get all library items with chapters
    func getAllLibraryItems() throws -> [LibraryItem] {
        guard let db else { return [] }

        var items: [LibraryItem] = []
        for row in try db.prepare(libraryItems.order(liDateAdded.desc)) {
            let itemId = row[liId]

            // Load chapters
            let chapterRows = try db.prepare(
                libraryChapters.filter(lcItemId == itemId).order(lcChapterIndex.asc)
            )
            let chapters = chapterRows.map { ch in
                Chapter(
                    id: UUID(uuidString: ch[lcId]) ?? UUID(),
                    title: ch[lcTitle],
                    textContent: ch[lcTextContent],
                    startTime: ch[lcStartTime],
                    endTime: ch[lcEndTime],
                    audioFileURL: ch[lcAudioFileURL]
                )
            }

            // Decode generatedChunks
            var generatedChunks: [Int: String] = [:]
            if let json = row[liGeneratedChunks],
               let data = json.data(using: .utf8) {
                generatedChunks = (try? JSONDecoder().decode([Int: String].self, from: data)) ?? [:]
            }

            let item = LibraryItem(
                id: UUID(uuidString: itemId) ?? UUID(),
                title: row[liTitle],
                author: row[liAuthor],
                source: ContentSource(rawValue: row[liSource]) ?? .text,
                sourceURL: row[liSourceURL],
                coverImageData: row[liCoverImage].map { Data($0.bytes) },
                dateAdded: Date(timeIntervalSince1970: row[liDateAdded]),
                lastPlayed: row[liLastPlayed].map { Date(timeIntervalSince1970: $0) },
                duration: row[liDuration],
                currentPosition: row[liCurrentPosition],
                textContent: row[liTextContent],
                chapters: chapters,
                selectedVoiceID: row[liSelectedVoiceId],
                status: ItemStatus(rawValue: row[liItemStatus]) ?? .pending,
                audioFileURL: row[liAudioFileURL],
                progress: row[liProgress],
                generatedChunks: generatedChunks
            )
            items.append(item)
        }
        return items
    }

    /// Get a single library item by ID
    func getLibraryItem(id: String) throws -> LibraryItem? {
        guard let db else { return nil }

        guard let row = try db.pluck(libraryItems.filter(liId == id)) else { return nil }
        let itemId = row[liId]

        let chapterRows = try db.prepare(
            libraryChapters.filter(lcItemId == itemId).order(lcChapterIndex.asc)
        )
        let chapters = chapterRows.map { ch in
            Chapter(
                id: UUID(uuidString: ch[lcId]) ?? UUID(),
                title: ch[lcTitle],
                textContent: ch[lcTextContent],
                startTime: ch[lcStartTime],
                endTime: ch[lcEndTime],
                audioFileURL: ch[lcAudioFileURL]
            )
        }

        var generatedChunks: [Int: String] = [:]
        if let json = row[liGeneratedChunks],
           let data = json.data(using: .utf8) {
            generatedChunks = (try? JSONDecoder().decode([Int: String].self, from: data)) ?? [:]
        }

        return LibraryItem(
            id: UUID(uuidString: itemId) ?? UUID(),
            title: row[liTitle],
            author: row[liAuthor],
            source: ContentSource(rawValue: row[liSource]) ?? .text,
            sourceURL: row[liSourceURL],
            coverImageData: row[liCoverImage].map { Data($0.bytes) },
            dateAdded: Date(timeIntervalSince1970: row[liDateAdded]),
            lastPlayed: row[liLastPlayed].map { Date(timeIntervalSince1970: $0) },
            duration: row[liDuration],
            currentPosition: row[liCurrentPosition],
            textContent: row[liTextContent],
            chapters: chapters,
            selectedVoiceID: row[liSelectedVoiceId],
            status: ItemStatus(rawValue: row[liItemStatus]) ?? .pending,
            audioFileURL: row[liAudioFileURL],
            progress: row[liProgress],
            generatedChunks: generatedChunks
        )
    }

    /// Update a library item (upsert)
    func updateLibraryItem(_ item: LibraryItem) throws {
        try insertLibraryItem(item) // uses INSERT OR REPLACE
    }

    /// Delete a library item and its chapters
    func deleteLibraryItem(id: String) throws {
        guard let db else { return }
        try db.transaction {
            try db.run(libraryChapters.filter(lcItemId == id).delete())
            try db.run(libraryItems.filter(liId == id).delete())
        }
        NSLog("[SynthesisDB] Deleted library item \(id)")
    }

    /// Update just the status of a library item
    func updateLibraryItemStatus(id: String, status: ItemStatus) throws {
        try db?.run(libraryItems.filter(liId == id).update(
            liItemStatus <- status.rawValue
        ))
    }

    /// Update just the progress of a library item
    func updateLibraryItemProgress(id: String, progress: Double) throws {
        try db?.run(libraryItems.filter(liId == id).update(
            liProgress <- progress
        ))
    }

    /// Bulk migrate library items from legacy storage
    func migrateLibraryItems(_ items: [LibraryItem]) throws {
        guard let db else { return }
        try db.transaction {
            for item in items {
                // Skip if already exists
                let exists = try db.scalar(libraryItems.filter(liId == item.id.uuidString).count) > 0
                if !exists {
                    try insertLibraryItem(item)
                }
            }
        }
        NSLog("[SynthesisDB] Migrated \(items.count) library items to SQLite")
    }

    /// Check if library_items table has any data
    func hasLibraryItems() throws -> Bool {
        guard let db else { return false }
        return try db.scalar(libraryItems.count) > 0
    }

    // MARK: - Custom Voice Operations

    /// Insert or replace a custom voice (stores VoiceProfile as JSON blob)
    func insertCustomVoice(_ voice: VoiceProfile) throws {
        guard let db else { return }
        let data = try JSONEncoder().encode(voice)
        try db.run(customVoices.insert(or: .replace,
            cvId <- voice.id,
            cvName <- voice.name,
            cvData <- SQLite.Blob(bytes: [UInt8](data))
        ))
    }

    /// Get all custom voices
    func getAllCustomVoices() throws -> [VoiceProfile] {
        guard let db else { return [] }
        var voices: [VoiceProfile] = []
        for row in try db.prepare(customVoices) {
            let data = Data(row[cvData].bytes)
            if let voice = try? JSONDecoder().decode(VoiceProfile.self, from: data) {
                voices.append(voice)
            }
        }
        return voices
    }

    /// Delete a custom voice
    func deleteCustomVoice(id: String) throws {
        try db?.run(customVoices.filter(cvId == id).delete())
    }

    /// Migrate custom voices from UserDefaults
    func migrateCustomVoices() throws {
        guard let db else { return }
        let key = "custom_voices"
        guard let data = UserDefaults.standard.data(forKey: key),
              let voices = try? JSONDecoder().decode([VoiceProfile].self, from: data) else {
            return
        }
        try db.transaction {
            for voice in voices {
                let exists = try db.scalar(customVoices.filter(cvId == voice.id).count) > 0
                if !exists {
                    try insertCustomVoice(voice)
                }
            }
        }
        UserDefaults.standard.removeObject(forKey: key)
        NSLog("[SynthesisDB] Migrated \(voices.count) custom voices to SQLite")
    }
}
