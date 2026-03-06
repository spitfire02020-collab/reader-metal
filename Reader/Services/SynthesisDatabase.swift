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

final class SynthesisDatabase {
    static let shared = SynthesisDatabase()

    private var db: Connection?

    // Tables
    private let synthesisItems = Table("synthesis_items")
    private let chunks = Table("chunks")

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

    private init() {
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

        // Indexes
        try db?.run(chunks.createIndex(chItemId, ifNotExists: true))
        try db?.run(chunks.createIndex(chStatus, ifNotExists: true))
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
        guard let db else { return }

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

        // Create chunk rows
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

        NSLog("[SynthesisDB] Created item \(id) with \(totalChunks) chunks")
    }

    /// Get synthesis item by ID
    func getItem(id: String) throws -> SynthesisItemRow? {
        guard let db else { return nil }

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

    /// Get all synthesis items
    func getAllItems() throws -> [SynthesisItemRow] {
        guard let db else { return [] }

        let query = synthesisItems.order(siUpdatedAt.desc)
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
        guard let db else { return [] }

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
        guard let db else { return 0 }

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
}
