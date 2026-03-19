import Foundation
import AVFoundation

// MARK: - Audio Player Protocol

@MainActor
protocol AudioPlayerProtocol: AnyObject {
    var isPlaying: Bool { get }
    var isPaused: Bool { get }
    nonisolated var currentTime: TimeInterval { get }
    nonisolated var duration: TimeInterval { get }
    var currentChunkIndex: Int { get }
    var hasAudioFiles: Bool { get }

    func loadAudio(url: URL, title: String, artist: String?)
    func play()
    func pause()
    func stop()
    func seek(to time: TimeInterval)
    func playNext()
    func playPrevious()
}

// MARK: - Synthesis Engine Protocol

protocol SynthesisEngineProtocol: AnyObject {
    var isLoaded: Bool { get }

    func loadModels() async throws
    func synthesize(
        text: String,
        preChunkedText: [String]?,
        referenceAudioURL: URL?,
        outputURL: URL,
        onChunkReady: @escaping (URL) -> Void,
        onProgress: ((Double) -> Void)?
    ) async throws

    var onChunkComplete: ((URL) -> Void)? { get set }
}

// MARK: - Model Download Protocol

protocol ModelDownloadProtocol: AnyObject {
    var isModelReady: Bool { get }
    var downloadProgress: Double { get }

    func checkModelAvailability() async
    func downloadModels(progress: @escaping (Double) -> Void) async throws
}

// MARK: - Database Protocol

protocol DatabaseProtocol: AnyObject {
    func createItem(id: String, title: String, text: String, voiceId: String, settings: [String: Any]) throws
    func getItem(id: String) -> SynthesisItem?
    func updateItemStatus(id: String, status: SynthesisItemStatus) throws
    func addCompletedChunk(itemId: String, chunkIndex: Int, filePath: String) throws
    func getCompletedChunks(itemId: String) -> [SynthesisChunk]
    func deleteItem(id: String) throws
}

// MARK: - Text Processing Protocol

protocol TextProcessingProtocol: AnyObject {
    func chunkText(_ text: String) -> [String]
    func estimateDuration(for text: String, wordsPerMinute: Double) -> TimeInterval
}
