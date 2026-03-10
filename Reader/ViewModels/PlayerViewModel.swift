import Foundation
import SwiftUI
import Combine

// MARK: - Player View Model

@MainActor
final class PlayerViewModel: ObservableObject {
    @Published var item: LibraryItem
    @Published var selectedVoice: VoiceProfile = .defaultVoice
    @Published var synthesisSettings = SynthesisSettings.defaults
    @Published var showChapterList = false
    @Published var showVoiceSelector = false
    @Published var showSettings = false
    @Published var isSynthesizing = false
    @Published var synthesisProgress: Double = 0
    @Published var errorMessage: String?
    @Published var showModelDownload = false
    @Published var isDownloadingModels = false
    /// True while streaming playback is active (first chunk has started but synthesis isn't done).
    @Published var isStreamingAudio = false
    /// Reference to current synthesis task - used for cancellation
    private var synthesisTask: Task<Void, Never>? = nil
    /// True if generation is paused
    @Published var isPaused = false
    /// Selected chunk indices for batch selection
    @Published var selectedChunks: Set<Int> = []
    /// Current chunk being played (for highlighting)
    @Published var currentPlaybackChunkIndex: Int = -1
    /// Cached paragraphs for display (computed once to avoid re-parsing on every render)
    @Published private(set) var cachedParagraphs: [String] = []
    /// Cached sentence splits for each paragraph (computed once)
    @Published private(set) var cachedParagraphSentences: [[String]] = []
    /// Cached paragraph index to sentence start index mapping
    @Published private(set) var sentencesIndicesStart: [Int: Int] = [:]
    /// Fast O(1) lookup map: text -> index for updatePlayingIndex
    private var textToIndexMap: [String: Int] = [:]
    /// Flattened sentences for efficient display with index-based highlighting
    /// Each item: (sentenceText, globalSentenceIndex, paragraphIndex)
    @Published private(set) var flattenedSentences: [(text: String, index: Int, paragraphIndex: Int)] = []
    /// Current playing sentence index for highlighting (tracks which sentence is playing)
    @Published var currentPlayingIndex: Int = -1
    /// Cached attributed strings for paragraphs (with highlighting) - rebuilt when highlight changes
    @Published private(set) var cachedAttributedParagraphs: [AttributedString] = []

    let audioPlayer: AudioPlayerService
    private let engine: ChatterboxEngine
    private let downloadService: ModelDownloadService
    private let synthesisDB: SynthesisDatabase

    /// Total duration - always use estimated full duration, not cumulative chunk duration
    var totalDuration: TimeInterval {
        // Always use estimated duration from full text, not the cumulative chunk duration
        // The audioPlayer.duration only contains chunks that have been generated so far
        return item.duration ?? TextChunker.estimateTotalDuration(for: item.textContent)
    }

    /// Formatted total duration string
    var formattedTotalDuration: String {
        formatTime(totalDuration)
    }

    /// Text to display in player - shows "current / total" when playing
    var totalDurationText: String {
        if audioPlayer.duration > 0 && audioPlayer.isPlaying {
            // Show current position / total when playing
            return "\(audioPlayer.formattedCurrentTime) / \(formattedTotalDuration)"
        }
        return formattedTotalDuration
    }

    /// Playback progress based on total estimated duration (not just loaded chunks)
    var playbackProgress: Double {
        if isSynthesizing {
            // Show synthesis progress when generating
            return synthesisProgress
        }
        // Calculate progress based on total estimated duration
        // Need to account for: current time in current chunk + duration of all completed chunks
        let totalEst = totalDuration
        guard totalEst > 0 else { return 0 }

        // Get the total played duration (current chunk time + all previous chunks)
        let currentTimeInChunk = audioPlayer.currentTime
        let currentChunkIdx = audioPlayer.currentChunkIndex

        // Calculate completed chunks duration (approximate based on average chunk duration)
        let avgChunkDuration = totalEst / Double(textChunks.count)
        let completedChunksTime = Double(currentChunkIdx) * avgChunkDuration

        let totalPlayedTime = currentTimeInChunk + completedChunksTime
        return min(1.0, totalPlayedTime / totalEst)
    }

    private func formatTime(_ time: TimeInterval) -> String {
        let totalSeconds = Int(time)
        let hours = totalSeconds / 3600
        let minutes = (totalSeconds % 3600) / 60
        let seconds = totalSeconds % 60

        if hours > 0 {
            return String(format: "%d:%02d:%02d", hours, minutes, seconds)
        }
        return String(format: "%d:%02d", minutes, seconds)
    }

    // MARK: - Chunk Persistence

    /// Get the directory for storing chunks for this item
    private var chunkDirectory: URL {
        let docsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docsDir.appendingPathComponent("Audio/\(item.id.uuidString)", isDirectory: true)
    }

    /// Get the URL for a specific chunk
    func chunkURL(for index: Int) -> URL {
        return chunkDirectory.appendingPathComponent("chunk_\(index).wav")
    }

    /// Load existing chunks from disk
    func loadExistingChunks() {
        let dir = chunkDirectory
        let audioBaseDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!.appendingPathComponent("Audio")
        NSLog("[PlayerVM] loadExistingChunks: item=\(item.id.uuidString)")
        NSLog("[PlayerVM] loadExistingChunks: docDir=\(FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!.path)")
        NSLog("[PlayerVM] loadExistingChunks: audioBaseDir exists=\(FileManager.default.fileExists(atPath: audioBaseDir.path))")
        NSLog("[PlayerVM] loadExistingChunks: chunkDir=\(dir.path)")

        // Migration: Check for old flat directory files and migrate to subdirectory
        let oldFlatDir = audioBaseDir
        let itemId = item.id.uuidString

        if FileManager.default.fileExists(atPath: oldFlatDir.path) {
            // Check for old format files: {uuid}_part0.wav, {uuid}_part1.wav, etc.
            do {
                let oldFiles = try FileManager.default.contentsOfDirectory(at: oldFlatDir, includingPropertiesForKeys: nil)
                let oldChunkFiles = oldFiles.filter { file in
                    let name = file.deletingPathExtension().lastPathComponent
                    return name.hasPrefix("\(itemId)_part")
                }

                if !oldChunkFiles.isEmpty {
                    NSLog("[PlayerVM] loadExistingChunks: Found \(oldChunkFiles.count) old-format files, migrating to subdirectory")

                    // Create subdirectory if needed
                    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

                    // Sort old files by index before migrating to ensure correct order
                    let sortedOldFiles = oldChunkFiles.sorted { file1, file2 in
                        let name1 = file1.deletingPathExtension().lastPathComponent
                        let name2 = file2.deletingPathExtension().lastPathComponent
                        let idx1 = name1.split(separator: "_").last?.replacingOccurrences(of: "part", with: "") ?? ""
                        let idx2 = name2.split(separator: "_").last?.replacingOccurrences(of: "part", with: "") ?? ""
                        return Int(idx1) ?? 0 < Int(idx2) ?? 0
                    }

                    // Migrate files to new location with new naming convention
                    for oldFile in sortedOldFiles {
                        // Extract index from old filename: {uuid}_part0.wav -> 0
                        let name = oldFile.deletingPathExtension().lastPathComponent
                        if let partIndex = name.split(separator: "_").last?.replacingOccurrences(of: "part", with: ""),
                           let index = Int(partIndex) {
                            let newFile = dir.appendingPathComponent("chunk_\(index).wav")

                            // Only copy if destination doesn't exist
                            if !FileManager.default.fileExists(atPath: newFile.path) {
                                try FileManager.default.copyItem(at: oldFile, to: newFile)
                                NSLog("[PlayerVM] loadExistingChunks: Migrated chunk \(index) to subdirectory")
                            }
                        }
                    }
                }
            } catch {
                NSLog("[PlayerVM] loadExistingChunks: Migration error: \(error)")
            }
        }

        guard FileManager.default.fileExists(atPath: dir.path) else {
            NSLog("[PlayerVM] loadExistingChunks: directory does not exist")
            return
        }

        // Clear existing audio files and stale playback state before loading
        audioPlayer.clearPlaybackState(item.id)

        do {
            let files = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            NSLog("[PlayerVM] loadExistingChunks: found \(files.count) files")

            // Collect all valid chunk indices and their URLs
            var chunks: [(index: Int, url: URL)] = []
            for file in files where file.pathExtension == "wav" {
                // Handle both regular files (chunk_0.wav) and _part files (chunk_0_part0.wav)
                // Extract index from filename - handle "chunk_X" or "chunk_X_partY" formats
                let filename = file.deletingLastPathComponent().deletingPathExtension().lastPathComponent
                var index: Int?

                if filename.contains("_part") {
                    // Format: chunk_X_partY - extract the first number
                    let parts = filename.split(separator: "_")
                    if parts.count >= 2, let firstNum = Int(parts[1]) {
                        index = firstNum
                    }
                } else {
                    // Format: chunk_X - extract the last number
                    if let indexStr = filename.split(separator: "_").last,
                       let idx = Int(indexStr) {
                        index = idx
                    }
                }

                if let idx = index {
                    chunks.append((index: idx, url: file))
                }
            }

            // Sort chunks by index to ensure correct playback order
            chunks.sort { $0.index < $1.index }
            NSLog("[PlayerVM] loadExistingChunks: sorted chunks: \(chunks.map { $0.index })")

            // Now load in sorted order - first chunk becomes main audio
            for chunk in chunks {
                item.generatedChunks[chunk.index] = chunk.url.path
                audioPlayer.loadChunk(chunk.url)
                NSLog("[PlayerVM] Loaded chunk \(chunk.index) from \(chunk.url.lastPathComponent)")
            }

            NSLog("[PlayerVM] loadExistingChunks: loaded \(item.generatedChunks.count) chunks, audioPlayer.hasAudioFiles = \(audioPlayer.hasAudioFiles)")
        } catch {
            NSLog("[PlayerVM] Failed to load existing chunks: \(error)")
        }
    }

    /// Save chunk path when synthesis completes
    private func saveChunkPath(_ url: URL, for index: Int) {
        item.generatedChunks[index] = url.path
    }

    private var cancellables = Set<AnyCancellable>()
    /// Callback when item is updated (e.g., voice selection changed)
    var onItemUpdate: ((LibraryItem) -> Void)?

    @MainActor init(
        item: LibraryItem,
        audioPlayer: AudioPlayerService? = nil,
        engine: ChatterboxEngine? = nil,
        downloadService: ModelDownloadService? = nil,
        synthesisDB: SynthesisDatabase? = nil,
        onItemUpdate: ((LibraryItem) -> Void)? = nil
    ) {
        self.item = item
        self.audioPlayer = audioPlayer ?? AudioPlayerService()
        self.engine = engine ?? ChatterboxEngine.shared
        self.downloadService = downloadService ?? ModelDownloadService()
        self.synthesisDB = synthesisDB ?? SynthesisDatabase()
        self.onItemUpdate = onItemUpdate

        // Set up callback for syncing text highlight when playback starts
        self.audioPlayer.onChunkPlaybackStarted = { [weak self] chunkIndex in
            Task { @MainActor in
                self?.updatePlayingIndex()
            }
        }

        // Stop any current playback and cancel synthesis from previous article
        // Use Task to defer to avoid "Publishing changes from within view updates" error
        Task { @MainActor in
            self.audioPlayer.stop()
            if let playingID = self.audioPlayer.currentPlayingItemID {
                self.audioPlayer.clearPlaybackState(playingID)
            }
        }

        // Cancel any ongoing synthesis task
        self.synthesisTask?.cancel()
        self.isSynthesizing = false

        // Cache paragraphs and sentences once
        cacheTextData()

        // Load existing chunks if any have been generated
        loadExistingChunks()

        // Set selected voice from item
        if let voiceID = item.selectedVoiceID {
            // First check built-in voices
            if let voice = VoiceProfile.builtInVoices.first(where: { $0.id == voiceID }) {
                selectedVoice = voice
            } else {
                // Then check custom voices from UserDefaults
                if let data = UserDefaults.standard.data(forKey: "custom_voices"),
                   let customVoices = try? JSONDecoder().decode([VoiceProfile].self, from: data),
                   let voice = customVoices.first(where: { $0.id == voiceID }) {
                    selectedVoice = voice
                } else {
                    selectedVoice = .defaultVoice
                }
            }
        }

        // Load audio if ready
        if item.status == .ready, let audioPath = item.audioFileURL {
            let url = URL(fileURLWithPath: audioPath)
            if FileManager.default.fileExists(atPath: audioPath) {
                self.audioPlayer.loadAudio(
                    url: url,
                    title: item.title,
                    artist: item.displayAuthor
                )
            }
        }
    }

    /// Save voice selection to item and notify parent
    func saveVoiceSelection(_ voice: VoiceProfile) {
        // Update both the item AND the currently selected voice
        selectedVoice = voice
        var updatedItem = item
        updatedItem.selectedVoiceID = voice.id
        item = updatedItem
        onItemUpdate?(updatedItem)
        NSLog("[PlayerVM] Saved voice selection: \(voice.id) for item: \(item.id)")

        // If audio is currently playing or synthesized chunks exist, restart with new voice
        if isStreamingAudio || audioPlayer.isPlaying || !item.generatedChunks.isEmpty {
            NSLog("[PlayerVM] Voice changed during playback - restarting synthesis")
            restartWithNewVoice()
        } else if !item.textContent.isEmpty {
            // Nothing playing but text exists - start synthesis immediately
            NSLog("[PlayerVM] Voice selected - starting synthesis with new voice")
            Task {
                await startSynthesis()
            }
        }
    }

    /// Restart synthesis with current voice (used when voice/settings change during playback)
    /// Uses database for full resume capability
    private func restartWithNewVoice() {
        // Stop current playback first to avoid race condition
        audioPlayer.stop()
        audioPlayer.clearAudioFiles()
        synthesisTask?.cancel()
        isSynthesizing = false
        isStreamingAudio = false
        isPaused = false

        // Get current position AFTER stopping
        let currentChunkIndex = audioPlayer.currentChunkIndex
        let itemId = item.id.uuidString
        NSLog("[PlayerVM] Voice changed at chunk \(currentChunkIndex), restarting from there using DB")

        // Reset highlighting
        currentPlayingIndex = -1

        // Clear in-memory chunks from current position onwards
        let chunksToRemove = item.generatedChunks.keys.filter { $0 >= currentChunkIndex }
        for key in chunksToRemove {
            item.generatedChunks.removeValue(forKey: key)
        }

        // Clear audio files on disk from current position onwards
        let outputDir = chunkDirectory
        if let files = try? FileManager.default.contentsOfDirectory(at: outputDir, includingPropertiesForKeys: nil) {
            for file in files {
                let filename = file.deletingPathExtension().lastPathComponent
                if let partRange = filename.range(of: "_part") {
                    let indexStr = String(filename[partRange.upperBound...])
                    if let index = Int(indexStr), index >= currentChunkIndex {
                        try? FileManager.default.removeItem(at: file)
                        NSLog("[PlayerVM] Deleted chunk file: \(file.lastPathComponent)")
                    }
                }
            }
        }

        // Update database: reset voice and reset chunks from current position
        Task { @MainActor in
            // Update voice in database
            try? self.synthesisDB.updateItemVoice(id: itemId, voiceId: self.selectedVoice.id)

            // Reset chunks from current position in database (they'll be re-synthesized)
            try? self.synthesisDB.resetChunksFromIndex(itemId: itemId, fromIndex: currentChunkIndex)

            // Update status to synthesizing
            try? self.synthesisDB.updateItemStatus(id: itemId, status: SynthesisItemStatus.synthesizing)

            // Update item progress in database
            try? self.synthesisDB.updateItemProgress(id: itemId)

            NSLog("[PlayerVM] Database updated for voice change - resuming from chunk \(currentChunkIndex)")

            // Start synthesis from current chunk - this uses the remaining text
            await self.startSynthesisFromChunk(currentChunkIndex)
        }
    }

    /// Handle preset change - apply settings and restart if playing
    func applyPreset(_ preset: VoicePreset) {
        synthesisSettings = preset.settings
        NSLog("[PlayerVM] Applied preset: \(preset.name)")

        // If audio is currently playing or synthesized chunks exist, restart with new settings
        if isStreamingAudio || audioPlayer.isPlaying || !item.generatedChunks.isEmpty {
            NSLog("[PlayerVM] Preset changed during playback - restarting synthesis")
            restartWithNewVoice()
        } else if !item.textContent.isEmpty {
            // Nothing playing but text exists - start synthesis immediately
            NSLog("[PlayerVM] Preset changed - starting synthesis with new settings")
            Task {
                await startSynthesis()
            }
        }
    }

    /// Cache paragraph and sentence data to avoid re-computing on every render
    private func cacheTextData() {
        NSLog("[PlayerVM] cacheTextData: starting for text length \(item.textContent.count)")

        // Cache paragraphs
        cachedParagraphs = item.textContent.components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        NSLog("[PlayerVM] cacheTextData: paragraphs count \(cachedParagraphs.count)")

        // CRITICAL FIX: Use TextChunker for BOTH textChunks AND flattenedSentences
        // This ensures they are 1:1 aligned for accurate index mapping
        let chunks = TextChunker.chunkText(item.textContent)
        cachedTextChunks = chunks

        // Build flattenedSentences from the SAME chunks with paragraph tracking
        var flattened: [(text: String, index: Int, paragraphIndex: Int)] = []
        var currentPara = 0
        var paraStartIndex = 0

        // Split text by paragraphs to assign paragraph indices
        let paragraphTexts = cachedParagraphs.map { $0.trimmingCharacters(in: .whitespaces) }

        for chunk in chunks {
            // Find which paragraph this chunk belongs to
            var foundPara = currentPara
            var found = false
            for (pi, paraText) in paragraphTexts.enumerated() {
                if paraText.isEmpty { continue }
                if chunk.contains(paraText) || paraText.contains(chunk.trimmingCharacters(in: .whitespaces).prefix(30)) {
                    foundPara = pi
                    found = true
                    break
                }
            }

            // If we moved to a new paragraph, record the start index
            if foundPara > currentPara {
                sentencesIndicesStart[currentPara] = paraStartIndex
                paraStartIndex = flattened.count
                currentPara = foundPara
            }

            flattened.append((text: chunk, index: flattened.count, paragraphIndex: currentPara))
        }

        // Record last paragraph start
        sentencesIndicesStart[currentPara] = paraStartIndex

        flattenedSentences = flattened

        // Build cachedParagraphSentences grouped by paragraph
        cachedParagraphSentences = cachedParagraphs.enumerated().map { idx, _ in
            flattened.filter { $0.paragraphIndex == idx }.map { $0.text }
        }

        // Build O(1) lookup map for text-to-index
        textToIndexMap = [:]
        for (idx, chunk) in chunks.enumerated() {
            textToIndexMap[chunk] = idx
        }

        // Build attributed strings (without highlight initially)
        rebuildAttributedStrings()

        NSLog("[PlayerVM] cacheTextData: done, textChunks=\(cachedTextChunks.count), flattenedSentences=\(flattenedSentences.count)")
    }

    /// Build attributed strings for all paragraphs with current highlight
    private func rebuildAttributedStrings() {
        cachedAttributedParagraphs = cachedParagraphs.enumerated().map { index, paragraph in
            let sentences = cachedParagraphSentences.indices.contains(index)
                ? cachedParagraphSentences[index] : []
            let startIdx = sentencesIndicesStart[index] ?? 0
            return buildAttributedString(sentences: sentences, startIndex: startIdx)
        }
    }

    /// Build attributed string for a paragraph with sentence highlighting
    private func buildAttributedString(sentences: [String], startIndex: Int) -> AttributedString {
        var result = AttributedString()

        for (idx, sentence) in sentences.enumerated() {
            let globalIdx = startIndex + idx
            let isActive = globalIdx == currentPlayingIndex
            let isPast = globalIdx < currentPlayingIndex && currentPlayingIndex >= 0

            var attr = AttributedString(sentence)

            if isActive {
                attr.foregroundColor = Color.appAccent
                attr.backgroundColor = Color.appAccent.opacity(0.15)
            } else if isPast {
                attr.foregroundColor = Color.appTextSecondary.opacity(0.5)
            } else {
                attr.foregroundColor = Color.appTextPrimary
            }

            result.append(attr)

            // Add space between sentences
            if idx < sentences.count - 1 {
                result.append(AttributedString(" "))
            }
        }

        return result
    }

    /// Split text into sentences - now uses TextChunker for consistency with audio
    private func splitIntoSentences(_ text: String) -> [String] {
        // Use TextChunker to get consistent sentence boundaries with audio synthesis
        return TextChunker.chunkText(text)
    }

    // MARK: - Playback Control Helpers

    /// The sentence chunks for TTS - now cached
    private var cachedTextChunks: [String] = []

    /// Cleaned text for display - matches chunk formatting exactly
    var cleanedText: String {
        TextChunker.cleanTextForDisplay(item.textContent)
    }

    var textChunks: [String] {
        if cachedTextChunks.isEmpty {
            cachedTextChunks = TextChunker.chunkText(item.textContent)
        }
        return cachedTextChunks
    }

    /// The original text split into paragraphs - uses cached value
    var paragraphs: [String] {
        cachedParagraphs
    }

    /// Current sentence being played (for highlighting)
    /// Uses currentChunkIndex directly - this is the chunk index from audio player
    var currentSentence: String? {
        guard currentChunkIndex >= 0, currentChunkIndex < textChunks.count else { return nil }
        return textChunks[currentChunkIndex]
    }

    /// Update currentPlayingIndex when chunk changes - called from view
    func updatePlayingIndex() {
        let previousIndex = currentPlayingIndex

        guard currentChunkIndex >= 0 else {
            currentPlayingIndex = -1
            if previousIndex != -1 {
                rebuildAttributedStrings()
            }
            return
        }

        // Direct index mapping: chunk index maps directly to sentence index
        // Both textChunks and flattenedSentences use TextChunker.chunkText(), so they're 1:1 aligned
        let foundIndex = currentChunkIndex

        if foundIndex >= 0 && foundIndex != previousIndex && foundIndex < flattenedSentences.count {
            currentPlayingIndex = foundIndex
            rebuildAttributedStrings()
        }
    }

    /// Current chunk index - use audio player's actual chapter index for accuracy
    var currentChunkIndex: Int {
        audioPlayer.currentChunkIndex
    }

    var canPlay: Bool {
        // Ready with a saved file OR has generated chunks OR actively streaming.
        // This supports both single audio file (audioFileURL) and chunk-based (generatedChunks) playback.
        (item.status == .ready && (item.audioFileURL != nil || !item.generatedChunks.isEmpty)) || isStreamingAudio
    }

    /// Stop audio playback and cancel any ongoing synthesis
    func stopGeneration() {
        audioPlayer.stop()
        synthesisTask?.cancel()
        synthesisTask = nil
        isSynthesizing = false
        isStreamingAudio = false
        NSLog("[PlayerVM] stopGeneration called")
    }

    /// Play from current position (resume or start from where user last tapped)
    func playFromCurrentPosition() {
        // If already playing, pause - also stop synthesis
        if audioPlayer.isPlaying {
            audioPlayer.pause()
            // Cancel the synthesis task
            synthesisTask?.cancel()
            synthesisTask = nil
            NSLog("[PlayerVM] Paused and cancelled synthesis")
            return
        }

        // If paused with audio loaded (either playing was paused OR we have loaded audio), resume playback
        // Check both duration > 0 and hasAudioFiles to handle all resume scenarios
        if audioPlayer.duration > 0 || audioPlayer.hasAudioFiles {
            // If we have audio files loaded, just resume
            if audioPlayer.hasAudioFiles {
                audioPlayer.play()
                NSLog("[PlayerVM] Resumed existing audio playback")
                return
            }
            // If duration is set but hasAudioFiles is false, still try to resume
            if audioPlayer.duration > 0 {
                audioPlayer.play()
                NSLog("[PlayerVM] Resumed audio with duration \(audioPlayer.duration)")
                return
            }
        }

        // If we have existing generated chunks but they're not loaded, load and play them
        if !item.generatedChunks.isEmpty {
            // Try to load existing chunks
            let sortedIndices = item.generatedChunks.keys.sorted()
            for chunkIndex in sortedIndices {
                guard let path = item.generatedChunks[chunkIndex],
                      FileManager.default.fileExists(atPath: path) else { continue }
                let url = URL(fileURLWithPath: path)
                audioPlayer.loadChunk(url)
            }

            if audioPlayer.hasAudioFiles {
                audioPlayer.play()
                NSLog("[PlayerVM] Loaded and playing existing chunks")
                return
            }
        }

        // Otherwise start synthesis from beginning in a Task so we can cancel it later
        NSLog("[PlayerVM] No existing audio/chunks found, starting fresh synthesis")
        synthesisTask = Task {
            await self.startSynthesisInternal()
        }
    }

    /// Internal synthesis that can be cancelled
    private func startSynthesisInternal(startingFromChunk: Int = 0) async {
        // Use regular startSynthesis which handles existing chunks properly
        await startSynthesis()
    }

    /// Start synthesis from a specific chunk index
    func startSynthesisFromChunk(_ chunkIndex: Int) async {
        guard chunkIndex >= 0, chunkIndex < textChunks.count else {
            await startSynthesis()
            return
        }

        downloadService.checkModelAvailability()
        guard downloadService.isModelReady else {
            showModelDownload = true
            return
        }

        isSynthesizing = true
        synthesisProgress = 0
        errorMessage = nil

        do {
            if !engine.isLoaded {
                NSLog("[PlayerVM] Loading models...")
                try await engine.loadModels()
                NSLog("[PlayerVM] Models loaded successfully")
            } else {
                NSLog("[PlayerVM] Models already loaded")
            }

            // Use chunk directory for persistent storage
            let outputDir = chunkDirectory
            try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

            let outputURL = outputDir.appendingPathComponent("chunk_\(chunkIndex).wav")

            var refAudioURL: URL?
            if let path = selectedVoice.resolvedReferenceAudioPath {
                refAudioURL = URL(fileURLWithPath: path)
            }

            // Pre-generate ALL chunks first to build up a queue
            var allChunkURLs: [URL] = []

            // Pass pre-chunked text to avoid rechunking which can cause boundary mismatches
            let chunksToSynthesize = Array(textChunks[chunkIndex...])

            try await engine.synthesize(
                text: "",  // Not used when preChunkedText is provided
                preChunkedText: chunksToSynthesize,
                referenceAudioURL: refAudioURL,
                outputURL: outputURL,
                onChunkReady: { chunkURL in
                    allChunkURLs.append(chunkURL)
                },
                onProgress: { [weak self] progress in
                    self?.synthesisProgress = progress
                },
                seed: synthesisSettings.seed,
                exaggeration: Float(synthesisSettings.exaggeration),
                cfgWeight: Float(synthesisSettings.cfgWeight),
                speedFactor: Float(synthesisSettings.speed)
            )

            // Save chunk paths to item
            for (index, url) in allChunkURLs.enumerated() {
                item.generatedChunks[chunkIndex + index] = url.path
            }

            NSLog("[PlayerVM] All \(allChunkURLs.count) chunks ready, starting playback")

            // Play all chunks - start with first, append rest
            if let firstURL = allChunkURLs.first {
                audioPlayer.startStreaming(
                    firstChunkURL: firstURL,
                    title: item.title,
                    artist: item.displayAuthor
                )

                // Queue up remaining chunks
                for url in allChunkURLs.dropFirst() {
                    audioPlayer.appendStreamChunk(url)
                }

                // All chunks are ready - no more expected
                audioPlayer.isExpectingMoreChunks = false

                isStreamingAudio = true
            }

        } catch {
            errorMessage = error.localizedDescription
            NSLog("[PlayerVM] Error: \(error)")
        }

        isSynthesizing = false
    }

    var playbackRateLabel: String {
        let rate = audioPlayer.playbackRate
        if rate == Float(Int(rate)) {
            return "\(Int(rate))x"
        }
        return String(format: "%.1fx", rate)
    }

    static let availableRates: [Float] = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    func cyclePlaybackRate() {
        let rates = Self.availableRates
        let current = audioPlayer.playbackRate
        if let index = rates.firstIndex(of: current) {
            let nextIndex = (index + 1) % rates.count
            audioPlayer.setPlaybackRate(rates[nextIndex])
        } else {
            audioPlayer.setPlaybackRate(1.0)
        }
    }

    // MARK: - Generation Control

    /// Toggle pause/resume of audio generation
    func toggleGenerationPause() {
        if isPaused {
            // Resume - update DB status and restart playback
            try? synthesisDB.updateItemStatus(id: item.id.uuidString, status: SynthesisItemStatus.synthesizing)
            // Resume audio playback if there was audio loaded
            if audioPlayer.duration > 0 {
                audioPlayer.play()
            }
        } else {
            // Pause - cancel synthesis, pause audio, and update DB status
            synthesisTask?.cancel()
            synthesisTask = nil
            audioPlayer.isExpectingMoreChunks = false
            // Also pause audio playback
            audioPlayer.pause()
            try? synthesisDB.updateItemStatus(id: item.id.uuidString, status: SynthesisItemStatus.paused)
            isSynthesizing = false
            isStreamingAudio = false
        }
        isPaused.toggle()
    }

    /// Cancel audio generation
    func cancelGeneration() {
        synthesisTask?.cancel()
        try? synthesisDB.updateItemStatus(id: item.id.uuidString, status: SynthesisItemStatus.cancelled)
        isSynthesizing = false
        isStreamingAudio = false
        isPaused = false
        NSLog("[PlayerVM] Generation cancelled")
    }

    // MARK: - Synthesis

    func startSynthesis() async {
        NSLog("[PlayerVM] startSynthesis called, item.status=\(item.status.rawValue), generatedChunks.count=\(item.generatedChunks.count), audioPlayer.hasAudioFiles=\(audioPlayer.hasAudioFiles)")

        // If audio is already ready (single file), play the existing file
        if item.status == .ready, let audioPath = item.audioFileURL, !audioPath.isEmpty {
            let url = URL(fileURLWithPath: audioPath)
            if FileManager.default.fileExists(atPath: audioPath) {
                audioPlayer.loadAudio(url: url, title: item.title, artist: item.displayAuthor)
                audioPlayer.play()
                NSLog("[PlayerVM] Playing existing audio file")
                return
            }
        }

        // If chunks already exist (from generateOnly), play them
        // Don't require specific status - presence of chunks is sufficient
        // (status may be .pending if app restarted during synthesis but chunks exist on disk)
        if !item.generatedChunks.isEmpty || audioPlayer.hasAudioFiles {
            NSLog("[PlayerVM] Using existing chunks for playback, status=\(item.status.rawValue)")
            // loadExistingChunks was already called in init, just start playing
            if audioPlayer.hasAudioFiles {
                audioPlayer.play()
                NSLog("[PlayerVM] Started playing chunks")
                return
            } else {
                // Chunks exist in item.generatedChunks but not loaded into audioPlayer
                // Try to load them now - MUST sort by index to preserve playback order
                NSLog("[PlayerVM] WARNING: chunks exist but audioPlayer.hasAudioFiles is false, attempting to load")
                let sortedIndices = item.generatedChunks.keys.sorted()
                for chunkIndex in sortedIndices {
                    guard let path = item.generatedChunks[chunkIndex],
                          FileManager.default.fileExists(atPath: path) else { continue }
                    let url = URL(fileURLWithPath: path)
                    audioPlayer.loadChunk(url)
                    NSLog("[PlayerVM] Loaded chunk \(chunkIndex) from existing path")
                }

                if audioPlayer.hasAudioFiles {
                    audioPlayer.play()
                    NSLog("[PlayerVM] Started playing chunks after manual load")
                    return
                }
            }
        } else {
            NSLog("[PlayerVM] Cannot use chunks: status=\(item.status.rawValue), generatedChunks.isEmpty=\(item.generatedChunks.isEmpty), hasAudioFiles=\(audioPlayer.hasAudioFiles)")
        }

        // Check if models are downloaded first
        downloadService.checkModelAvailability()
        guard downloadService.isModelReady else {
            NSLog("[PlayerVM] Models not ready, showing download")
            showModelDownload = true
            return
        }

        NSLog("[PlayerVM] Models ready, starting synthesis")
        isSynthesizing = true
        synthesisProgress = 0
        errorMessage = nil

        // Check for existing synthesis in database (resume support)
        let itemId = item.id.uuidString
        var isResuming = false
        var isAlreadyCompleted = false
        if let existingItem = try? synthesisDB.getItem(id: itemId) {
            if existingItem.status == .completed {
                // Already completed - load chunks and skip synthesis
                NSLog("[PlayerVM] Synthesis already completed for \(itemId)")
                isAlreadyCompleted = true
                // Load completed chunks into memory
                if let completedChunks = try? synthesisDB.getCompletedChunks(itemId: itemId) {
                    for chunk in completedChunks {
                        if let path = chunk.filePath {
                            item.generatedChunks[chunk.chunkIndex] = path
                        }
                    }
                }
            } else if existingItem.status == .paused || existingItem.status == .error || existingItem.status == .cancelled {
                NSLog("[PlayerVM] Resuming synthesis for \(itemId), progress: \(existingItem.completedChunks)/\(existingItem.totalChunks)")
                isResuming = true
                // Load completed chunks into memory
                if let completedChunks = try? synthesisDB.getCompletedChunks(itemId: itemId) {
                    for chunk in completedChunks {
                        if let path = chunk.filePath {
                            item.generatedChunks[chunk.chunkIndex] = path
                        }
                    }
                }
            } else if existingItem.status == .synthesizing {
                // Already synthesizing - might be interrupted, treat as resume
                NSLog("[PlayerVM] Found synthesizing item, resuming...")
                isResuming = true
            }
        }

        // If already completed, skip synthesis and start playback
        if isAlreadyCompleted {
            await playCompletedAudio()
            return
        }

        // Create new synthesis in database if not resuming
        if !isResuming {
            do {
                try synthesisDB.createItem(
                    id: itemId,
                    title: item.title,
                    text: item.textContent,
                    voiceId: selectedVoice.id,
                    settings: synthesisSettings
                )
                NSLog("[PlayerVM] Created new synthesis in database")
            } catch {
                NSLog("[PlayerVM] Failed to create synthesis in DB: \(error)")
            }
        }

        // Mark as synthesizing in database
        try? synthesisDB.updateItemStatus(id: itemId, status: SynthesisItemStatus.synthesizing)

        do {
            if !engine.isLoaded {
                NSLog("[PlayerVM] Loading models...")
                try await engine.loadModels()
                NSLog("[PlayerVM] Models loaded successfully")
            } else {
                NSLog("[PlayerVM] Models already loaded")
            }

            // Write chunks to subdirectory matching loadExistingChunks expectation
            let outputDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
                .appendingPathComponent("Audio/\(item.id.uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

            let outputURL = outputDir.appendingPathComponent("chunk.wav")

            // Get reference audio for voice cloning
            var refAudioURL: URL?
            if let path = selectedVoice.resolvedReferenceAudioPath {
                refAudioURL = URL(fileURLWithPath: path)
            }

            // Skip already completed chunks - filter text to only include missing chunks
            var chunksToSynthesize: [String] = []
            let allChunks = TextChunker.chunkText(item.textContent)

            for (index, chunk) in allChunks.enumerated() {
                if let existingPath = item.generatedChunks[index],
                   FileManager.default.fileExists(atPath: existingPath) {
                    // Chunk already exists, skip it
                    continue
                }
                chunksToSynthesize.append(chunk)
            }

            // If all chunks exist, skip synthesis
            if chunksToSynthesize.isEmpty && !item.generatedChunks.isEmpty {
                NSLog("[PlayerVM] All chunks already exist, skipping synthesis")
                await playCompletedAudio()
                return
            }

            // Load existing completed chunks into audio player for immediate playback
            var existingChunkURLs: [URL] = []
            for i in 0..<allChunks.count {
                if let path = item.generatedChunks[i], FileManager.default.fileExists(atPath: path) {
                    existingChunkURLs.append(URL(fileURLWithPath: path))
                }
            }
            if !existingChunkURLs.isEmpty {
                audioPlayer.loadChapters(urls: existingChunkURLs, title: item.title, artist: item.displayAuthor)
                NSLog("[PlayerVM] Loaded \(existingChunkURLs.count) existing chunks for playback")
            }

            NSLog("[PlayerVM] Starting synthesizeStream, text length: \(item.textContent.count), chunks to synthesize: \(chunksToSynthesize.count)/\(allChunks.count)")
            // Use callback-based synthesis instead of broken AsyncThrowingStream
            // The stream has Swift concurrency issues with @MainActor
            // Wrap in Task.detached to run ONNX inference off main thread to prevent UI freeze
            audioPlayer.isExpectingMoreChunks = true
            NSLog("[PlayerVM] starting callback-based synthesis")

            // Use actor-like isolation with a class wrapper to track first chunk
            class ChunkTracker: @unchecked Sendable {
                var firstURL: URL? = nil
            }
            let tracker = ChunkTracker()

            // Capture UI state for use in callbacks
            let itemTitle = item.title
            let itemArtist = item.displayAuthor
            let itemTextContent = item.textContent
            let itemId = item.id.uuidString

            // Run synthesis on background thread to prevent UI freezing
            try await Task.detached(priority: .userInitiated) { [weak self] in
                guard let self = self else { return }
                let engine = self.engine
                let audioPlayer = self.audioPlayer

                // Use preChunkedText to only synthesize missing chunks
                let missingChunks = chunksToSynthesize
                try await engine.synthesize(
                    text: "",  // Not used when preChunkedText provided
                    preChunkedText: missingChunks,
                    referenceAudioURL: refAudioURL,
                    outputURL: outputURL,
                    onChunkReady: { [weak self] chunkURL in
                        guard let self = self else { return }
                        NSLog("[PlayerVM] onChunkReady callback received: \(chunkURL.lastPathComponent)")

                        // Extract chunk index from filename (format: uuid_part0.wav)
                        let filename = chunkURL.deletingPathExtension().lastPathComponent
                        if let partRange = filename.range(of: "_part") {
                            let indexStr = String(filename[partRange.upperBound...])
                            if let chunkIndex = Int(indexStr) {
                                // Mark chunk as completed in database and update progress
                                Task { @MainActor in
                                    try? self.synthesisDB.markChunkCompleted(
                                        itemId: itemId,
                                        chunkIndex: chunkIndex,
                                        filePath: chunkURL.path,
                                        duration: nil
                                    )
                                    try? self.synthesisDB.updateItemProgress(id: itemId)
                                    if let progress = try? self.synthesisDB.getProgress(itemId: itemId) {
                                        self.synthesisProgress = progress
                                    }
                                }
                            }
                        }

                        // All UI updates must be dispatched to main actor
                        Task { @MainActor in
                            if tracker.firstURL == nil {
                                // First chunk: start streaming playback
                                tracker.firstURL = chunkURL
                                NSLog("[PlayerVM] First chunk, starting streaming playback")
                                self.audioPlayer.startStreaming(
                                    firstChunkURL: chunkURL,
                                    title: itemTitle,
                                    artist: itemArtist
                                )
                                self.isStreamingAudio = true
                            } else {
                                // Subsequent chunks: append to stream
                                NSLog("[PlayerVM] Appending chunk to stream")
                                self.audioPlayer.appendStreamChunk(chunkURL)
                            }
                        }
                    },
                    onProgress: { [weak self] progress in
                        Task { @MainActor in
                            self?.synthesisProgress = progress
                        }
                    },
                    seed: synthesisSettings.seed,
                    exaggeration: Float(synthesisSettings.exaggeration),
                    cfgWeight: Float(synthesisSettings.cfgWeight),
                    speedFactor: Float(synthesisSettings.speed)
                )
            }.value

            // Synthesis complete
            NSLog("[PlayerVM] callback synthesis complete")

            // Mark synthesis as completed in database
            try? synthesisDB.updateItemStatus(id: itemId, status: SynthesisItemStatus.completed)

            audioPlayer.isExpectingMoreChunks = false
            item.audioFileURL = outputURL.path
            item.status = .ready
            // Save library with generated chunks so they persist across app restarts
            onItemUpdate?(item)
            isStreamingAudio = false

            // If the player finished the streaming chunks while we were synthesizing,
            // load the full file so seek/waveform work correctly.
            if !audioPlayer.isPlaying {
                audioPlayer.loadAudio(
                    url: outputURL,
                    title: item.title,
                    artist: item.displayAuthor
                )
            }

        } catch {
            NSLog("[PlayerVM] Synthesis error: \(error)")
            item.status = .error
            isStreamingAudio = false
            errorMessage = error.localizedDescription
        }

        isSynthesizing = false
    }

    func downloadModels() async {
        isDownloadingModels = true
        errorMessage = nil
        await downloadService.downloadModels()
        isDownloadingModels = false

        if downloadService.isModelReady {
            showModelDownload = false
            await startSynthesis()
        } else if let error = downloadService.errorMessage {
            errorMessage = error
        }
    }

    /// Generate audio files only without playing - for library background download
    func generateOnly() async {
        NSLog("[PlayerVM] generateOnly called for: \(item.title)")

        // Stop any current playback and cancel synthesis from previous article
        // This ensures we don't have multiple syntheses playing simultaneously
        audioPlayer.stop()
        audioPlayer.clearAudioFiles()
        synthesisTask?.cancel()
        synthesisTask = nil
        isSynthesizing = false
        isStreamingAudio = false

        // Check if already ready (either single file or chunks)
        if item.status == .ready && (item.audioFileURL != nil || !item.generatedChunks.isEmpty) {
            NSLog("[PlayerVM] Audio already ready")
            return
        }

        // Check if models are available
        downloadService.checkModelAvailability()
        guard downloadService.isModelReady else {
            NSLog("[PlayerVM] Models not ready")
            showModelDownload = true
            return
        }

        isSynthesizing = true
        synthesisProgress = 0
        errorMessage = nil

        do {
            if !engine.isLoaded {
                try await engine.loadModels()
            }

            // Use chunk directory for persistent storage
            let outputDir = chunkDirectory
            try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

            // Get reference audio for voice cloning
            var refAudioURL: URL?
            if let path = selectedVoice.resolvedReferenceAudioPath {
                refAudioURL = URL(fileURLWithPath: path)
            }

            // Generate all chunks and save them
            let chunks = TextChunker.chunkText(item.textContent)
            var allChunkURLs: [URL] = []

            for (index, chunkText) in chunks.enumerated() {
                let outputURL = outputDir.appendingPathComponent("chunk_\(index).wav")

                // Skip if already exists, but still track it in generatedChunks
                if FileManager.default.fileExists(atPath: outputURL.path) {
                    allChunkURLs.append(outputURL)
                    item.generatedChunks[index] = outputURL.path
                    NSLog("[PlayerVM] Chunk \(index) already exists, tracking: \(outputURL.path)")
                    continue
                }

                // Generate this chunk
                NSLog("[PlayerVM] Generating chunk \(index)/\(chunks.count): \(chunkText.prefix(50))...")
                try await engine.synthesize(
                    text: chunkText,
                    referenceAudioURL: refAudioURL,
                    outputURL: outputURL,
                    onChunkReady: { url in
                        allChunkURLs.append(url)
                    },
                    onProgress: { [weak self] progress in
                        // Update local progress
                        let chunkProgress = Double(index) / Double(chunks.count)
                        let inChunkProgress = progress / Double(chunks.count)
                        self?.synthesisProgress = chunkProgress + inChunkProgress
                        // Update shared progress for library view
                        self?.audioPlayer.updateSynthesisProgress(self?.item.id ?? UUID(), progress: chunkProgress + inChunkProgress)
                    },
                    seed: synthesisSettings.seed,
                exaggeration: Float(synthesisSettings.exaggeration),
                cfgWeight: Float(synthesisSettings.cfgWeight),
                speedFactor: Float(synthesisSettings.speed)
                )

                // Save chunk path
                item.generatedChunks[index] = outputURL.path
            }

            // Calculate and save total duration
            item.duration = TextChunker.estimateTotalDuration(for: item.textContent)

            // Mark as ready
            item.status = .ready
            // Save library with generated chunks so they persist across app restarts
            onItemUpdate?(item)

            // Clear synthesis progress
            audioPlayer.clearSynthesisProgress(item.id)

            NSLog("[PlayerVM] generateOnly complete: \(allChunkURLs.count) chunks saved, duration: \(item.duration ?? 0)")

        } catch {
            NSLog("[PlayerVM] generateOnly error: \(error)")
            errorMessage = error.localizedDescription
            item.status = .error
            // Clear synthesis progress on error
            audioPlayer.clearSynthesisProgress(item.id)
        }

        isSynthesizing = false
    }

    // MARK: - Chapter Navigation

    /// Get chapters - generate from paragraphs if empty
    var displayChapters: [Chapter] {
        if !item.chapters.isEmpty {
            return item.chapters
        }
        // Generate chapters from paragraphs if empty
        return cachedParagraphs.enumerated().map { index, text in
            Chapter(
                title: "Paragraph \(index + 1)",
                textContent: String(text.prefix(100))
            )
        }
    }

    var currentChapterTitle: String {
        guard !displayChapters.isEmpty, audioPlayer.currentChunkIndex < displayChapters.count else {
            return item.title
        }
        return item.chapters[audioPlayer.currentChunkIndex].title
    }

    func selectChapter(_ index: Int) {
        NSLog("[PlayerVM] selectChapter: \(index)")

        // Close the sheet first
        showChapterList = false

        // If audio files exist, go to that chapter
        if audioPlayer.hasAudioFiles && index < audioPlayer.audioFileCount {
            audioPlayer.goToChapter(index)
        } else {
            // No audio yet - synthesize from this chunk
            Task {
                await startSynthesisFromChunk(index)
            }
        }
    }

    // MARK: - Text for Current Position

    var currentTextSnippet: String {
        let chunks = TextChunker.chunkText(item.textContent)
        guard !chunks.isEmpty else { return "" }

        if audioPlayer.duration > 0 {
            let chunkIndex = Int(audioPlayer.progress * Double(chunks.count - 1))
            let safeIndex = max(0, min(chunkIndex, chunks.count - 1))
            return chunks[safeIndex]
        }

        return chunks.first ?? ""
    }

    // MARK: - Selection Playback

    /// Track current streaming chunk index for highlighting
    @Published var currentStreamingChunkIndex: Int = -1

    /// Play from selected text position to end of content
    func playFromSelection(_ selectedText: String) async {
        guard !selectedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }

        // Check if models are available
        downloadService.checkModelAvailability()
        guard downloadService.isModelReady else {
            errorMessage = "Please download the voice model first"
            showModelDownload = true
            return
        }

        // Find the position of selected text in the full content
        let fullText = item.textContent
        guard let range = fullText.range(of: selectedText, options: .caseInsensitive) else {
            // If exact match not found, just play the selection
            await playSelectionOnly(selectedText)
            return
        }

        // Get text from selection to end
        let textFromSelection = String(fullText[range.lowerBound...])

        isSynthesizing = true
        synthesisProgress = 0
        errorMessage = nil

        do {
            if !engine.isLoaded {
                NSLog("[PlayerVM] Loading models...")
                try await engine.loadModels()
                NSLog("[PlayerVM] Models loaded successfully")
            } else {
                NSLog("[PlayerVM] Models already loaded")
            }

            let outputDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
                .appendingPathComponent("Audio", isDirectory: true)
            try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

            let outputURL = outputDir.appendingPathComponent("\(item.id.uuidString)_from_selection.wav")

            // Get reference audio for voice cloning
            var refAudioURL: URL?
            if let path = selectedVoice.resolvedReferenceAudioPath {
                refAudioURL = URL(fileURLWithPath: path)
            }

            // Find which chunk the selection starts in
            let chunks = TextChunker.chunkText(textFromSelection)
            currentStreamingChunkIndex = 0

            var firstChunkPlayed = false

            try await engine.synthesize(
                text: textFromSelection,
                referenceAudioURL: refAudioURL,
                outputURL: outputURL,
                onChunkReady: { [weak self] chunkURL in
                    guard let self else { return }

                    // Update highlight to current chunk
                    _ = self.currentStreamingChunkIndex

                    if !firstChunkPlayed {
                        firstChunkPlayed = true
                        self.audioPlayer.startStreaming(
                            firstChunkURL: chunkURL,
                            title: "From Selection",
                            artist: self.item.displayAuthor
                        )
                        self.isStreamingAudio = true
                    } else {
                        self.audioPlayer.appendStreamChunk(chunkURL)
                    }
                },
                onProgress: { [weak self] progress in
                    guard let self else { return }
                    self.synthesisProgress = progress

                    // Update chunk index for highlighting based on progress
                    if !chunks.isEmpty {
                        let chunkIndex = Int(progress * Double(chunks.count))
                        self.currentStreamingChunkIndex = min(chunkIndex, chunks.count - 1)
                    }
                },
                seed: synthesisSettings.seed,
                exaggeration: Float(synthesisSettings.exaggeration),
                cfgWeight: Float(synthesisSettings.cfgWeight),
                speedFactor: Float(synthesisSettings.speed)
            )

            isStreamingAudio = false
            isSynthesizing = false
            currentStreamingChunkIndex = -1

        } catch {
            errorMessage = error.localizedDescription
            isSynthesizing = false
            isStreamingAudio = false
            currentStreamingChunkIndex = -1
        }
    }

    /// Play already completed audio (skip synthesis)
    private func playCompletedAudio() async {
        // Check if we have completed chunks to play
        guard !item.generatedChunks.isEmpty else {
            NSLog("[PlayerVM] No completed chunks found, starting fresh synthesis")
            await startSynthesis()
            return
        }

        // Load audio files from disk
        var audioFiles: [URL] = []

        for i in 0..<textChunks.count {
            let chunkPath = item.generatedChunks[i]
            if let path = chunkPath, FileManager.default.fileExists(atPath: path) {
                audioFiles.append(URL(fileURLWithPath: path))
            } else {
                // Missing chunk - fall back to synthesis
                NSLog("[PlayerVM] Missing chunk \(i), need to synthesize")
                await startSynthesis()
                return
            }
        }

        // Load into audio player and start playback
        audioPlayer.loadChapters(
            urls: audioFiles,
            title: item.title,
            artist: item.displayAuthor
        )
        audioPlayer.play()
        isStreamingAudio = true

        isSynthesizing = false
        NSLog("[PlayerVM] Started playback of completed audio, \(audioFiles.count) chunks")
    }

    /// Play only the selected text (legacy behavior)
    private func playSelectionOnly(_ text: String) async {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }

        downloadService.checkModelAvailability()
        guard downloadService.isModelReady else {
            errorMessage = "Please download the voice model first"
            showModelDownload = true
            return
        }

        isSynthesizing = true
        synthesisProgress = 0
        errorMessage = nil

        do {
            if !engine.isLoaded {
                NSLog("[PlayerVM] Loading models...")
                try await engine.loadModels()
                NSLog("[PlayerVM] Models loaded successfully")
            } else {
                NSLog("[PlayerVM] Models already loaded")
            }

            let outputDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
                .appendingPathComponent("Audio", isDirectory: true)
            try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

            let outputURL = outputDir.appendingPathComponent("\(item.id.uuidString)_selection.wav")

            var refAudioURL: URL?
            if let path = selectedVoice.resolvedReferenceAudioPath {
                refAudioURL = URL(fileURLWithPath: path)
            }

            var firstChunkPlayed = false
            let chunks = TextChunker.chunkText(text)
            currentStreamingChunkIndex = 0

            try await engine.synthesize(
                text: text,
                referenceAudioURL: refAudioURL,
                outputURL: outputURL,
                onChunkReady: { [weak self] chunkURL in
                    guard let self else { return }
                    if !firstChunkPlayed {
                        firstChunkPlayed = true
                        self.audioPlayer.startStreaming(
                            firstChunkURL: chunkURL,
                            title: "Selection",
                            artist: self.item.displayAuthor
                        )
                        self.isStreamingAudio = true
                    } else {
                        self.audioPlayer.appendStreamChunk(chunkURL)
                    }
                },
                onProgress: { [weak self] progress in
                    guard let self else { return }
                    self.synthesisProgress = progress
                    if !chunks.isEmpty {
                        let chunkIndex = Int(progress * Double(chunks.count))
                        self.currentStreamingChunkIndex = min(chunkIndex, chunks.count - 1)
                    }
                },
                seed: synthesisSettings.seed,
                exaggeration: Float(synthesisSettings.exaggeration),
                cfgWeight: Float(synthesisSettings.cfgWeight),
                speedFactor: Float(synthesisSettings.speed)
            )

            isStreamingAudio = false
            isSynthesizing = false
            currentStreamingChunkIndex = -1

        } catch {
            errorMessage = error.localizedDescription
            isSynthesizing = false
            isStreamingAudio = false
            currentStreamingChunkIndex = -1
        }
    }

    /// Get the chunk index for a given text position
    func chunkIndexForPosition(_ position: Int) -> Int {
        let chunks = textChunks
        guard !chunks.isEmpty else { return 0 }

        var currentPos = 0
        for (index, chunk) in chunks.enumerated() {
            if position < currentPos + chunk.count {
                return index
            }
            currentPos += chunk.count
        }
        return chunks.count - 1
    }

    /// Play from a specific chunk index to the end
    func playFromChunkIndex(_ chunkIndex: Int) async {
        let chunks = textChunks
        guard chunkIndex >= 0, chunkIndex < chunks.count else { return }

        // Stop current audio if playing
        audioPlayer.stop()
        isStreamingAudio = false

        // Check if models are available
        downloadService.checkModelAvailability()
        guard downloadService.isModelReady else {
            errorMessage = "Please download the voice model first"
            showModelDownload = true
            return
        }

        isSynthesizing = true
        synthesisProgress = 0
        errorMessage = nil

        // Highlight the clicked chunk immediately
        currentPlaybackChunkIndex = chunkIndex
        currentStreamingChunkIndex = chunkIndex

        do {
            if !engine.isLoaded {
                NSLog("[PlayerVM] Loading models...")
                try await engine.loadModels()
                NSLog("[PlayerVM] Models loaded successfully")
            } else {
                NSLog("[PlayerVM] Models already loaded")
            }

            // Write chunks to subdirectory matching loadExistingChunks expectation
            let outputDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
                .appendingPathComponent("Audio/\(item.id.uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

            let outputURL = outputDir.appendingPathComponent("chunk.wav")

            var refAudioURL: URL?
            if let path = selectedVoice.resolvedReferenceAudioPath {
                refAudioURL = URL(fileURLWithPath: path)
            }

            // Pre-generate all chunks before starting playback
            // This ensures all audio is ready before playback begins
            var allChunkURLs: [URL] = []

            let tempOnChunkReady: (URL) -> Void = { chunkURL in
                allChunkURLs.append(chunkURL)
            }

            try await engine.synthesize(
                text: item.textContent,
                referenceAudioURL: refAudioURL,
                outputURL: outputURL,
                onChunkReady: tempOnChunkReady,
                onProgress: nil,
                seed: synthesisSettings.seed,
                exaggeration: Float(synthesisSettings.exaggeration),
                cfgWeight: Float(synthesisSettings.cfgWeight),
                speedFactor: Float(synthesisSettings.speed)
            )

            // Now play all the chunks
            NSLog("[PlayerVM] All \(allChunkURLs.count) chunks ready, starting playback")

            if let firstURL = allChunkURLs.first {
                audioPlayer.startStreaming(
                    firstChunkURL: firstURL,
                    title: item.title,
                    artist: item.displayAuthor
                )

                // Append remaining chunks
                for url in allChunkURLs.dropFirst() {
                    audioPlayer.appendStreamChunk(url)
                }

                isStreamingAudio = true
            }

            // Track progress separately (no synthesis happening anymore)
            var progressUpdateTask: Task<Void, Never>? = nil
            progressUpdateTask = Task { @MainActor in
                while audioPlayer.isPlaying || audioPlayer.currentTime > 0 {
                    try? await Task.sleep(nanoseconds: 500_000_000) // 0.5 seconds
                }
                isStreamingAudio = false
            }

            // Wait for playback to complete
            while audioPlayer.isPlaying {
                try? await Task.sleep(nanoseconds: 100_000_000) // 0.1 second
            }

            // Reset state
            progressUpdateTask?.cancel()
            isStreamingAudio = false
            isSynthesizing = false

        } catch {
            errorMessage = error.localizedDescription
            isSynthesizing = false
            isStreamingAudio = false
        }
    }
}
