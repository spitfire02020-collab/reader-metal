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
    /// Flattened sentences for efficient display with index-based highlighting
    /// Each item: (sentenceText, globalSentenceIndex, paragraphIndex)
    @Published private(set) var flattenedSentences: [(text: String, index: Int, paragraphIndex: Int)] = []
    /// Current playing sentence index for highlighting (tracks which sentence is playing)
    @Published var currentPlayingIndex: Int = -1
    /// Cached attributed strings for paragraphs (with highlighting) - rebuilt when highlight changes
    @Published private(set) var cachedAttributedParagraphs: [AttributedString] = []

    let audioPlayer: AudioPlayerService
    private let engine = ChatterboxEngine()
    private let downloadService = ModelDownloadService.shared

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
        NSLog("[PlayerVM] loadExistingChunks: checking \(dir.path)")
        guard FileManager.default.fileExists(atPath: dir.path) else {
            NSLog("[PlayerVM] loadExistingChunks: directory does not exist")
            return
        }

        // Clear existing audio files before loading to avoid duplicates
        audioPlayer.clearAudioFiles()

        // Clear any stale synthesis progress
        audioPlayer.clearSynthesisProgress(item.id)

        do {
            let files = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            NSLog("[PlayerVM] loadExistingChunks: found \(files.count) files")

            // Collect all valid chunk indices and their URLs
            var chunks: [(index: Int, url: URL)] = []
            for file in files where file.pathExtension == "wav" {
                // Skip temporary "part" files from interrupted synthesis
                if file.lastPathComponent.contains("_part") {
                    continue
                }
                // Extract chunk index from filename like "chunk_0.wav"
                if let indexStr = file.deletingPathExtension().lastPathComponent.split(separator: "_").last,
                   let index = Int(indexStr) {
                    chunks.append((index: index, url: file))
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

    @MainActor init(item: LibraryItem, audioPlayer: AudioPlayerService? = nil) {
        self.item = item
        self.audioPlayer = audioPlayer ?? AudioPlayerService.shared

        // Cache paragraphs and sentences once
        cacheTextData()

        // Load existing chunks if any have been generated
        loadExistingChunks()

        // Set selected voice from item
        if let voiceID = item.selectedVoiceID {
            selectedVoice = VoiceProfile.builtInVoices.first { $0.id == voiceID } ?? .defaultVoice
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

    /// Cache paragraph and sentence data to avoid re-computing on every render
    private func cacheTextData() {
        NSLog("[PlayerVM] cacheTextData: starting for text length \(item.textContent.count)")

        // Cache paragraphs
        cachedParagraphs = item.textContent.components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        NSLog("[PlayerVM] cacheTextData: paragraphs count \(cachedParagraphs.count)")

        // Cache sentence splits for each paragraph
        cachedParagraphSentences = cachedParagraphs.map { paragraph in
            splitIntoSentences(paragraph)
        }

        // Build flattened list for efficient display AND cache indices mapping
        var flattened: [(text: String, index: Int, paragraphIndex: Int)] = []
        var indicesMap: [Int: Int] = [:]
        var globalIndex = 0
        for (paragraphIndex, sentences) in cachedParagraphSentences.enumerated() {
            indicesMap[paragraphIndex] = globalIndex
            for sentence in sentences {
                flattened.append((text: sentence, index: globalIndex, paragraphIndex: paragraphIndex))
                globalIndex += 1
            }
        }
        flattenedSentences = flattened
        sentencesIndicesStart = indicesMap

        // Build attributed strings (without highlight initially)
        rebuildAttributedStrings()

        NSLog("[PlayerVM] cacheTextData: done, total sentences \(flattenedSentences.count)")
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

    /// Split text into sentences - used for caching
    private func splitIntoSentences(_ text: String) -> [String] {
        var sentences: [String] = []
        var currentSentence = ""
        let chars = Array(text)

        for char in chars {
            currentSentence.append(char)

            if ".?!".contains(char) {
                let trimmed = currentSentence.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmed.count > 2 {
                    sentences.append(trimmed)
                    currentSentence = ""
                }
            }
        }

        let remaining = currentSentence.trimmingCharacters(in: .whitespacesAndNewlines)
        if !remaining.isEmpty {
            sentences.append(remaining)
        }

        return sentences
    }

    // MARK: - Playback Control Helpers

    /// The sentence chunks for TTS - now cached
    private var cachedTextChunks: [String] = []

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

        // Map chunk index directly to sentence index
        // Each chunk corresponds to sentences from that chunk's text
        let chunkText = currentChunkIndex < textChunks.count ? textChunks[currentChunkIndex] : ""

        // Find the starting sentence index for this chunk by looking at where this chunk's text appears
        // Use a more robust matching - find any sentence that starts this chunk
        var foundIndex = -1

        // First try exact match
        if let idx = flattenedSentences.firstIndex(where: { $0.text == chunkText }) {
            foundIndex = idx
        } else {
            // Try to find by partial match - sentence that starts the chunk
            for (idx, sentence) in flattenedSentences.enumerated() {
                if chunkText.hasPrefix(sentence.text) || sentence.text.hasPrefix(chunkText.prefix(min(50, sentence.text.count))) {
                    foundIndex = idx
                    break
                }
            }
        }

        // Fallback: approximate index based on chunk position
        if foundIndex < 0 && !flattenedSentences.isEmpty {
            let approximateIndex = Int(Double(currentChunkIndex) / Double(textChunks.count) * Double(flattenedSentences.count))
            foundIndex = min(approximateIndex, flattenedSentences.count - 1)
        }

        if foundIndex >= 0 && foundIndex != previousIndex {
            currentPlayingIndex = foundIndex
            rebuildAttributedStrings()
        }
    }

    /// Current chunk index - use audio player's actual chapter index for accuracy
    var currentChunkIndex: Int {
        audioPlayer.currentChunkIndex
    }

    var canPlay: Bool {
        // Ready with a saved file OR actively streaming the first synthesized chunks.
        (item.status == .ready && item.audioFileURL != nil) || isStreamingAudio
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

        // If paused with audio loaded, resume playback only (don't re-synthesize)
        if audioPlayer.currentTime > 0 && audioPlayer.duration > 0 {
            audioPlayer.play()
            return
        }

        // Otherwise start synthesis from beginning in a Task so we can cancel it later
        synthesisTask = Task {
            await self.startSynthesisInternal()
        }
    }

    /// Internal synthesis that can be cancelled
    private func startSynthesisInternal() async {
        await startSynthesis()
    }

    /// Start synthesis from a specific chunk index
    func startSynthesisFromChunk(_ chunkIndex: Int) async {
        guard chunkIndex >= 0, chunkIndex < textChunks.count else {
            await startSynthesis()
            return
        }

        // Get text from this chunk to end
        let textFromChunk = textChunks[chunkIndex...].joined(separator: " ")

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
            if let path = selectedVoice.referenceAudioPath {
                refAudioURL = URL(fileURLWithPath: path)
            }

            // Pre-generate ALL chunks first to build up a queue
            var allChunkURLs: [URL] = []

            try await engine.synthesize(
                text: textFromChunk,
                referenceAudioURL: refAudioURL,
                outputURL: outputURL,
                onChunkReady: { chunkURL in
                    allChunkURLs.append(chunkURL)
                },
                onProgress: { [weak self] progress in
                    self?.synthesisProgress = progress
                },
                seed: synthesisSettings.seed
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
        isPaused.toggle()
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
        // Check both .ready and .processing status - if chunks exist, use them
        if (!item.generatedChunks.isEmpty || audioPlayer.hasAudioFiles) && item.status != .pending {
            NSLog("[PlayerVM] Using existing chunks for playback, status=\(item.status.rawValue)")
            // loadExistingChunks was already called in init, just start playing
            if audioPlayer.hasAudioFiles {
                audioPlayer.play()
                NSLog("[PlayerVM] Started playing chunks")
                return
            } else {
                NSLog("[PlayerVM] WARNING: chunks exist but audioPlayer.hasAudioFiles is false!")
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

            let outputURL = outputDir.appendingPathComponent("\(item.id.uuidString).wav")

            // Get reference audio for voice cloning
            var refAudioURL: URL?
            if let path = selectedVoice.referenceAudioPath {
                refAudioURL = URL(fileURLWithPath: path)
            }

            NSLog("[PlayerVM] Starting synthesizeStream, text length: \(item.textContent.count)")
            // Use async stream for proper backpressure handling
            let stream = engine.synthesizeStream(
                text: item.textContent,
                referenceAudioURL: refAudioURL,
                outputURL: outputURL
            )

            // Tell delegate to wait for more chunks if playback catches up
            audioPlayer.isExpectingMoreChunks = true
            NSLog("[PlayerVM] set isExpectingMoreChunks = true, starting stream")

            // Check for cancellation at start of each chunk iteration
            for try await chunk in stream {
                // Check if synthesis was cancelled (user paused)
                if Task.isCancelled {
                    NSLog("[PlayerVM] Synthesis cancelled by user")
                    break
                }

                // Update progress
                synthesisProgress = chunk.progress

                // Handle chunk playback with proper backpressure
                if chunk.isFirst {
                    // First chunk: start streaming playback
                    audioPlayer.startStreaming(
                        firstChunkURL: chunk.url,
                        title: item.title,
                        artist: item.displayAuthor
                    )
                    isStreamingAudio = true
                    // First chunk started
                } else {
                    // No backpressure - just append chunks as they arrive
                    NSLog("[PlayerVM] appending chunk, isFirst=\(chunk.isFirst), isLast=\(chunk.isLast), progress=\(chunk.progress)")
                    audioPlayer.appendStreamChunk(chunk.url)
                }
            }

            // Synthesis complete — persist the final concatenated file.
            NSLog("[PlayerVM] stream complete, setting isExpectingMoreChunks = false")
            audioPlayer.isExpectingMoreChunks = false
            item.audioFileURL = outputURL.path
            item.status = .ready
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

        // Check if already ready
        if item.status == .ready && item.audioFileURL != nil {
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
            if let path = selectedVoice.referenceAudioPath {
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
                    seed: synthesisSettings.seed
                )

                // Save chunk path
                item.generatedChunks[index] = outputURL.path
            }

            // Calculate and save total duration
            item.duration = TextChunker.estimateTotalDuration(for: item.textContent)

            // Mark as ready
            item.status = .ready

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
            if let path = selectedVoice.referenceAudioPath {
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
                seed: synthesisSettings.seed
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
            if let path = selectedVoice.referenceAudioPath {
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
                seed: synthesisSettings.seed
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

            let outputDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
                .appendingPathComponent("Audio", isDirectory: true)
            try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

            let outputURL = outputDir.appendingPathComponent("\(item.id.uuidString).wav")

            var refAudioURL: URL?
            if let path = selectedVoice.referenceAudioPath {
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
                seed: synthesisSettings.seed
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
