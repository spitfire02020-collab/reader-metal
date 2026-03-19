import Foundation
import UIKit
@preconcurrency import AVFoundation
import MediaPlayer
import Combine
import SQLite
import os.log

// MARK: - Logging

/// Structured logger for AudioPlayerService
/// Use os.Logger instead of NSLog for better production debugging with log levels
private let audioLogger = Logger(subsystem: "com.reader.app", category: "AudioPlayer")

// MARK: - Audio Player Service
// Handles audio playback with background support, Now Playing info, and controls

@MainActor
final class AudioPlayerService: NSObject, ObservableObject {
    static let shared = AudioPlayerService()

    // Playback state
    @Published var isPlaying = false
    @Published var currentTime: TimeInterval = 0
    @Published var duration: TimeInterval = 0
    @Published var progress: Double = 0
    @Published var playbackRate: Float = 1.0
    @Published var currentChunkIndex: Int = 0

    /// Check if audio files are loaded and ready for chapter navigation
    var hasAudioFiles: Bool {
        !audioFiles.isEmpty
    }

    /// Get the number of loaded audio files (chapters)
    var audioFileCount: Int {
        audioFiles.count
    }

    // Flag set by PlayerViewModel when synthesis is still in progress
    // Delegate uses this to decide whether to wait for more chunks
    @Published var isExpectingMoreChunks = false

    // Guard flag to prevent race condition between delegate and polling
    private var isTransitioningChapter = false

    /// Active poll task — cancel-before-replace to prevent concurrent polls
    private var pollTask: Task<Void, Never>? = nil

    /// Callback triggered when a new chunk starts playing - used for syncing text highlight
    var onChunkPlaybackStarted: ((Int) -> Void)?

    /// Current item ID for checking database for chunk progress
    private var currentItemID: UUID?

    // MARK: - Item Queue for Background Playback

    /// Queue of items waiting to play (stored as item IDs)
    @Published private(set) var playbackQueue: [UUID] = []

    /// Currently playing item ID
    @Published private(set) var currentPlayingItemID: UUID?

    /// Progress tracking per item (itemID -> progress 0-1)
    @Published private(set) var itemProgress: [UUID: Double] = [:]

    /// Synthesis progress tracking per item (itemID -> progress 0-1)
    /// This is updated during background generation (generateOnly)
    @Published private(set) var synthesisProgress: [UUID: Double] = [:]

    /// Update synthesis progress for an item
    func updateSynthesisProgress(_ itemID: UUID, progress: Double) {
        synthesisProgress[itemID] = progress
    }

    // MARK: - Combine Publishers for Notifications
    /// Combine cancellables for notification observers
    private var cancellables = Set<AnyCancellable>()

    /// Clear synthesis progress for an item (when done)
    func clearSynthesisProgress(_ itemID: UUID) {
        synthesisProgress.removeValue(forKey: itemID)
        // Also clear itemProgress to avoid stale playback progress showing
        itemProgress.removeValue(forKey: itemID)
    }

    /// Clear all loaded audio files and reset state for fresh playback
    /// Called when restarting synthesis with new voice/preset
    func clearAudioFiles() {
        audioFiles.removeAll()
        chunkDurations.removeAll()
        // Reset all playback state to avoid stale state causing issues
        duration = 0
        currentTime = 0
        progress = 0
        currentChunkIndex = 0
        isExpectingMoreChunks = false
        currentItemID = nil
        // Stop any currently playing audio
        audioPlayer?.stop()
        audioPlayer = nil
    }

    /// Set the current item ID for database tracking
    func setCurrentItemID(_ id: UUID?) {
        currentItemID = id
    }

    /// Get the number of completed chunks from database
    func getCompletedChunkCount() -> Int {
        guard let itemID = currentItemID else { return audioFiles.count }
        do {
            let completedChunks = try synthesisDB.getCompletedChunks(itemId: itemID.uuidString)
            return completedChunks.count
        } catch {
            return audioFiles.count
        }
    }

    /// Get the number of remaining chunks from database
    func getRemainingChunkCount() -> Int {
        guard let itemID = currentItemID else { return 0 }
        do {
            return try synthesisDB.getRemainingChunks(itemId: itemID.uuidString)
        } catch {
            return 0
        }
    }

    /// Clear playback state for an item - called when loading existing chunks
    func clearPlaybackState(_ itemID: UUID) {
        playbackQueue.removeAll { $0 == itemID }
        if currentPlayingItemID == itemID {
            currentPlayingItemID = nil
        }
        itemProgress.removeValue(forKey: itemID)
        synthesisProgress.removeValue(forKey: itemID)
    }

    /// Check if a specific item is in queue or playing
    func isItemQueued(_ itemID: UUID) -> Bool {
        currentPlayingItemID == itemID || playbackQueue.contains(itemID)
    }

    /// Get progress for a specific item
    func progressForItem(_ itemID: UUID) -> Double {
        itemProgress[itemID] ?? 0
    }

    /// Add item to playback queue
    func enqueueItem(_ itemID: UUID) {
        guard !isItemQueued(itemID) else { return }
        playbackQueue.append(itemID)
        itemProgress[itemID] = 0
        NSLog("[AudioPlayer] Enqueued item: \(itemID), queue: \(playbackQueue)")
    }

    /// Update progress for current item
    func updateItemProgress(_ itemID: UUID, progress: Double) {
        itemProgress[itemID] = progress
    }

    /// Mark item as complete and move to next
    func itemDidFinish(_ itemID: UUID) {
        // Remove from queue if still there
        if let index = playbackQueue.firstIndex(of: itemID) {
            playbackQueue.remove(at: index)
        }

        // Clear current if it's this item
        if currentPlayingItemID == itemID {
            currentPlayingItemID = nil
        }

        // Remove progress tracking
        itemProgress.removeValue(forKey: itemID)

        // Start next item if available
        if let nextItemID = playbackQueue.first {
            currentPlayingItemID = nextItemID
            // Notification will be sent to play next item
            NSLog("[AudioPlayer] Moving to next item: \(nextItemID)")
        }
    }

    /// Start playing an item (called when starting from library)
    func startPlayingItem(_ itemID: UUID) {
        // If something else is playing, enqueue this item
        if currentPlayingItemID != nil && currentPlayingItemID != itemID {
            enqueueItem(itemID)
        } else {
            currentPlayingItemID = itemID
            itemProgress[itemID] = 0
        }
    }

    // Crossfade settings (matching server: 20ms equal-power crossfade)
    private let crossfadeDuration: TimeInterval = 0.020  // 20ms

    // Audio engine for crossfade playback
    private var audioEngine: AVAudioEngine?
    private var playerNodeA: AVAudioPlayerNode?
    private var playerNodeB: AVAudioPlayerNode?
    private var mixerNode: AVAudioMixerNode?
    private var isUsingNodeA = true  // Toggle between nodes for crossfade

    // Current audio buffers
    private var bufferA: AVAudioPCMBuffer?
    private var bufferB: AVAudioPCMBuffer?

    // Legacy player for non-crossfade mode
    private var audioPlayer: AVAudioPlayer?
    private var displayLink: CADisplayLink?
    private var audioFiles: [URL] = []
    private(set) var chunkDurations: [TimeInterval] = []
    private var chapterTimestamps: [(start: TimeInterval, end: TimeInterval)] = []

    /// Cumulative duration of all chunks before the given index
    func cumulativeDurationUpToChunk(_ index: Int) -> TimeInterval {
        guard index > 0, !chunkDurations.isEmpty else { return 0 }
        let end = min(index, chunkDurations.count)
        return chunkDurations[..<end].reduce(0, +)
    }

    /// Total actual audio duration across all loaded chunks
    var totalActualDuration: TimeInterval {
        chunkDurations.reduce(0, +)
    }

    // Now Playing info
    private var nowPlayingTitle: String = ""
    private var nowPlayingArtist: String = ""
    private var nowPlayingImage: UIImage?

    /// Database for checking chunk progress
    private let synthesisDB = SynthesisDatabase.shared

    override init() {
        super.init()
        setupAudioSession()
        setupRemoteTransportControls()
        setupAudioEngine()
        setupAudioInterruptionObservers()

        // Listen for app lifecycle to properly manage audio session
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(appDidBecomeActive),
            name: UIApplication.didBecomeActiveNotification,
            object: nil
        )
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(appWillResignActive),
            name: UIApplication.willResignActiveNotification,
            object: nil
        )
    }

    @objc private func appDidBecomeActive() {
        reactivateAudioSession()
    }

    @objc private func appWillResignActive() {
        deactivateAudioSession()
    }

    // MARK: - Audio Session Setup

    private func setupAudioSession() {
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .spokenAudio, options: [.allowBluetoothA2DP])
            try session.setActive(true)
        } catch {
            print("Audio session setup failed: \(error)")
        }
    }

    /// Called when app becomes active - reconfigure audio session
    func reactivateAudioSession() {
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setActive(true)
        } catch {
            print("Audio session reactivation failed: \(error)")
        }
    }

    /// Called when app resigns active - deactivate audio session so other apps can play
    func deactivateAudioSession() {
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setActive(false, options: .notifyOthersOnDeactivation)
        } catch {
            print("Audio session deactivation failed: \(error)")
        }
    }

    // MARK: - Audio Interruption Handling

    /// Setup audio session interruption and route change observers
    private func setupAudioInterruptionObservers() {
        // Audio interruption (phone calls, Siri, other apps)
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleAudioInterruption),
            name: AVAudioSession.interruptionNotification,
            object: AVAudioSession.sharedInstance()
        )

        // Memory warning - release non-essential resources
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleMemoryWarning),
            name: UIApplication.didReceiveMemoryWarningNotification,
            object: nil
        )

        // Audio route change (headphones disconnected, etc.)
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleRouteChange),
            name: AVAudioSession.routeChangeNotification,
            object: AVAudioSession.sharedInstance()
        )
    }

    /// Handle audio session interruptions (phone calls, Siri, etc.)
    @objc private func handleAudioInterruption(_ notification: Notification) {
        guard let userInfo = notification.userInfo,
              let typeValue = userInfo[AVAudioSessionInterruptionTypeKey] as? UInt,
              let type = AVAudioSession.InterruptionType(rawValue: typeValue) else {
            return
        }

        NSLog("[AudioPlayer] Interruption type: \(type)")

        switch type {
        case .began:
            // Audio interrupted - pause playback
            NSLog("[AudioPlayer] Interruption began - pausing")
            pause()

        case .ended:
            // Interruption ended - check if we should resume
            guard let optionsValue = userInfo[AVAudioSessionInterruptionOptionKey] as? UInt else {
                return
            }

            let options = AVAudioSession.InterruptionOptions(rawValue: optionsValue)
            if options.contains(.shouldResume) {
                // Resume playback
                NSLog("[AudioPlayer] Interruption ended - resuming playback")
                play()
            }

        @unknown default:
            NSLog("[AudioPlayer] Unknown interruption type")
        }
    }

    /// Handle audio route changes (headphones disconnected, etc.)
    @objc private func handleRouteChange(_ notification: Notification) {
        guard let userInfo = notification.userInfo,
              let reasonValue = userInfo[AVAudioSessionRouteChangeReasonKey] as? UInt,
              let reason = AVAudioSession.RouteChangeReason(rawValue: reasonValue) else {
            return
        }

        NSLog("[AudioPlayer] Route change reason: \(reason)")

        switch reason {
        case .oldDeviceUnavailable:
            // Headphones were unplugged - pause playback
            NSLog("[AudioPlayer] Audio device removed - pausing playback")
            pause()

        case .newDeviceAvailable:
            // New audio device connected (e.g., headphones)
            NSLog("[AudioPlayer] New audio device available")

        default:
            break
        }
    }

    /// Handle memory warning - log but do NOT clear audioFiles (they are URLs, not data)
    @objc private func handleMemoryWarning() {
        NSLog("[AudioPlayer] Memory warning received - audioFiles are URLs only, nothing to release")
    }

    // MARK: - Audio Engine Setup (for crossfade)

    private func setupAudioEngine() {
        audioEngine = AVAudioEngine()
        playerNodeA = AVAudioPlayerNode()
        playerNodeB = AVAudioPlayerNode()
        mixerNode = AVAudioMixerNode()

        guard let engine = audioEngine,
              let nodeA = playerNodeA,
              let nodeB = playerNodeB,
              let mixer = mixerNode else { return }

        engine.attach(nodeA)
        engine.attach(nodeB)
        engine.attach(mixer)

        // Connect: playerNodeA -> mixer -> mainMixerNode -> output
        //          playerNodeB -> mixer -> mainMixerNode -> output
        let format = AVAudioFormat(standardFormatWithSampleRate: 24000, channels: 1)!

        engine.connect(nodeA, to: mixer, format: format)
        engine.connect(nodeB, to: mixer, format: format)
        engine.connect(mixer, to: engine.mainMixerNode, format: format)

        // Set initial volumes
        nodeA.volume = 1.0
        nodeB.volume = 0.0

        do {
            try engine.start()
        } catch {
            audioLogger.error("Failed to start audio engine: \(error.localizedDescription)")
        }
    }

    /// Load audio file into AVAudioPCMBuffer
    private func loadAudioBuffer(from url: URL) -> AVAudioPCMBuffer? {
        do {
            let file = try AVAudioFile(forReading: url)
            let format = file.processingFormat
            let frameCount = AVAudioFrameCount(file.length)

            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
                return nil
            }

            try file.read(into: buffer)
            return buffer
        } catch {
            audioLogger.error("Failed to load audio buffer: \(error.localizedDescription)")
            return nil
        }
    }

    /// Perform crossfade between current and next chunk
    private func performCrossfade(to nextBuffer: AVAudioPCMBuffer, completion: @escaping () -> Void) {
        guard audioEngine != nil,
              let nodeA = playerNodeA,
              let nodeB = playerNodeB else {
            completion()
            return
        }

        let activeNode = isUsingNodeA ? nodeA : nodeB
        let incomingNode = isUsingNodeA ? nodeB : nodeA

        // Load next buffer into incoming node
        if isUsingNodeA {
            bufferB = nextBuffer
        } else {
            bufferA = nextBuffer
        }

        // Schedule the next buffer
        incomingNode.scheduleBuffer(nextBuffer, at: nil, options: []) {
            // Buffer finished
        }

        // Equal-power crossfade: cos² fade out, sin² fade in
        let fadeSteps = 20
        let fadeInterval = crossfadeDuration / Double(fadeSteps)

        for i in 0...fadeSteps {
            let progress = Float(i) / Float(fadeSteps)
            // Equal-power curves: cos² for fade out, sin² for fade in
            let fadeOutVolume = cos(progress * .pi / 2)
            let fadeInVolume = sin(progress * .pi / 2)

            DispatchQueue.main.asyncAfter(deadline: .now() + fadeInterval * Double(i)) {
                activeNode.volume = fadeOutVolume
                incomingNode.volume = fadeInVolume
            }
        }

        // Start playing incoming node after crossfade starts
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.001) {
            incomingNode.play()
        }

        // Toggle for next crossfade
        isUsingNodeA.toggle()

        // Stop and reset active node after crossfade
        DispatchQueue.main.asyncAfter(deadline: .now() + crossfadeDuration) {
            activeNode.stop()
            completion()
        }
    }

    // MARK: - Remote Controls (Lock Screen / Control Center)

    private func setupRemoteTransportControls() {
        let center = MPRemoteCommandCenter.shared()

        center.playCommand.addTarget { [weak self] _ in
            self?.play()
            return .success
        }

        center.pauseCommand.addTarget { [weak self] _ in
            self?.pause()
            return .success
        }

        center.togglePlayPauseCommand.addTarget { [weak self] _ in
            self?.togglePlayPause()
            return .success
        }

        center.skipForwardCommand.preferredIntervals = [15]
        center.skipForwardCommand.addTarget { [weak self] _ in
            self?.skipForward(seconds: 15)
            return .success
        }

        center.skipBackwardCommand.preferredIntervals = [15]
        center.skipBackwardCommand.addTarget { [weak self] _ in
            self?.skipBackward(seconds: 15)
            return .success
        }

        center.changePlaybackPositionCommand.addTarget { [weak self] event in
            if let event = event as? MPChangePlaybackPositionCommandEvent {
                self?.seek(to: event.positionTime)
            }
            return .success
        }

        center.changePlaybackRateCommand.supportedPlaybackRates = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        center.changePlaybackRateCommand.addTarget { [weak self] event in
            if let event = event as? MPChangePlaybackRateCommandEvent {
                self?.setPlaybackRate(event.playbackRate)
            }
            return .success
        }

        center.nextTrackCommand.addTarget { [weak self] _ in
            self?.nextChapter()
            return .success
        }

        center.previousTrackCommand.addTarget { [weak self] _ in
            self?.previousChapter()
            return .success
        }
    }

    // MARK: - Load Audio

    /// Load a single audio file for playback
    func loadAudio(url: URL, title: String, artist: String, coverImage: UIImage? = nil) {
        nowPlayingTitle = title
        nowPlayingArtist = artist
        nowPlayingImage = coverImage
        audioFiles = [url]
        chunkDurations = []
        chapterTimestamps = []

        do {
            audioPlayer = try AVAudioPlayer(contentsOf: url)
            audioPlayer?.delegate = self
            audioPlayer?.prepareToPlay()
            audioPlayer?.enableRate = true
            audioPlayer?.rate = playbackRate
            duration = audioPlayer?.duration ?? 0
            currentTime = 0
            progress = 0

            updateNowPlayingInfo()
        } catch {
            print("Failed to load audio: \(error)")
        }
    }

    /// Load a persistent chunk into the audio queue
    func loadChunk(_ url: URL) {
        guard FileManager.default.fileExists(atPath: url.path) else { return }

        if audioFiles.isEmpty {
            // First chunk - load as main audio
            audioFiles = [url]
            do {
                audioPlayer = try AVAudioPlayer(contentsOf: url)
                audioPlayer?.delegate = self
                audioPlayer?.prepareToPlay()
                duration = audioPlayer?.duration ?? 0
                chunkDurations = [duration]
            } catch {
                audioLogger.error("Failed to load chunk: \(error.localizedDescription)")
            }
        } else {
            // Additional chunk - append to queue
            audioFiles.append(url)
            // Update duration using AVAudioFile (more efficient than AVAudioPlayer)
            do {
                let audioFile = try AVAudioFile(forReading: url)
                let fileDuration = Double(audioFile.length) / audioFile.fileFormat.sampleRate
                duration += fileDuration
                chunkDurations.append(fileDuration)
            } catch {
                audioLogger.error("Failed to get chunk duration: \(error.localizedDescription)")
            }
        }
    }

    /// Load multiple chapter audio files
    func loadChapters(urls: [URL], title: String, artist: String, coverImage: UIImage? = nil) {
        audioFiles = urls
        nowPlayingTitle = title
        nowPlayingArtist = artist
        nowPlayingImage = coverImage
        currentChunkIndex = 0

        if let firstURL = urls.first {
            loadAudio(url: firstURL, title: title, artist: artist, coverImage: coverImage)
        }
    }

    // MARK: - Progressive Streaming

    /// Start playback from the first synthesized chunk and prepare for more chunks to follow.
    /// Call `appendStreamChunk(_:)` for each subsequent chunk as it becomes available.
    /// The `AVAudioPlayerDelegate` auto-advances through the queue as each chunk finishes.
    func startStreaming(firstChunkURL: URL, title: String, artist: String) {
        NSLog("[AudioPlayer] startStreaming called with URL: \(firstChunkURL.lastPathComponent)")
        nowPlayingTitle  = title
        nowPlayingArtist = artist
        nowPlayingImage  = nil
        audioFiles       = [firstChunkURL]
        chunkDurations   = []
        chapterTimestamps = []
        currentChunkIndex = 0
        isExpectingMoreChunks = true  // Will be set to false after all chunks appended
        NSLog("[AudioPlayer] startStreaming: audioFiles initialized to 1 file, isExpectingMoreChunks = true")

        do {
            audioPlayer = try AVAudioPlayer(contentsOf: firstChunkURL)
            NSLog("[AudioPlayer] Created player, duration: \(audioPlayer?.duration ?? 0)")
            audioPlayer?.delegate = self
            audioPlayer?.numberOfLoops = 0  // Don't loop - play once then advance to next chunk
            audioPlayer?.prepareToPlay()
            audioPlayer?.enableRate = true
            audioPlayer?.rate = playbackRate
            duration    = audioPlayer?.duration ?? 0
            chunkDurations = [duration]
            currentTime = 0
            progress    = 0
            updateNowPlayingInfo()
            NSLog("[AudioPlayer] Calling play(), isPlaying will be: \(audioPlayer?.isPlaying ?? false)")
            play()
            NSLog("[AudioPlayer] After play(), isPlaying: \(audioPlayer?.isPlaying ?? false)")
        } catch {
            audioLogger.error("Failed to start streaming: \(error.localizedDescription)")
        }
    }

    /// Append a newly synthesized chunk URL to the playback queue.
    /// Called on each subsequent chunk while the current one is still playing.
    func appendStreamChunk(_ url: URL) {
        NSLog("[AudioPlayer] appendStreamChunk: \(url.lastPathComponent), current count: \(audioFiles.count)")
        audioFiles.append(url)
        NSLog("[AudioPlayer] appendStreamChunk: appended, total files now: \(audioFiles.count)")

        // Update total duration using AVAudioFile (more efficient)
        do {
            let audioFile = try AVAudioFile(forReading: url)
            let fileDuration = Double(audioFile.length) / audioFile.fileFormat.sampleRate
            duration += fileDuration
            chunkDurations.append(fileDuration)
            NSLog("[AudioPlayer] Updated total duration: \(duration)")
        } catch {
            audioLogger.error("Failed to get chunk duration: \(error.localizedDescription)")
        }
    }

    /// Reload completed chunks from database that haven't been loaded yet
    /// Called during polling to catch chunks generated between delegate callbacks
    func reloadPendingChunks() {
        guard let itemID = currentItemID else { return }

        do {
            let completedChunks = try synthesisDB.getCompletedChunks(itemId: itemID.uuidString)
            let existingPaths = Set(audioFiles.map { $0.path })

            for chunk in completedChunks {
                guard let filePath = chunk.filePath else { continue }
                let chunkURL = URL(fileURLWithPath: filePath)

                // Skip if already loaded
                guard !existingPaths.contains(chunkURL.path) else { continue }

                NSLog("[AudioPlayer] reloadPendingChunks: loading chunk \(chunk.chunkIndex)")
                audioFiles.append(chunkURL)

                // Update duration
                if let player = try? AVAudioPlayer(contentsOf: chunkURL) {
                    player.prepareToPlay()
                    duration += player.duration
                    chunkDurations.append(player.duration)
                }
            }

            NSLog("[AudioPlayer] reloadPendingChunks: total loaded: \(audioFiles.count)")
        } catch {
            NSLog("[AudioPlayer] reloadPendingChunks: error \(error)")
        }
    }

    // MARK: - Playback Controls

    func play() {
        guard let player = audioPlayer else {
            NSLog("[AudioPlayer] play() FAILED: audioPlayer is nil")
            return
        }
        NSLog("[AudioPlayer] play() called, audioPlayer is nil: false")
        player.rate = playbackRate
        let playResult = player.play()
        NSLog("[AudioPlayer] play() result: \(String(describing: playResult)), isPlaying: \(player.isPlaying)")
        isPlaying = true
        startProgressUpdates()
        updateNowPlayingInfo()

        // Trigger highlight sync callback - fires when audio actually starts playing
        let chunkIndex = currentChunkIndex
        onChunkPlaybackStarted?(chunkIndex)
    }

    func pause() {
        guard let player = audioPlayer else {
            NSLog("[AudioPlayer] pause() FAILED: audioPlayer is nil")
            return
        }
        player.pause()
        isPlaying = false
        // Note: do NOT set isExpectingMoreChunks = false here.
        // Pausing audio playback should not claim synthesis is complete.
        stopProgressUpdates()
        updateNowPlayingInfo()
    }

    func togglePlayPause() {
        if isPlaying { pause() } else { play() }
    }

    func stop() {
        guard let player = audioPlayer else {
            NSLog("[AudioPlayer] stop() FAILED: audioPlayer is nil")
            return
        }
        player.stop()
        player.currentTime = 0
        isExpectingMoreChunks = false  // Reset flag when stopping
        DispatchQueue.main.async { [weak self] in
            self?.isPlaying = false
            self?.currentTime = 0
            self?.progress = 0
        }
        stopProgressUpdates()
    }

    func seek(to globalTime: TimeInterval) {
        let maxTime = totalActualDuration
        let clampedTime = max(0, min(globalTime, maxTime))
        
        // Find which chunk this global time falls into
        var accumulated: TimeInterval = 0
        var targetIndex = 0
        
        for (index, chunkDur) in chunkDurations.enumerated() {
            if accumulated + chunkDur > clampedTime {
                targetIndex = index
                break
            }
            accumulated += chunkDur
            targetIndex = index // Fallback to last chunk if exactly at end
        }
        
        let localTimeInChunk = max(0, clampedTime - accumulated)
        
        if targetIndex != currentChunkIndex || audioPlayer == nil {
            // Need to load a different chunk
            let wasPlaying = isPlaying
            goToChapter(targetIndex)
            audioPlayer?.currentTime = localTimeInChunk
            if wasPlaying && audioPlayer?.isPlaying == false {
                audioPlayer?.play()
            }
        } else {
            // Same chunk, just seek
            audioPlayer?.currentTime = localTimeInChunk
        }
        
        currentTime = localTimeInChunk
        duration = maxTime
        progress = maxTime > 0 ? clampedTime / maxTime : 0
        updateNowPlayingInfo()
    }

    func seekToProgress(_ value: Double) {
        let globalTime = value * totalActualDuration
        seek(to: globalTime)
    }

    func skipForward(seconds: TimeInterval = 15) {
        seek(to: currentTime + seconds)
    }

    func skipBackward(seconds: TimeInterval = 15) {
        seek(to: currentTime - seconds)
    }

    func setPlaybackRate(_ rate: Float) {
        playbackRate = rate
        audioPlayer?.rate = rate
        updateNowPlayingInfo()
    }

    // MARK: - Chapter Navigation

    func nextChapter() {
        NSLog("[AudioPlayer] nextChapter CALLED: current=\(currentChunkIndex), totalFiles=\(audioFiles.count)")
        // Guard against race condition: prevent double-calls from delegate + polling
        guard !isTransitioningChapter else {
            NSLog("[AudioPlayer] nextChapter: SKIPPED (already transitioning)")
            return
        }
        isTransitioningChapter = true
        defer { isTransitioningChapter = false }

        guard currentChunkIndex + 1 < audioFiles.count else {
            NSLog("[AudioPlayer] nextChapter: NO MORE CHAPTERS, stopping")
            isPlaying = false
            isExpectingMoreChunks = false  // Reset state
            stopProgressUpdates()
            return
        }
        currentChunkIndex += 1
        NSLog("[AudioPlayer] nextChapter: advancing to chapter \(currentChunkIndex)")
        loadAndPlayChapter(at: currentChunkIndex)
    }

    func previousChapter() {
        // If more than 3 seconds in, restart current chapter
        if currentTime > 3 {
            seek(to: 0)
            return
        }
        guard currentChunkIndex > 0 else { return }
        currentChunkIndex -= 1
        loadAndPlayChapter(at: currentChunkIndex)
    }

    func goToChapter(_ index: Int) {
        guard index >= 0, index < audioFiles.count else { return }
        currentChunkIndex = index
        loadAndPlayChapter(at: index)
    }

    private func loadAndPlayChapter(at index: Int) {
        NSLog("[AudioPlayer] loadAndPlayChapter: index=\(index), file=\(audioFiles[index].lastPathComponent)")
        let wasPlaying = isPlaying
        do {
            audioPlayer = try AVAudioPlayer(contentsOf: audioFiles[index])
            NSLog("[AudioPlayer] loadAndPlayChapter: created player, duration=\(audioPlayer?.duration ?? 0)")
            audioPlayer?.delegate = self
            audioPlayer?.numberOfLoops = 0
            audioPlayer?.prepareToPlay()
            audioPlayer?.enableRate = true
            audioPlayer?.rate = playbackRate
            duration = audioPlayer?.duration ?? 0
            currentTime = 0
            progress = 0

            updateNowPlayingInfo()

            if wasPlaying {
                NSLog("[AudioPlayer] loadAndPlayChapter: calling play()")
                play()
            }
        } catch {
            NSLog("[AudioPlayer] loadAndPlayChapter: FAILED - \(error)")
        }
    }

    // MARK: - Progress Updates

    private func startProgressUpdates() {
        stopProgressUpdates()
        displayLink = CADisplayLink(target: self, selector: #selector(updateProgress))
        displayLink?.preferredFrameRateRange = CAFrameRateRange(minimum: 10, maximum: 30)
        displayLink?.add(to: .main, forMode: .common)
    }

    private func stopProgressUpdates() {
        displayLink?.invalidate()
        displayLink = nil
    }

    @objc private func updateProgress() {
        guard let player = audioPlayer else { return }
        currentTime = player.currentTime
        // Maintain global duration instead of overwriting with single chunk duration
        let maxTime = totalActualDuration
        duration = maxTime > 0 ? maxTime : player.duration
        
        let globalCurrentTime = cumulativeDurationUpToChunk(currentChunkIndex) + currentTime
        progress = duration > 0 ? globalCurrentTime / duration : 0
    }

    // MARK: - Now Playing Info

    private func updateNowPlayingInfo() {
        var info: [String: Any] = [
            MPMediaItemPropertyTitle: nowPlayingTitle,
            MPMediaItemPropertyArtist: nowPlayingArtist,
            MPNowPlayingInfoPropertyElapsedPlaybackTime: currentTime,
            MPMediaItemPropertyPlaybackDuration: duration,
            MPNowPlayingInfoPropertyPlaybackRate: isPlaying ? playbackRate : 0,
            MPNowPlayingInfoPropertyDefaultPlaybackRate: 1.0,
        ]

        if let image = nowPlayingImage {
            let artwork = MPMediaItemArtwork(boundsSize: image.size) { _ in image }
            info[MPMediaItemPropertyArtwork] = artwork
        }

        MPNowPlayingInfoCenter.default().nowPlayingInfo = info
    }

    // MARK: - Waveform Generation

    // MARK: - Formatted Time

    var formattedCurrentTime: String { formatTime(currentTime) }
    var formattedDuration: String { formatTime(duration) }
    var formattedRemaining: String { formatTime(max(0, duration - currentTime)) }

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
}

// MARK: - AVAudioPlayerDelegate

/// Polling configuration for waiting on synthesis chunks
private enum ChunkPolling {
    static let maxIterations = 60
    static let sleepIntervalNs: UInt64 = 500_000_000  // 0.5 seconds
}

extension AudioPlayerService: AVAudioPlayerDelegate {
    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        NSLog("[AudioPlayer] delegate: didFinishPlaying called, flag=\(flag)")
        Task { @MainActor in
            // Re-read values at decision time to avoid stale closure captures
            let currentIndex = self.currentChunkIndex
            let totalFiles = self.audioFiles.count
            let expectingMore = self.isExpectingMoreChunks

            NSLog("[AudioPlayer] delegate CHECK: currentIndex=\(currentIndex), totalFiles=\(totalFiles), expectingMore=\(expectingMore)")

            NSLog("[AudioPlayer] delegate: captured chapter=\(currentIndex), totalFiles=\(totalFiles), expectingMore=\(expectingMore)")
            NSLog("[AudioPlayer] delegate: audioFiles = \(self.audioFiles.map { $0.lastPathComponent })")
            NSLog("[AudioPlayer] delegate: currentChunkIndex now = \(self.currentChunkIndex)")

            // Guard: Don't advance if already at/past available chunks
            // This prevents re-playing chunks when audioFiles array updates mid-playback
            let moreChunksAvailable = currentIndex + 1 < totalFiles
            NSLog("[AudioPlayer] delegate: condition (currentIndex + 1 < totalFiles) = \(moreChunksAvailable) (\(currentIndex) + 1 < \(totalFiles))")

            if moreChunksAvailable {
                // More chunks available - advance to next
                NSLog("[AudioPlayer] delegate: calling nextChapter() - should advance from \(currentIndex)")
                self.nextChapter()
            } else if expectingMore {
                // No more chunks currently but synthesis may still be going
                // Use database to check, not the in-memory flag
                let remaining = self.getRemainingChunkCount()
                NSLog("[AudioPlayer] delegate: no more chunks yet, remaining in DB=\(remaining)")

                if remaining > 0 || expectingMore {
                    NSLog("[AudioPlayer] delegate: waiting for synthesis (remaining=\(remaining))...")
                    // Cancel any previous poll before starting a new one
                    self.pollTask?.cancel()
                    self.pollTask = Task.detached { [weak self] in
                        guard let self = self else { return }
                        await self.pollForNewChunks()
                    }
                } else {
                    // No more chunks and synthesis complete - stop
                    NSLog("[AudioPlayer] delegate: playback complete, stopping")
                    self.isPlaying = false
                }
            } else {
                // Neither more chunks available nor expecting more from synthesis
                // Final check: query DB in case isExpectingMoreChunks was cleared prematurely
                let remaining = self.getRemainingChunkCount()
                if remaining > 0 {
                    NSLog("[AudioPlayer] delegate: flags say done but DB shows \(remaining) remaining, polling")
                    self.pollTask?.cancel()
                    self.pollTask = Task.detached { [weak self] in
                        guard let self = self else { return }
                        await self.pollForNewChunks()
                    }
                } else {
                    NSLog("[AudioPlayer] delegate: playback complete, stopping")
                    self.isPlaying = false
                }
            }
        }
    }

    /// Polls for new chunks in background without blocking MainActor.
    /// Uses the database as the source of truth for chunk availability.
    /// Only one poll task should be active at a time (enforced by cancel-before-replace in delegate).
    private func pollForNewChunks() async {
        var previousChunkCount = self.getCompletedChunkCount()

        NSLog("[AudioPlayer] pollForNewChunks: starting DB-driven polling (completed=\(previousChunkCount))")

        for iteration in 0..<ChunkPolling.maxIterations {
            // Exit early if this poll was cancelled (another one replaced us)
            if Task.isCancelled {
                NSLog("[AudioPlayer] pollForNewChunks[\(iteration)]: cancelled, exiting")
                return
            }

            try? await Task.sleep(nanoseconds: ChunkPolling.sleepIntervalNs)

            if Task.isCancelled { return }

            let newChunkCount = self.getCompletedChunkCount()
            let remaining = self.getRemainingChunkCount()

            NSLog("[AudioPlayer] pollForNewChunks[\(iteration)]: completed=\(newChunkCount), remaining=\(remaining)")

            // New chunks arrived — reload and try to advance
            if newChunkCount > previousChunkCount {
                NSLog("[AudioPlayer] pollForNewChunks: new chunk! \(previousChunkCount)->\(newChunkCount)")
                previousChunkCount = newChunkCount

                let advanced = await MainActor.run { () -> Bool in
                    self.reloadPendingChunks()
                    if self.currentChunkIndex + 1 < self.audioFiles.count {
                        NSLog("[AudioPlayer] pollForNewChunks: advancing to next chapter")
                        self.nextChapter()
                        return true
                    }
                    return false
                }
                if advanced { return }  // delegate will spawn a new poll if needed
                continue
            }

            // Check if we can advance (chunks may have been appended by the callback)
            let canAdvance = await MainActor.run {
                self.currentChunkIndex + 1 < self.audioFiles.count
            }
            if canAdvance {
                await MainActor.run { self.nextChapter() }
                return
            }

            // Synthesis complete (DB says nothing remaining) — stop waiting
            if remaining == 0 {
                NSLog("[AudioPlayer] pollForNewChunks: DB says synthesis done (remaining=0)")
                break
            }
        }

        // Final check after loop exits (either break or maxIterations)
        if Task.isCancelled { return }

        await MainActor.run {
            self.reloadPendingChunks()
            if self.currentChunkIndex + 1 < self.audioFiles.count {
                NSLog("[AudioPlayer] pollForNewChunks: final check — chunks available, resuming")
                self.nextChapter()
            } else {
                // Truly done or timed out
                let remaining = self.getRemainingChunkCount()
                if remaining == 0 {
                    NSLog("[AudioPlayer] pollForNewChunks: playback complete")
                    self.isPlaying = false
                    self.stopProgressUpdates()
                } else {
                    // Timed out but synthesis still running — DON'T kill playback.
                    // Restart the poll instead.
                    NSLog("[AudioPlayer] pollForNewChunks: poll timed out but \(remaining) chunks remaining, restarting poll")
                    self.pollTask?.cancel()
                    self.pollTask = Task.detached { [weak self] in
                        guard let self = self else { return }
                        await self.pollForNewChunks()
                    }
                }
            }
        }
    }
}
