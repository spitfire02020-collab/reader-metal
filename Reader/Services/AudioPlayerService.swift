import Foundation
import UIKit
@preconcurrency import AVFoundation
import MediaPlayer
import Combine

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

    // Waveform data for visualization
    @Published var waveformSamples: [Float] = []

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
    private var nextChunkBuffer: AVAudioPCMBuffer?  // Pre-loaded next chunk

    // Legacy player for non-crossfade mode
    private var audioPlayer: AVAudioPlayer?
    private var displayLink: CADisplayLink?
    private var audioFiles: [URL] = []
    private var chapterTimestamps: [(start: TimeInterval, end: TimeInterval)] = []

    // Now Playing info
    private var nowPlayingTitle: String = ""
    private var nowPlayingArtist: String = ""
    private var nowPlayingImage: UIImage?

    private override init() {
        super.init()
        setupAudioSession()
        setupRemoteTransportControls()
        setupAudioEngine()

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
            NSLog("[AudioPlayer] Failed to start audio engine: \(error)")
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
            NSLog("[AudioPlayer] Failed to load audio buffer: \(error)")
            return nil
        }
    }

    /// Perform crossfade between current and next chunk
    private func performCrossfade(to nextBuffer: AVAudioPCMBuffer, completion: @escaping () -> Void) {
        guard let engine = audioEngine,
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
        incomingNode.scheduleBuffer(nextBuffer, at: nil, options: []) { [weak self] in
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

            DispatchQueue.main.asyncAfter(deadline: .now() + fadeInterval * Double(i)) { [weak self] in
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
        DispatchQueue.main.asyncAfter(deadline: .now() + crossfadeDuration) { [weak self] in
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

            generateWaveformData(from: url)
            updateNowPlayingInfo()
        } catch {
            print("Failed to load audio: \(error)")
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
            currentTime = 0
            progress    = 0
            generateWaveformData(from: firstChunkURL)
            updateNowPlayingInfo()
            NSLog("[AudioPlayer] Calling play(), isPlaying will be: \(audioPlayer?.isPlaying ?? false)")
            play()
            NSLog("[AudioPlayer] After play(), isPlaying: \(audioPlayer?.isPlaying ?? false)")
        } catch {
            NSLog("[AudioPlayer] Failed to start streaming: \(error)")
        }
    }

    /// Append a newly synthesized chunk URL to the playback queue.
    /// Called on each subsequent chunk while the current one is still playing.
    func appendStreamChunk(_ url: URL) {
        NSLog("[AudioPlayer] appendStreamChunk: \(url.lastPathComponent), current count: \(audioFiles.count)")
        audioFiles.append(url)
        NSLog("[AudioPlayer] appendStreamChunk: appended, total files now: \(audioFiles.count)")

        // Update total duration by calculating duration of new chunk
        do {
            let player = try AVAudioPlayer(contentsOf: url)
            player.prepareToPlay()
            duration += player.duration
            NSLog("[AudioPlayer] Updated total duration: \(duration)")
        } catch {
            NSLog("[AudioPlayer] Failed to get chunk duration: \(error)")
        }
    }

    // MARK: - Playback Controls

    func play() {
        audioPlayer?.rate = playbackRate
        audioPlayer?.play()
        isPlaying = true
        startProgressUpdates()
        updateNowPlayingInfo()
    }

    func pause() {
        audioPlayer?.pause()
        isPlaying = false
        stopProgressUpdates()
        updateNowPlayingInfo()
    }

    func togglePlayPause() {
        if isPlaying { pause() } else { play() }
    }

    func stop() {
        audioPlayer?.stop()
        audioPlayer?.currentTime = 0
        isPlaying = false
        currentTime = 0
        progress = 0
        stopProgressUpdates()
    }

    func seek(to time: TimeInterval) {
        let clampedTime = max(0, min(time, duration))
        audioPlayer?.currentTime = clampedTime
        currentTime = clampedTime
        progress = duration > 0 ? clampedTime / duration : 0
        updateNowPlayingInfo()
    }

    func seekToProgress(_ value: Double) {
        let time = value * duration
        seek(to: time)
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
        guard currentChunkIndex + 1 < audioFiles.count else {
            NSLog("[AudioPlayer] nextChapter: NO MORE CHAPTERS, stopping")
            isPlaying = false
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

            generateWaveformData(from: audioFiles[index])
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
        duration = player.duration
        progress = duration > 0 ? currentTime / duration : 0
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

    private func generateWaveformData(from url: URL) {
        let capturedURL = url
        Task.detached {
            do {
                let file = try AVAudioFile(forReading: capturedURL)
                let frameCount = AVAudioFrameCount(file.length)
                let format = file.processingFormat
                guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else { return }
                try file.read(into: buffer)

                guard let channelData = buffer.floatChannelData?[0] else { return }

                let sampleCount = 200 // Number of bars in waveform
                let samplesPerBar = max(1, Int(frameCount) / sampleCount)
                var bars: [Float] = []

                for i in 0..<sampleCount {
                    let start = i * samplesPerBar
                    let end = min(start + samplesPerBar, Int(frameCount))
                    guard start < end else {
                        bars.append(0)
                        continue
                    }

                    var maxAmp: Float = 0
                    for j in start..<end {
                        let amp = abs(channelData[j])
                        if amp > maxAmp { maxAmp = amp }
                    }
                    bars.append(maxAmp)
                }

                // Normalize
                let peak = bars.max() ?? 1
                let normalizedBars: [Float]
                if peak > 0 {
                    normalizedBars = bars.map { $0 / peak }
                } else {
                    normalizedBars = bars
                }

                await MainActor.run { [normalizedBars] in
                    self.waveformSamples = normalizedBars
                }
            } catch {
                let randomBars: [Float] = (0..<200).map { _ in Float.random(in: 0.1...0.8) }
                await MainActor.run { [randomBars] in
                    self.waveformSamples = randomBars
                }
            }
        }
    }

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

extension AudioPlayerService: AVAudioPlayerDelegate {
    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        NSLog("[AudioPlayer] delegate: didFinishPlaying called, flag=\(flag)")
        Task { @MainActor in
            // Capture current state at the time of delegate callback
            let currentIndex = self.currentChunkIndex
            let totalFiles = self.audioFiles.count
            let expectingMore = self.isExpectingMoreChunks

            NSLog("[AudioPlayer] delegate CHECK: currentIndex=\(currentIndex), totalFiles=\(totalFiles), expectingMore=\(expectingMore)")

            NSLog("[AudioPlayer] delegate: captured chapter=\(currentIndex), totalFiles=\(totalFiles), expectingMore=\(expectingMore)")
            NSLog("[AudioPlayer] delegate: audioFiles = \(self.audioFiles.map { $0.lastPathComponent })")
            NSLog("[AudioPlayer] delegate: currentChunkIndex now = \(self.currentChunkIndex)")

            let moreChunksAvailable = currentIndex + 1 < totalFiles
            NSLog("[AudioPlayer] delegate: condition (currentIndex + 1 < totalFiles) = \(moreChunksAvailable) (\(currentIndex) + 1 < \(totalFiles))")

            if moreChunksAvailable {
                // More chunks available - advance to next
                NSLog("[AudioPlayer] delegate: calling nextChapter() - should advance from \(currentIndex)")
                self.nextChapter()
            } else if expectingMore {
                // No more chunks currently but synthesis is still going
                // Wait and poll for new chunks
                NSLog("[AudioPlayer] delegate: no more chunks yet, waiting for synthesis...")
                // Don't set isPlaying = false here! We'll resume playback when next chunk arrives
                // and loadAndPlayChapter checks wasPlaying

                // Poll for new chunks arriving
                for pollIteration in 0..<60 { // Wait up to 30 seconds
                    try? await Task.sleep(nanoseconds: 500_000_000) // 0.5s

                    let newTotalFiles = self.audioFiles.count
                    let newExpectingMore = self.isExpectingMoreChunks
                    let currentNow = self.currentChunkIndex

                    NSLog("[AudioPlayer] delegate: poll[\(pollIteration)] - totalFiles=\(newTotalFiles) (was \(totalFiles)), expectingMore=\(newExpectingMore), currentChunkIndex=\(currentNow)")
                    NSLog("[AudioPlayer] delegate: audioFiles now = \(self.audioFiles.map { $0.lastPathComponent })")

                    // FIRST: Check if more chunks have arrived (before checking expectingMore)
                    // This fixes race condition where synthesis completes between delegate call and polling
                    if newTotalFiles > totalFiles {
                        // New chunk arrived! Advance to next chapter
                        NSLog("[AudioPlayer] delegate: new chunk arrived! totalFiles changed \(totalFiles) -> \(newTotalFiles), resuming playback")
                        self.nextChapter()
                        return
                    }

                    // SECOND: Check if synthesis is done - only stop if no more chunks coming
                    if !newExpectingMore {
                        // Synthesis finished and no new chunks arrived
                        NSLog("[AudioPlayer] delegate: breaking - expectingMore=\(newExpectingMore), no more synthesis coming")
                        break
                    }
                }

                // Either synthesis done or timed out
                self.isPlaying = false
                self.stopProgressUpdates()
                NSLog("[AudioPlayer] delegate: Playback complete (after waiting)")
            } else {
                // No more chunks and not expecting any
                self.isPlaying = false
                self.stopProgressUpdates()
                NSLog("[AudioPlayer] delegate: Playback complete")
            }
        }
    }
}
