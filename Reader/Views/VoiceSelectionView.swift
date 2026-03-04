import SwiftUI
import AVFoundation
import UniformTypeIdentifiers

// MARK: - Voice Selection View

struct VoiceSelectionView: View {
    @Binding var selectedVoice: VoiceProfile
    @Binding var synthesisSettings: SynthesisSettings
    /// Callback when voice selection changes - saves to LibraryItem
    var onVoiceSelected: ((VoiceProfile) -> Void)?
    @Environment(\.dismiss) private var dismiss
    @State private var showRecordVoice = false
    @State private var isRecording = false
    @State private var recordingDuration: TimeInterval = 0
    @State private var audioRecorder: AVAudioRecorder?
    @State private var recordingTimer: Timer?
    @State private var customVoices: [VoiceProfile] = []
    @State private var showFileImporter = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 24) {
                    // Header
                    headerSection

                    // Built-in Voices (using default reference audio)
                    voiceSection(title: "Built-in Voices (Default)", voices: VoiceProfile.builtInVoices)

                    // Custom Voices (cloned)
                    if !customVoices.isEmpty {
                        voiceSection(title: "Your Voices", voices: customVoices, allowDelete: true) { voice in
                            deleteCustomVoice(voice)
                        }
                    }

                    // Clone Voice Section
                    cloneVoiceSection

                    // Synthesis Settings
                    voiceSettingsSection

                    // About
                    aboutSection
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 40)
            }
            .background(Color.appBackground)
            .navigationTitle("Voice")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
            .sheet(isPresented: $showRecordVoice) {
                recordVoiceSheet
            }
            .fileImporter(
                isPresented: $showFileImporter,
                allowedContentTypes: [
                    .audio,
                    UTType.wav,
                    UTType.mp3,
                    UTType(filenameExtension: "wav") ?? .audio,
                    UTType(filenameExtension: "mp3") ?? .audio
                ],
                allowsMultipleSelection: false
            ) { result in
                handleFileImport(result)
            }
            .onAppear {
                loadCustomVoices()
            }
        }
        .preferredColorScheme(.dark)
    }

    // MARK: - Header

    private var headerSection: some View {
        VStack(spacing: 8) {
            Text("Choose a Voice")
                .font(.system(size: 22, weight: .bold))
                .foregroundStyle(Color.appTextPrimary)

            Text("Select a built-in voice, import an audio file, or clone your own voice for consistent TTS output")
                .font(.system(size: 14))
                .foregroundStyle(Color.appTextSecondary)
                .multilineTextAlignment(.center)

            // Tip about consistent generation
            HStack(spacing: 4) {
                Image(systemName: "sparkles")
                    .font(.system(size: 10))
                Text("Tip: Use a fixed seed + cloned voice for reproducible results")
                    .font(.system(size: 11))
            }
            .foregroundStyle(Color.appAccent)
            .padding(.top, 4)
        }
        .padding(.top, 8)
    }

    // MARK: - Voice Section

    private func voiceSection(title: String, voices: [VoiceProfile], allowDelete: Bool = false, onDelete: ((VoiceProfile) -> Void)? = nil) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(Color.appTextTertiary)
                .textCase(.uppercase)
                .tracking(0.5)

            VStack(spacing: 6) {
                ForEach(voices) { voice in
                    if allowDelete, let onDelete = onDelete {
                        voiceCard(voice)
                            .contextMenu {
                                Button(role: .destructive) {
                                    onDelete(voice)
                                } label: {
                                    Label("Delete", systemImage: "trash")
                                }
                            }
                    } else {
                        voiceCard(voice)
                    }
                }
            }
        }
    }

    private func voiceCard(_ voice: VoiceProfile) -> some View {
        Button {
            withAnimation(.easeInOut(duration: 0.2)) {
                selectedVoice = voice
                // Notify parent to save voice selection
                onVoiceSelected?(voice)
            }
        } label: {
            HStack(spacing: 14) {
                // Avatar
                ZStack {
                    Circle()
                        .fill(
                            selectedVoice.id == voice.id
                            ? Color.appAccent.opacity(0.2)
                            : Color.appSurfaceHover
                        )
                        .frame(width: 44, height: 44)

                    Text(String(voice.name.prefix(1)))
                        .font(.system(size: 18, weight: .bold))
                        .foregroundStyle(
                            selectedVoice.id == voice.id
                            ? Color.appAccent
                            : Color.appTextSecondary
                        )
                }

                VStack(alignment: .leading, spacing: 3) {
                    Text(voice.name)
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(Color.appTextPrimary)

                    Text(voice.description)
                        .font(.system(size: 13))
                        .foregroundStyle(Color.appTextSecondary)
                        .lineLimit(1)
                }

                Spacer()

                // Tags
                HStack(spacing: 4) {
                    ForEach(voice.tags.prefix(2), id: \.self) { tag in
                        Text(tag)
                            .font(.system(size: 10, weight: .medium))
                            .foregroundStyle(Color.appTextTertiary)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 3)
                            .background(
                                Capsule().fill(Color.appSurfaceElevated)
                            )
                    }
                }

                // Selection indicator
                ZStack {
                    Circle()
                        .stroke(
                            selectedVoice.id == voice.id ? Color.appAccent : Color.appTextTertiary,
                            lineWidth: 2
                        )
                        .frame(width: 22, height: 22)

                    if selectedVoice.id == voice.id {
                        Circle()
                            .fill(Color.appAccent)
                            .frame(width: 14, height: 14)
                    }
                }
            }
            .padding(14)
            .background(
                RoundedRectangle(cornerRadius: 14)
                    .fill(Color.appSurface)
                    .overlay(
                        RoundedRectangle(cornerRadius: 14)
                            .stroke(
                                selectedVoice.id == voice.id ? Color.appAccent.opacity(0.5) : Color.clear,
                                lineWidth: 1.5
                            )
                    )
            )
        }
        .buttonStyle(.plain)
    }

    // MARK: - Clone Voice Section

    private var cloneVoiceSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Voice Cloning")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(Color.appTextTertiary)
                .textCase(.uppercase)
                .tracking(0.5)

            Button {
                showRecordVoice = true
            } label: {
                HStack(spacing: 14) {
                    ZStack {
                        Circle()
                            .fill(Color.appAccentSubtle)
                            .frame(width: 44, height: 44)

                        Image(systemName: "mic.fill")
                            .font(.system(size: 18))
                            .foregroundStyle(Color.appAccent)
                    }

                    VStack(alignment: .leading, spacing: 3) {
                        Text("Clone Your Voice")
                            .font(.system(size: 15, weight: .semibold))
                            .foregroundStyle(Color.appTextPrimary)

                        Text("Record 10-30 seconds of speech to create a custom voice")
                            .font(.system(size: 13))
                            .foregroundStyle(Color.appTextSecondary)
                            .lineLimit(2)
                    }

                    Spacer()

                    Image(systemName: "chevron.right")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(Color.appTextTertiary)
                }
                .padding(14)
                .background(
                    RoundedRectangle(cornerRadius: 14)
                        .fill(Color.appSurface)
                        .overlay(
                            RoundedRectangle(cornerRadius: 14)
                                .strokeBorder(style: StrokeStyle(lineWidth: 1, dash: [6, 4]))
                                .foregroundStyle(Color.appAccent.opacity(0.3))
                        )
                )
            }
            .buttonStyle(.plain)

            // Import audio file button
            Button {
                showFileImporter = true
            } label: {
                HStack(spacing: 14) {
                    ZStack {
                        Circle()
                            .fill(Color.appAccentSubtle.opacity(0.5))
                            .frame(width: 44, height: 44)

                        Image(systemName: "square.and.arrow.down")
                            .font(.system(size: 18))
                            .foregroundStyle(Color.appAccent)
                    }

                    VStack(alignment: .leading, spacing: 3) {
                        Text("Import Audio File")
                            .font(.system(size: 15, weight: .semibold))
                            .foregroundStyle(Color.appTextPrimary)

                        Text("Import a .wav or .mp3 file as reference audio")
                            .font(.system(size: 13))
                            .foregroundStyle(Color.appTextSecondary)
                            .lineLimit(2)
                    }

                    Spacer()

                    Image(systemName: "chevron.right")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(Color.appTextTertiary)
                }
                .padding(14)
                .background(
                    RoundedRectangle(cornerRadius: 14)
                        .fill(Color.appSurface)
                        .overlay(
                            RoundedRectangle(cornerRadius: 14)
                                .strokeBorder(style: StrokeStyle(lineWidth: 1, dash: [6, 4]))
                                .foregroundStyle(Color.appAccent.opacity(0.3))
                        )
                )
            }
            .buttonStyle(.plain)
        }
    }

    // MARK: - Voice Settings

    private var voiceSettingsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Synthesis Settings")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(Color.appTextTertiary)
                .textCase(.uppercase)
                .tracking(0.5)

            VStack(spacing: 16) {
                // Seed input with consistent generation info
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Seed")
                            .font(.system(size: 14))
                            .foregroundStyle(Color.appTextPrimary)

                        Spacer()

                        TextField("Random", value: $synthesisSettings.seed, format: .number)
                            .keyboardType(.numberPad)
                            .multilineTextAlignment(.trailing)
                            .frame(width: 120)
                            .padding(8)
                            .background(Color.appSurfaceElevated)
                            .cornerRadius(8)
                            .foregroundStyle(Color.appTextPrimary)
                    }

                    // Seed + Voice combo tip
                    HStack(spacing: 6) {
                        Image(systemName: "info.circle")
                            .font(.system(size: 11))
                            .foregroundStyle(Color.appAccent)

                        Text("Combine with a fixed voice (built-in or cloned) + non-zero seed for consistent output across generations.")
                            .font(.system(size: 11))
                            .foregroundStyle(Color.appTextSecondary)
                    }
                }

                Divider().background(Color.appTextTertiary.opacity(0.3))

                // Tip about reproducible output
                HStack(spacing: 6) {
                    Image(systemName: "arrow.triangle.2.circlepath")
                        .font(.system(size: 11))
                        .foregroundStyle(Color.appAccent)

                    Text("Set a seed (e.g., 42) to get the same output every time. Use 0 for random.")
                        .font(.system(size: 12))
                        .foregroundStyle(Color.appTextSecondary)
                }
            }
            .padding(16)
            .background(
                RoundedRectangle(cornerRadius: 14)
                    .fill(Color.appSurface)
            )
        }
    }

    // MARK: - About Section

    private var aboutSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("About Chatterbox Turbo")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(Color.appTextTertiary)
                .textCase(.uppercase)
                .tracking(0.5)

            VStack(alignment: .leading, spacing: 10) {
                featureRow(icon: "cpu", title: "On-Device", detail: "350M parameters, runs locally")
                featureRow(icon: "waveform", title: "24kHz Audio", detail: "High-quality speech synthesis")
                featureRow(icon: "face.smiling", title: "Expressive", detail: "Supports [laugh], [cough], [chuckle] tags")
                featureRow(icon: "person.2", title: "Zero-Shot Cloning", detail: "Clone any voice from a short sample")
                featureRow(icon: "lock.shield", title: "Private", detail: "All processing happens on your device")
            }
            .padding(16)
            .background(
                RoundedRectangle(cornerRadius: 14)
                    .fill(Color.appSurface)
            )
        }
    }

    private func featureRow(icon: String, title: String, detail: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 14))
                .foregroundStyle(Color.appAccent)
                .frame(width: 20)

            Text(title)
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(Color.appTextPrimary)

            Spacer()

            Text(detail)
                .font(.system(size: 12))
                .foregroundStyle(Color.appTextTertiary)
        }
    }

    // MARK: - Record Voice Sheet

    private var recordVoiceSheet: some View {
        NavigationStack {
            VStack(spacing: 32) {
                Spacer()

                // Recording visualization
                ZStack {
                    Circle()
                        .fill(Color.appAccentSubtle)
                        .frame(width: 160, height: 160)

                    Circle()
                        .fill(isRecording ? Color.appError.opacity(0.2) : Color.appSurfaceElevated)
                        .frame(width: 120, height: 120)
                        .scaleEffect(isRecording ? 1.1 : 1.0)
                        .animation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true), value: isRecording)

                    Image(systemName: isRecording ? "stop.fill" : "mic.fill")
                        .font(.system(size: 36))
                        .foregroundStyle(isRecording ? Color.appError : Color.appAccent)
                }

                // Instructions
                VStack(spacing: 8) {
                    Text(isRecording ? "Recording..." : "Record Voice Sample")
                        .font(.system(size: 20, weight: .bold))
                        .foregroundStyle(Color.appTextPrimary)

                    if isRecording {
                        Text(String(format: "%.0f seconds", recordingDuration))
                            .font(.system(size: 16, weight: .medium, design: .monospaced))
                            .foregroundStyle(Color.appAccent)
                    } else {
                        Text("Read any text aloud for 10-30 seconds in your natural voice. Speak clearly in a quiet environment.")
                            .font(.system(size: 14))
                            .foregroundStyle(Color.appTextSecondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 24)
                    }
                }

                // Sample text to read
                if !isRecording {
                    VStack(spacing: 6) {
                        Text("Try reading this:")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(Color.appTextTertiary)

                        Text("\"The quick brown fox jumps over the lazy dog. Technology continues to advance at an incredible pace, reshaping how we live, work, and communicate with one another.\"")
                            .font(.system(size: 14))
                            .foregroundStyle(Color.appTextSecondary)
                            .italic()
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 24)
                    }
                }

                Spacer()

                // Record button
                Button {
                    if isRecording {
                        stopRecording()
                    } else {
                        startRecording()
                    }
                } label: {
                    HStack {
                        Image(systemName: isRecording ? "stop.fill" : "mic.fill")
                        Text(isRecording ? "Stop Recording" : "Start Recording")
                    }
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
                    .background(
                        Capsule().fill(isRecording ? Color.appError : Color.appAccent)
                    )
                }
                .padding(.horizontal, 24)
                .padding(.bottom, 24)
            }
            .background(Color.appBackground)
            .navigationTitle("Voice Cloning")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Cancel") { showRecordVoice = false }
                }
            }
        }
        .preferredColorScheme(.dark)
    }

    // MARK: - Recording

    private func startRecording() {
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.record, mode: .default)
            try session.setActive(true)
        } catch {
            return
        }

        let docsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let voicesDir = docsURL.appendingPathComponent("VoiceSamples", isDirectory: true)
        try? FileManager.default.createDirectory(at: voicesDir, withIntermediateDirectories: true)

        let fileURL = voicesDir.appendingPathComponent("\(UUID().uuidString).wav")

        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: 24000,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
        ]

        do {
            audioRecorder = try AVAudioRecorder(url: fileURL, settings: settings)
            audioRecorder?.record()
            isRecording = true
            recordingDuration = 0

            recordingTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
                Task { @MainActor in
                    recordingDuration += 0.1
                    if recordingDuration >= 30 {
                        stopRecording()
                    }
                }
            }
        } catch {
            print("Recording failed: \(error)")
        }
    }

    private func stopRecording() {
        recordingTimer?.invalidate()
        recordingTimer = nil
        audioRecorder?.stop()
        isRecording = false

        guard let url = audioRecorder?.url, recordingDuration >= 5 else {
            // Too short
            return
        }

        // Create custom voice profile
        let voiceName = "My Voice \(customVoices.count + 1)"
        let voice = VoiceProfile(
            id: UUID().uuidString,
            name: voiceName,
            description: "Custom cloned voice",
            isBuiltIn: false,
            referenceAudioPath: url.path,
            sampleRate: 24000,
            language: "en",
            tags: ["custom", "cloned"]
        )

        customVoices.append(voice)
        saveCustomVoices()
        selectedVoice = voice
        showRecordVoice = false

        // Notify parent to save voice selection
        onVoiceSelected?(voice)

        // Reset audio session
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .spokenAudio)
    }

    // MARK: - File Import

    private func handleFileImport(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let sourceURL = urls.first else { return }

            // Start accessing security-scoped resource
            guard sourceURL.startAccessingSecurityScopedResource() else {
                NSLog("[VoiceSelection] Failed to access security-scoped resource")
                return
            }
            defer { sourceURL.stopAccessingSecurityScopedResource() }

            // Copy file to app's documents
            let docsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            let voicesDir = docsURL.appendingPathComponent("VoiceSamples", isDirectory: true)
            try? FileManager.default.createDirectory(at: voicesDir, withIntermediateDirectories: true)

            let fileName = sourceURL.lastPathComponent
            let destURL = voicesDir.appendingPathComponent("\(UUID().uuidString)_\(fileName)")

            do {
                try FileManager.default.copyItem(at: sourceURL, to: destURL)

                // Create voice profile from imported file
                let voiceName = sourceURL.deletingPathExtension().lastPathComponent
                let voice = VoiceProfile(
                    id: UUID().uuidString,
                    name: voiceName,
                    description: "Imported voice from file",
                    isBuiltIn: false,
                    referenceAudioPath: destURL.path,
                    sampleRate: 24000,
                    language: "en",
                    tags: ["imported"]
                )

                customVoices.append(voice)
                saveCustomVoices()
                selectedVoice = voice

                // Notify parent to save
                onVoiceSelected?(voice)

                NSLog("[VoiceSelection] Imported voice: \(voiceName) from \(fileName)")
            } catch {
                NSLog("[VoiceSelection] Failed to import file: \(error)")
            }

        case .failure(let error):
            NSLog("[VoiceSelection] File import error: \(error)")
        }
    }

    // MARK: - Custom Voice Persistence

    private func loadCustomVoices() {
        if let data = UserDefaults.standard.data(forKey: "custom_voices"),
           let voices = try? JSONDecoder().decode([VoiceProfile].self, from: data) {
            customVoices = voices
        }
    }

    private func saveCustomVoices() {
        if let data = try? JSONEncoder().encode(customVoices) {
            UserDefaults.standard.set(data, forKey: "custom_voices")
        }
    }

    private func deleteCustomVoice(_ voice: VoiceProfile) {
        // Remove from array
        customVoices.removeAll { $0.id == voice.id }
        saveCustomVoices()

        // Delete the audio file if it exists
        if let path = voice.referenceAudioPath {
            try? FileManager.default.removeItem(atPath: path)
        }

        // If deleted voice was selected, switch to default
        if selectedVoice.id == voice.id {
            selectedVoice = .defaultVoice
            onVoiceSelected?(.defaultVoice)
        }

        NSLog("[VoiceSelection] Deleted custom voice: \(voice.name)")
    }
}
