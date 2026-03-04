import SwiftUI

// MARK: - Player View (Text-focused reader with floating controls) - Redesigned

struct PlayerView: View {
    @StateObject private var viewModel: PlayerViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var showTextView = false
    @State private var selectedText = ""
    @State private var isPlayingSelection = false
    @State private var showControls = true
    @State private var showChapterSheet = false
    // Auto-scroll state
    @State private var autoScrollEnabled = true
    @State private var scrollProxy: ScrollViewProxy?
    @State private var lastScrolledParagraph = -1

    init(item: LibraryItem) {
        _viewModel = StateObject(wrappedValue: PlayerViewModel(item: item))
    }

    var body: some View {
        NavigationStack {
            ZStack {
                // Background with noise texture
                Color.appBackground
                    .ignoresSafeArea()
                    .noiseTexture(opacity: 0.02)

                // Main content - full screen text area
                ZStack {
                    // Full-screen text section
                    textSection
                        .gesture(tapGesture)

                    // Floating controls overlay with glassmorphism - properly inset from safe area
                    if showControls {
                        VStack {
                            Spacer()
                            floatingControls
                        }
                    }
                }
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        dismiss()
                    } label: {
                        Image(systemName: "chevron.down")
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundStyle(Color.appTextSecondary)
                            .padding(10)
                            .background(Circle().fill(Color.appSurfaceElevated))
                    }
                }

                ToolbarItem(placement: .principal) {
                    VStack(spacing: 2) {
                        Text(viewModel.item.title)
                            .font(AppTypography.headlineSmall)
                            .foregroundStyle(Color.appTextPrimary)
                            .lineLimit(1)
                    }
                }

                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        Button {
                            showTextView = true
                        } label: {
                            Label("View Text", systemImage: "doc.text")
                        }

                        Button {
                            showChapterSheet = true
                        } label: {
                            Label("Chapters", systemImage: "list.bullet")
                        }

                        Button {
                            viewModel.showVoiceSelector = true
                        } label: {
                            Label("Change Voice", systemImage: "person.wave.2")
                        }

                        Button {
                            viewModel.showSettings = true
                        } label: {
                            Label("Settings", systemImage: "gear")
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle.fill")
                            .font(.system(size: 18))
                            .foregroundStyle(Color.appTextSecondary)
                    }
                }
            }
            .sheet(isPresented: $showChapterSheet) {
                chapterListSheet
            }
            .sheet(isPresented: $viewModel.showVoiceSelector) {
                VoiceSelectionView(
                    selectedVoice: $viewModel.selectedVoice,
                    synthesisSettings: $viewModel.synthesisSettings
                )
            }
            .sheet(isPresented: $showTextView) {
                textViewSheet
            }
            .sheet(isPresented: $viewModel.showModelDownload) {
                modelDownloadSheet
            }
            .onChange(of: viewModel.audioPlayer.isPlaying) { _, isPlaying in
                if !isPlaying {
                    withAnimation(AppAnimation.smooth) {
                        showControls = true
                    }
                }
            }
        }
        .preferredColorScheme(.dark)
    }

    // MARK: - Tap Gesture

    private var tapGesture: some Gesture {
        TapGesture()
            .onEnded {
                withAnimation(AppAnimation.quick) {
                    showControls.toggle()
                }
            }
    }

    // MARK: - Text Section

    private var textSection: some View {
        ZStack(alignment: .top) {
            ScrollViewReader { proxy in
                ScrollView(showsIndicators: true) {
                    LazyVStack(alignment: .leading, spacing: 20) {
                        ForEach(Array(viewModel.cachedParagraphs.enumerated()), id: \.offset) { paragraphIndex, paragraph in
                            paragraphWithHighlighting(paragraph: paragraph, paragraphIndex: paragraphIndex)
                                .id(paragraphIndex)
                        }
                    }
                    .padding(.horizontal, 20)
                    .padding(.top, 16)
                    .padding(.bottom, 180)
                }
                .onAppear {
                    scrollProxy = proxy
                }
                .onChange(of: viewModel.currentChunkIndex) { _, newIndex in
                    viewModel.updatePlayingIndex()
                    // Auto-scroll only when paragraph actually changes
                    if autoScrollEnabled, newIndex >= 0 {
                        let targetParagraph = findParagraphIndex(for: newIndex)
                        if targetParagraph != lastScrolledParagraph && targetParagraph >= 0 {
                            lastScrolledParagraph = targetParagraph
                            scrollToParagraph(chunkIndex: newIndex, proxy: proxy)
                        }
                    }
                }
                .simultaneousGesture(
                    DragGesture(minimumDistance: 10)
                        .onChanged { _ in
                            // Disable auto-scroll when user starts dragging
                            autoScrollEnabled = false
                        }
                )
            }

            // Back to Current button
            if !autoScrollEnabled && viewModel.currentChunkIndex >= 0 {
                backToCurrentButton(proxy: scrollProxy)
            }
        }
    }

    /// Scroll to the paragraph containing the given chunk index
    private func scrollToParagraph(chunkIndex: Int, proxy: ScrollViewProxy) {
        // Find which paragraph contains this chunk
        let paragraphIndex = findParagraphIndex(for: chunkIndex)
        guard paragraphIndex >= 0 else { return }

        withAnimation(AppAnimation.smooth) {
            proxy.scrollTo(paragraphIndex, anchor: .center)
        }
    }

    /// Find the paragraph index for a given chunk index
    private func findParagraphIndex(for chunkIndex: Int) -> Int {
        guard chunkIndex >= 0, chunkIndex < viewModel.textChunks.count else { return 0 }
        let sentence = viewModel.textChunks[chunkIndex]

        // Find which paragraph contains this sentence
        for (paragraphIndex, sentences) in viewModel.cachedParagraphSentences.enumerated() {
            if sentences.contains(where: { $0.contains(sentence) || sentence.contains($0) }) {
                return paragraphIndex
            }
        }

        // Fallback: estimate based on chunk distribution
        guard !viewModel.cachedParagraphs.isEmpty else { return 0 }
        let avgChunksPerParagraph = Double(viewModel.textChunks.count) / Double(viewModel.cachedParagraphs.count)
        return min(Int(Double(chunkIndex) / avgChunksPerParagraph), viewModel.cachedParagraphs.count - 1)
    }

    /// Back to Current button
    private func backToCurrentButton(proxy: ScrollViewProxy?) -> some View {
        Button {
            autoScrollEnabled = true
            lastScrolledParagraph = -1 // Reset to force scroll
            if let proxy = proxy {
                scrollToParagraph(chunkIndex: viewModel.currentChunkIndex, proxy: proxy)
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "arrow.down.to.line")
                Text("Back to Current")
            }
            .font(AppTypography.captionLarge)
            .foregroundColor(.black)
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(
                Capsule()
                    .fill(AppGradients.accent)
            )
            .shadow(color: Color.appAccent.opacity(0.4), radius: 8, y: 4)
        }
        .padding(.top, 8)
    }

    /// Show paragraph text with inline sentence highlighting
    private func paragraphWithHighlighting(paragraph: String, paragraphIndex: Int) -> some View {
        // Use cached attributed string directly (already has highlighting built-in)
        let attributed = viewModel.cachedAttributedParagraphs.indices.contains(paragraphIndex)
            ? viewModel.cachedAttributedParagraphs[paragraphIndex]
            : AttributedString(paragraph)

        // Get sentences for tap action
        let sentences = viewModel.cachedParagraphSentences.indices.contains(paragraphIndex)
            ? viewModel.cachedParagraphSentences[paragraphIndex] : []

        return Text(attributed)
            .font(AppTypography.bodyLarge)
            .lineSpacing(8)
            .padding(.vertical, 8)
            .padding(.horizontal, 12)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.appSurface.opacity(0.3))
            )
            .onTapGesture {
                if let firstSentence = sentences.first,
                   let chunkIndex = viewModel.textChunks.firstIndex(of: firstSentence) {
                    Task {
                        await viewModel.startSynthesisFromChunk(chunkIndex)
                    }
                }
            }
    }

    // MARK: - Floating Controls - Redesigned with Glassmorphism

    private var floatingControls: some View {
        VStack(spacing: 0) {
            // Progress indicator
            VStack(spacing: 10) {
                // Progress bar with glow - now draggable
                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        // Track background
                        RoundedRectangle(cornerRadius: 3)
                            .fill(Color.appSurfaceElevated.opacity(0.5))
                            .frame(height: 6)

                        // Progress fill
                        RoundedRectangle(cornerRadius: 3)
                            .fill(AppGradients.accent)
                            .frame(width: max(0, geometry.size.width * (viewModel.audioPlayer.duration > 0 ? viewModel.audioPlayer.progress : viewModel.synthesisProgress)), height: 6)
                            .shadow(color: Color.appAccent.opacity(0.5), radius: 6, y: 0)
                    }
                    .contentShape(Rectangle())
                    .gesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { value in
                                let progress = max(0, min(1, value.location.x / geometry.size.width))
                                // Calculate target chunk based on progress
                                let targetChunkIndex = Int(progress * Double(viewModel.textChunks.count - 1))
                                let safeChunkIndex = max(0, min(targetChunkIndex, viewModel.textChunks.count - 1))
                                // If we have existing chunks, seek; otherwise start synthesis from that chunk
                                if viewModel.item.generatedChunks[safeChunkIndex] != nil || viewModel.audioPlayer.hasAudioFiles {
                                    viewModel.audioPlayer.seekToProgress(progress)
                                } else {
                                    // Start synthesis from target chunk
                                    Task {
                                        await viewModel.startSynthesisFromChunk(safeChunkIndex)
                                    }
                                }
                            }
                    )
                }
                .frame(height: 6)

                // Status text
                HStack {
                    if viewModel.isSynthesizing {
                        HStack(spacing: 6) {
                            AnimatedWaveformView(barCount: 12, accentColor: Color.appAccent)
                                .frame(width: 40, height: 16)

                            Text("Generating \(Int(viewModel.synthesisProgress * 100))%")
                                .font(AppTypography.captionLarge)
                                .foregroundStyle(Color.appAccent)
                        }
                    } else if viewModel.audioPlayer.isPlaying {
                        HStack(spacing: 6) {
                            Image(systemName: "speaker.wave.2.fill")
                                .font(.system(size: 10))
                                .foregroundStyle(Color.appAccent)

                            Text(viewModel.audioPlayer.formattedCurrentTime)
                                .font(AppTypography.mono)
                                .foregroundStyle(Color.appTextSecondary)
                        }
                    } else {
                        Text("Tap to play")
                            .font(AppTypography.captionLarge)
                            .foregroundStyle(Color.appTextTertiary)
                    }

                    Spacer()

                    if !viewModel.audioPlayer.isPlaying && viewModel.item.status != .ready {
                        Text("\(viewModel.textChunks.count) chunks")
                            .font(AppTypography.captionMedium)
                            .foregroundStyle(Color.appTextTertiary)
                    }

                    Spacer()

                    Text(viewModel.totalDurationText)
                        .font(AppTypography.mono)
                        .foregroundStyle(Color.appTextTertiary)
                }
            }
            .padding(.horizontal, 24)
            .padding(.top, 16)

            // Main controls
            HStack {
                // Left actions
                HStack(spacing: 20) {
                    QuickActionButton(icon: "gobackward.15", label: "15s") {
                        viewModel.audioPlayer.skipBackward()
                    }
                    .opacity(viewModel.canPlay || viewModel.audioPlayer.isPlaying ? 1 : 0.3)
                    .disabled(!viewModel.canPlay && !viewModel.audioPlayer.isPlaying)

                    QuickActionButton(icon: "goforward.15", label: "15s") {
                        viewModel.audioPlayer.skipForward()
                    }
                    .opacity(viewModel.canPlay || viewModel.audioPlayer.isPlaying ? 1 : 0.3)
                    .disabled(!viewModel.canPlay && !viewModel.audioPlayer.isPlaying)
                }

                Spacer()

                // Main play button
                Button {
                    handleMainButtonTap()
                } label: {
                    ZStack {
                        // Outer glow
                        if viewModel.audioPlayer.isPlaying || viewModel.isSynthesizing {
                            Circle()
                                .fill(Color.appAccent.opacity(0.3))
                                .frame(width: 80, height: 80)
                                .blur(radius: 12)
                        }

                        // Gradient background
                        Circle()
                            .fill(AppGradients.accent)
                            .frame(width: 68, height: 68)
                            .shadow(color: Color.appAccent.opacity(0.5), radius: 16, y: 6)

                        // Icon
                        if viewModel.isSynthesizing || isPlayingSelection {
                            Image(systemName: viewModel.isPaused ? "play.fill" : "stop.fill")
                                .font(.system(size: 26, weight: .bold))
                                .foregroundStyle(.white)
                        } else if viewModel.audioPlayer.isPlaying {
                            Image(systemName: "pause.fill")
                                .font(.system(size: 26, weight: .bold))
                                .foregroundStyle(.white)
                        } else {
                            Image(systemName: "play.fill")
                                .font(.system(size: 26, weight: .bold))
                                .foregroundStyle(.white)
                                .offset(x: 2)
                        }
                    }
                }
                .buttonStyle(PlayButtonStyle())

                Spacer()

                // Right actions
                HStack(spacing: 20) {
                    QuickActionButton(icon: "speedometer", label: viewModel.playbackRateLabel) {
                        viewModel.cyclePlaybackRate()
                    }

                    QuickActionButton(icon: "person.wave.2", label: nil) {
                        viewModel.showVoiceSelector = true
                    }
                }
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 32)
        }
        .background(
            LinearGradient(
                colors: [
                    Color.appBackground.opacity(0),
                    Color.appBackground.opacity(0.8),
                    Color.appBackground.opacity(0.98),
                    Color.appBackground
                ],
                startPoint: .top,
                endPoint: .bottom
            )
        )
    }

    private func handleMainButtonTap() {
        withAnimation(AppAnimation.quick) {
            showControls = true
        }

        if viewModel.isSynthesizing {
            viewModel.stopGeneration()
        } else if isPlayingSelection {
            viewModel.audioPlayer.stop()
            isPlayingSelection = false
        } else {
            viewModel.playFromCurrentPosition()
        }
    }

    // MARK: - Chapter List Sheet

    private var chapterListSheet: some View {
        NavigationStack {
            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(Array(viewModel.displayChapters.enumerated()), id: \.offset) { index, chapter in
                        Button {
                            NSLog("[PlayerView] Chapter tapped: \(index)")
                            viewModel.selectChapter(index)
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(chapter.title)
                                        .font(AppTypography.headlineSmall)
                                        .foregroundStyle(
                                            index == viewModel.audioPlayer.currentChunkIndex
                                            ? Color.appAccent : Color.appTextPrimary
                                        )

                                    Text("\(chapter.textContent.prefix(60))...")
                                        .font(AppTypography.bodySmall)
                                        .foregroundStyle(Color.appTextTertiary)
                                        .lineLimit(2)
                                        .multilineTextAlignment(.leading)
                                }

                                Spacer()

                                if index == viewModel.audioPlayer.currentChunkIndex {
                                    Image(systemName: "speaker.wave.2.fill")
                                        .font(.system(size: 14))
                                        .foregroundStyle(Color.appAccent)
                                }

                                Image(systemName: "chevron.right")
                                    .font(.system(size: 12, weight: .semibold))
                                    .foregroundStyle(Color.appTextTertiary)
                            }
                            .padding(.horizontal, 16)
                            .padding(.vertical, 14)
                            .background(
                                index == viewModel.audioPlayer.currentChunkIndex
                                ? Color.appAccent.opacity(0.1)
                                : Color.clear
                            )
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
            .background(Color.appBackground)
            .navigationTitle("Chapters")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { showChapterSheet = false }
                        .foregroundStyle(Color.appAccent)
                }
            }
        }
        .presentationDetents([.medium, .large])
        .preferredColorScheme(.dark)
    }

    // MARK: - Model Download Sheet

    private var modelDownloadSheet: some View {
        NavigationStack {
            VStack(spacing: 28) {
                Spacer()

                ZStack {
                    Circle()
                        .fill(Color.appAccent.opacity(0.2))
                        .frame(width: 80, height: 80)
                        .blur(radius: 20)

                    Image(systemName: "arrow.down.circle.fill")
                        .font(.system(size: 40, weight: .ultraLight))
                        .foregroundStyle(Color.appAccent)
                }

                VStack(spacing: 8) {
                    Text("Download Voice Model")
                        .font(AppTypography.displaySmall)
                        .foregroundStyle(Color.appTextPrimary)

                    Text("Chatterbox Turbo (~558 MB)\nOne-time download required")
                        .font(AppTypography.bodyMedium)
                        .foregroundStyle(Color.appTextSecondary)
                        .multilineTextAlignment(.center)
                }

                if viewModel.isDownloadingModels {
                    VStack(spacing: 12) {
                        ProgressView(value: ModelDownloadService.shared.overallProgress)
                            .tint(Color.appAccent)
                            .padding(.horizontal, 40)

                        Text("\(Int(ModelDownloadService.shared.overallProgress * 100))%")
                            .font(AppTypography.monoLarge)
                            .foregroundStyle(Color.appAccent)
                    }
                } else {
                    Button {
                        Task { await viewModel.downloadModels() }
                    } label: {
                        Text("Download Model")
                            .font(AppTypography.headlineMedium)
                            .foregroundStyle(.black)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 16)
                            .background(Capsule().fill(AppGradients.accent))
                            .shadow(color: Color.appAccent.opacity(0.4), radius: 12, y: 4)
                    }
                    .padding(.horizontal, 40)
                }

                if let error = viewModel.errorMessage {
                    Text(error)
                        .font(AppTypography.captionLarge)
                        .foregroundStyle(Color.appError)
                        .multilineTextAlignment(.center)
                }

                Spacer()
            }
            .background(Color.appBackground)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Cancel") { viewModel.showModelDownload = false }
                }
            }
        }
        .presentationDetents([.medium])
        .preferredColorScheme(.dark)
    }

    // MARK: - Text View Sheet

    private var textViewSheet: some View {
        NavigationStack {
            SelectableTextView(
                text: viewModel.item.textContent,
                chunks: viewModel.paragraphs,
                selectedText: $selectedText,
                onPlaySelection: {
                    isPlayingSelection = true
                    Task {
                        await viewModel.playFromSelection(selectedText)
                        isPlayingSelection = false
                    }
                },
                isPlaying: isPlayingSelection,
                currentChunkIndex: viewModel.currentStreamingChunkIndex
            )
            .background(Color.appBackground)
            .navigationTitle("Text")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        selectedText = ""
                        showTextView = false
                    }
                    .foregroundStyle(Color.appAccent)
                }
            }
        }
        .presentationDetents([.large])
        .preferredColorScheme(.dark)
    }
}

// MARK: - Quick Action Button - Redesigned

struct QuickActionButton: View {
    let icon: String
    let label: String?
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 5) {
                Image(systemName: icon)
                    .font(.system(size: 20))
                    .foregroundStyle(Color.appTextSecondary)

                if let label = label {
                    Text(label)
                        .font(AppTypography.captionMedium)
                        .foregroundStyle(Color.appTextTertiary)
                }
            }
            .frame(width: 56, height: 50)
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Text Chunk View

struct TextChunkView: View {
    let text: String
    let isActive: Bool
    let isPast: Bool
    let isSelected: Bool
    var onTap: (() -> Void)?

    var body: some View {
        Text(text)
            .font(AppTypography.bodyLarge)
            .foregroundStyle(textColor)
            .lineSpacing(8)
            .padding(.vertical, 10)
            .padding(.horizontal, 14)
            .background(backgroundColor)
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(borderColor, lineWidth: isSelected ? 2 : 0)
            )
            .animation(AppAnimation.smooth, value: isActive)
            .onTapGesture {
                onTap?()
            }
    }

    private var textColor: Color {
        if isActive { return Color.appAccent }
        else if isPast { return Color.appTextSecondary.opacity(0.6) }
        else { return Color.appTextPrimary }
    }

    private var backgroundColor: Color {
        if isActive { return Color.appAccent.opacity(0.12) }
        else if isSelected { return Color.appAccent.opacity(0.08) }
        else { return Color.clear }
    }

    private var borderColor: Color {
        if isActive { return Color.appAccent.opacity(0.5) }
        else if isSelected { return Color.appAccent.opacity(0.3) }
        else { return Color.clear }
    }
}

// MARK: - Paragraph View with Sentence Highlighting

struct ParagraphView: View {
    let paragraph: String
    let sentences: [String]
    let currentSentence: String?
    let isActive: Bool
    var onSentenceTap: ((String) -> Void)?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(sentences.enumerated()), id: \.offset) { index, sentence in
                sentenceView(sentence, index: index)
            }
        }
    }

    private func sentenceView(_ sentence: String, index: Int) -> some View {
        let isCurrentSentence = sentence == currentSentence

        return Text(sentence)
            .font(AppTypography.bodyLarge)
            .foregroundStyle(sentenceColor(isCurrentSentence: isCurrentSentence))
            .lineSpacing(8)
            .padding(.vertical, 6)
            .padding(.horizontal, 10)
            .background(sentenceBackground(isCurrentSentence: isCurrentSentence))
            .cornerRadius(8)
            .animation(AppAnimation.smooth, value: isCurrentSentence)
            .onTapGesture {
                onSentenceTap?(sentence)
            }
    }

    private func sentenceColor(isCurrentSentence: Bool) -> Color {
        if isCurrentSentence { return Color.appAccent }
        else if currentSentence != nil { return Color.appTextSecondary }
        else { return Color.appTextPrimary }
    }

    private func sentenceBackground(isCurrentSentence: Bool) -> Color {
        if isCurrentSentence { return Color.appAccent.opacity(0.15) }
        return Color.clear
    }
}

// MARK: - Sentence View

struct SentenceView: View {
    let text: String
    let isActive: Bool
    let isPast: Bool
    let paragraphIndex: Int
    var onTap: (() -> Void)?

    private var topPadding: CGFloat {
        paragraphIndex == 0 ? 0 : 8
    }

    var body: some View {
        Text(text)
            .font(AppTypography.bodyLarge)
            .foregroundStyle(textColor)
            .lineSpacing(8)
            .padding(.vertical, 6)
            .padding(.horizontal, 10)
            .background(backgroundColor)
            .cornerRadius(8)
            .padding(.top, topPadding)
            .animation(AppAnimation.quick, value: isActive)
            .onTapGesture {
                onTap?()
            }
    }

    private var textColor: Color {
        if isActive { return Color.appAccent }
        else if isPast { return Color.appTextSecondary.opacity(0.5) }
        else { return Color.appTextPrimary }
    }

    private var backgroundColor: Color {
        if isActive { return Color.appAccent.opacity(0.15) }
        return Color.clear
    }
}

// MARK: - Text Paragraph View

struct TextParagraphView: View {
    let text: String
    let isActive: Bool
    let isPast: Bool
    var onTap: (() -> Void)?

    var body: some View {
        Text(text)
            .font(AppTypography.bodyLarge)
            .foregroundStyle(textColor)
            .lineSpacing(10)
            .padding(.vertical, 12)
            .padding(.horizontal, 16)
            .background(backgroundColor)
            .cornerRadius(12)
            .onTapGesture {
                onTap?()
            }
    }

    private var textColor: Color {
        if isActive { return Color.appAccent }
        else if isPast { return Color.appTextSecondary.opacity(0.6) }
        else { return Color.appTextPrimary }
    }

    private var backgroundColor: Color {
        if isActive { return Color.appAccent.opacity(0.12) }
        else { return Color.clear }
    }
}
