import SwiftUI
import UniformTypeIdentifiers

// MARK: - Library View (Main Screen) - Redesigned with Glassmorphism

struct LibraryView: View {
    @StateObject private var viewModel = LibraryViewModel()
    @StateObject private var audioPlayer = AudioPlayerService.shared
    @StateObject private var downloadService = ModelDownloadService.shared
    /// Non-nil while the PlayerView sheet is open. sheet(item:) sets this to nil on dismiss.
    @State private var selectedItem: LibraryItem?
    /// Persists after PlayerView is dismissed – keeps the mini-player visible.
    @State private var nowPlayingItem: LibraryItem?
    @State private var showFileImporter = false
    @State private var showModelSetup = false
    @State private var selectedVariant: ModelVariant = .q4f16

    var body: some View {
        NavigationStack {
            ZStack(alignment: .bottom) {
                // Main content
                ScrollView {
                    VStack(spacing: 0) {
                        // Model status banner
                        if !downloadService.isModelReady {
                            modelBanner
                        }

                        // Filter tabs
                        filterTabs

                        // Content list
                        if viewModel.filteredItems.isEmpty {
                            emptyState
                        } else {
                            itemsList
                        }
                    }
                    .padding(.bottom, audioPlayer.isPlaying ? 90 : 20)
                }
                .background(Color.appBackground)
                .noiseTexture(opacity: 0.025)
                .scrollContentBackground(.hidden)

                // Mini player — uses nowPlayingItem which persists after sheet dismiss
                if audioPlayer.isPlaying || audioPlayer.currentTime > 0, let item = nowPlayingItem {
                    MiniPlayerView(item: item, audioPlayer: audioPlayer) {
                        selectedItem = nowPlayingItem   // re-open the player sheet
                    }
                    .padding(.bottom, 8)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                }
            }
            .navigationTitle("Reader")
            .toolbarColorScheme(.dark, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        viewModel.showAddContent = true
                    } label: {
                        Image(systemName: "plus.circle.fill")
                            .font(.system(size: 22))
                            .foregroundStyle(Color.appAccent)
                            .shadow(color: Color.appAccent.opacity(0.4), radius: 6, y: 0)
                    }
                }

                ToolbarItem(placement: .topBarLeading) {
                    NavigationLink {
                        SettingsView()
                    } label: {
                        Image(systemName: "gearshape.fill")
                            .foregroundStyle(Color.appTextSecondary)
                    }
                }
            }
            .searchable(text: $viewModel.searchText, prompt: "Search library")
            .sheet(isPresented: $viewModel.showAddContent) {
                AddContentView(viewModel: viewModel, showFileImporter: $showFileImporter)
                    .presentationDetents([.large])
                    .presentationDragIndicator(.visible)
            }
            .sheet(item: $selectedItem) { item in
                PlayerView(item: item) { updatedItem in
                    // Find and update the item in viewModel
                    if let index = viewModel.items.firstIndex(where: { $0.id == updatedItem.id }) {
                        viewModel.items[index] = updatedItem
                        viewModel.saveLibrary()
                    }
                }
            }
            .sheet(isPresented: $showModelSetup) {
                modelSetupSheet
            }
            .fileImporter(
                isPresented: $showFileImporter,
                allowedContentTypes: [
                    UTType.epub ?? .data,
                    .pdf,
                    .plainText
                ],
                allowsMultipleSelection: false
            ) { result in
                switch result {
                case .success(let urls):
                    if let url = urls.first {
                        Task { await viewModel.addFromFile(at: url) }
                    }
                case .failure(let error):
                    viewModel.errorMessage = error.localizedDescription
                    viewModel.showError = true
                }
            }
            .alert("Error", isPresented: $viewModel.showError) {
                Button("OK") {}
            } message: {
                Text(viewModel.errorMessage ?? "An error occurred.")
            }
            .onAppear {
                viewModel.loadLibrary()
                downloadService.checkModelAvailability()
            }
        }
        .preferredColorScheme(.dark)
        .onReceive(NotificationCenter.default.publisher(for: .startGenerationFromLibrary)) { notification in
            if let item = notification.userInfo?["item"] as? LibraryItem {
                Task {
                    await handleStartGeneration(item)
                }
            }
        }
    }

    // Handle start generation from library (background download - no playback)
    private func handleStartGeneration(_ item: LibraryItem) async {
        NSLog("[LibraryView] handleStartGeneration called for: \(item.title), id: \(item.id)")
        // Find the item in viewModel to get the latest state
        guard let itemIndex = viewModel.items.firstIndex(where: { $0.id == item.id }) else {
            NSLog("[LibraryView] ERROR: item not found in viewModel.items")
            return
        }

        NSLog("[LibraryView] Found item at index \(itemIndex), current status: \(viewModel.items[itemIndex].status.rawValue)")

        // Update status to processing
        viewModel.items[itemIndex].status = .processing
        viewModel.saveLibrary()

        // Create a temporary PlayerViewModel to handle synthesis
        // Pass audioPlayer so it can update synthesis progress
        let vm = PlayerViewModel(item: viewModel.items[itemIndex], audioPlayer: audioPlayer) { updatedItem in
            viewModel.items[itemIndex] = updatedItem
        }
        await vm.generateOnly()

        NSLog("[LibraryView] generateOnly complete, vm.item.status: \(vm.item.status.rawValue), generatedChunks: \(vm.item.generatedChunks.count)")

        // Update the item in viewModel with generated chunks and status
        viewModel.items[itemIndex] = vm.item
        NSLog("[LibraryView] Updated viewModel.items[\(itemIndex)], new status: \(viewModel.items[itemIndex].status.rawValue)")
        viewModel.saveLibrary()
    }

    // MARK: - Model Status Banner - Redesigned

    private var modelBanner: some View {
        Button {
            showModelSetup = true
        } label: {
            HStack(spacing: 16) {
                // Icon with glow
                ZStack {
                    Circle()
                        .fill(Color.appAccent.opacity(0.2))
                        .frame(width: 48, height: 48)
                        .blur(radius: 8)

                    Circle()
                        .fill(Color.appAccentSubtle)
                        .frame(width: 44, height: 44)

                    Image(systemName: "cpu.fill")
                        .font(.system(size: 20, weight: .medium))
                        .foregroundStyle(Color.appAccent)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text("Set Up Voice Engine")
                        .font(AppTypography.headlineMedium)
                        .foregroundStyle(Color.appTextPrimary)

                    Text("Download Chatterbox Turbo model for on-device TTS")
                        .font(AppTypography.bodySmall)
                        .foregroundStyle(Color.appTextSecondary)
                }

                Spacer()

                Image(systemName: "arrow.down.circle.fill")
                    .font(.system(size: 22))
                    .foregroundStyle(Color.appAccent)
                    .shadow(color: Color.appAccent.opacity(0.5), radius: 6, y: 0)
            }
            .padding(18)
            .glassCard(cornerRadius: 16)
            .padding(.horizontal, 16)
            .padding(.top, 12)
        }
        .buttonStyle(.plain)
    }

    // MARK: - Filter Tabs - Redesigned

    private var filterTabs: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 10) {
                ForEach(LibraryViewModel.ContentFilter.allCases, id: \.self) { filter in
                    FilterTabButton(
                        title: filter.rawValue,
                        icon: filter.icon,
                        isSelected: viewModel.selectedFilter == filter
                    ) {
                        withAnimation(AppAnimation.smooth) {
                            viewModel.selectedFilter = filter
                        }
                    }
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
        }
    }

    // MARK: - Items List - Redesigned

    private var itemsList: some View {
        LazyVStack(spacing: 10) {
            ForEach(Array(viewModel.filteredItems.enumerated()), id: \.element.id) { index, item in
                itemRow(for: item)
            }
        }
    }

    @ViewBuilder
    private func itemRow(for item: LibraryItem) -> some View {
        LibraryItemRow(item: item, onTap: {
            nowPlayingItem = item
            selectedItem = item
        }, audioPlayer: audioPlayer)
        .contextMenu {
            if item.status == .pending {
                Button {
                    Task { await viewModel.synthesize(item: item) }
                } label: {
                    Label("Generate Audio", systemImage: "waveform")
                }
            }

            Button(role: .destructive) {
                viewModel.deleteItem(item)
            } label: {
                Label("Delete", systemImage: "trash")
            }
        }
    }

    // MARK: - Empty State - Redesigned

    private var emptyState: some View {
        VStack(spacing: 24) {
            Spacer()
                .frame(height: 60)

            // Animated icon
            ZStack {
                // Glow
                Circle()
                    .fill(Color.appAccent.opacity(0.15))
                    .frame(width: 140, height: 140)
                    .blur(radius: 30)

                // Background circle
                Circle()
                    .fill(Color.appSurfaceElevated)
                    .frame(width: 100, height: 100)

                // Icon
                Image(systemName: "headphones")
                    .font(.system(size: 40, weight: .ultraLight))
                    .foregroundStyle(
                        LinearGradient(
                            colors: [Color.appAccent, Color.appCoral],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
            }

            VStack(spacing: 8) {
                Text("Your Library is Empty")
                    .font(AppTypography.displaySmall)
                    .foregroundStyle(Color.appTextPrimary)

                Text("Add articles, books, or paste text to start listening")
                    .font(AppTypography.bodyMedium)
                    .foregroundStyle(Color.appTextSecondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 40)
            }

            // Styled Add button
            Button {
                viewModel.showAddContent = true
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "plus")
                    Text("Add Content")
                }
                .font(AppTypography.headlineSmall)
                .foregroundStyle(.black)
                .padding(.horizontal, 28)
                .padding(.vertical, 14)
                .background(
                    Capsule()
                        .fill(AppGradients.accent)
                )
                .shadow(color: Color.appAccent.opacity(0.4), radius: 12, y: 4)
            }
            .buttonStyle(.plain)
            .padding(.top, 8)
        }
    }

    // MARK: - Model Setup Sheet - Redesigned

    private var modelSetupSheet: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 28) {
                    // Header with glow
                    VStack(spacing: 12) {
                        ZStack {
                            Circle()
                                .fill(Color.appAccent.opacity(0.2))
                                .frame(width: 80, height: 80)
                                .blur(radius: 20)

                            Image(systemName: "waveform.badge.plus")
                                .font(.system(size: 36, weight: .ultraLight))
                                .foregroundStyle(Color.appAccent)
                        }

                        Text("Chatterbox Turbo")
                            .font(AppTypography.displayMedium)
                            .foregroundStyle(Color.appTextPrimary)

                        Text("350M parameter on-device text-to-speech model with voice cloning capabilities")
                            .font(AppTypography.bodyMedium)
                            .foregroundStyle(Color.appTextSecondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 24)
                    }
                    .padding(.top, 24)

                    // Model variants
                    VStack(spacing: 12) {
                        ForEach([ModelVariant.q4f16, .q4, .fp16], id: \.rawValue) { variant in
                            ModelVariantCard(
                                variant: variant,
                                isSelected: selectedVariant == variant
                            ) {
                                withAnimation(AppAnimation.quick) {
                                    selectedVariant = variant
                                }
                            }
                        }
                    }
                    .padding(.horizontal, 20)

                    // Download progress
                    if downloadService.isDownloading {
                        VStack(spacing: 12) {
                            ProgressView(value: downloadService.overallProgress)
                                .tint(Color.appAccent)
                                .scaleEffect(y: 1.5)

                            if let component = downloadService.currentComponent {
                                Text("Downloading \(component.displayName)...")
                                    .font(AppTypography.captionLarge)
                                    .foregroundStyle(Color.appTextSecondary)
                            }

                            Text("\(Int(downloadService.overallProgress * 100))%")
                                .font(AppTypography.monoLarge)
                                .foregroundStyle(Color.appAccent)
                        }
                        .padding(.horizontal, 20)
                    }

                    Spacer(minLength: 20)

                    // Download button
                    Button {
                        Task { await downloadService.downloadModels(variant: selectedVariant) }
                    } label: {
                        HStack {
                            if downloadService.isDownloading {
                                ProgressView()
                                    .tint(.black)
                                    .scaleEffect(0.8)
                            } else {
                                Image(systemName: "arrow.down.circle.fill")
                            }
                            Text(downloadService.isDownloading ? "Downloading..." : "Download Model (~558 MB)")
                        }
                        .font(AppTypography.headlineMedium)
                        .foregroundStyle(.black)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 16)
                        .background(
                            Capsule()
                                .fill(AppGradients.accent)
                        )
                        .shadow(color: Color.appAccent.opacity(0.4), radius: 16, y: 6)
                    }
                    .disabled(downloadService.isDownloading)
                    .padding(.horizontal, 20)
                    .padding(.bottom, 24)
                }
            }
            .background(Color.appBackground)
            .navigationTitle("Voice Engine Setup")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { showModelSetup = false }
                        .foregroundStyle(Color.appAccent)
                }
            }
        }
        .preferredColorScheme(.dark)
    }
}

// MARK: - Filter Tab Button

struct FilterTabButton: View {
    let title: String
    let icon: String
    let isSelected: Bool
    let action: () -> Void

    @State private var isPressed = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.system(size: 12, weight: .medium))
                Text(title)
                    .font(AppTypography.captionLarge)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(
                Capsule()
                    .fill(isSelected ? AnyShapeStyle(AppGradients.accent) : AnyShapeStyle(Color.appSurfaceElevated))
            )
            .foregroundStyle(isSelected ? .black : Color.appTextSecondary)
            .scaleEffect(isPressed ? 0.96 : 1.0)
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Model Variant Card

struct ModelVariantCard: View {
    let variant: ModelVariant
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(variant.displayName)
                        .font(AppTypography.headlineSmall)
                        .foregroundStyle(Color.appTextPrimary)

                    if variant == .q4f16 {
                        HStack(spacing: 4) {
                            Image(systemName: "checkmark.seal.fill")
                                .font(.system(size: 10))
                            Text("Recommended for mobile")
                                .font(AppTypography.captionMedium)
                        }
                        .foregroundStyle(Color.appAccent)
                    }
                }

                Spacer()

                // Custom radio button
                ZStack {
                    Circle()
                        .stroke(isSelected ? Color.appAccent : Color.appTextTertiary, lineWidth: 2)
                        .frame(width: 24, height: 24)

                    if isSelected {
                        Circle()
                            .fill(Color.appAccent)
                            .frame(width: 14, height: 14)
                            .shadow(color: Color.appAccent.opacity(0.5), radius: 4, y: 0)
                    }
                }
            }
            .padding(16)
            .background(
                RoundedRectangle(cornerRadius: 14)
                    .fill(isSelected ? Color.appAccentSubtle : Color.appSurfaceElevated)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 14)
                    .stroke(isSelected ? Color.appAccent.opacity(0.5) : Color.clear, lineWidth: 2)
            )
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Library Item Row - Redesigned

struct LibraryItemRow: View {
    let item: LibraryItem
    let onTap: () -> Void
    @ObservedObject var audioPlayer: AudioPlayerService

    @State private var isPressed = false
    @State private var isGenerating = false

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 14) {
                // Cover thumbnail with glow
                coverView
                    .frame(width: 60, height: 60)

                // Content info
                VStack(alignment: .leading, spacing: 5) {
                    Text(item.title)
                        .font(AppTypography.headlineSmall)
                        .foregroundStyle(Color.appTextPrimary)
                        .lineLimit(2)
                        .multilineTextAlignment(.leading)

                    HStack(spacing: 6) {
                        Text(item.displayAuthor)
                            .font(AppTypography.bodySmall)
                            .foregroundStyle(Color.appTextSecondary)
                            .lineLimit(1)

                        if item.duration != nil {
                            Text("·")
                                .foregroundStyle(Color.appTextTertiary)
                            Text(item.formattedDuration)
                                .font(AppTypography.mono)
                                .foregroundStyle(Color.appTextTertiary)
                        }
                    }

                    // Status & progress
                    statusRow
                }

                Spacer()

                // Download/Play button or progress
                downloadButton

                // Chevron with animation
                Image(systemName: "chevron.right")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(Color.appTextTertiary)
                    .opacity(isPressed ? 0.5 : 1)
            }
            .padding(14)
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(Color.appSurface)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(Color.appDivider, lineWidth: 1)
            )
            .scaleEffect(isPressed ? 0.98 : 1.0)
        }
        .buttonStyle(.plain)
        .contentShape(Rectangle())
        .onTapGesture {
            onTap()
        }
        .onLongPressGesture(minimumDuration: .infinity, pressing: { pressing in
            isPressed = pressing
        }, perform: {})
    }

    private var coverView: some View {
        ZStack {
            // Glow when processing
            if item.status == .processing {
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.appAccent.opacity(0.2))
                    .blur(radius: 8)
            }

            RoundedRectangle(cornerRadius: 12)
                .fill(Color.appSurfaceElevated)

            if let data = item.coverImageData,
               let uiImage = UIImage(data: data) {
                Image(uiImage: uiImage)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            } else {
                Image(systemName: iconForSource(item.source))
                    .font(.system(size: 22, weight: .medium))
                    .foregroundStyle(statusIconColor)
            }

            // Status overlay
            if item.status == .processing {
                ZStack {
                    Circle()
                        .fill(Color.black.opacity(0.5))
                        .frame(width: 28, height: 28)

                    ProgressView()
                        .scaleEffect(0.7)
                        .tint(.white)
                }
            }
        }
    }

    private var statusRow: some View {
        HStack(spacing: 8) {
            statusBadge

            if item.progress > 0 && item.progress < 1 {
                CompactWaveformView(progress: item.progress, isPlaying: false)
                    .frame(width: 50, height: 16)
            }
        }
    }

    private var statusBadge: some View {
        HStack(spacing: 5) {
            Circle()
                .fill(statusColor)
                .frame(width: 6, height: 6)
                .shadow(color: statusColor.opacity(0.5), radius: 2, y: 0)

            Text(statusText)
                .font(AppTypography.captionMedium)
                .foregroundStyle(statusColor)
        }
    }

    private var statusColor: Color {
        switch item.status {
        case .pending: return Color.appTextTertiary
        case .processing: return Color.appWarning
        case .ready: return Color.appSuccess
        case .error: return Color.appError
        }
    }

    private var statusText: String {
        switch item.status {
        case .pending: return "Pending"
        case .processing: return "Generating..."
        case .ready: return "Ready"
        case .error: return "Error"
        }
    }

    /// Download/Generate button for quick-start from library
    @ViewBuilder
    private var downloadButton: some View {
        // Use synthesis progress for background generation, fallback to playback progress
        let progress = audioPlayer.synthesisProgress[item.id] ?? audioPlayer.progressForItem(item.id)
        let isCurrentItem = audioPlayer.currentPlayingItemID == item.id
        let isQueued = audioPlayer.isItemQueued(item.id)

        // Show progress circle if generating/queued
        if item.status == .processing || isQueued || progress > 0 {
            progressCircle(progress: progress)
        }
        // Show download button for pending/error items
        else if item.status == .pending || item.status == .error {
            generateButton
        }
        // Show play button for ready items or currently playing
        else {
            playIconButton(isCurrentItem: isCurrentItem)
        }
    }

    private func progressCircle(progress: Double) -> some View {
        ZStack {
            Circle()
                .stroke(Color.appAccent.opacity(0.3), lineWidth: 3)
                .frame(width: 36, height: 36)

            Circle()
                .trim(from: 0, to: progress)
                .stroke(Color.appAccent, style: StrokeStyle(lineWidth: 3, lineCap: .round))
                .frame(width: 36, height: 36)
                .rotationEffect(.degrees(-90))

            Text("\(Int(progress * 100))")
                .font(.system(size: 9, weight: .bold))
                .foregroundStyle(Color.appAccent)
        }
    }

    private var generateButton: some View {
        Button {
            handleGenerateTap()
        } label: {
            Image(systemName: "arrow.down.circle.fill")
                .font(.system(size: 22))
                .foregroundStyle(Color.appAccent)
        }
        .buttonStyle(.plain)
    }

    private func playIconButton(isCurrentItem: Bool) -> some View {
        Button {
            // Handled by mini player
        } label: {
            Image(systemName: isCurrentItem ? "pause.circle.fill" : "play.circle.fill")
                .font(.system(size: 22))
                .foregroundStyle(Color.appAccent)
        }
        .buttonStyle(.plain)
    }

    private func handleGenerateTap() {
        // Add to queue
        audioPlayer.startPlayingItem(item.id)

        // Post notification to trigger generation
        NotificationCenter.default.post(
            name: .startGenerationFromLibrary,
            object: nil,
            userInfo: ["item": item]
        )
    }

    private var statusIconColor: Color {
        switch item.status {
        case .ready: return Color.appAccent
        case .processing: return Color.appWarning
        default: return Color.appTextSecondary
        }
    }

    private func iconForSource(_ source: ContentSource) -> String {
        switch source {
        case .webpage: return "globe"
        case .epub: return "book.closed"
        case .pdf: return "doc.text"
        case .text: return "text.alignleft"
        }
    }
}

// MARK: - UTType Extension

extension UTType {
    static let epub = UTType("org.idpf.epub-container")
}
