import SwiftUI

// MARK: - Mini Player (persistent bottom bar) - Redesigned with Glassmorphism

struct MiniPlayerView: View {
    let item: LibraryItem
    let audioPlayer: AudioPlayerService
    let onTap: () -> Void

    @State private var isPressed = false

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 14) {
                // Cover / Icon with glow effect
                coverView
                    .frame(width: 48, height: 48)

                // Title & Progress
                VStack(alignment: .leading, spacing: 4) {
                    Text(item.title)
                        .font(AppTypography.headlineSmall)
                        .foregroundStyle(Color.appTextPrimary)
                        .lineLimit(1)

                    // Progress bar with time labels
                    progressSection
                }

                Spacer()

                // Play/Pause Button with glow
                playPauseButton
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(glassBackground)
            .overlay(backgroundBorder)
            .shadow(color: Color.appAccent.opacity(0.15), radius: 20, x: 0, y: -4)
            .scaleEffect(isPressed ? 0.98 : 1.0)
            .animation(AppAnimation.quick, value: isPressed)
            .padding(.horizontal, 8)
        }
        .buttonStyle(.plain)
        .onLongPressGesture(minimumDuration: .infinity, pressing: { pressing in
            isPressed = pressing
        }, perform: {})
    }

    // MARK: - Cover View with animated glow

    private var coverView: some View {
        ZStack {
            // Glow effect when playing
            if audioPlayer.isPlaying {
                RoundedRectangle(cornerRadius: 10)
                    .fill(Color.appAccent.opacity(0.3))
                    .blur(radius: 8)
            }

            RoundedRectangle(cornerRadius: 10)
                .fill(Color.appSurfaceElevated)

            if let data = item.coverImageData,
               let uiImage = UIImage(data: data) {
                Image(uiImage: uiImage)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
            } else {
                Image(systemName: iconForSource)
                    .font(.system(size: 18, weight: .medium))
                    .foregroundStyle(audioPlayer.isPlaying ? Color.appAccent : Color.appTextSecondary)
            }
        }
    }

    private var iconForSource: String {
        switch item.source {
        case .webpage: return "globe"
        case .epub: return "book.closed"
        case .pdf: return "doc.text"
        case .text: return "text.alignleft"
        }
    }

    // MARK: - Progress Section

    private var progressSection: some View {
        HStack(spacing: 8) {
            Text(audioPlayer.formattedCurrentTime)
                .font(AppTypography.mono)
                .foregroundStyle(Color.appTextTertiary)
                .monospacedDigit()

            // Custom progress bar
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    // Track
                    Capsule()
                        .fill(Color.appSurfaceHover)
                        .frame(height: 4)

                    // Progress with gradient
                    Capsule()
                        .fill(AppGradients.accent)
                        .frame(width: max(0, geo.size.width * audioPlayer.progress), height: 4)
                        .shadow(color: Color.appAccent.opacity(0.5), radius: 4, y: 0)
                }
            }
            .frame(height: 4)

            Text(audioPlayer.formattedRemaining)
                .font(AppTypography.mono)
                .foregroundStyle(Color.appTextTertiary)
                .monospacedDigit()
        }
    }

    // MARK: - Play/Pause Button

    private var playPauseButton: some View {
        Button {
            audioPlayer.togglePlayPause()
        } label: {
            ZStack {
                // Glow background when playing
                if audioPlayer.isPlaying {
                    Circle()
                        .fill(Color.appAccent.opacity(0.2))
                        .frame(width: 48, height: 48)
                        .blur(radius: 6)
                }

                Circle()
                    .fill(audioPlayer.isPlaying ? AnyShapeStyle(AppGradients.accent) : AnyShapeStyle(Color.appSurfaceElevated))
                    .frame(width: 40, height: 40)
                    .overlay(
                        Circle()
                            .stroke(audioPlayer.isPlaying ? Color.clear : Color.appGlassBorder, lineWidth: 1)
                    )

                Image(systemName: audioPlayer.isPlaying ? "pause.fill" : "play.fill")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundStyle(audioPlayer.isPlaying ? .white : Color.appTextPrimary)
                    .offset(x: audioPlayer.isPlaying ? 0 : 1)
            }
        }
        .buttonStyle(PlayButtonStyle())
    }

    // MARK: - Glass Background

    private var glassBackground: some View {
        RoundedRectangle(cornerRadius: 18)
            .fill(.ultraThinMaterial)
            .overlay(
                RoundedRectangle(cornerRadius: 18)
                    .fill(
                        LinearGradient(
                            colors: [
                                Color.white.opacity(0.1),
                                Color.white.opacity(0.02)
                            ],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
            )
    }

    private var backgroundBorder: some View {
        RoundedRectangle(cornerRadius: 18)
            .stroke(
                LinearGradient(
                    colors: [
                        Color.white.opacity(0.15),
                        Color.white.opacity(0.05)
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ),
                lineWidth: 1
            )
    }
}

// MARK: - Custom Button Style for Play Button

struct PlayButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.9 : 1.0)
            .animation(AppAnimation.quick, value: configuration.isPressed)
    }
}
