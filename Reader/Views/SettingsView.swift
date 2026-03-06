import SwiftUI

// MARK: - Settings View - Redesigned

struct SettingsView: View {
    @StateObject private var downloadService = ModelDownloadService.shared
    @AppStorage("default_playback_speed") private var defaultSpeed: Double = 1.0
    @AppStorage("auto_generate_audio") private var autoGenerate = false
    @AppStorage("high_quality_mode") private var highQuality = true
    @AppStorage("skip_silence") private var skipSilence = false
    @State private var showDeleteConfirmation = false

    var body: some View {
        List {
            // Voice Engine Section
            Section {
                voiceEngineRow

                if downloadService.isModelReady {
                    storageRow
                    deleteModelButton
                }
            } header: {
                Text("Voice Engine")
                    .font(AppTypography.captionMedium)
                    .foregroundStyle(Color.appTextTertiary)
            }

            // Playback Section
            Section {
                speedPicker
                autoGenerateToggle
                skipSilenceToggle
                highQualityToggle
            } header: {
                Text("Playback")
                    .font(AppTypography.captionMedium)
                    .foregroundStyle(Color.appTextTertiary)
            }

            // About Section
            Section {
                versionRow
                modelLink
                licenseRow
                poweredBySection
            } header: {
                Text("About")
                    .font(AppTypography.captionMedium)
                    .foregroundStyle(Color.appTextTertiary)
            }
        }
        .listStyle(.insetGrouped)
        .scrollContentBackground(.hidden)
        .background(Color.appBackground)
        .noiseTexture(opacity: 0.025)
        .navigationTitle("Settings")
        .toolbarColorScheme(.dark, for: .navigationBar)
        .alert("Delete Model?", isPresented: $showDeleteConfirmation) {
            Button("Cancel", role: .cancel) {}
            Button("Delete", role: .destructive) {
                downloadService.deleteModels()
            }
        } message: {
            Text("This will free up \(downloadService.modelSizeOnDisk()) of storage. You'll need to download the model again to generate audio.")
        }
        .task {
            await downloadService.checkModelAvailabilityAsync()
        }
    }

    // MARK: - Voice Engine Row

    private var voiceEngineRow: some View {
        HStack(spacing: 14) {
            settingsIcon("cpu.fill", color: .appAccent)

            VStack(alignment: .leading, spacing: 3) {
                Text("Chatterbox Turbo")
                    .font(AppTypography.headlineSmall)
                    .foregroundStyle(Color.appTextPrimary)

                Text(downloadService.isModelReady ? "Installed" : "Not installed")
                    .font(AppTypography.bodySmall)
                    .foregroundStyle(downloadService.isModelReady ? Color.appSuccess : Color.appTextTertiary)
            }

            Spacer()

            if downloadService.isModelReady {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 20))
                    .foregroundStyle(Color.appSuccess)
                    .shadow(color: Color.appSuccess.opacity(0.4), radius: 4, y: 0)
            }
        }
        .padding(.vertical, 4)
    }

    // MARK: - Storage Row

    private var storageRow: some View {
        HStack(spacing: 14) {
            settingsIcon("internaldrive.fill", color: .appTextSecondary)

            Text("Storage Used")
                .font(AppTypography.headlineSmall)
                .foregroundStyle(Color.appTextPrimary)

            Spacer()

            Text(downloadService.modelSizeOnDisk())
                .font(AppTypography.bodyMedium)
                .foregroundStyle(Color.appTextSecondary)
        }
        .padding(.vertical, 4)
    }

    // MARK: - Delete Model Button

    private var deleteModelButton: some View {
        Button(role: .destructive) {
            showDeleteConfirmation = true
        } label: {
            HStack(spacing: 14) {
                settingsIcon("trash.fill", color: .appError)

                Text("Delete Model")
                    .font(AppTypography.headlineSmall)
                    .foregroundStyle(Color.appError)
            }
            .padding(.vertical, 4)
        }
    }

    // MARK: - Speed Picker

    private var speedPicker: some View {
        HStack(spacing: 14) {
            settingsIcon("gauge.with.dots.needle.50percent", color: .appAccent)

            Text("Default Speed")
                .font(AppTypography.headlineSmall)
                .foregroundStyle(Color.appTextPrimary)

            Spacer()

            Picker("", selection: $defaultSpeed) {
                Text("0.5x").tag(0.5)
                Text("0.75x").tag(0.75)
                Text("1.0x").tag(1.0)
                Text("1.25x").tag(1.25)
                Text("1.5x").tag(1.5)
                Text("1.75x").tag(1.75)
                Text("2.0x").tag(2.0)
            }
            .pickerStyle(.menu)
            .tint(Color.appAccent)
        }
    }

    // MARK: - Auto Generate Toggle

    private var autoGenerateToggle: some View {
        Toggle(isOn: $autoGenerate) {
            HStack(spacing: 14) {
                settingsIcon("waveform.badge.plus", color: .appAccent)

                Text("Auto-Generate Audio")
                    .font(AppTypography.headlineSmall)
                    .foregroundStyle(Color.appTextPrimary)
            }
        }
        .tint(Color.appAccent)
    }

    // MARK: - Skip Silence Toggle

    private var skipSilenceToggle: some View {
        Toggle(isOn: $skipSilence) {
            HStack(spacing: 14) {
                settingsIcon("forward.fill", color: .appAccent)

                Text("Skip Silence")
                    .font(AppTypography.headlineSmall)
                    .foregroundStyle(Color.appTextPrimary)
            }
        }
        .tint(Color.appAccent)
    }

    // MARK: - High Quality Toggle

    private var highQualityToggle: some View {
        Toggle(isOn: $highQuality) {
            HStack(spacing: 14) {
                settingsIcon("sparkles", color: .appAccent)

                VStack(alignment: .leading, spacing: 3) {
                    Text("High Quality Mode")
                        .font(AppTypography.headlineSmall)
                        .foregroundStyle(Color.appTextPrimary)

                    Text("Uses more battery but produces better audio")
                        .font(AppTypography.captionMedium)
                        .foregroundStyle(Color.appTextTertiary)
                }
            }
        }
        .tint(Color.appAccent)
    }

    // MARK: - Version Row

    private var versionRow: some View {
        HStack(spacing: 14) {
            settingsIcon("info.circle.fill", color: .appTextSecondary)

            Text("Version")
                .font(AppTypography.headlineSmall)
                .foregroundStyle(Color.appTextPrimary)

            Spacer()

            Text("1.0.0")
                .font(AppTypography.bodyMedium)
                .foregroundStyle(Color.appTextTertiary)
        }
    }

    // MARK: - Model Link

    private var modelLink: some View {
        Link(destination: URL(string: "https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX")!) {
            HStack(spacing: 14) {
                settingsIcon("link", color: .appTextSecondary)

                Text("Chatterbox Turbo Model")
                    .font(AppTypography.headlineSmall)
                    .foregroundStyle(Color.appTextPrimary)

                Spacer()

                Image(systemName: "arrow.up.right")
                    .font(.system(size: 12))
                    .foregroundStyle(Color.appTextTertiary)
            }
        }
    }

    // MARK: - License Row

    private var licenseRow: some View {
        HStack(spacing: 14) {
            settingsIcon("scroll.fill", color: .appTextSecondary)

            VStack(alignment: .leading, spacing: 3) {
                Text("License")
                    .font(AppTypography.headlineSmall)
                    .foregroundStyle(Color.appTextPrimary)

                Text("MIT License")
                    .font(AppTypography.captionMedium)
                    .foregroundStyle(Color.appTextTertiary)
            }
        }
    }

    // MARK: - Powered By Section

    private var poweredBySection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Powered by")
                .font(AppTypography.captionMedium)
                .foregroundStyle(Color.appTextTertiary)

            HStack(spacing: 12) {
                techBadge("Chatterbox Turbo", subtitle: "Resemble AI")
                techBadge("ONNX Runtime", subtitle: "Microsoft")
            }
        }
        .padding(.vertical, 8)
    }

    // MARK: - Helpers

    private func settingsIcon(_ name: String, color: Color) -> some View {
        ZStack {
            RoundedRectangle(cornerRadius: 8)
                .fill(color.opacity(0.15))
                .frame(width: 32, height: 32)

            Image(systemName: name)
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(color)
        }
    }

    private func techBadge(_ name: String, subtitle: String) -> some View {
        VStack(spacing: 4) {
            Text(name)
                .font(AppTypography.captionMedium)
                .foregroundStyle(Color.appTextPrimary)

            Text(subtitle)
                .font(AppTypography.captionSmall)
                .foregroundStyle(Color.appTextTertiary)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color.appSurfaceElevated)
        )
    }
}

#Preview {
    NavigationStack {
        SettingsView()
    }
    .preferredColorScheme(.dark)
}
