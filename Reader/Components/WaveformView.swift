import SwiftUI

// MARK: - Waveform Visualizer - Redesigned with Amber Glow

struct WaveformView: View {
    let samples: [Float]
    let progress: Double
    let accentColor: Color
    let inactiveColor: Color

    @State private var animatedProgress: Double = 0

    init(
        samples: [Float],
        progress: Double = 0,
        accentColor: Color = .appWaveformActive,
        inactiveColor: Color = .appWaveformInactive
    ) {
        self.samples = samples
        self.progress = progress
        self.accentColor = accentColor
        self.inactiveColor = inactiveColor
    }

    var body: some View {
        GeometryReader { geometry in
            let barCount = samples.count
            let spacing: CGFloat = 2
            let totalSpacing = spacing * CGFloat(barCount - 1)
            let barWidth = max(2, (geometry.size.width - totalSpacing) / CGFloat(barCount))
            let midY = geometry.size.height / 2

            ZStack {
                // Glow layer behind active waveform
                Canvas { context, size in
                    for (index, sample) in samples.enumerated() {
                        let barProgress = Double(index) / Double(max(1, barCount - 1))
                        if barProgress <= animatedProgress {
                            let x = CGFloat(index) * (barWidth + spacing)
                            let barHeight = max(3, CGFloat(sample) * size.height * 0.8)

                            let glowRect = CGRect(
                                x: x - 2,
                                y: midY - barHeight / 2 - 2,
                                width: barWidth + 4,
                                height: barHeight + 4
                            )

                            let glowPath = RoundedRectangle(cornerRadius: (barWidth + 4) / 2)
                                .path(in: glowRect)

                            context.fill(
                                glowPath,
                                with: .color(accentColor.opacity(0.3))
                            )
                        }
                    }
                }

                // Main waveform
                Canvas { context, size in
                    for (index, sample) in samples.enumerated() {
                        let x = CGFloat(index) * (barWidth + spacing)
                        let barHeight = max(3, CGFloat(sample) * size.height * 0.8)

                        let rect = CGRect(
                            x: x,
                            y: midY - barHeight / 2,
                            width: barWidth,
                            height: barHeight
                        )

                        let barProgress = Double(index) / Double(max(1, barCount - 1))
                        let isActive = barProgress <= animatedProgress

                        let path = RoundedRectangle(cornerRadius: barWidth / 2)
                            .path(in: rect)

                        context.fill(
                            path,
                            with: .color(isActive ? accentColor : inactiveColor)
                        )
                    }
                }
            }
        }
        .onAppear {
            withAnimation(AppAnimation.smooth) {
                animatedProgress = progress
            }
        }
        .onChange(of: progress) { _, newValue in
            withAnimation(AppAnimation.smooth) {
                animatedProgress = newValue
            }
        }
    }
}

// MARK: - Animated Waveform (for synthesis progress) - Redesigned

struct AnimatedWaveformView: View {
    @State private var phase: CGFloat = 0
    @State private var isVisible = false
    let barCount: Int
    let isAnimating: Bool
    let accentColor: Color

    init(barCount: Int = 40, isAnimating: Bool = true, accentColor: Color = .appAccent) {
        self.barCount = barCount
        self.isAnimating = isAnimating
        self.accentColor = accentColor
    }

    var body: some View {
        GeometryReader { geometry in
            HStack(spacing: 2) {
                ForEach(0..<barCount, id: \.self) { index in
                    let height = barHeight(for: index, totalHeight: geometry.size.height)

                    RoundedRectangle(cornerRadius: 2)
                        .fill(
                            LinearGradient(
                                colors: [accentColor, accentColor.opacity(0.6)],
                                startPoint: .bottom,
                                endPoint: .top
                            )
                        )
                        .frame(width: 3, height: height)
                        .shadow(color: accentColor.opacity(0.5), radius: 2, y: 0)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
            .opacity(isVisible ? 1 : 0)
        }
        .onAppear {
            if isAnimating {
                isVisible = true
                withAnimation(.linear(duration: 2.5).repeatForever(autoreverses: false)) {
                    phase = .pi * 2
                }
            }
        }
    }

    private func barHeight(for index: Int, totalHeight: CGFloat) -> CGFloat {
        let normalized = CGFloat(index) / CGFloat(barCount)
        let wave1 = sin(normalized * .pi * 4 + phase) * 0.35
        let wave2 = sin(normalized * .pi * 7 + phase * 1.5) * 0.25
        let wave3 = sin(normalized * .pi * 2 + phase * 0.7) * 0.15
        let envelope = sin(normalized * .pi) // Fade edges
        let amplitude = (0.15 + wave1 + wave2 + wave3) * envelope
        return max(4, CGFloat(amplitude) * totalHeight * 0.9)
    }
}

// MARK: - Seekable Waveform - Redesigned

struct SeekableWaveformView: View {
    let samples: [Float]
    @Binding var progress: Double
    let onSeek: (Double) -> Void

    @State private var isDragging = false
    @State private var dragProgress: Double = 0

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Background track
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.appWaveformInactive.opacity(0.3))

                // Progress fill with glow
                WaveformView(
                    samples: samples,
                    progress: isDragging ? dragProgress : progress
                )
                .mask(
                    RoundedRectangle(cornerRadius: 4)
                        .frame(width: geometry.size.width * (isDragging ? dragProgress : progress))
                )

                // Scrubber handle
                Circle()
                    .fill(Color.appAccent)
                    .frame(width: isDragging ? 16 : 12, height: isDragging ? 16 : 12)
                    .shadow(color: Color.appAccent.opacity(0.6), radius: isDragging ? 8 : 4)
                    .position(x: geometry.size.width * (isDragging ? dragProgress : progress), y: geometry.size.height / 2)
                    .animation(AppAnimation.quick, value: isDragging)
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        isDragging = true
                        dragProgress = max(0, min(1, Double(value.location.x / geometry.size.width)))
                    }
                    .onEnded { value in
                        let finalProgress = max(0, min(1, Double(value.location.x / geometry.size.width)))
                        onSeek(finalProgress)
                        isDragging = false
                    }
            )
        }
        .frame(height: 50)
    }
}

// MARK: - Compact Waveform (for list items)

struct CompactWaveformView: View {
    let progress: Double
    let isPlaying: Bool

    @State private var animatedProgress: Double = 0

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                // Background bars
                HStack(spacing: 1.5) {
                    ForEach(0..<20, id: \.self) { index in
                        let height = CGFloat.random(in: 4...geometry.size.height * 0.8)
                        RoundedRectangle(cornerRadius: 1)
                            .fill(Color.appWaveformInactive.opacity(0.4))
                            .frame(width: 2, height: height)
                    }
                }

                // Progress overlay
                HStack(spacing: 1.5) {
                    ForEach(0..<20, id: \.self) { index in
                        let barProgress = Double(index) / 19.0
                        let height = CGFloat.random(in: 4...geometry.size.height * 0.8)

                        if barProgress <= animatedProgress {
                            RoundedRectangle(cornerRadius: 1)
                                .fill(
                                    LinearGradient(
                                        colors: [Color.appAccent, Color.appCoral],
                                        startPoint: .bottom,
                                        endPoint: .top
                                    )
                                )
                                .frame(width: 2, height: height)
                                .shadow(color: Color.appAccent.opacity(0.4), radius: 2, y: 0)
                        }
                    }
                }
                .mask(RoundedRectangle(cornerRadius: 2))
            }
        }
        .onAppear {
            if isPlaying {
                withAnimation(AppAnimation.smooth) {
                    animatedProgress = progress
                }
            }
        }
        .onChange(of: progress) { _, newValue in
            withAnimation(AppAnimation.smooth) {
                animatedProgress = newValue
            }
        }
    }
}

#Preview {
    VStack(spacing: 40) {
        WaveformView(
            samples: (0..<100).map { _ in Float.random(in: 0.1...1.0) },
            progress: 0.4
        )
        .frame(height: 60)

        AnimatedWaveformView()
            .frame(height: 80)

        CompactWaveformView(progress: 0.6, isPlaying: true)
            .frame(height: 24)
    }
    .padding()
    .background(Color.appBackground)
}
