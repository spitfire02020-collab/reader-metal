import SwiftUI
import UIKit

// MARK: - Midnight Studio Color Palette

extension Color {
    // MARK: - Backgrounds (Deep black with subtle warmth)
    static let appBackground = Color(hex: "0A0A0A")
    static let appSurface = Color(hex: "141414")
    static let appSurfaceElevated = Color(hex: "1C1C1E")
    static let appSurfaceHover = Color(hex: "252528")

    // MARK: - Primary Accent (Amber/Gold - warm, studio lamp feel)
    static let appAccent = Color(hex: "F59E0B")        // Amber
    static let appAccentLight = Color(hex: "FBBF24")   // Light Amber
    static let appAccentDark = Color(hex: "D97706")    // Dark Amber
    static let appAccentSubtle = Color(hex: "F59E0B").opacity(0.15)

    // MARK: - Supporting Accents
    static let appCoral = Color(hex: "F97316")         // Coral
    static let appTeal = Color(hex: "14B8A6")         // Teal
    static let appLavender = Color(hex: "A78BFA")     // Lavender
    static let appRose = Color(hex: "F43F5E")         // Rose (for errors)

    // MARK: - Text Colors
    static let appTextPrimary = Color(hex: "FAFAFA")
    static let appTextSecondary = Color(hex: "A1A1AA")
    static let appTextTertiary = Color(hex: "71717A")

    // MARK: - Semantic Colors
    static let appSuccess = Color(hex: "22C55E")
    static let appWarning = Color(hex: "F59E0B")
    static let appError = Color(hex: "EF4444")

    // MARK: - Waveform Colors
    static let appWaveformActive = Color(hex: "F59E0B")
    static let appWaveformInactive = Color(hex: "3F3F46")

    // MARK: - Gradient Colors
    static let appGradientStart = Color(hex: "F59E0B")
    static let appGradientEnd = Color(hex: "F97316")

    // MARK: - Glassmorphism
    static let appGlass = Color.white.opacity(0.08)
    static let appGlassBorder = Color.white.opacity(0.12)
    static let appGlassShine = Color.white.opacity(0.04)

    // MARK: - Dividers
    static let appDivider = Color.white.opacity(0.06)
    static let appDividerStrong = Color.white.opacity(0.1)
}

// MARK: - Glow Effects

struct GlowModifier: ViewModifier {
    let color: Color
    let radius: CGFloat

    func body(content: Content) -> some View {
        content
            .shadow(color: color.opacity(0.5), radius: radius, x: 0, y: 0)
    }
}

extension View {
    func glow(color: Color, radius: CGFloat = 10) -> some View {
        modifier(GlowModifier(color: color, radius: radius))
    }

    func glassCard(cornerRadius: CGFloat = 16) -> some View {
        self
            .background(
                RoundedRectangle(cornerRadius: cornerRadius)
                    .fill(Color.appGlass)
            )
            .background(
                RoundedRectangle(cornerRadius: cornerRadius)
                    .stroke(Color.appGlassBorder, lineWidth: 1)
            )
    }

    func ambientGlow(color: Color) -> some View {
        self
            .shadow(color: color.opacity(0.3), radius: 20, x: 0, y: 8)
    }
}

// MARK: - Hex Color Initializer

extension Color {
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a, r, g, b: UInt64
        switch hex.count {
        case 3: // RGB (12-bit)
            (a, r, g, b) = (255, (int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6: // RGB (24-bit)
            (a, r, g, b) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        case 8: // ARGB (32-bit)
            (a, r, g, b) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default:
            (a, r, g, b) = (255, 0, 0, 0)
        }
        self.init(
            .sRGB,
            red: Double(r) / 255,
            green: Double(g) / 255,
            blue: Double(b) / 255,
            opacity: Double(a) / 255
        )
    }
}

// MARK: - Typography System

struct AppTypography {
    // MARK: - Display (Large titles, headers)
    static let displayLarge = Font.system(size: 32, weight: .bold, design: .default)
    static let displayMedium = Font.system(size: 24, weight: .bold, design: .default)
    static let displaySmall = Font.system(size: 20, weight: .semibold, design: .default)

    // MARK: - Headlines
    static let headlineLarge = Font.system(size: 18, weight: .semibold, design: .default)
    static let headlineMedium = Font.system(size: 16, weight: .semibold, design: .default)
    static let headlineSmall = Font.system(size: 15, weight: .medium, design: .default)

    // MARK: - Body
    static let bodyLarge = Font.system(size: 16, weight: .regular, design: .default)
    static let bodyMedium = Font.system(size: 14, weight: .regular, design: .default)
    static let bodySmall = Font.system(size: 13, weight: .regular, design: .default)

    // MARK: - Caption
    static let captionLarge = Font.system(size: 12, weight: .medium, design: .default)
    static let captionMedium = Font.system(size: 11, weight: .medium, design: .default)
    static let captionSmall = Font.system(size: 10, weight: .medium, design: .default)

    // MARK: - Monospace (Timestamps, technical)
    static let mono = Font.system(size: 12, weight: .medium, design: .monospaced)
    static let monoLarge = Font.system(size: 14, weight: .medium, design: .monospaced)
}

// MARK: - Animation Presets

struct AppAnimation {
    static let quick = SwiftUI.Animation.easeOut(duration: 0.2)
    static let standard = SwiftUI.Animation.easeInOut(duration: 0.3)
    static let smooth = SwiftUI.Animation.spring(response: 0.4, dampingFraction: 0.7)
    static let bouncy = SwiftUI.Animation.spring(response: 0.5, dampingFraction: 0.6)
    static let gradual = SwiftUI.Animation.easeInOut(duration: 0.5)

    static func spring(response: Double, dampingFraction: Double = 0.7) -> SwiftUI.Animation {
        SwiftUI.Animation.spring(response: response, dampingFraction: dampingFraction)
    }
}

// MARK: - Corner Radius System

struct AppCornerRadius {
    static let small: CGFloat = 6
    static let medium: CGFloat = 10
    static let large: CGFloat = 14
    static let extraLarge: CGFloat = 20
    static let pill: CGFloat = 100
}

// MARK: - Spacing System

struct AppSpacing {
    static let xxs: CGFloat = 4
    static let xs: CGFloat = 8
    static let sm: CGFloat = 12
    static let md: CGFloat = 16
    static let lg: CGFloat = 20
    static let xl: CGFloat = 24
    static let xxl: CGFloat = 32
}

// MARK: - Gradient Presets

struct AppGradients {
    static let accent = LinearGradient(
        colors: [Color.appAccent, Color.appCoral],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    static let subtle = LinearGradient(
        colors: [Color.appAccent.opacity(0.3), Color.appCoral.opacity(0.3)],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    static let background = LinearGradient(
        colors: [Color.appBackground, Color.appSurface],
        startPoint: .top,
        endPoint: .bottom
    )

    static let glass = LinearGradient(
        colors: [Color.appGlassShine, Color.appGlass, Color.appGlass.opacity(0.5)],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )
}

// MARK: - Noise/Grain Texture Overlay

struct NoiseTextureModifier: ViewModifier {
    let opacity: Double
    let blendMode: BlendMode

    init(opacity: Double = 0.03, blendMode: BlendMode = .overlay) {
        self.opacity = opacity
        self.blendMode = blendMode
    }

    func body(content: Content) -> some View {
        content
            .overlay {
                NoiseView(opacity: opacity)
                    .blendMode(blendMode)
                    .allowsHitTesting(false)
            }
    }
}

// MARK: - Animated Noise View

struct NoiseView: View {
    let opacity: Double
    @State private var noiseOffset: CGPoint = .zero

    var body: some View {
        Canvas { context, size in
            var path = Path()
            let gridSize: CGFloat = 3

            for y in stride(from: 0, to: size.height, by: gridSize) {
                for x in stride(from: 0, to: size.width, by: gridSize) {
                    let alpha = Double.random(in: 0...1)
                    if alpha > 0.6 {
                        let rect = CGRect(x: x, y: y, width: gridSize, height: gridSize)
                        path.addRect(rect)
                    }
                }
            }

            context.fill(path, with: .color(.white.opacity(opacity)))
        }
        .blendMode(.overlay)
    }
}

// MARK: - View Extension for Noise

extension View {
    /// Adds a subtle noise/grain texture overlay for that premium audio software feel
    func noiseTexture(_ enabled: Bool = true, opacity: Double = 0.025) -> some View {
        self.modifier(NoiseTextureModifier(opacity: enabled ? opacity : 0))
    }

    /// Adds noise with a specific blend mode
    func noiseOverlay(opacity: Double = 0.03, blendMode: BlendMode = .overlay) -> some View {
        self.modifier(NoiseTextureModifier(opacity: opacity, blendMode: blendMode))
    }
}

// MARK: - Animated Noise (with subtle movement)

struct AnimatedNoiseModifier: ViewModifier {
    let opacity: Double
    @State private var offset: CGPoint = .zero

    func body(content: Content) -> some View {
        content
            .overlay {
                GeometryReader { geometry in
                    NoiseCanvas(opacity: opacity)
                        .offset(x: offset.x, y: offset.y)
                }
                .blendMode(.overlay)
                .allowsHitTesting(false)
            }
            .onAppear {
                withAnimation(.linear(duration: 20).repeatForever(autoreverses: false)) {
                    offset = CGPoint(x: -50, y: -50)
                }
            }
    }
}

struct NoiseCanvas: View {
    let opacity: Double

    var body: some View {
        Canvas { context, size in
            let cellSize: CGFloat = 2
            var path = Path()

            for y in stride(from: 0, to: size.height, by: cellSize) {
                for x in stride(from: 0, to: size.width, by: cellSize) {
                    let value = Double.random(in: 0...1)
                    if value > 0.7 {
                        path.addRect(CGRect(x: x, y: y, width: cellSize, height: cellSize))
                    }
                }
            }

            context.fill(path, with: .color(.white.opacity(opacity)))
        }
    }
}

extension View {
    /// Adds animated noise texture that slowly drifts
    func animatedNoise(opacity: Double = 0.02) -> some View {
        self.modifier(AnimatedNoiseModifier(opacity: opacity))
    }
}
