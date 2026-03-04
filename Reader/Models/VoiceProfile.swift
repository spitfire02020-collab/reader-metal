import Foundation

// MARK: - Voice Profile

struct VoiceProfile: Identifiable, Codable, Hashable {
    let id: String
    var name: String
    var description: String
    var isBuiltIn: Bool
    var referenceAudioPath: String?
    var sampleRate: Int
    var language: String
    var tags: [String]

    static let defaultVoice = VoiceProfile(
        id: "default",
        name: "Nova",
        description: "Clear, professional voice with natural intonation",
        isBuiltIn: true,
        referenceAudioPath: nil,
        sampleRate: 24000,
        language: "en",
        tags: ["neutral", "professional"]
    )

    static let builtInVoices: [VoiceProfile] = [
        defaultVoice,
        VoiceProfile(
            id: "warm",
            name: "Aria",
            description: "Warm, conversational tone perfect for stories",
            isBuiltIn: true,
            referenceAudioPath: nil,
            sampleRate: 24000,
            language: "en",
            tags: ["warm", "storytelling"]
        ),
        VoiceProfile(
            id: "energetic",
            name: "Kai",
            description: "Energetic and clear, great for articles and news",
            isBuiltIn: true,
            referenceAudioPath: nil,
            sampleRate: 24000,
            language: "en",
            tags: ["energetic", "news"]
        ),
        VoiceProfile(
            id: "calm",
            name: "Luna",
            description: "Calm and soothing, ideal for long reading sessions",
            isBuiltIn: true,
            referenceAudioPath: nil,
            sampleRate: 24000,
            language: "en",
            tags: ["calm", "relaxing"]
        ),
        VoiceProfile(
            id: "deep",
            name: "Atlas",
            description: "Deep, authoritative voice for non-fiction",
            isBuiltIn: true,
            referenceAudioPath: nil,
            sampleRate: 24000,
            language: "en",
            tags: ["deep", "authoritative"]
        ),
    ]
}

// MARK: - Synthesis Settings

struct SynthesisSettings: Codable {
    var speed: Double          // 0.5 - 2.0, default 1.0
    var stability: Double      // 0.0 - 1.0
    var clarity: Double        // 0.0 - 1.0
    var maxNewTokens: Int      // max generation tokens
    var repetitionPenalty: Double
    var seed: Int             // seed for reproducible output, 0 = random

    static let defaults = SynthesisSettings(
        speed: 1.0,
        stability: 0.5,
        clarity: 0.75,
        maxNewTokens: 1024,
        repetitionPenalty: 1.2,
        seed: 0
    )
}
