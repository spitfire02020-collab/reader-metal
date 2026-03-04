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
    var speed: Double          // 0.5 - 2.0, default 1.0 (maps to speedFactor)
    var exaggeration: Double  // 0.25 - 2.0, default 0.5 (controls expressiveness)
    var cfgWeight: Double      // 0.2 - 1.0, default 0.5 (classifier-free guidance)
    var seed: Int             // seed for reproducible output, 0 = random

    static let defaults = SynthesisSettings(
        speed: 1.0,
        exaggeration: 0.5,
        cfgWeight: 0.5,
        seed: 0
    )
}

// MARK: - Preset

struct VoicePreset: Identifiable, Codable {
    let id: String
    var name: String
    var description: String
    var settings: SynthesisSettings
    var sampleText: String
    var tags: [String]

    static let presets: [VoicePreset] = [
        // Turbo Paralinguistic Presets (using special tags)
        VoicePreset(
            id: "turbo_tech_support",
            name: "Turbo: Tech Support Meltdown",
            description: "Tech support script with [laugh] and [sigh] tags",
            settings: SynthesisSettings(speed: 1.0, exaggeration: 1.2, cfgWeight: 0.5, seed: 42),
            sampleText: "Hello, thank you for calling tech support. I'm [sigh] having a terrible day. First, my computer crashed, then the printer caught fire, and now you're telling me the internet is down? [laugh] That's just great. Let me transfer you to my supervisor.",
            tags: ["turbo", "paralinguistic", "comedy"]
        ),
        VoicePreset(
            id: "turbo_dramatic_chef",
            name: "Turbo: The Overly Dramatic Chef",
            description: "Cooking show with dramatic flair",
            settings: SynthesisSettings(speed: 0.9, exaggeration: 1.4, cfgWeight: 0.6, seed: 123),
            sampleText: "And now, we add just a pinch of salt. [clears throat] But WAIT! Don't just add any salt. No, no, no. We need to add the SALT OF LIFE. [chuckle] The salt that will transform this mere mortal dish into a culinary masterwork!",
            tags: ["turbo", "paralinguistic", "drama"]
        ),
        VoicePreset(
            id: "turbo_conspiracy",
            name: "Turbo: Conspiracy Podcast Host",
            description: "Suscious podcast host delivery",
            settings: SynthesisSettings(speed: 0.85, exaggeration: 0.8, cfgWeight: 0.55, seed: 666),
            sampleText: "They don't want you to know this, but I've been investigating for years. [sigh] The evidence is... compelling. Some say I'm crazy. But what if I told you that the moon landing was... [whisper] actually real? [laugh] Got you!",
            tags: ["turbo", "paralinguistic", "podcast"]
        ),
        VoicePreset(
            id: "turbo_first_jump",
            name: "Turbo: First-Time Skydiver",
            description: "Nervous skydiving experience",
            settings: SynthesisSettings(speed: 1.1, exaggeration: 1.3, cfgWeight: 0.5, seed: 777),
            sampleText: "So, uh, yeah, I'm about to jump out of a plane. [clears throat] No big deal, right? It's not like I'm FREAKING OUT or anything. [gasp] Wait, did I just hear a weird noise? Oh, that's just the door. Cool. Cool cool cool. [laugh]",
            tags: ["turbo", "paralinguistic", "comedy"]
        ),
        VoicePreset(
            id: "turbo_bedtime_story",
            name: "Turbo: Bedtime Story",
            description: "Tired parent's bedtime story",
            settings: SynthesisSettings(speed: 0.8, exaggeration: 0.6, cfgWeight: 0.5, seed: 999),
            sampleText: "Once upon a time, in a faraway land... [yawn] there lived a little princess who... [sniff] loved to eat pizza. And they all lived happily ever after. The end. [groan] Finally!",
            tags: ["turbo", "paralinguistic", "story"]
        ),
        // Standard Voice Style Presets
        VoicePreset(
            id: "noir_detective",
            name: "Noir Detective Monologue",
            description: "Classic film noir style",
            settings: SynthesisSettings(speed: 0.85, exaggeration: 0.4, cfgWeight: 0.6, seed: 101),
            sampleText: "It was raining again. The kind of rain that makes you wonder if the sky has it in for you. She walked into my office like she had somewhere to be, and somewhere to hide. I poured myself a drink. It was going to be that kind of night.",
            tags: ["style", "noir", "narrative"]
        ),
        VoicePreset(
            id: "children_story",
            name: "Children's Story Narrator",
            description: "Warm and friendly storytelling",
            settings: SynthesisSettings(speed: 0.9, exaggeration: 0.7, cfgWeight: 0.5, seed: 202),
            sampleText: "Once upon a time, in a magical forest, there lived a little bunny named Fluffy. Fluffy had the softest white fur and the gentlest heart. Every morning, she would hop through the meadow, saying hello to all her friends the butterflies and the flowers.",
            tags: ["style", "children", "story"]
        ),
        VoicePreset(
            id: "motivational",
            name: "Motivational Speech",
            description: "Inspiring and energetic",
            settings: SynthesisSettings(speed: 1.0, exaggeration: 0.9, cfgWeight: 0.5, seed: 303),
            sampleText: "You have within you, right now, everything you need to achieve greatness. Don't let anyone tell you otherwise. You are capable of AMAZING things. Get out there and show the world what you're made of!",
            tags: ["style", "motivational", "speech"]
        ),
        VoicePreset(
            id: "scientific",
            name: "Scientific Abstract",
            description: "Professional academic reading",
            settings: SynthesisSettings(speed: 0.85, exaggeration: 0.3, cfgWeight: 0.6, seed: 404),
            sampleText: "This study examines the correlation between neural network architecture and training efficiency. Through systematic analysis of 10,000 iterations, we demonstrate that certain structural optimizations yield significant improvements in computational resource utilization.",
            tags: ["style", "academic", "professional"]
        ),
        VoicePreset(
            id: "fairy_tale_villain",
            name: "Fairy Tale Villain",
            description: "Dark and menacing narration",
            settings: SynthesisSettings(speed: 0.75, exaggeration: 0.8, cfgWeight: 0.55, seed: 505),
            sampleText: "Ah, you foolish child. You thought you could defeat ME? [evil laugh] I have lived for a thousand years. I have conquered kingdoms and crushed dreams. And YOU... you are nothing but a speck of dust before my magnificence!",
            tags: ["style", "villain", "drama"]
        )
    ]
}
