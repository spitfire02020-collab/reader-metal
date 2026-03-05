import XCTest
@testable import Reader

final class VoiceProfileTests: XCTestCase {

    // MARK: - Default Voice Tests

    func testDefaultVoiceHasCorrectID() {
        // Given & When
        let defaultVoice = VoiceProfile.defaultVoice

        // Then
        XCTAssertEqual(defaultVoice.id, "default", "Default voice should have ID 'default'")
    }

    func testDefaultVoiceHasCorrectName() {
        // Given & When
        let defaultVoice = VoiceProfile.defaultVoice

        // Then
        XCTAssertEqual(defaultVoice.name, "Nova", "Default voice should have name 'Nova'")
    }

    func testDefaultVoiceIsBuiltIn() {
        // Given & When
        let defaultVoice = VoiceProfile.defaultVoice

        // Then
        XCTAssertTrue(defaultVoice.isBuiltIn, "Default voice should be marked as built-in")
    }

    // MARK: - Built-in Voices Tests

    func testBuiltInVoicesArrayIsNotEmpty() {
        // Given & When
        let builtInVoices = VoiceProfile.builtInVoices

        // Then
        XCTAssertFalse(builtInVoices.isEmpty, "Built-in voices array should not be empty")
    }

    func testBuiltInVoicesContainsDefaultVoice() {
        // Given & When
        let builtInVoices = VoiceProfile.builtInVoices

        // Then
        XCTAssertTrue(builtInVoices.contains { $0.id == "default" }, "Built-in voices should contain the default voice")
    }

    func testBuiltInVoicesCount() {
        // Given & When
        let builtInVoices = VoiceProfile.builtInVoices

        // Then
        XCTAssertEqual(builtInVoices.count, 5, "Built-in voices should contain exactly 5 voices")
    }

    func testAllBuiltInVoicesAreMarkedAsBuiltIn() {
        // Given & When
        let builtInVoices = VoiceProfile.builtInVoices

        // Then
        for voice in builtInVoices {
            XCTAssertTrue(voice.isBuiltIn, "Voice with id '\(voice.id)' should be marked as built-in")
        }
    }

    // MARK: - JSON Encoding/Decoding Tests

    func testVoiceProfileCanBeEncodedToJSON() {
        // Given
        let voice = VoiceProfile.defaultVoice

        // When
        let encoder = JSONEncoder()
        XCTAssertNoThrow(try encoder.encode(voice), "VoiceProfile should be encodable to JSON")
    }

    func testVoiceProfileCanBeDecodedFromJSON() throws {
        // Given
        let jsonString = """
        {
            "id": "test_voice",
            "name": "Test Voice",
            "description": "A test voice",
            "isBuiltIn": false,
            "referenceAudioPath": "/path/to/audio.wav",
            "sampleRate": 24000,
            "language": "en",
            "tags": ["test", "custom"]
        }
        """

        // When
        let decoder = JSONDecoder()
        let voice = try decoder.decode(VoiceProfile.self, from: Data(jsonString.utf8))

        // Then
        XCTAssertEqual(voice.id, "test_voice")
        XCTAssertEqual(voice.name, "Test Voice")
        XCTAssertEqual(voice.description, "A test voice")
        XCTAssertFalse(voice.isBuiltIn)
        XCTAssertEqual(voice.referenceAudioPath, "/path/to/audio.wav")
        XCTAssertEqual(voice.sampleRate, 24000)
        XCTAssertEqual(voice.language, "en")
        XCTAssertEqual(voice.tags, ["test", "custom"])
    }

    func testVoiceProfileRoundTripEncoding() throws {
        // Given
        let originalVoice = VoiceProfile(
            id: "roundtrip_test",
            name: "Roundtrip Test",
            description: "Testing encoding and decoding",
            isBuiltIn: false,
            referenceAudioPath: "/test/path.wav",
            sampleRate: 22050,
            language: "en",
            tags: ["roundtrip", "test"]
        )

        // When
        let encoder = JSONEncoder()
        let encodedData = try encoder.encode(originalVoice)

        let decoder = JSONDecoder()
        let decodedVoice = try decoder.decode(VoiceProfile.self, from: encodedData)

        // Then
        XCTAssertEqual(decodedVoice.id, originalVoice.id)
        XCTAssertEqual(decodedVoice.name, originalVoice.name)
        XCTAssertEqual(decodedVoice.description, originalVoice.description)
        XCTAssertEqual(decodedVoice.isBuiltIn, originalVoice.isBuiltIn)
        XCTAssertEqual(decodedVoice.referenceAudioPath, originalVoice.referenceAudioPath)
        XCTAssertEqual(decodedVoice.sampleRate, originalVoice.sampleRate)
        XCTAssertEqual(decodedVoice.language, originalVoice.language)
        XCTAssertEqual(decodedVoice.tags, originalVoice.tags)
    }

    func testDefaultVoiceCanBeEncodedAndDecoded() throws {
        // Given & When
        let encoder = JSONEncoder()
        let encodedData = try encoder.encode(VoiceProfile.defaultVoice)

        let decoder = JSONDecoder()
        let decodedVoice = try decoder.decode(VoiceProfile.self, from: encodedData)

        // Then
        XCTAssertEqual(decodedVoice.id, VoiceProfile.defaultVoice.id)
        XCTAssertEqual(decodedVoice.name, VoiceProfile.defaultVoice.name)
        XCTAssertEqual(decodedVoice.description, VoiceProfile.defaultVoice.description)
        XCTAssertEqual(decodedVoice.isBuiltIn, VoiceProfile.defaultVoice.isBuiltIn)
        XCTAssertEqual(decodedVoice.sampleRate, VoiceProfile.defaultVoice.sampleRate)
        XCTAssertEqual(decodedVoice.language, VoiceProfile.defaultVoice.language)
        XCTAssertEqual(decodedVoice.tags, VoiceProfile.defaultVoice.tags)
    }

    func testBuiltInVoicesCanBeEncodedAndDecoded() throws {
        // Given
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        // When & Then
        for originalVoice in VoiceProfile.builtInVoices {
            let encodedData = try encoder.encode(originalVoice)
            let decodedVoice = try decoder.decode(VoiceProfile.self, from: encodedData)

            XCTAssertEqual(decodedVoice.id, originalVoice.id, "Voice ID should match after round-trip")
            XCTAssertEqual(decodedVoice.name, originalVoice.name, "Voice name should match after round-trip")
            XCTAssertEqual(decodedVoice.description, originalVoice.description, "Voice description should match after round-trip")
            XCTAssertEqual(decodedVoice.isBuiltIn, originalVoice.isBuiltIn, "isBuiltIn should match after round-trip")
            XCTAssertEqual(decodedVoice.sampleRate, originalVoice.sampleRate, "Sample rate should match after round-trip")
            XCTAssertEqual(decodedVoice.language, originalVoice.language, "Language should match after round-trip")
            XCTAssertEqual(decodedVoice.tags, originalVoice.tags, "Tags should match after round-trip")
        }
    }

    func testVoiceProfileWithNilReferenceAudioPath() throws {
        // Given
        let jsonString = """
        {
            "id": "nil_ref",
            "name": "Nil Ref Voice",
            "description": "Voice with nil reference audio path",
            "isBuiltIn": true,
            "referenceAudioPath": null,
            "sampleRate": 24000,
            "language": "en",
            "tags": []
        }
        """

        // When
        let decoder = JSONDecoder()
        let voice = try decoder.decode(VoiceProfile.self, from: Data(jsonString.utf8))

        // Then
        XCTAssertNil(voice.referenceAudioPath, "referenceAudioPath should be nil when not provided")
    }

    // MARK: - Hashable Tests

    func testVoiceProfileIsHashable() {
        // Given
        let voice1 = VoiceProfile.defaultVoice
        let voice2 = VoiceProfile.defaultVoice

        // When & Then
        let hash1 = voice1.hashValue
        let hash2 = voice2.hashValue
        XCTAssertEqual(hash1, hash2, "Equal VoiceProfiles should have equal hash values")
    }

    func testVoiceProfileEquatable() {
        // Given
        let voice1 = VoiceProfile(
            id: "test",
            name: "Test",
            description: "Desc",
            isBuiltIn: true,
            referenceAudioPath: nil,
            sampleRate: 24000,
            language: "en",
            tags: ["tag"]
        )

        let voice2 = VoiceProfile(
            id: "test",
            name: "Test",
            description: "Desc",
            isBuiltIn: true,
            referenceAudioPath: nil,
            sampleRate: 24000,
            language: "en",
            tags: ["tag"]
        )

        // Then
        XCTAssertEqual(voice1, voice2, "VoiceProfiles with same values should be equal")
    }
}
