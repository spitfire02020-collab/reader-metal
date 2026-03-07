import Foundation

// MARK: - Service Container
// Simple dependency injection container for services
// Used to avoid singletons while maintaining convenience

@MainActor
final class ServiceContainer {
    static let shared = ServiceContainer()

    // Lazy services - created on first access
    lazy var audioPlayer: AudioPlayerService = {
        AudioPlayerService()
    }()

    lazy var synthesisDatabase: SynthesisDatabase = {
        SynthesisDatabase()
    }()

    lazy var modelDownloadService: ModelDownloadService = {
        ModelDownloadService()
    }()

    // For services that need initialization
    func makeChatterboxEngine() -> ChatterboxEngine {
        ChatterboxEngine()
    }

    func makeWebContentExtractor() -> WebContentExtractor {
        WebContentExtractor()
    }

    private init() {}
}

// MARK: - Factory for ViewModel Injection
// Use this in SwiftUI views to inject services

struct ServicesKey: EnvironmentKey {
    static let defaultValue: ServiceContainer = ServiceContainer.shared
}

extension EnvironmentValues {
    var services: ServiceContainer {
        get { self[ServicesKey.self] }
        set { self[ServicesKey.self] = newValue }
    }
}
