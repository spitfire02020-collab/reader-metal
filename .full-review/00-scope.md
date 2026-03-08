# Review Scope

## Target

Full Reader app review - SwiftUI iOS app with Chatterbox TTS

## Files

All Swift files in Reader/:
- **Services/**: AudioPlayerService, ChatterboxEngine, ModelDownloadService, TextChunker, SynthesisDatabase, WebContentExtractor, TokenizerService, ServiceContainer, ServiceProtocols, ONNXGraphFixer, BookParser
- **ViewModels/**: PlayerViewModel, LibraryViewModel
- **Views/**: PlayerView, LibraryView, SettingsView, AddContentView, VoiceSelectionView
- **Components/**: WaveformView, SelectableTextView, MiniPlayerView, SkeletonLoading
- **Models/**: LibraryItem, VoiceProfile
- **App/**: ReaderApp, ContentView

## Flags

- Security Focus: no
- Performance Critical: no
- Strict Mode: no
- Framework: SwiftUI/iOS

## Review Phases

1. Code Quality & Architecture
2. Security & Performance
3. Testing & Documentation
4. Best Practices & Standards
5. Consolidated Report
