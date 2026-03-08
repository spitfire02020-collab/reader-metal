# Review Scope

## Target

**Reader** - A SwiftUI iOS app with Chatterbox Turbo TTS (Text-to-Speech) that runs entirely on-device using ONNX Runtime.

## Files

### Core Services (7 files)
- `ChatterboxEngine.swift` (1,332 lines) - ONNX TTS engine, audio generation
- `AudioPlayerService.swift` (913 lines) - Audio playback, streaming, chunk management
- `ModelDownloadService.swift` (594 lines) - Model download/management
- `TokenizerService.swift` (289 lines) - Text tokenization for TTS
- `TextChunker.swift` (355 lines) - Text chunking for streaming synthesis
- `WebContentExtractor.swift` (438 lines) - Web content extraction
- `BookParser.swift` (451 lines) - EPUB/PDF parsing

### ViewModels (2 files)
- `PlayerViewModel.swift` (1,393 lines) - Main player logic
- `LibraryViewModel.swift` (282 lines) - Library management

### Views (6 files)
- `PlayerView.swift` (868 lines) - Player UI
- `LibraryView.swift` (849 lines) - Library screen UI
- `VoiceSelectionView.swift` (825 lines) - Voice selection UI
- `AddContentView.swift` (425 lines) - Add content UI
- `SettingsView.swift` (323 lines) - Settings UI
- `ContentView.swift` - Root content view

### Components (4 files)
- `WaveformView.swift` (285 lines) - Audio visualization
- `MiniPlayerView.swift` - Mini player
- `SelectableTextView.swift` - Text selection
- `SkeletonLoading.swift` - Loading states

### Models & Other (6 files)
- `LibraryItem.swift` - Library item model
- `VoiceProfile.swift` (220 lines) - Voice profiles
- `Color+Extensions.swift` (490 lines) - UI styling
- `SynthesisDatabase.swift` (451 lines) - SQLite storage
- `ServiceContainer.swift` - Dependency injection
- `ServiceProtocols.swift` - Protocol definitions

**Total: 27 Swift files, ~12,255 lines of code**

## Flags

- Security Focus: no
- Performance Critical: no (but important for TTS)
- Strict Mode: no
- Framework: SwiftUI/iOS

## Review Phases

1. Code Quality & Architecture
2. Security & Performance
3. Testing & Documentation
4. Best Practices & Standards
5. Consolidated Report
