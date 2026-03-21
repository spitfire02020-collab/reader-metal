# Review Scope

## Target

Full Reader app — an iOS SwiftUI app with Chatterbox TTS (text-to-speech) engine using ONNX models for neural synthesis.

## Files

### Core App (Reader/)
**Services (11 files):**
- `Reader/Services/ChatterboxEngine.swift` — Core TTS engine, ONNX inference pipeline
- `Reader/Services/TokenizerService.swift` — GPT-2 BPE tokenizer
- `Reader/Services/ModelDownloadService.swift` — Model download + caching
- `Reader/Services/AudioPlayerService.swift` — Audio playback via AVFoundation
- `Reader/Services/SynthesisDatabase.swift` — SQLite storage for synthesis items
- `Reader/Services/ServiceProtocols.swift` — Protocol definitions
- `Reader/Services/ServiceContainer.swift` — Dependency injection container
- `Reader/Services/TextChunker.swift` — Text splitting for synthesis
- `Reader/Services/BookParser.swift` — PDF/EPUB content extraction
- `Reader/Services/WebContentExtractor.swift` — URL content extraction
- `Reader/Services/ONNXGraphFixer.swift` — ONNX graph manipulation utility

**Views (5 files):**
- `Reader/Views/PlayerView.swift`, `VoiceSelectionView.swift`, `SettingsView.swift`, `LibraryView.swift`, `AddContentView.swift`

**ViewModels (2 files):**
- `Reader/ViewModels/PlayerViewModel.swift`, `LibraryViewModel.swift`

**Components (4 files):**
- `Reader/Components/WaveformView.swift`, `SelectableTextView.swift`, `MiniPlayerView.swift`, `SkeletonLoading.swift`

**Models (2 files):**
- `Reader/Models/LibraryItem.swift`, `VoiceProfile.swift`

**Extensions (2 files):**
- `Reader/Extensions/Color+Extensions.swift`, `HapticFeedback.swift`

**App Root (2 files):**
- `Reader/ReaderApp.swift`, `Reader/ContentView.swift`

**Test Files:**
- `Reader/ReaderTests/Services/TextChunkerTests.swift`
- `Reader/ReaderTests/Models/VoiceProfileTests.swift`

**Config:**
- `project.yml` — XcodeGen configuration

## Flags

- Security Focus: no (general review)
- Performance Critical: yes (ONNX inference, audio streaming)
- Strict Mode: no
- Framework: SwiftUI/iOS, ONNX Runtime (Swift PM), SQLite.swift

## Review Phases

1. Code Quality & Architecture
2. Security & Performance
3. Testing & Documentation
4. Best Practices & Standards
5. Consolidated Report
