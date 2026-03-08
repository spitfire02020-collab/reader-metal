# Review Scope

## Target

Full codebase review of the Reader iOS app (Chatterbox TTS text-to-speech reader)

## Files (25 Swift files)

### Core Services
- `Reader/Services/ChatterboxEngine.swift` - TTS inference engine
- `Reader/Services/AudioPlayerService.swift` - Audio playback service
- `Reader/Services/TokenizerService.swift` - GPT-2 tokenizer
- `Reader/Services/TextChunker.swift` - Text chunking logic
- `Reader/Services/ModelDownloadService.swift` - Model download manager
- `Reader/Services/SynthesisDatabase.swift` - SQLite database
- `Reader/Services/BookParser.swift` - Content parsing
- `Reader/Services/WebContentExtractor.swift` - Web content extraction
- `Reader/Services/ONNXGraphFixer.swift` - ONNX utilities

### ViewModels
- `Reader/ViewModels/PlayerViewModel.swift` - Player state management
- `Reader/ViewModels/LibraryViewModel.swift` - Library state management

### Views
- `Reader/Views/PlayerView.swift` - Main player UI
- `Reader/Views/LibraryView.swift` - Library UI
- `Reader/Views/AddContentView.swift` - Add content UI
- `Reader/Views/SettingsView.swift` - Settings UI
- `Reader/Views/VoiceSelectionView.swift` - Voice selection UI
- `Reader/ContentView.swift` - Root content view
- `Reader/ReaderApp.swift` - App entry point

### Components
- `Reader/Components/WaveformView.swift` - Waveform visualization
- `Reader/Components/SelectableTextView.swift` - Text with highlighting
- `Reader/Components/MiniPlayerView.swift` - Mini player
- `Reader/Components/SkeletonLoading.swift` - Loading skeletons

### Models & Extensions
- `Reader/Models/LibraryItem.swift` - Content item model
- `Reader/Models/VoiceProfile.swift` - Voice profile model
- `Reader/Extensions/Color+Extensions.swift` - Color extensions
- `Reader/Extensions/HapticFeedback.swift` - Haptic feedback

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
