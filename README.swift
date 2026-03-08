// ============================================================
//  Reader - ElevenLabs-like iOS App for Listening to Content
//  Powered by Chatterbox Turbo ONNX (ResembleAI)
// ============================================================
//
//  SETUP INSTRUCTIONS:
//
//  1. Open Reader.xcodeproj in Xcode 15+
//
//  2. Add ONNX Runtime Swift Package:
//     File → Add Package Dependencies → Enter:
//     https://github.com/spitfire02020-collab/onnxruntime-swift-package-manager
//     Select: onnxruntime-objc
//
//  3. (Optional) Add ZIPFoundation for EPUB support on device:
//     https://github.com/weichsel/ZIPFoundation
//
//  4. Build & Run on iOS 17+ device or simulator
//
//  5. First launch: Download the Chatterbox Turbo model
//     (Settings → Voice Engine Setup → Download)
//
//  ARCHITECTURE:
//
//  ├── ReaderApp.swift           → App entry point
//  ├── ContentView.swift         → Root view + onboarding
//  ├── Models/
//  │   ├── LibraryItem.swift     → Content data model
//  │   └── VoiceProfile.swift    → Voice configuration
//  ├── Services/
//  │   ├── ChatterboxEngine.swift    → ONNX TTS inference pipeline
//  │   ├── AudioPlayerService.swift  → AVFoundation playback + Now Playing
//  │   ├── WebContentExtractor.swift → Web page text extraction
//  │   ├── BookParser.swift          → EPUB/PDF parsing
//  │   ├── ModelDownloadService.swift→ HuggingFace model download
//  │   ├── TextChunker.swift         → Text splitting for TTS
//  │   ├── TokenizerService.swift     → GPT-2 BPE tokenizer
//  │   └── SynthesisDatabase.swift    → SQLite persistence
//  ├── ViewModels/
//  │   ├── LibraryViewModel.swift    → Library state management
//  │   └── PlayerViewModel.swift     → Player state management
//  ├── Views/
//  │   ├── LibraryView.swift         → Main library screen
//  │   ├── PlayerView.swift          → Full-screen audio player
//  │   ├── AddContentView.swift      → Add URL/file/text
//  │   ├── VoiceSelectionView.swift  → Voice picker + cloning
//  │   └── SettingsView.swift        → App settings
//  ├── Components/
//  │   ├── WaveformView.swift        → Audio waveform visualizer
//  │   └── MiniPlayerView.swift      → Persistent mini player
//  └── Extensions/
//      └── Color+Extensions.swift    → App color palette
//
//  MODEL DETAILS (Chatterbox Turbo ONNX):
//  - 350M parameters
//  - 4 ONNX components: speech_encoder, embed_tokens, language_model, conditional_decoder
//  - 24kHz sample rate
//  - GPT-2 BPE tokenizer with emotion tags
//  - Zero-shot voice cloning from reference audio
//  - MIT License
//
//  CHANGELOG:
//
//  2026-03-07: Audio Playback Fixes
//  - Fixed memory leak in waveform generation (added task cancellation)
//  - Added error logging to SynthesisDatabase for debugging
//  - Added debounce to Combine subscription in PlayerViewModel
//  - Replaced NSLock with Swift Actor in ChatterboxEngine
//
//  2026-03-01: Core TTS Fixes
//  - Fixed wrong tokenizer (onnx-community → ResembleAI Turbo)
//  - Fixed KV cache layer ordering
//  - Removed incorrect logit masking
//  - Added SDK patches for Float16 support
//
//  TROUBLESHOOTING:
//
//  Q: Audio doesn't play after synthesis?
//  A: Check console for "HALC_ProxyIOContext" errors - may indicate simulator audio issues
//
//  Q: Synthesis is slow?
//  A: Use iOS device instead of simulator for better performance
//
//  Q: Models fail to download?
//  A: Ensure internet connection; check Settings → Voice Engine Setup
//
//  Q: App crashes on long texts?
//  A: Memory pressure - synthesis creates multiple chunks. Try shorter texts.
//
