# Phase 1: Code Quality & Architecture Review

## Code Quality Findings

### Critical Issues

1. **PlayerViewModel Exceeds Single Responsibility Principle**
   - Severity: Critical
   - Location: `PlayerViewModel.swift` (~1000+ lines)
   - Issue: 40+ @Published properties, 30+ methods handling text caching, playback, synthesis, database, voice management, highlighting
   - Recommendation: Break into TextCacheManager, SynthesisCoordinator, PlaybackController

2. **Memory Leak Risk with Cached Text Chunks**
   - Severity: Critical
   - Location: `PlayerViewModel.swift` lines 434-441
   - Issue: `cachedTextChunks` not thread-safe, never invalidated when `item` changes
   - Recommendation: Use proper cache with invalidation or computed property

### High Severity Issues

3. **Inconsistent Logging (NSLog vs os.Logger)**
   - Severity: High
   - Location: Multiple files
   - Issue: Mixed usage of NSLog and os.Logger throughout codebase

4. **Debug Logging Left in Production Code**
   - Severity: High
   - Location: `AudioPlayerService.swift` lines 700-712, 900-933
   - Issue: Excessive NSLog statements for debugging remain in production

5. **TextChunker Static Cache Not Thread-Safe**
   - Severity: High
   - Location: `TextChunker.swift` lines 31-62
   - Issue: Race condition in static cache - sync used only for read, not write

6. **Deeply Nested Logic in TextChunker**
   - Severity: High
   - Location: `TextChunker.swift` lines 223-342
   - Issue: 8+ levels of nesting in splitByPunctuation method

7. **Missing Error Propagation in ChatterboxEngine**
   - Severity: High
   - Location: `ChatterboxEngine.swift` lines 340-440
   - Issue: Synthesis failures logged but not propagated to caller

### Medium Severity Issues

8. **Heavy Objects Created in ViewModel Init**
9. **Code Duplication in PlayerViewModel** (restartWithNewVoice vs applyPreset)
10. **Magic Numbers Without Constants**
11. **Swallowed Errors with try?**
12. **Large Method in AudioPlayerService** (pollForNewChunks 70+ lines)

---

## Architecture Findings

### Critical Issues

1. **PlayerViewModel God Class**
   - Severity: Critical
   - Location: `PlayerViewModel.swift`
   - Issue: Grown to ~1400 lines with excessive responsibilities
   - Recommendation: Extract TextCacheManager, SynthesisCoordinator, PlaybackCoordinator

### High Severity Issues

2. **Singleton Overuse Creates Hidden Dependencies**
   - Severity: High
   - Location: AudioPlayerService.shared, SynthesisDatabase.shared, ServiceContainer.shared
   - Issue: Makes unit testing impossible, violates Dependency Inversion Principle

3. **Direct Service Access from Views Breaks MVVM**
   - Severity: High
   - Location: `PlayerView.swift` lines 125, 336
   - Issue: Views directly access `viewModel.audioPlayer.isPlaying` instead of ViewModel abstractions

4. **Service Layer Boundary Violation**
   - Severity: High
   - Location: `AudioPlayerService.swift` line 213
   - Issue: AudioPlayerService depends on SynthesisDatabase - violates Single Responsibility

### Medium Severity Issues

5. **Data Model Inconsistency** - LibraryItem has both audioFileURL and generatedChunks
6. **Static Methods in TextChunker** - Prevent testability
7. **Inconsistent Protocol Usage** - ServiceProtocols.swift exists but rarely used
8. **Missing Error Handling Architecture** - No typed error enums

---

## Critical Issues for Phase 2 Context

The following findings from Phase 1 should inform the Security and Performance review:

1. **Memory leak risk** in cached text chunks - potential DoS vector
2. **Thread safety issues** in TextChunker static cache - concurrency vulnerability
3. **Missing error propagation** - synthesis failures silently dropped
4. **Singleton overuse** - makes security auditing difficult (hidden global state)
5. **Debug logging in production** - potential information disclosure
