# Phase 1: Code Quality & Architecture Review

## Code Quality Findings

### Critical Issues

None identified.

### High Severity Issues

1. **No Unit Tests**
   - Severity: High
   - Location: Entire codebase - no test files found
   - Issue: Critical services like TextChunker, TokenizerService, and BookParser lack test coverage
   - Recommendation: Add unit tests for core services

2. **AudioPlayerService - Large Class (1014 lines)**
   - Severity: High
   - Location: AudioPlayerService.swift
   - Issue: Violates Single Responsibility Principle - handles playback, crossfading, remote controls, Now Playing, database queries
   - Recommendation: Extract into AVPlayerAdapter, CrossfadeController, NowPlayingManager, RemoteControlHandler

3. **PlayerViewModel - God Class (~1800 lines)**
   - Severity: High
   - Location: PlayerViewModel.swift
   - Issue: Multiple responsibilities - audio control, text caching, synthesis orchestration, state management
   - Recommendation: Extract TextCacheManager, PlaybackCoordinator, SynthesisCoordinator

### Medium Severity Issues

4. **HTML Entity Decoding Triplicated** - WebContentExtractor, BookParser, LibraryItem
5. **Magic Numbers Scattered** - Need Constants.swift
6. **Inconsistent Error Handling** - Mixed try?, custom enums, throws
7. **TextChunker Deep Nesting** - 8+ levels in splitByPunctuation
8. **Large PlayerView** - 871 lines, should extract subviews
9. **Hardcoded Voice Mappings** - Should be data-driven

### Low Severity Issues

10. **Mixed Logging** - NSLog, os.Logger, print() inconsistency
11. **Missing Accessibility Labels** - Some buttons lack labels
12. **Memory-Intensive Cache** - TextChunker cache could use memory-based eviction

---

## Architecture Findings

### Critical Issues

None identified.

### High Severity Issues

1. **Inconsistent Data Access Layer**
   - Severity: High
   - Location: LibraryViewModel.swift (lines 74-107)
   - Issue: ViewModel directly handles JSON encoding/decoding, FileManager, UserDefaults migration
   - Recommendation: Create LibraryRepository protocol

2. **Singleton Overuse**
   - Severity: High
   - Location: ModelDownloadService, AudioPlayerService, ServiceContainer
   - Issue: Creates tight coupling, hidden dependencies, testing difficulties
   - Recommendation: Use ServiceContainer consistently, remove static shared instances

### Medium Severity Issues

3. **Direct Service Instantiation** - ViewModels create services directly instead of injection
4. **Weakly-Typed Protocol Parameters** - `[String: Any]` in ServiceProtocols.swift
5. **Deep Link in App Entry Point** - Business logic in ReaderApp.swift
6. **LibraryItem Monolithic Model** - Conflates domain model with view state

### Low Severity Issues

7. **Callback-Based API** - Could use AsyncSequence for streaming
8. **Notification-Based Communication** - Could use Combine publishers
9. **Missing Value Objects** - Primitive types where value objects would improve type safety

---

## Critical Issues for Phase 2 Context

The following findings from Phase 1 should inform the Security and Performance review:

1. **No unit tests** - security-critical paths untested
2. **Singleton overuse** - hidden global state complicates security auditing
3. **Memory-intensive cache** - potential DoS vector
4. **Mixed error handling** - silent failures could mask security issues
5. **Deep link handling** - URL validation should be reviewed
