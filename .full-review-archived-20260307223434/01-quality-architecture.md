# Phase 1: Code Quality & Architecture Review

## Code Quality Findings

### Summary

| Severity | Count |
|----------|-------|
| Critical | 0 |
| High | 6 |
| Medium | 6 |
| Low | 3 |

---

## Critical Issues

None identified at Critical level.

---

## High Priority Issues

### 1. Large File Sizes - Maintainability Risk

**Files:**
- `PlayerView.swift` (869 lines)
- `AudioPlayerService.swift` (1007 lines)
- `PlayerViewModel.swift` (~1000+ lines)
- `VoiceSelectionView.swift` (825 lines)
- `SynthesisDatabase.swift` (496 lines)

**Recommendation:** Break into smaller, focused files.

---

### 2. Silent Error Handling - Risk of Undetected Failures

**Locations:**
- `SynthesisDatabase.swift` lines 158-161: Silent return on database errors
- `LibraryViewModel.swift` lines 68-82: Uses `try?` swallowing decode errors

**Recommendation:** Use Result types or throw errors properly.

---

### 3. High Cyclomatic Complexity in TextChunker

**File:** `TextChunker.swift` lines 206-298

The `splitByPunctuation` function has deeply nested switch/if statements.

**Recommendation:** Extract each case into separate functions.

---

### 4. Complex State Management in PlayerViewModel

Too many responsibilities:
- Audio playback control
- Text caching/preprocessing
- Database operations
- Voice selection management
- Synthesis orchestration
- Chunk file management

**Recommendation:** Extract into separate managers.

---

### 5. Memory Leak Risk with Closures

**File:** `PlayerViewModel.swift` lines 521-525, 799

Some closures may capture `self` strongly without `[weak self]`.

**Recommendation:** Audit all closures for strong capture.

---

### 6. Inconsistent Logging Approach

Mixes `NSLog` and `os.Logger` throughout codebase.

**Recommendation:** Standardize on `os.Logger`.

---

## Medium Priority Issues

### 7. Code Duplication - File URL Validation

Similar URL validation in multiple locations:
- `LibraryViewModel.swift` lines 93-99
- `ModelDownloadService.swift` lines 517-519

### 8. Unused Properties/Variables

- `AudioPlayerService.swift` line 200: `nextChunkBuffer` never used
- Audio engine properties may be incomplete

### 9. Hardcoded Values

Scattered magic numbers:
- `ChatterboxEngine.swift` line 43: `maxNewTokens: Int = 500`
- `TextChunker.swift` line 12: `minChunkSize = 100`

### 10. Duplicate Synthesis Logic

Similar synthesis code in `startSynthesis()`, `startSynthesisFromChunk()`, `startSynthesisInternal()`

### 11. Service Dependencies in ViewModels

ViewModels creating own service instances - tight coupling.

### 12. Incomplete Error Messages

Some errors logged without actionable context.

---

## Low Priority Issues

### 13. Missing Access Control Modifiers

### 14. Type Aliases Could Improve Readability

### 15. Inconsistent Naming

---

## Architecture Findings

### Summary

| Severity | Count |
|----------|-------|
| Critical | 1 |
| High | 4 |
| Medium | 5 |
| Low | 4 |

### Critical Issue

1. **Singleton Pattern Overuse** - Multiple singletons create hidden dependencies and make testing difficult

### High Issues

1. **Layer Boundary Violation** - AudioPlayerService imports SQLite directly
2. **No Protocol Abstraction for Engine** - Direct instantiation
3. **PlayerViewModel Has Excessive Responsibilities** - 17+ @Published properties
4. **Inconsistent Concurrency Models** - Mix of @MainActor, Task.detached, NSLock

---

## Recommendations Summary

| Priority | Issue | Impact | Effort |
|----------|-------|--------|--------|
| 1 | Break up large files | Maintainability | High |
| 2 | Fix silent error handling | Reliability | Medium |
| 3 | Reduce PlayerViewModel complexity | Maintainability | High |
| 4 | Simplify TextChunker complexity | Maintainability | Medium |
| 5 | Standardize logging | Debugging | Low |
| 6 | Extract duplicate synthesis logic | DRY | Medium |
