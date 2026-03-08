# Code Refactoring Analysis - Reader iOS App

## Executive Summary

Analyzed 4 key files (4,545 lines total) for code smells and refactoring opportunities.

| Issue Type | Count | Severity | Effort |
|------------|-------|----------|--------|
| Duplicate time formatting | 3 locations | Medium | Low |
| Magic numbers | 15+ | Medium | Low |
| Large files (>500 lines) | 4 files | High | Medium |
| Duplicate engine calls | 6 pairs | Medium | Low |

---

## Findings

### 1. Duplicate Time Formatting (Medium - Low Effort)

**Locations:**
- `AudioPlayerService.swift:886-888`
- `LibraryItem.swift:50-52`
- `PlayerViewModel.swift:103-105`

**Issue:** Same time formatting logic duplicated 3 times.

**Solution:** Extract to shared utility.

### 2. Magic Numbers (Medium - Low Effort)

**Examples:**
- `sampleRate: Int = 24000` (ChatterboxEngine.swift:33)
- `startSpeechToken: Int = 6561` (ChatterboxEngine.swift:34)
- `stopSpeechToken: Int = 6562` (ChatterboxEngine.swift:35)
- `silenceToken: Int = 4299` (ChatterboxEngine.swift:36)
- `maxNewTokens: Int = 1500` (various locations)

**Solution:** Group into configuration struct/enum.

### 3. Large Files (High - Medium Effort)

| File | Lines | Recommendation |
|------|-------|----------------|
| PlayerViewModel.swift | 1524 | Split into smaller services |
| ChatterboxEngine.swift | 1408 | Extract ONNX helpers |
| AudioPlayerService.swift | 1007 | Split playback/queue concerns |

### 4. Duplicate Engine Calls (Medium - Low Effort)

**Pattern:** `try await engine.loadModels()` followed by `try await engine.synthesize()` repeated 6 times.

**Solution:** Create synthesis method that combines both.

---

## Refactoring Plan

### Phase 1: Quick Wins (30 min)

1. Extract time formatting to `TimeFormatter` utility
2. Create `ChatterboxConfig` struct for magic numbers
3. Combine loadModels+synthesize into single method

### Phase 2: Medium Effort (2 hours)

4. Extract chunk management from AudioPlayerService
5. Create separate ONNX helper class

### Phase 3: Large Effort (Future)

6. Split PlayerViewModel into smaller ViewModels
