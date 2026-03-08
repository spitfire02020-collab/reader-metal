# Phase 2: Security & Performance Review

## Security Findings

### Summary

| Severity | Count |
|----------|-------|
| Critical | 1 |
| High | 2 |
| Medium | 4 |
| Low | 2 |

---

### Critical Issues

#### 1. Path Traversal in EPUB Extraction
**CWE:** CWE-22 (Path Traversal)
**File:** `BookParser.swift` lines 360-390

The EPUB extraction allows files to be written outside intended directory via `../` in filenames.

**Remediation:** Validate paths stay within destination:
```swift
let resolvedPath = filePath.resolvingSymlinksInPath()
guard resolvedPath.hasPrefix(destinationPath) else { continue }
```

---

### High Issues

#### 2. Missing URL Scheme Validation
**CWE:** CWE-346 (Origin Validation)
**File:** `LibraryViewModel.swift` lines 93-99

Only checks scheme != nil, not restricted to http/https.

#### 3. Insufficient Input Sanitization
**CWE:** CWE-79 (XSS)
**File:** `WebContentExtractor.swift` lines 121-146

Extracted content stored without sanitization.

---

## Performance Findings

### Summary

| Severity | Count |
|----------|-------|
| Critical | 3 |
| High | 4 |
| Medium | 5 |
| Low | 3 |

---

### Critical Issues

#### 1. Busy-Wait Polling with No Backoff
**File:** `AudioPlayerService.swift` lines 936-1006

Polls every 0.5s for up to 60 iterations. 60× more CPU than needed.

#### 2. Temporary AVAudioPlayer for Duration
**File:** `AudioPlayerService.swift` lines 587-597

Creates full AVAudioPlayer just to read duration - 5-10× more memory than needed.

#### 3. O(n²) Sentence Search
**File:** `PlayerViewModel.swift` lines 457-478

Linear scan through all sentences twice - could use dictionary lookup O(1).

---

### High Issues

1. Missing TextChunker Regex Caching
2. Full Paragraph Rebuild on Single Change
3. Synchronous WAV File Writes
4. Multiple AVAudioPlayer Instances

---

## Critical Issues for Phase 3

1. Security: Path traversal needs immediate fix
2. Performance: Polling mechanism affects all users during synthesis
3. These findings affect testing requirements for input validation
