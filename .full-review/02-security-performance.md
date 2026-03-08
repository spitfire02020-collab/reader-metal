# Phase 2: Security & Performance Review

## Security Findings

### Critical Issues

1. **Unbounded Memory Cache - Potential DoS**
   - Severity: Critical (CVSS 7.5)
   - Location: TextChunker.swift (lines 31-48)
   - Issue: Cache can grow to 100 entries but each can be MBs of text
   - Recommendation: Add byte-based size limits

### High Severity Issues

2. **Deprecated UserDefaults Storage**
   - Severity: High (CVSS 6.5)
   - Location: ReaderApp.swift (lines 27-44)
   - Issue: Deep link handler uses deprecated UserDefaults instead of secure file storage

3. **Incomplete URL Scheme Validation**
   - Severity: High (CVSS 5.3)
   - Location: ReaderApp.swift (lines 24-48)
   - Issue: Only scheme checked, no host whitelist

### Medium Severity Issues

4. **Singleton Pattern Overuse** - Hidden global state
5. **Silent Error Handling** - Could mask security issues
6. **Potential SSRF** - WebContentExtractor lacks internal IP blocking
7. **Background Mode Security** - Audio/processing modes declared

### Low Severity Issues

8. **Debug Logging** - Exposes internal state
9. **Custom ONNXRuntime Fork** - May miss security patches

---

## Performance Findings

### Critical Issues

bounded Text Caching in PlayerViewModel1. **Un**
   - Severity: Critical
   - Location: PlayerViewModel.swift (lines 68-81)
   - Impact: OOM for large documents

### High Severity Issues

2. **N+1 Database Queries**
   - Severity: High
   - Location: SynthesisDatabase.swift (lines 280-295)
   - Impact: O(n) queries for progress updates

3. **Synchronous I/O on Main Thread**
   - Severity: High
   - Location: PlayerViewModel.swift (lines 159-203)
   - Impact: UI freezes during chunk loading

4. **Redundant Text Processing**
   - Severity: High
   - Location: PlayerViewModel.swift
   - Impact: Double processing - TextCacheManager AND cached properties

5. **@MainActor Isolation**
   - Severity: High
   - Location: SynthesisDatabase.swift (line 58)
   - Impact: Thread starvation, UI freezes

### Medium Severity Issues

6. **Missing DB Indexes** - Slow queries at scale
7. **Large View Rebuilds** - Unnecessary re-renders
8. **No Pagination** - Memory grows with library
9. **AudioPlayerService Singleton** - Holds large state

---

## Critical Issues for Phase 3 Context

1. No unit tests - security and performance paths untested
2. Unbounded caching - needs stress testing
3. Synchronous I/O - needs concurrent test scenarios
4. Error handling - needs edge case testing
