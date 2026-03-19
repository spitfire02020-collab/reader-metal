# Codebase Audit Design (Targeted Static Analysis)
Date: 2026-03-20
Topic: Targeted Codebase Audit & Architecture Review

## Objective
Proactively identify and resolve structural, concurrency, and memory bugs within the iOS Reader application before they manifest as runtime crashes or silent failures.

## Phase 1: Concurrency & Memory Audit
Focuses on ensuring safe concurrent execution and preventing resource exhaustion.

### Threading & Main Actor Violations
- Trace `Task` and `Task.detached` blocks in `PlayerViewModel` and `ChatterboxEngine`.
- Verify UI-bound properties (`@Published`, `@State`) are mutated strictly on `@MainActor`.
- Ensure heavy operations (ONNX synthesis, filesystem I/O) do not block the main thread.

### Retain Cycles & Memory Leaks
- Audit all asynchronous closures and delegate patterns.
- Validate the explicit use of `[weak self]` in `AudioPlayerService` callbacks and background timers (e.g., `CADisplayLink`).
- Check memory footprint of `SynthesisDatabase` operations (e.g., loading large arrays of strings/paths into memory simultaneously).

## Phase 2: Data Integrity & Application State
Focuses on ensuring persistent state matches the physical filesystem and reactive UI states.

### Database & File System Consistency
- Track audio chunk lifecycle: generation, renaming, and database committing.
- Identify "dirty" states where a file is saved but DB insertion fails, or vice-versa.
- Standardize UUID and path resolution (e.g., `resolveChunkPath`).

### State Synchronization
- Audit `SynthesisItemStatus` (.pending, .synthesizing, .ready, .paused, .error).
- Ensure transitions are atomic and consistent across `PlayerViewModel` and `LibraryViewModel`.
- Verify the UI never enters an unrecoverable "loading" state.

## Phase 3: Error Recovery & Edge Cases
Focuses on graceful degradation and recovery from failures.

### Silent Failures
- Scan for standalone `try?` statements that swallow critical errors.
- Ensure proper logging and error bubbling up to the `errorMessage` properties.

### Background Boundary State
- Test extreme user boundaries: spamming controls, rapid voice switching, and forced app exits mid-synthesis.
- Ensure resources (e.g., `AVAudioEngine`, ONNX models) are cleanly initialized and released.

## Execution Strategy
1. The AI agent will systematically review the files related to each phase.
2. If vulnerabilities or anti-patterns are found, they will be documented and fixed sequentially.
3. Tests or builds will be verifying patches locally.
