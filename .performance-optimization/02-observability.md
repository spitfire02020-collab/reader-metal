# Phase 2: Observability Assessment - Reader iOS App

## Current Observability State

### Logging
- **NSLog** usage: Scattered throughout codebase
  - `AudioPlayerService.swift`: ~20 NSLog statements
  - `ChatterboxEngine.swift`: ~15 NSLog statements
  - `SynthesisDatabase.swift`: Recently added error logging
  - `PlayerViewModel.swift`: ~10 NSLog statements

### Issues
1. **No structured logging**: NSLog doesn't support log levels
2. **No performance metrics**: No timing information for ONNX inference
3. **No memory warnings**: No memory pressure handling
4. **No crash reporting**: No third-party SDK integration

### Recommendations for iOS

#### 1. Migrate to os.Logger (Swift 5.3+)
```swift
import os.log

private let logger = Logger(subsystem: "com.reader.app", category: "AudioPlayer")

// Replace NSLog with:
logger.debug("Audio chunk loaded: \(chunkIndex)")
logger.error("Failed to load chunk: \(error.localizedDescription)")
```

#### 2. Add Performance Metrics
```swift
// In ChatterboxEngine
private let perfLogger = PerformanceLogger()

func synthesize(...) async throws {
    let start = CFAbsoluteTimeGetCurrent()
    // ... inference ...
    perfLogger.log("synthesizeChunk", duration: CFAbsoluteTimeGetCurrent() - start)
}
```

#### 3. Memory Pressure Handling
```swift
// Add to AppDelegate or SceneDelegate
func applicationDidReceiveMemoryWarning(_ application: UIApplication) {
    // Notify services to release caches
    NotificationCenter.default.post(name: .didReceiveMemoryWarning, object: nil)
}
```

#### 4. ONNX Session Management
- Consider creating/destroying sessions based on memory pressure
- Add session reuse metrics

### Deliverables Status
- [x] Current observability assessment
- [x] Gap identification
- [x] Recommendations
- [ ] Implementation

## Phase 2: Database & Memory Optimization

Since this is an iOS app, Phase 2 focuses on:
1. SQLite query optimization
2. Memory management for ONNX models
3. Audio buffer optimization
