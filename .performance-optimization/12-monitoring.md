# Phase 12: Production Monitoring - Reader iOS App

## iOS Monitoring Options

### Built-in
- **os.Logger**: Structured logging (Swift 5.3+)
- **MetricKit**: Crash reporting, performance metrics (iOS 13+)

### Third-party (Optional)
- **Firebase Crashlytics**: Crash reporting
- **Firebase Performance**: Performance monitoring
- **Datadog**: Custom metrics

## Implementation

### Replace NSLog with os.Logger

```swift
import os.log

private let logger = Logger(subsytem: "com.reader.app", category: "AudioPlayer")

// Replace
NSLog("[AudioPlayer] chunk loaded")

// With
logger.debug("chunk loaded")
```

### Add Performance Logging

```swift
func synthesize(...) async throws {
    let start = CFAbsoluteTimeGetCurrent()
    // ... inference ...
    logger.info("synthesis completed in \(
        CFAbsoluteTimeGetCurrent() - start, format: ".2f")s")
}
```

## Implementation Status

| Item | Status |
|------|--------|
| os.Logger migration | 📋 Backlog |
| Performance metrics | 📋 Backlog |
| Crash reporting | 📋 Optional |
