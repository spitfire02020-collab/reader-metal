# Phase 6: iOS-Specific Optimizations - Reader App

## Not Applicable
This is an iOS app, not a distributed system. Skip Phase 6 (Distributed Systems Optimization).

## iOS-Specific Considerations

### 1. Instruments Profiling
- Use Time Profiler to measure ONNX inference
- Use Allocations to track memory
- Use Core Animation for UI jank

### 2. Device Testing
- Test on actual iOS devices (not just simulator)
- ONNX performance varies by chip (A-series vs M-series)

### 3. Battery Impact
- Background audio drains battery
- ONNX inference is CPU-intensive

## Recommendations for Device Testing

1. **CPU Profiling**: Focus on ChatterboxEngine.synthesizeChunk()
2. **Memory**: Track ONNX session allocation/deallocation
3. **Battery**: Monitor background audio playback
