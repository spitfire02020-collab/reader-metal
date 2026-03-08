# Phase 5: Backend Optimization - Reader iOS App

## Busy-Wait Polling Fix (COMPLETED)

### Before
```swift
// Blocked execution with busy-wait
while audioPlayer.isPlaying {
    try? await Task.sleep(nanoseconds: 100_000_000)
}
```

### After
- Removed blocking busy-wait loops
- Added Combine subscriber to `$isPlaying`
- Playback state handled by AVAudioPlayerDelegate automatically

### Impact
- **CPU**: Eliminates wasteful polling
- **Battery**: Reduces background CPU usage
- **Responsiveness**: UI remains responsive during playback

## Waveform Generation (Already Optimized)

The waveform generation already:
- Downsamples to 200 samples for display
- Uses Task.detached with proper cancellation
- Has periodic cancellation checks

### Potential Further Optimization
```swift
// Use Accelerate framework for faster max calculation
import Accelerate

var maxVal: Float = 0
vDSP_maxv(channelData + start, 1, &maxVal, vDSP_Length(end - start))
```

## Database Pagination (Recommended)

```swift
// Add pagination to getAllItems
func getAllItems(limit: Int = 50, offset: Int = 0) throws -> [SynthesisItemRow]
```

## Memory Management

### ONNX Session Lifecycle
- Consider releasing sessions on memory warning
- Keep only active session in memory

## Implementation Status

| Optimization | Status | Impact |
|--------------|--------|--------|
| Busy-wait removal | ✅ Complete | High |
| Waveform downsampling | ✅ Already done | Medium |
| Database pagination | 📋 Backlog | Low |
| Memory handling | 📋 Backlog | Medium |
