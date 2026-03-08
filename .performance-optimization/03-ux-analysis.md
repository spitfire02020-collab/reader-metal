# Phase 3: User Experience Analysis - Reader iOS App

## UX Performance Metrics

### Perceived Performance
| Operation | Target | Current | Status |
|-----------|--------|---------|--------|
| App launch | < 2s | ~3-5s | ⚠️ |
| First chunk ready | < 3s | ~5-8s | ⚠️ |
| Chunk-to-chunk gap | < 100ms | 0ms (streaming) | ✅ |
| UI responsiveness | 60fps | 60fps | ✅ |
| Waveform render | < 500ms | ~1s | ⚠️ |

### User Journeys
1. **Add URL → Listen**: ~10-15s to first audio
2. **Resume playback**: ~2s to resume
3. **Switch voice**: ~5s to re-synthesize

## iOS UX Findings

### 1. Busy-Wait Polling (Critical)
**Location**: PlayerViewModel.swift:1504-1512
```swift
while audioPlayer.isPlaying || audioPlayer.currentTime > 0 {
    try? await Task.sleep(nanoseconds: 500_000_000) // 0.5 seconds
}
```
**Impact**: Wastes CPU, drains battery
**Fix**: Use AVAudioPlayerDelegate or Combine publisher

### 2. Waveform Generation
**Location**: AudioPlayerService.swift:871-940
**Impact**: Processes full audio samples for visualization
**Fix**: Downsample to 100-200 points for display

### 3. Eager Text Processing
**Location**: PlayerViewModel.swift:358-384
```swift
cachedAttributedParagraphs = cachedParagraphs.enumerated().map { index, paragraph in
    // Creates attributed string for ALL paragraphs upfront
}
```
**Impact**: Memory allocation on load
**Fix**: Lazy loading on scroll

### 4. ONNX Inference Time
- speech_encoder: ~500ms
- language_model (1500 tokens): ~3-5s
- conditional_decoder: ~1-2s
- **Total per chunk**: ~5-8s

## Recommendations

### Quick Wins
1. Replace busy-wait with delegate callbacks
2. Downsample waveform data for display
3. Lazy-load paragraph attributes

### Medium-term
1. Add loading indicators during synthesis
2. Pre-generate next chunk ahead of playback
3. Cache ONNX sessions

### Long-term
1. Background audio processing
2. Progressive UI updates

## Deliverables Status
- [x] Core metrics analysis
- [x] User journey performance
- [x] Recommendations
