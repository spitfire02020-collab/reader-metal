# Phase 10: Load Testing - Reader iOS App

## iOS Load Testing

### Not Traditional Web Testing
iOS apps don't have traditional "load testing" - but we can test:

1. **Long Audio Playback**: Test with 1+ hour audio
2. **Many Chunks**: Test with 100+ chunks
3. **Memory Pressure**: Test with limited memory (low-power mode)

### Test Scenarios

| Scenario | Metric | Target |
|----------|--------|--------|
| First chunk synthesis | Time | < 5s |
| Chunk-to-chunk gap | Time | < 100ms |
| Memory (idle) | MB | < 100 |
| Memory (synthesizing) | MB | < 700 |
| Battery (background play) | %/hour | < 10% |

## Implementation Status

| Test | Status |
|------|--------|
| Long playback | Manual |
| Memory profiling | Manual |
| Battery drain | Manual |
