# Phase 4: Database & Memory Optimization - Reader iOS App

## SQLite Optimizations

### Issue 1: No Pagination in getAllItems()
**Location**: SynthesisDatabase.swift:227-248
**Problem**: Loads all items into memory
**Solution**: Add limit/offset support

```swift
func getAllItems(limit: Int = 50, offset: Int = 0) throws -> [SynthesisItemRow] {
    guard let db else { return [] }
    let query = synthesisItems
        .order(siUpdatedAt.desc)
        .limit(limit, offset: offset)
    return try db.prepare(query).map { row in ... }
}
```

### Issue 2: N+1 Query in updateItemProgress
**Location**: SynthesisDatabase.swift:262-277
**Problem**: Two separate count queries
**Solution**: Single query with aggregation

### Issue 3: Missing Indexes
**Recommendation**: Add composite index on (chItemId, chChunkIndex)

## Memory Optimizations

### Issue 1: ONNX Session Memory
**Location**: ChatterboxEngine.swift
**Problem**: Sessions kept in memory indefinitely
**Solution**: Add memory pressure handling

```swift
func handleMemoryWarning() {
    // Release non-essential sessions
    speechEncoderSession = nil
    embedTokensSession = nil
    // Keep only essential for playback
}
```

### Issue 2: Waveform Buffer Size
**Location**: AudioPlayerService.swift:871-940
**Problem**: Processes full sample count
**Solution**: Downsample for visualization

```swift
// Downsample to 200 points for display
let downsampleFactor = max(1, sampleCount / 200)
let displaySamples = stride(from: 0, to: sampleCount, by: downsampleFactor)
    .map { samples[$0] }
```

## Implementation Priority

1. **P0**: Waveform downsampling (quick win)
2. **P0**: Busy-wait replacement (CPU/battery)
3. **P1**: Pagination for getAllItems
4. **P1**: Memory pressure handling
5. **P2**: Database indexes
