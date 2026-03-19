# Replace Hand-Rolled Sentence Parser with NLTokenizer

## Goal
Replace `TextChunker.splitByPunctuation()` and `splitIntoSentences()` with Apple's `NLTokenizer(.sentence)` for more robust, linguistically-aware sentence splitting while keeping all other TextChunker behaviour (cleaning, cue preservation, caching, duration estimation) unchanged.

## Tasks

- [x] **1. Add `import NaturalLanguage`** to `TextChunker.swift` → Verify: file compiles
- [x] **2. Rewrite `splitIntoSentences()`** to use `NLTokenizer(.sentence)`, remove `splitByPunctuation()`, remove `splitBulletPointText()`, remove `splitLongSentence()`, remove the `abbreviations` set and `bulletPointPattern` regex → Verify: all 19 existing tests pass
- [x] **3. Handle bullet points** inside the new `splitIntoSentences()` — `NLTokenizer` treats each bullet as a sentence naturally, so verify with a new test case → Verify: new test passes
- [x] **4. Add new edge-case tests** covering known weaknesses of the old parser:
  - Abbreviations mid-sentence: `"Dr. Smith said Mr. Jones arrived."`
  - Ellipsis: `"Wait... What happened? I don't know."`
  - Unicode sentence boundaries: `"Ciao! ¿Cómo estás? Bien."`
  - Numbered list: `"1. First item 2. Second item"`
  → Verify: new tests pass
- [x] **5. Run full test suite** → Verify: `xcodebuild test` exits 0

## Done When
- [x] `splitByPunctuation`, `splitBulletPointText`, `splitLongSentence`, `abbreviations`, `bulletPointPattern` are all deleted
- [x] All 19 existing tests + 4 new tests pass
- [x] No changes to the public API (`chunkText`, `cleanTextForDisplay`, `estimateDuration`, `estimateTotalDuration`)

## Notes
- `NLTokenizer` handles abbreviations, Unicode, quotes, and bullet lists out of the box — it replaces ~170 lines of hand-rolled parsing with ~15 lines.
- The public API surface is unchanged, so all 25 call sites in `ChatterboxEngine`, `PlayerViewModel`, `LibraryViewModel`, `SynthesisDatabase` need **zero** changes.
- The non-verbal cue placeholder system is retained (it runs *before* sentence splitting).
- Some existing test counts may shift by ±1 if `NLTokenizer` groups sentences differently from the old parser (e.g. `"Hello 世界! こんにちは!"` — old parser returned 3, NLTokenizer may return 2). We'll adjust test expectations to match linguistically-correct behaviour.
