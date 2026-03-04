#!/usr/bin/env python3
"""
Test Chatterbox-TTS pipeline on Mac using ONNX Runtime.

This script tests the same pipeline that the iOS app uses.
We use ONNX Runtime directly since CoreML conversion requires additional tools.
"""

import os
import sys
import time
import resource
import numpy as np
import onnxruntime as ort
from scipy.io import wavfile

# ── Configuration ────────────────────────────────────────────────────────────
MODELS_DIR = "/tmp/fresh_models"
TOKENIZER_PATH = MODELS_DIR + "/tokenizer.json"

SAMPLE_RATE = 24000
START_SPEECH_TOK = 6561
STOP_SPEECH_TOK = 6562
SILENCE_TOKEN = 4299
NUM_KV_HEADS = 16
HEAD_DIM = 64
MAX_NEW_TOKENS = 1024
REP_PENALTY = 1.2
TEST_TEXT = "He was here for a reason. Maintaining his place in the sect was absolutely critical, both because of his Quests and because"
OUTPUT_WAV = "/Users/rockymoon/Downloads/Reader/test_coreml_output.wav"
REF_WAV = "/tmp/fresh_models/default_voice.wav"


def mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


class RepetitionPenaltyLogitsProcessor:
    """Exact match of the reference implementation."""
    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        score = np.take_along_axis(scores, input_ids, axis=1)
        score = np.where(score < 0, score * self.penalty, score / self.penalty)
        scores_processed = scores.copy()
        np.put_along_axis(scores_processed, input_ids, score, axis=1)
        return scores_processed


def load_onnx_model(name: str):
    """Load an ONNX model."""
    onnx_path = os.path.join(MODELS_DIR, f"{name}_q4f16.onnx")
    print(f"Loading {name} from {onnx_path}...")

    # Create session with CPU only (the CoreML provider has issues with GatherBlockQuantized)
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)

    t0 = time.time()
    # Warmup
    inputs = {inp.name: np.zeros([1, 1], dtype=np.int64) for inp in session.get_inputs()}
    if "audio_values" in [inp.name for inp in session.get_inputs()]:
        inputs = {"audio_values": np.zeros([1, 24000], dtype=np.float32)}
    _ = session.run(None, inputs)
    print(f"  Loaded in {time.time()-t0:.2f}s (RSS {mb():.0f} MB)")
    return session


def main():
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")
    print(f"Models dir:  {MODELS_DIR}")
    print(f"Initial RSS: {mb():.0f} MB\n")

    # ── 1. Load models ─────────────────────────────────────────────────────
    print("── Loading models ──\n")

    speech_enc = load_onnx_model("speech_encoder")
    embed_tok = load_onnx_model("embed_tokens")
    lang_model = load_onnx_model("language_model")
    cond_dec = load_onnx_model("conditional_decoder")

    print(f"\nAll loaded. Peak RSS: {mb():.0f} MB\n")

    # ── 2. Tokenize text ─────────────────────────────────────────────────
    print(f"── Tokenizing: '{TEST_TEXT}' ──")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    enc = tokenizer.encode(TEST_TEXT)
    input_ids = np.array([enc.ids], dtype=np.int64)
    print(f"  Tokens ({len(enc.ids)}): {enc.ids}")

    # ── 3. Reference audio → speaker context ─────────────────────────────
    ref_wav = REF_WAV
    if os.path.exists(ref_wav):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref_sr, ref_data = wavfile.read(ref_wav)
        if ref_data.dtype != np.float32:
            ref_data = ref_data.astype(np.float32) / 32767.0
        if ref_sr != SAMPLE_RATE:
            ratio = SAMPLE_RATE / ref_sr
            ref_data = ref_data[::max(1, int(1/ratio))] if ratio < 1 else ref_data
        audio_values = ref_data.reshape(1, -1)
        print(f"\n── Speech encoder on reference voice ({audio_values.shape[1]/SAMPLE_RATE:.2f}s) ──")
    else:
        print("ERROR: No reference voice found at", ref_wav)
        sys.exit(1)

    t0 = time.time()
    # Run speech encoder
    enc_outs = speech_enc.run(None, {"audio_values": audio_values})
    # ONNX returns list of outputs
    audio_features = enc_outs[0]        # [1, S, 1024]
    audio_tokens = enc_outs[1]          # [1, audio_seq]
    speaker_embeddings = enc_outs[2]     # [1, 192]
    speaker_features = enc_outs[3]       # [1, F, 80]
    print(f"  Done in {time.time()-t0:.2f}s")
    print(f"  audio_features:     {audio_features.shape} {audio_features.dtype}")
    print(f"  audio_tokens:       {audio_tokens.shape} {audio_tokens.dtype}")
    print(f"  speaker_embeddings: {speaker_embeddings.shape}")
    print(f"  speaker_features:   {speaker_features.shape}")

    # ── 4. Generation loop ───────────────────────────────────────────────
    print(f"\n── LM decode (max {MAX_NEW_TOKENS} tokens, greedy + rep_penalty={REP_PENALTY}) ──")

    # Get input names for KV cache
    lm_input_names = [inp.name for inp in lang_model.get_inputs()]
    num_kv_layers = sum(1 for n in lm_input_names if n.startswith("past_key_values.") and n.endswith(".key"))
    print(f"  KV layers: {num_kv_layers}")

    # Determine KV dtype from model
    kv_dtype = np.float32
    for inp in lang_model.get_inputs():
        if "past_key_values" in inp.name:
            if "fp16" in str(inp.type).lower() or "float16" in str(inp.type).lower():
                kv_dtype = np.float16
            break
    print(f"  KV dtype: {kv_dtype}")

    repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=REP_PENALTY)
    generate_tokens = np.array([[START_SPEECH_TOK]], dtype=np.int64)

    past_key_values = {}
    attention_mask = None
    position_ids = None
    batch_size = 1

    # Get ordered KV input names from model's input list
    kv_input_names = [inp.name for inp in lang_model.get_inputs() if "past_key_values" in inp.name]

    for i in range(MAX_NEW_TOKENS):
        t0 = time.time()

        # Embed current input_ids
        inputs_embeds = embed_tok.run(None, {"input_ids": input_ids})[0]

        if i == 0:
            # First step: prepend audio features
            inputs_embeds = np.concatenate((audio_features, inputs_embeds), axis=1)

            # Initialize KV cache
            batch_size, seq_len, _ = inputs_embeds.shape
            past_key_values = {
                inp.name: np.zeros([batch_size, NUM_KV_HEADS, 0, HEAD_DIM], dtype=kv_dtype)
                for inp in lang_model.get_inputs()
                if "past_key_values" in inp.name
            }
            attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
            position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        # Run language model
        lm_feed = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **past_key_values,
        )
        lm_outputs = lang_model.run(None, lm_feed)

        # Handle different output formats
        logits = lm_outputs[0]
        present_key_values = lm_outputs[1:]

        dt = time.time() - t0

        # Get last-position logits
        logits = logits[:, -1, :]

        # Apply repetition penalty
        next_token_logits = repetition_penalty_processor(generate_tokens, logits)

        # Greedy decoding
        input_ids = np.argmax(next_token_logits, axis=-1, keepdims=True).astype(np.int64)
        generate_tokens = np.concatenate((generate_tokens, input_ids), axis=-1)

        tok_val = int(input_ids.flatten()[0])
        if i < 5 or i % 50 == 0:
            print(f"  step {i:4d}: tok={tok_val:5d} ({dt:.2f}s) RSS={mb():.0f}MB")

        if (input_ids.flatten() == STOP_SPEECH_TOK).all():
            print(f"  [STOP_SPEECH] at step {i}")
            break

        # Update masks
        attention_mask = np.concatenate([attention_mask, np.ones((batch_size, 1), dtype=np.int64)], axis=1)
        position_ids = position_ids[:, -1:] + 1

        # Update KV cache
        for j, key in enumerate(kv_input_names):
            if j < len(present_key_values):
                past_key_values[key] = present_key_values[j]

    # ── 5. Extract speech tokens ──────────────────────────────────────────
    all_gen = generate_tokens.flatten().tolist()
    print(f"\n  All generated tokens ({len(all_gen)}): {all_gen[:30]}{'...' if len(all_gen)>30 else ''}")

    speech_tokens = generate_tokens[:, 1:-1]
    print(f"  Speech tokens: {speech_tokens.shape}")
    if speech_tokens.size == 0:
        print("  No speech tokens — skipping decoder")
        return
    print(f"  Token range: {speech_tokens.min()} – {speech_tokens.max()}")

    # ── 6. Conditional decoder ─────────────────────────────────────────────
    silence_tokens = np.full((1, 3), SILENCE_TOKEN, dtype=np.int64)
    decoder_input = np.concatenate([audio_tokens, speech_tokens, silence_tokens], axis=1)

    print(f"\n── Conditional decoder ──")
    print(f"  decoder_input: {decoder_input.shape}")

    t0 = time.time()
    wav = cond_dec.run(None, {
        "speech_tokens": decoder_input,
        "speaker_embeddings": speaker_embeddings,
        "speaker_features": speaker_features,
    })[0]

    print(f"  Done in {time.time()-t0:.2f}s shape={wav.shape} dtype={wav.dtype}")

    # ── 7. Save WAV ───────────────────────────────────────────────────────
    audio = wav.squeeze(axis=0).flatten().astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)

    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))
    print(f"\n  Audio: peak={peak:.4f} RMS={rms:.4f}")
    print(f"  Duration: {len(audio)/SAMPLE_RATE:.2f}s ({len(audio)} samples)")

    audio_i16 = (audio * 32767).astype(np.int16)
    wavfile.write(OUTPUT_WAV, SAMPLE_RATE, audio_i16)
    print(f"  Saved → {OUTPUT_WAV}")
    print(f"\nDone! Final RSS: {mb():.0f} MB")


if __name__ == "__main__":
    main()
