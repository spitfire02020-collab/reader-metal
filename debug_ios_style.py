#!/usr/bin/env python3
"""
Debug script to trace iOS-style inference step by step.
This replicates the iOS ChatterboxEngine logic exactly.
"""

import os, sys, time
import numpy as np
import onnxruntime as ort
from scipy.io import wavfile

# Configuration
MODELS_DIR = (
    "/Users/rockymoon/Library/Developer/CoreSimulator/Devices/"
    "43BFF268-600A-4DB9-AECF-B6730357F0D6/data/Containers/Data/"
    "Application/5D69A735-FB46-4581-904E-D0A311AC3B9E/Documents/ChatterboxModels"
)
TOKENIZER_PATH = MODELS_DIR + "/tokenizer.json"
DEFAULT_VOICE_PATH = MODELS_DIR + "/default_voice.wav"
OUTPUT_WAV = "/Users/rockymoon/Downloads/Reader/test_ios_style.wav"

SAMPLE_RATE = 24000
START_SPEECH_TOK = 6561
STOP_SPEECH_TOK = 6562
SILENCE_TOKEN = 4299
NUM_KV_HEADS = 16
HEAD_DIM = 64
MAX_NEW_TOKENS = 1500
REP_PENALTY = 1.2

TEST_TEXT = "Hello, this is a test of chatterbox text to speech."


def make_session(name):
    onnx_path = os.path.join(MODELS_DIR, f"{name}_q4f16.onnx")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
    return sess


class RepetitionPenaltyLogitsProcessor:
    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        score = np.take_along_axis(scores, input_ids, axis=1)
        score = np.where(score < 0, score * self.penalty, score / self.penalty)
        scores_processed = scores.copy()
        np.put_along_axis(scores_processed, input_ids, score, axis=1)
        return scores_processed


def load_audio_as_float(path, target_sr=24000):
    """Load audio file and convert to Float array at target sample rate - matches iOS"""
    sr, data = wavfile.read(path)
    if data.dtype != np.float32:
        data = data.astype(np.float32) / 32767.0
    if sr != target_sr:
        # Simple resample
        ratio = target_sr / sr
        if ratio > 1:
            # Upsample - repeat samples
            data = np.repeat(data, int(ratio))
        else:
            # Downsample - take every Nth sample
            data = data[::int(1/ratio)]
    return data.reshape(1, -1)


def create_int64_tensor(data, shape):
    """Create int64 tensor - matches iOS createInt64Tensor"""
    return np.array(data, dtype=np.int64).reshape(shape)


def concat_int64_tensors(a, b):
    """Concatenate int64 tensors along axis 1 - matches iOS concatInt64Tensors"""
    return np.concatenate([a, b], axis=1)


def extract_float_array(value):
    """Extract float array from tensor - matches iOS extractFloatArray"""
    if value.dtype == np.float16:
        return value.astype(np.float32)
    return value


def greedy_next_token(logits, previous, penalty=1.2):
    """Greedy decode with repetition penalty - matches iOS greedyNextToken"""
    adj = logits.copy()
    for token in set(previous):
        if token < len(adj):
            if adj[token] > 0:
                adj[token] /= penalty
            else:
                adj[token] *= penalty
    return int(np.argmax(adj))


def main():
    print("=== iOS-Style Inference Debug ===")
    print(f"ORT: {ort.__version__}")
    print(f"Models: {MODELS_DIR}")
    print(f"Text: '{TEST_TEXT}'")
    print()

    # Load models
    print("Loading models...")
    speech_enc = make_session("speech_encoder")
    embed_tok = make_session("embed_tokens")
    lang_model = make_session("language_model")
    cond_dec = make_session("conditional_decoder")

    # Load tokenizer
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    enc = tokenizer.encode(TEST_TEXT)
    input_ids = np.array([enc.ids], dtype=np.int64)
    print(f"Token IDs ({len(enc.ids)}): {enc.ids}")

    # Load reference audio - matches iOS createDefaultSpeakerContext
    print(f"\nLoading reference voice: {DEFAULT_VOICE_PATH}")
    audio_values = load_audio_as_float(DEFAULT_VOICE_PATH, SAMPLE_RATE)
    print(f"Audio shape: {audio_values.shape}")

    # Speech encoder - matches iOS encodeSpeakerVoiceFromSamples
    print("\n=== Speech Encoder ===")
    enc_outs = speech_enc.run(None, {"audio_values": audio_values})
    audio_features = enc_outs[0]   # [1, S, 1024]
    audio_tokens = enc_outs[1]      # [1, T] - prompt_token
    speaker_embeddings = enc_outs[2]  # [1, 192]
    speaker_features = enc_outs[3]   # [1, F, 80]

    print(f"audio_features: {audio_features.shape} dtype={audio_features.dtype}")
    print(f"audio_tokens (prompt): {audio_tokens.shape} dtype={audio_tokens.dtype}")
    print(f"speaker_embeddings: {speaker_embeddings.shape} dtype={speaker_embeddings.dtype}")
    print(f"speaker_features: {speaker_features.shape} dtype={speaker_features.dtype}")

    # LM inference
    print("\n=== LM Decode ===")
    lm_input_names = [i.name for i in lang_model.get_inputs()]
    num_kv_layers = sum(1 for n in lm_input_names if n.startswith("past_key_values.") and n.endswith(".key"))
    print(f"KV layers: {num_kv_layers}")

    # Embed text
    inputs_embeds = embed_tok.run(None, {"input_ids": input_ids})[0]

    # Prepend audio features - matches iOS concatEmbeddings
    inputs_embeds = np.concatenate([audio_features, inputs_embeds], axis=1)
    total_seq_len = inputs_embeds.shape[1]
    print(f"Combined inputs_embeds shape: {inputs_embeds.shape}")

    # Initialize KV cache
    past_key_values = {
        inp.name: np.zeros([1, NUM_KV_HEADS, 0, HEAD_DIM], dtype=np.float16)
        for inp in lang_model.get_inputs()
        if "past_key_values" in inp.name
    }
    attention_mask = np.ones((1, total_seq_len), dtype=np.int64)
    position_ids = np.arange(total_seq_len, dtype=np.int64).reshape(1, -1)

    # Generation loop
    generate_tokens = [START_SPEECH_TOK]
    speech_tokens = []

    for step in range(MAX_NEW_TOKENS):
        lm_feed = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **past_key_values,
        )
        lm_outputs = lang_model.run(None, lm_feed)
        logits = lm_outputs[0]

        # Get last position
        logits = logits[:, -1, :]

        # Apply repetition penalty
        next_token = greedy_next_token(logits.flatten(), generate_tokens, REP_PENALTY)
        generate_tokens.append(next_token)

        if step < 10:
            print(f"  step {step}: tok={next_token}")

        if next_token == STOP_SPEECH_TOK:
            print(f"  STOP at step {step}")
            break

        speech_tokens.append(next_token)

        # Update for next iteration
        attention_mask = np.concatenate([attention_mask, np.ones((1, 1), dtype=np.int64)], axis=1)
        position_ids = position_ids[:, -1:] + 1

        # Embed next token
        next_id = np.array([[next_token]], dtype=np.int64)
        next_emb = embed_tok.run(None, {"input_ids": next_id})[0]
        inputs_embeds = next_emb

        # Update KV cache
        present = lm_outputs[1:]
        kv_names = [inp.name for inp in lang_model.get_inputs() if "past_key_values" in inp.name]
        for j, key in enumerate(kv_names):
            past_key_values[key] = present[j]

    print(f"\nGenerated {len(speech_tokens)} speech tokens")
    print(f"First 10: {speech_tokens[:10]}")

    # Decoder - matches iOS
    print("\n=== Conditional Decoder ===")

    # Build decoder input: concat [audio_tokens, speech_tokens, silence×3]
    silence_tail = np.full((1, 3), SILENCE_TOKEN, dtype=np.int64)
    decoder_input = np.concatenate([audio_tokens, np.array([speech_tokens], dtype=np.int64), silence_tail], axis=1)

    print(f"audio_tokens: {audio_tokens.shape}")
    print(f"speech_tokens: {np.array([speech_tokens]).shape}")
    print(f"decoder_input: {decoder_input.shape}")

    # Run decoder
    wav = cond_dec.run(None, dict(
        speech_tokens=decoder_input,
        speaker_embeddings=speaker_embeddings,
        speaker_features=speaker_features,
    ))[0]

    print(f"waveform: {wav.shape} dtype={wav.dtype}")

    # Save
    audio = wav.squeeze(axis=0).flatten().astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)

    print(f"\nAudio: {len(audio)} samples, {len(audio)/SAMPLE_RATE:.2f}s")
    print(f"Peak: {np.max(np.abs(audio)):.4f}, RMS: {np.sqrt(np.mean(audio**2)):.4f}")

    audio_i16 = (audio * 32767).astype(np.int16)
    wavfile.write(OUTPUT_WAV, SAMPLE_RATE, audio_i16)
    print(f"Saved: {OUTPUT_WAV}")


if __name__ == "__main__":
    main()
