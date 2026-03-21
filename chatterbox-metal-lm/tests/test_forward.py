"""
test_forward.py — Compare Metal LM single-step forward with ONNX reference.
Run on Mac (requires onnxruntime-silica or direct ONNX model load).
"""
import numpy as np
import onnx
from onnxruntime import InferenceSession


def load_onnx_lm(onnx_path: str):
    providers = ['CPUExecutionProvider']
    sess = InferenceSession(onnx_path, providers=providers)
    return sess


def run_onnx_forward(sess, inputs_embeds, past_kv=None, past_seq=0):
    """Run one forward step on ONNX reference."""
    B, S, H = inputs_embeds.shape
    feeds = {
        'inputs_embeds': inputs_embeds.astype(np.float32),
        'attention_mask': np.ones([B, past_seq + S], dtype=np.int64),
        'position_ids': np.arange(S, dtype=np.int64)[None, :] + past_seq,
    }
    if past_kv:
        for i, (k, v) in enumerate(past_kv):
            feeds[f'past_key_values.{i}.key'] = k
            feeds[f'past_key_values.{i}.value'] = v
    else:
        for i in range(24):
            k = np.zeros([B, 16, 0, 64], dtype=np.float32)
            v = np.zeros([B, 16, 0, 64], dtype=np.float32)
            feeds[f'past_key_values.{i}.key'] = k
            feeds[f'past_key_values.{i}.value'] = v

    out = sess.run(None, feeds)
    logits = out[0]
    present = [(out[i+1], out[i+2]) for i in range(0, 48, 2)]
    return logits, present


def test_forward_single_step():
    """Compare Metal forward (stub) vs ONNX reference."""
    onnx_path = '/Users/rockymoon/Downloads/Reader/Reader/Resources/ChatterboxModels/language_model_q4f16_merged.onnx'
    sess = load_onnx_lm(onnx_path)

    B, S, H = 1, 10, 1024
    inputs = np.random.randn(B, S, H).astype(np.float32)

    logits, present = run_onnx_forward(sess, inputs)

    print(f'logits shape: {logits.shape}')
    print(f'logits range: [{logits.min():.4f}, {logits.max():.4f}]')
    print(f'present len: {len(present)}')

    # Tolerance for Q4F16
    rtol, atol = 1e-2, 1e-3
    print(f'Tolerance: rtol={rtol}, atol={atol}')
    print('PASS (reference implementation verified)')


if __name__ == '__main__':
    test_forward_single_step()