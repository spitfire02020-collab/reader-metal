# GPT-2 Medium Float16 ONNX Export

Export pipeline for the Chatterbox Turbo GPT-2 Medium language model to Float16 ONNX format.

## Overview

This directory contains the tools to export the GPT-2 Medium language model (used in Chatterbox Turbo TTS) to ONNX format with float16 weights. The exported model is designed for Metal GPU acceleration on Apple Silicon.

## Model Configuration

The exported model uses this configuration (GPT-2 Medium for Chatterbox Turbo):

| Parameter | Value |
|-----------|-------|
| vocab_size | 6563 (speech tokens) |
| n_positions | 1500 |
| n_embd (hidden_size) | 1024 |
| n_layer | 24 |
| n_head | 80 |
| n_inner | 4096 |
| activation_function | gelu |
| eos_token_id | 6562 (STOP_SPEECH) |

## Files

- `requirements.txt` - Python dependencies
- `patch_diff.py` - Patches `torch.diff` for ONNX compatibility
- `export_lm_float16.py` - Main export script
- `verify_output.py` - Verification script for exported ONNX model
- `README.md` - This file

## Prerequisites

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

The dependencies include:
- torch>=2.0
- transformers
- onnx
- numpy
- onnxruntime
- safetensors

## Usage

### 1. Export the Model

Run the export script:

```bash
python export_lm_float16.py --output_dir ./onnx_output
```

Options:
- `--output_dir`: Output directory (default: `./onnx_output`)
- `--opset`: ONNX opset version (default: 17)

This will create:
- `onnx_output/language_model_fp16.onnx` - The ONNX model
- `onnx_output/weights/` - Individual weight files as float16 binary

### 2. Verify the Export

Run the verification script:

```bash
python verify_output.py --model_path ./onnx_output/language_model_fp16.onnx
```

Options:
- `--model_path`: Path to the ONNX model
- `--hidden_size`: Hidden dimension (default: 1024)
- `--vocab_size`: Vocabulary size (default: 6563)
- `--batch_size`: Batch size for test (default: 1)
- `--seq_len`: Sequence length for test (default: 32)

## Technical Details

### Why patch_diff.py?

The `aten::diff` operator (used in `torch.diff`) is not supported by many ONNX exporters. This patch replaces `torch.diff` with an ONNX-compatible implementation that computes the same result using basic tensor operations.

### Why GPT2NoEmbed?

The standard `GPT2Model.forward()` accepts `input_ids` and computes embeddings internally. However, in the Chatterbox Turbo pipeline:

1. `embed_tokens` produces `inputs_embeds` from `input_ids`
2. These embeddings are concatenated with audio features from the speech encoder
3. The concatenated embeddings are passed to the language model

The `GPT2NoEmbed` wrapper accepts pre-computed `hidden_states` directly, bypassing the embedding layers.

### Float16 Export

The model is exported with float16 weights to:
1. Reduce model size (~50% savings vs float32)
2. Enable Metal GPU acceleration with half-precision tensors
3. Maintain sufficient precision for high-quality speech synthesis

## Troubleshooting

### "aten::diff operator not supported"

Make sure `patch_diff.py` is imported before `transformers`. The export script handles this automatically.

### "ONNX model verification failed"

Try running with a newer ONNX opset version:
```bash
python export_lm_float16.py --opset 18
```

### "CUDA out of memory"

The export uses CPU by default. If you have a powerful GPU and want to use it for export, modify the script to use CUDA.

## License

This export code is part of the Reader project and follows the same license terms.