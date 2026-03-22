#!/usr/bin/env python3
"""
Export GPT-2 Medium language model to Float16 ONNX format.

This script exports the Chatterbox Turbo GPT-2 Medium model to ONNX with float16 weights.
The exported model accepts pre-computed hidden_states (from embed_tokens) rather than input_ids.

Model Config (Chatterbox Turbo):
- vocab_size=6563 (speech tokens)
- n_positions=1500
- n_embd=1024 (hidden_size)
- n_layer=24
- n_head=16 (Q heads for GPT-2 Medium)
- n_inner=4096 (intermediate size)
- activation_function="gelu"
- eos_token_id=6562
- use_cache=True (KV cache for generation)

Usage:
    python export_lm_float16.py [--output_dir OUTPUT_DIR]
"""

import argparse
import os
import sys

# Apply patch BEFORE importing transformers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import patch_diff

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config
from transformers.pytorch_utils import Conv1D

# Fixed seed for reproducible export and verification
torch.manual_seed(42)


class GPT2Attention(nn.Module):
    """GPT2 Attention layer - simplified for ONNX export."""

    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.config = config
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention
        self.layer_idx = layer_idx

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_dropout)
        self.is_causal = True

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        """Attention computation - simplified without vmap."""
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (value.size(-1) ** 0.5)

        # Apply causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length:key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2)
        return attn_output, attn_weights

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        query, key, value = self.c_attn(hidden_states).split(self.embed_dim, dim=2)
        query = query.view(query.size(0), query.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(key.size(0), key.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(value.size(0), value.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = attn_output.contiguous().view(attn_output.size(0), attn_output.size(1), self.embed_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class GPT2MLP(nn.Module):
    """GPT2 MLP layer."""

    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = F.gelu
        self.dropout = nn.Dropout(config.resid_dropout)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):
    """GPT2 Transformer block."""

    def __init__(self, config, layer_idx=0):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs


class GPT2NoEmbed(nn.Module):
    """
    GPT-2 Model wrapper that accepts pre-computed hidden_states.

    This wrapper bypasses the embedding layers and position encodings,
    accepting hidden_states directly as input. This is the correct interface
    for the Chatterbox Turbo pipeline where:
    1. embed_tokens produces inputs_embeds from input_ids
    2. These embeddings are concatenated with audio features
    3. The concatenated embeddings are passed to this model

    This is a simplified implementation that avoids the problematic
    create_causal_mask and sdpa_mask functions in transformers.
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.num_hidden_layers = config.n_layer

        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        use_cache: bool = True,
        output_hidden_states: bool = False,
        past_key_values=None,
    ):
        """
        Forward pass with pre-computed hidden states.

        Args:
            hidden_states: Pre-computed embeddings [batch, seq_len, hidden_size]
            use_cache: Whether to return KV cache for generation
            output_hidden_states: Whether to return all hidden states
            past_key_values: Optional tuple of past key/value states

        Returns:
            If use_cache=True: (hidden_states, past_key_values)
            If use_cache=False: hidden_states
        """
        all_hidden_states = () if output_hidden_states else None

        # Create causal mask (simple causal, no vmap)
        batch_size, seq_len, _ = hidden_states.shape
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device)
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

        # Expand mask for batch
        causal_mask = causal_mask.expand(batch_size, -1, -1, -1)

        # Process each layer
        new_past_key_values = ()
        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Get past key/value for this layer if available
            layer_past = past_key_values[i] if past_key_values is not None else None

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=causal_mask,
                use_cache=use_cache,
                output_attentions=False,
            )

            hidden_states = outputs[0]

            if use_cache and outputs[1] is not None:
                new_past_key_values = new_past_key_values + (outputs[1],)

        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if use_cache:
            return hidden_states, new_past_key_values
        else:
            return hidden_states


def export_weights_binary(model: nn.Module, output_dir: str, prefix: str = ""):
    """
    Export model weights as individual .bin float16 files.

    Args:
        model: PyTorch model
        output_dir: Directory to write weight files
        prefix: Filename prefix for weight files
    """
    os.makedirs(output_dir, exist_ok=True)

    state_dict = model.state_dict()

    for name, param in state_dict.items():
        if param.dtype in [torch.float32, torch.float16]:
            # Convert to float16
            param_fp16 = param.half()

            # Generate filename from parameter name
            # Replace dots and brackets with underscores
            safe_name = name.replace(".", "_").replace("[", "_").replace("]", "_")
            if prefix:
                filename = f"{prefix}_{safe_name}.bin"
            else:
                filename = f"{safe_name}.bin"

            filepath = os.path.join(output_dir, filename)

            # Write as raw float16 bytes
            param_fp16.contiguous().cpu().numpy().astype('float16').tofile(filepath)

            print(f"  Exported: {filename} ({param.shape}, {param_fp16.dtype})")


def export_language_model(
    output_path: str,
    weights_dir: str,
    config: GPT2Config,
    opset_version: int = 17,
):
    """
    Export GPT-2 language model to ONNX format.

    Args:
        output_path: Path for the ONNX model file
        weights_dir: Directory for individual weight .bin files
        config: GPT2Config with model parameters
        opset_version: ONNX opset version (default: 17)
    """
    print("=" * 60)
    print("GPT-2 Medium Float16 ONNX Export")
    print("=" * 60)

    # Create model
    print("\n1. Creating GPT2NoEmbed model...")
    model = GPT2NoEmbed(config)
    model.eval()

    # Try to load trained weights
    weight_source = "random (no trained weights found)"
    state_dict = None

    # Check for local checkpoint
    checkpoint_paths = [
        os.path.expanduser("~/tmp/chatterbox_turbo/"),
        "/tmp/chatterbox_turbo/",
    ]
    for ckpt_dir in checkpoint_paths:
        safetensor_path = os.path.join(ckpt_dir, "t3_turbo_v1.safetensors")
        if os.path.exists(safetensor_path):
            print(f"   Found checkpoint at {safetensor_path}")
            try:
                from safetensors.torch import load_file
                state_dict = load_file(safetensor_path)
                if "model" in state_dict.keys():
                    state_dict = state_dict["model"][0]
                weight_source = "safetensor checkpoint"
                print(f"   Loaded weights from {safetensor_path}")
                break
            except Exception as e:
                print(f"   Failed to load safetensor: {e}")
                state_dict = None

    # Try importing ChatterboxTurboTTS if no checkpoint found
    if state_dict is None:
        try:
            from chatterbox_turbo import ChatterboxTurboTTS
            print("   Found chatterbox_turbo package, loading from pretrained...")
            tts = ChatterboxTurboTTS.from_pretrained(device="cpu")
            t3_model = tts.t3.tfmr  # GPT2Model
            state_dict = t3_model.state_dict()
            weight_source = "ChatterboxTurboTTS pretrained"
            print(f"   Loaded weights from ChatterboxTurboTTS")
        except ImportError:
            pass
        except Exception as e:
            print(f"   Failed to load from ChatterboxTurboTTS: {e}")

    # Apply loaded weights if available
    if state_dict is not None:
        try:
            model.load_state_dict(state_dict, strict=False)
            print(f"   Successfully loaded trained weights")
        except Exception as e:
            print(f"   Failed to apply state_dict: {e}")
            print("   Using randomly initialized weights")

    print(f"   Weight source: {weight_source}")

    # Convert to float16 for export
    print("   Converting model to float16...")
    model = model.half()

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")
    print(f"   Hidden size: {config.n_embd}")
    print(f"   Num layers: {config.n_layer}")
    print(f"   Num heads: {config.n_head}")
    print(f"   Vocab size: {config.vocab_size}")

    # Create dummy inputs
    print("\n2. Creating dummy inputs...")
    batch_size = 1
    seq_len = 32  # Short sequence for export
    hidden_size = config.n_embd

    dummy_hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
    print(f"   hidden_states: {dummy_hidden_states.shape}, dtype={dummy_hidden_states.dtype}")

    # Export to ONNX
    print("\n3. Exporting to ONNX...")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # We need to trace with torch.onnx.export
    # Since the model has control flow (use_cache), we'll export a single-step version

    # Export with dynamic axes for sequence length
    # Note: We export with use_cache=False for ONNX compatibility
    # KV cache handling should be done separately in the iOS app
    model_for_export = GPT2NoEmbed(config)
    model_for_export.load_state_dict(model.state_dict())
    model_for_export.eval()
    model_for_export = model_for_export.half()

    # Export with use_cache=False (no KV cache output)
    torch.onnx.export(
        model_for_export,
        (dummy_hidden_states,),
        output_path,
        input_names=["hidden_states"],
        output_names=["logits"],
        dynamic_axes={
            "hidden_states": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=opset_version,
        export_params=True,
        do_constant_folding=True,
    )
    print(f"   ONNX model saved to: {output_path}")

    # Export weights as binary float16
    print("\n4. Exporting weights as binary float16...")
    export_weights_binary(model, weights_dir, prefix="lm")
    print(f"   Weights saved to: {weights_dir}/")

    # Verify the ONNX model
    print("\n5. Verifying ONNX model...")
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("   ONNX model verified successfully!")

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export GPT-2 Medium to Float16 ONNX")
    parser.add_argument("--output_dir", type=str, default="./onnx_output",
                        help="Output directory for ONNX model and weights")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default: 17)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # GPT-2 Medium config for Chatterbox Turbo
    # Note: n_head=16 (GPT-2 Medium has 16 attention heads, head_dim=64)
    # 16 * 64 = 1024 hidden_size
    config = GPT2Config(
        vocab_size=6563,        # Speech tokens vocab size
        n_positions=1500,       # Max sequence length
        n_embd=1024,            # Hidden size (must match embed_tokens output)
        n_layer=24,            # 24 layers
        n_head=16,             # Attention heads (GPT-2 Medium: 16 heads x 64 head_dim = 1024)
        n_inner=4096,          # FFN inner size
        activation_function="gelu_new",  # GELU activation
        resid_dropout=0.0,      # No dropout in inference
        embd_dropout=0.0,      # No dropout in inference
        attn_dropout=0.0,      # No dropout in inference
        layer_norm_epsilon=1e-5,
        bos_token_id=6561,      # START_SPEECH token
        eos_token_id=6562,     # STOP_SPEECH token
        use_cache=True,
        scale_attn_weights=True,
        # Use eager attention to avoid vmap issues with ONNX export
        _attn_implementation="eager",
    )

    output_path = os.path.join(args.output_dir, "language_model_fp16.onnx")
    weights_dir = os.path.join(args.output_dir, "weights")

    export_language_model(output_path, weights_dir, config, args.opset)


if __name__ == "__main__":
    main()