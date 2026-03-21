#!/usr/bin/env python3
"""
Patch torch operations to make them ONNX-exportable.

The aten::diff operator is not supported by many ONNX exporters.
This patch replaces torch.diff with a simple ONNX-compatible implementation.

Install this patch BEFORE importing transformers.
"""

import torch


def onnx_safe_diff(input, dim=-1, prepend=None, out=None):
    """
    ONNX-compatible replacement for torch.diff.

    Computes the 1st order discrete difference along the specified dimension.
    This is equivalent to: output[i] = input[i] - input[i-1] for i > 0.

    Args:
        input: Input tensor
        dim: Dimension along which to compute the difference (default: -1)
        prepend: Values to prepend to input along dimension before computing diff
        out: Output buffer (unused, for API compatibility)

    Returns:
        Tensor of same shape as input, with the first element along dim unchanged
        and subsequent elements containing input[i] - input[i-1]
    """
    if prepend is not None:
        # Handle prepend by concatenating first
        input = torch.cat([prepend, input], dim=dim)

    # For 1D or last-dim diff: output[1:] = input[1:] - input[:-1]
    # First element remains unchanged (or 0 if prepend was used)

    if dim == -1 or dim == input.dim() - 1:
        # Simple case: work along last dimension
        output = torch.empty_like(input)
        output[..., 0] = input[..., 0]
        if input.shape[-1] > 1:
            output[..., 1:] = input[..., 1:] - input[..., :-1]
        return output
    else:
        # General case for other dimensions
        output = torch.empty_like(input)
        # Build slicing tuples
        slices_1 = [slice(None)] * input.dim()
        slices_1[dim] = slice(0, 1)
        slices_2 = [slice(None)] * input.dim()
        slices_2[dim] = slice(1, None)
        slices_3 = [slice(None)] * input.dim()
        slices_3[dim] = slice(None, -1)

        output[slices_1] = input[slices_1]
        output[slices_2] = input[slices_2] - input[slices_3]
        return output


def install_patch():
    """Install ONNX-safe patches."""
    # Patch torch.diff first
    torch.diff = onnx_safe_diff
    print("Patched torch.diff for ONNX compatibility")

    # Also patch the masking_utils to avoid vmap issues
    try:
        import transformers
        import transformers.masking_utils as masking_utils

        # Force using the older sdpa_mask that doesn't use vmap
        # This is needed because GPT2Model.create_causal_mask always calls sdpa_mask
        # even when using eager attention, and newer torch uses vmap which breaks ONNX

        if hasattr(masking_utils, 'sdpa_mask_recent_torch'):
            # Store original
            masking_utils._original_sdpa_mask = masking_utils.sdpa_mask_recent_torch

            # Replace with older version that doesn't use vmap
            if hasattr(masking_utils, 'sdpa_mask_older_torch'):
                masking_utils.sdpa_mask_recent_torch = masking_utils.sdpa_mask_older_torch
                masking_utils.sdpa_mask = masking_utils.sdpa_mask_older_torch
                print("Patched masking_utils.sdpa_mask to use older (vmap-free) version")

    except ImportError:
        print("transformers not yet imported, skipping masking_utils patch")
    except Exception as e:
        print(f"Warning: Could not patch masking_utils: {e}")


# Auto-install on import
install_patch()