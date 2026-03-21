#!/usr/bin/env python3
"""
Extract Q4F16 weights from language_model_q4f16_merged.onnx
and write them as binary .metalweight files for Metal loading.

Weight layout per quantized matmul (CONFIRMED from ONNX inspection):
  name.quant   -> uint8              [out_dim, block_count, 16]  <- 1 byte per element
  name.scales  -> float16             [out_dim, block_count]
  name.zp      -> uint8               [out_dim, block_count/16]  <- 1 zp per 16 blocks, NOT float16!

Dequantization formula:
  fp16_weight[row, block, col] = (quant[row, block, col] - zp[row, block/16]) * scale[row, block]
  (broadcast quant-zp over 16 cols, then scale)
"""
import argparse
import os
import struct
from pathlib import Path

import numpy as np
import onnx


def extract_q4f16_weights(onnx_path: str, output_dir: str) -> dict[str, dict]:
    """Extract all Q4F16 quantized weights from ONNX model."""
    model = onnx.load(onnx_path)
    initializers = {i.name: i for i in model.graph.initializer}

    q_weights = {k: v for k, v in initializers.items() if '_quant' in k}

    manifest = {}
    os.makedirs(output_dir, exist_ok=True)

    for qname, tensor in sorted(q_weights.items()):
        # Derive base name and reconstruct full scale/zp names from quant name
        # Two patterns exist:
        # 1. name_MatMul_weight_quant -> name, name_MatMul_weight_scales, name_MatMul_weight_zp (145 weights)
        # 2. name_weight_quant -> name, name_weight_scales, name_weight_zp (1 weight: transformer_wpe)
        if '_MatMul_weight_quant' in qname:
            base_name = qname.replace('_MatMul_weight_quant', '')
            scales_name = f'{base_name}_MatMul_weight_scales'
            zp_name = f'{base_name}_MatMul_weight_zp'
        else:
            base_name = qname.replace('_weight_quant', '')
            scales_name = f'{base_name}_weight_scales'
            zp_name = f'{base_name}_weight_zp'

        scales = initializers[scales_name]
        zp = initializers[zp_name]

        # Load raw bytes
        q_data = np.frombuffer(tensor.raw_data, dtype=np.uint8)
        scales_data = np.frombuffer(scales.raw_data, dtype=np.float16)
        zp_data = np.frombuffer(zp.raw_data, dtype=np.uint8)  # uint8 NOT float16

        out_dim = tensor.dims[0]
        block_count = tensor.dims[1]
        block_size = 16  # fixed for Q4F16

        manifest[base_name] = {
            'out_dim': out_dim,
            'block_count': block_count,
            'block_size': block_size,
            'quant_path': f'{base_name}.quant',
            'scales_path': f'{base_name}.scales',
            'zp_path': f'{base_name}.zp',
        }

        # Write binary files
        q_path = os.path.join(output_dir, f'{base_name}.quant')
        scales_path = os.path.join(output_dir, f'{base_name}.scales')
        zp_path = os.path.join(output_dir, f'{base_name}.zp')

        q_data.tofile(q_path)
        scales_data.tofile(scales_path)
        zp_data.tofile(zp_path)

    # Write manifest
    import json
    manifest_path = os.path.join(output_dir, 'weights_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'Extracted {len(manifest)} quantized weights to {output_dir}')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', required=True, help='Path to language_model_q4f16_merged.onnx')
    parser.add_argument('--output', required=True, help='Output directory for .metalweight files')
    args = parser.parse_args()
    extract_q4f16_weights(args.onnx, args.output)