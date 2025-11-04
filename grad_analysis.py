#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradient analysis: Analyze layer-wise gradient similarities across components
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

def extract_step_number(npz_file):
    """Extract step number from filename like 'gradients_step_0.npz'."""
    import re
    match = re.search(r'step[_\-]?(\d+)', os.path.basename(npz_file))
    if match:
        return int(match.group(1))
    return 0  # Default if no step number found

def load_gradients(npz_file):
    """Load gradients from npz file."""
    data = np.load(npz_file)
    print(f"Loaded {len(data.files)} gradient arrays from {npz_file}")
    return data

def group_params_by_layer(data):
    """Group parameters by layer."""
    all_params = data.files

    # Group by component first
    components = {}
    for param_name in all_params:
        comp_name = param_name.split('/')[0]  # correctness/format/length
        if comp_name not in components:
            components[comp_name] = []
        components[comp_name].append(param_name)

    print(f"\nFound components: {list(components.keys())}")

    # Group parameters by layer
    # Assume format: component/model.layers.X.submodule.param_name
    layer_params = {}

    for comp_name, params in components.items():
        for param_full in params:
            # Remove component prefix
            param_name = param_full.split('/', 1)[1]

            # Determine layer
            if 'layers.' in param_name:
                # Extract layer number: model.layers.X.xxx
                parts = param_name.split('.')
                for i, part in enumerate(parts):
                    if part == 'layers' and i + 1 < len(parts):
                        layer_num = int(parts[i + 1])
                        layer_key = f"layer_{layer_num}"
                        break
            elif 'embed_tokens' in param_name:
                layer_key = "embedding"
            elif 'lm_head' in param_name:
                layer_key = "lm_head"
            elif 'norm' in param_name:
                layer_key = "final_norm"
            else:
                layer_key = "other"

            # Store parameter
            if layer_key not in layer_params:
                layer_params[layer_key] = {}
            if param_name not in layer_params[layer_key]:
                layer_params[layer_key][param_name] = {}
            layer_params[layer_key][param_name][comp_name] = data[param_full]

    return layer_params

def compute_cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def plot_layer_similarities(layer_names, similarities, step_number, output_plot=None):
    """Plot layer-wise similarities."""
    plt.figure(figsize=(14, 6))

    x_indices = range(len(layer_names))

    # Plot three lines
    plt.plot(x_indices, similarities['correctness_format'],
             marker='o', label='Correctness vs Format', linewidth=2, markersize=5)
    plt.plot(x_indices, similarities['correctness_length'],
             marker='s', label='Correctness vs Length', linewidth=2, markersize=5)
    plt.plot(x_indices, similarities['format_length'],
             marker='^', label='Format vs Length', linewidth=2, markersize=5)

    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Cosine Similarity', fontsize=12)
    plt.title(f'Layer-wise Gradient Similarity Analysis (Step {step_number})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(x_indices, layer_names, rotation=45, ha='right', fontsize=9)
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()

    # Create plots directory if it doesn't exist
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)

    if output_plot:
        save_path = os.path.join(plots_dir, output_plot)
    else:
        save_path = os.path.join(plots_dir, f'layer_similarity_step_{step_number}.png')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")

    plt.close()

def analyze_layer_gradients(npz_file, output_file=None, output_plot=None):
    """Analyze gradients layer by layer."""
    data = load_gradients(npz_file)

    # Group parameters by layer
    layer_params = group_params_by_layer(data)

    # Sort layers (embedding, layer_0, layer_1, ..., final_norm, lm_head)
    layer_keys = sorted(layer_params.keys(), key=lambda x: (
        0 if x == "embedding" else
        1 if x.startswith("layer_") else
        2 if x == "final_norm" else
        3 if x == "lm_head" else
        4
    ))

    print(f"\nFound {len(layer_keys)} layer groups")

    # Prepare output
    results = []
    results.append("=" * 100)
    results.append("LAYER-WISE GRADIENT SIMILARITY ANALYSIS")
    results.append("=" * 100)

    # Store similarities for plotting
    layer_names = []
    similarities = {
        'correctness_format': [],
        'correctness_length': [],
        'format_length': []
    }

    # Analyze each layer
    for layer_key in layer_keys:
        params = layer_params[layer_key]

        results.append(f"\n{'='*100}")
        results.append(f"LAYER: {layer_key}")
        results.append(f"{'='*100}")
        results.append(f"Number of parameters: {len(params)}")

        # Concatenate all gradients for this layer across all parameters
        layer_grads = {'correctness': [], 'format': [], 'length': []}

        for param_name, comp_grads in params.items():
            if len(comp_grads) == 3:  # All three components present
                for comp in ['correctness', 'format', 'length']:
                    if comp in comp_grads:
                        layer_grads[comp].append(comp_grads[comp].flatten())

        # Concatenate all parameter gradients for this layer
        for comp in layer_grads:
            if len(layer_grads[comp]) > 0:
                layer_grads[comp] = np.concatenate(layer_grads[comp])

        # Check if all components are present
        if all(len(layer_grads[comp]) > 0 for comp in ['correctness', 'format', 'length']):
            results.append(f"\nLayer gradient sizes:")
            for comp in ['correctness', 'format', 'length']:
                norm = np.linalg.norm(layer_grads[comp])
                results.append(f"  {comp}: {len(layer_grads[comp])} elements, norm={norm:.6f}")

            # Compute cosine similarities
            sim_cf = compute_cosine_similarity(layer_grads['correctness'], layer_grads['format'])
            sim_cl = compute_cosine_similarity(layer_grads['correctness'], layer_grads['length'])
            sim_fl = compute_cosine_similarity(layer_grads['format'], layer_grads['length'])

            results.append(f"\nCosine Similarities:")
            results.append(f"  Correctness vs Format:  {sim_cf:.6f}")
            results.append(f"  Correctness vs Length:  {sim_cl:.6f}")
            results.append(f"  Format vs Length:       {sim_fl:.6f}")

            # Store for plotting
            layer_names.append(layer_key)
            similarities['correctness_format'].append(sim_cf)
            similarities['correctness_length'].append(sim_cl)
            similarities['format_length'].append(sim_fl)

            # Statistical summary
            results.append(f"\nGradient Statistics:")
            for comp in ['correctness', 'format', 'length']:
                grad = layer_grads[comp]
                results.append(f"  {comp}:")
                results.append(f"    mean={np.mean(grad):.6e}, std={np.std(grad):.6e}")
                results.append(f"    min={np.min(grad):.6e}, max={np.max(grad):.6e}")
        else:
            results.append(f"\nWarning: Not all components present for this layer")

    # Print results
    for line in results:
        print(line)

    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(results))
        print(f"\n\nResults saved to: {output_file}")

    # Generate plot
    if len(layer_names) > 0:
        step_number = extract_step_number(npz_file)
        plot_layer_similarities(layer_names, similarities, step_number, output_plot)
    else:
        print("\nWarning: No valid layers found for plotting")

def main():
    if len(sys.argv) < 2:
        print("Usage: python grad_analysis.py <gradients_step_X.npz> [output_text.txt] [output_plot.png]")
        print("\nExample:")
        print("  python grad_analysis.py gradient_analysis/gradients_step_0.npz")
        print("  python grad_analysis.py gradient_analysis/gradients_step_0.npz results.txt")
        print("  python grad_analysis.py gradient_analysis/gradients_step_0.npz results.txt similarity_plot.png")
        sys.exit(1)

    npz_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    output_plot = sys.argv[3] if len(sys.argv) > 3 else None

    if not os.path.exists(npz_file):
        print(f"Error: File not found: {npz_file}")
        sys.exit(1)

    analyze_layer_gradients(npz_file, output_file, output_plot)

if __name__ == "__main__":
    main()
