#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradient Analysis Over Time for a Specific FSDP Layer.

This script is adapted to handle flattened parameters from FSDP training,
allowing for exact matching of parameter names.
The plotting function is enhanced to draw smooth curves and hide the x-axis.
"""

import numpy as np
import sys
import os
import re
import glob
import matplotlib.pyplot as plt
import argparse
from scipy.interpolate import CubicSpline

# ============================================================================
# Helper Functions (Unchanged)
# ============================================================================
def extract_step_number(npz_file):
    """Extract step number from filename."""
    match = re.search(r'step[_\-]?(\d+)', os.path.basename(npz_file))
    if match: return int(match.group(1))
    return -1

def compute_cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(v1.flatten(), v2.flatten())
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0: return 0.0
    return dot_product / (norm1 * norm2)

# ============================================================================
# Core Logic (get_gradients_for_specific_param is unchanged)
# ============================================================================
def get_gradients_for_specific_param(data, exact_param_name):
    """
    From loaded npz data, extract gradients for an EXACT parameter name.
    """
    layer_grads = {'correctness': None, 'format': None, 'length': None}
    found = False
    for comp_name in layer_grads.keys():
        full_key = f"{comp_name}/{exact_param_name}"
        if full_key in data:
            layer_grads[comp_name] = data[full_key]
            found = True
    if not found:
        print(f"  - Warning: Parameter '{exact_param_name}' not found.")
        return None
    if any(grad is None for grad in layer_grads.values()):
        missing = [k for k, v in layer_grads.items() if v is None]
        print(f"  - Warning: Missing components {missing} for param '{exact_param_name}'.")
        return None
    return layer_grads

def plot_similarity_over_time(steps, similarities, param_name, output_plot_path):
    """Plots the evolution of gradient similarities with smooth curves."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 7))
    ax = plt.gca()

    plot_data = {
        'Correctness vs Format': ('o', similarities['correctness_format']),
        'Correctness vs Length': ('s', similarities['correctness_length']),
        'Format vs Length': ('^', similarities['format_length'])
    }
    
    if len(steps) > 2:
        x_smooth = np.linspace(min(steps), max(steps), 300)
        for label, (marker, y_data) in plot_data.items():
            spline = CubicSpline(steps, y_data)
            y_smooth = spline(x_smooth)
            line, = ax.plot(x_smooth, y_smooth, label=label, linewidth=2.5)
            ax.scatter(steps, y_data, marker=marker, color=line.get_color(), s=50, zorder=10)
    else:
        ax.plot(steps, similarities['correctness_format'], marker='o', label='Correctness vs Format')
        ax.plot(steps, similarities['correctness_length'], marker='s', label='Correctness vs Length')
        ax.plot(steps, similarities['format_length'], marker='^', label='Format vs Length')

    ax.set_xlabel('Training Progression', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    short_title = param_name.replace("_fsdp_wrapped_module", "").replace("._flat_param", "")
    ax.set_title(f'Gradient Similarity Dynamics (Layer: {short_title})', fontsize=16, fontweight='bold')
    
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.set_ylim(-1.1, 1.1)
    plt.tight_layout()

    plots_dir = os.path.dirname(output_plot_path)
    if plots_dir: os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_plot_path}")
    plt.close()

def main():
    """Main function to orchestrate the analysis."""
    parser = argparse.ArgumentParser(description="Analyze gradient similarity of a specific FSDP parameter across training steps.")
    parser.add_argument("gradient_dir", type=str, help="Directory with 'gradients_step_*.npz' files.")
    parser.add_argument("--param_name", type=str, required=True, help="The exact name of the parameter to analyze (the part after 'component/').")
    parser.add_argument("--output_plot", type=str, default="plots/fsdp_similarity_over_time.png", help="Path for the output plot.")
    args = parser.parse_args()
    search_pattern = os.path.join(args.gradient_dir, 'gradients_step_*.npz')
    npz_files = sorted(glob.glob(search_pattern), key=extract_step_number)
    if not npz_files:
        print(f"Error: No gradient files found in '{args.gradient_dir}'.")
        sys.exit(1)
    print(f"Found {len(npz_files)} files. Analyzing exact parameter: '{args.param_name}'")
    
    # --- THIS IS THE CORRECTED SECTION ---
    steps, similarities = [], {
        'correctness_format': [], 
        'correctness_length': [],  # Corrected typo here
        'format_length': []
    }
    # --- END OF CORRECTION ---

    for npz_file in npz_files:
        step = extract_step_number(npz_file)
        if step == -1: continue
        print(f"\nProcessing Step {step}...")
        try:
            data = np.load(npz_file)
        except Exception as e:
            print(f"  - Error: Could not load file. It might be corrupted. Details: {e}")
            continue
        param_grads = get_gradients_for_specific_param(data, args.param_name)
        if param_grads:
            sim_cf = compute_cosine_similarity(param_grads['correctness'], param_grads['format'])
            sim_cl = compute_cosine_similarity(param_grads['correctness'], param_grads['length'])
            sim_fl = compute_cosine_similarity(param_grads['format'], param_grads['length'])
            print(f"  - Similarities | C-F: {sim_cf:.4f} | C-L: {sim_cl:.4f} | F-L: {sim_fl:.4f}")
            steps.append(step)
            similarities['correctness_format'].append(sim_cf)
            # --- CORRECTED LINE ---
            similarities['correctness_length'].append(sim_cl) # Corrected typo here
            similarities['format_length'].append(sim_fl)
            
    if steps:
        plot_similarity_over_time(steps, similarities, args.param_name, args.output_plot)
    else:
        print(f"\nNo valid data processed for parameter '{args.param_name}'. No plot generated.")

if __name__ == "__main__":
    main()