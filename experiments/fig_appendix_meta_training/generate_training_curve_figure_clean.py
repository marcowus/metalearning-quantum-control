"""
Generate Training Curve Figure  
© 2025 The MITRE Corporation, All Rights Reserved 
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import argparse

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    # Type 1 font settings for publication
    'pdf.fonttype': 42,  # TrueType fonts (Type 42) - accepted by journals
    'ps.fonttype': 42,   # TrueType fonts in PostScript
    'text.usetex': True,  # Use LaTeX for proper Type 1 fonts
    'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',
})


def smooth_gaussian(x, sigma=5):
    """Apply Gaussian smoothing to a 1D array."""
    return gaussian_filter1d(x, sigma=sigma, mode='nearest')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', type=str,
                        default='../../checkpoints_gamma/training_history.json')
    parser.add_argument('--sigma', type=float, default=5,
                        help='Gaussian smoothing sigma')
    parser.add_argument('--output', type=str, default='training_curve_gamma_clean',
                        help='Output filename (without extension)')
    parser.add_argument('--max_iter', type=int, default=None,
                        help='Maximum iteration to plot (default: all)')
    args = parser.parse_args()

    # Load training history
    history_path = Path(__file__).parent / args.history
    if not history_path.exists():
        history_path = Path(args.history)
    if not history_path.exists():
        print(f"ERROR: Training history not found at {history_path}")
        print("Run training first to generate training_history.json")
        return

    print(f"Loading training history from: {history_path}")
    with open(history_path, 'r') as f:
        history = json.load(f)

    # Extract data
    iterations = np.array(history['iterations'])
    meta_loss = np.array(history['meta_loss'])
    grad_norms = np.array(history['grad_norms'])
    val_iterations = np.array(history.get('val_iteration', []))
    val_pre_adapt = np.array(history.get('val_pre_adapt', []))
    val_post_adapt = np.array(history.get('val_post_adapt', []))
    val_fidelity = np.array(history.get('val_fidelity', []))
    val_fidelity_std = np.array(history.get('val_fidelity_std', []))

    # Truncate to max_iter if specified
    if args.max_iter is not None:
        mask = iterations <= args.max_iter
        iterations = iterations[mask]
        meta_loss = meta_loss[mask]
        grad_norms = grad_norms[mask]
        if len(val_iterations) > 0:
            val_mask = val_iterations <= args.max_iter
            val_iterations = val_iterations[val_mask]
            val_pre_adapt = val_pre_adapt[val_mask]
            val_post_adapt = val_post_adapt[val_mask]
            val_fidelity = val_fidelity[val_mask]
            val_fidelity_std = val_fidelity_std[val_mask]

    n_iters = int(iterations[-1]) if len(iterations) > 0 else 0
    print(f"Training iterations: {len(iterations)} (max iter: {n_iters})")
    print(f"Validation points: {len(val_iterations)}")

    # Apply Gaussian smoothing
    smoothed_loss = smooth_gaussian(meta_loss, sigma=args.sigma)
    smoothed_grad = smooth_gaussian(grad_norms, sigma=args.sigma)

    # Colors
    color_train = '#3498db'      # Blue for training
    color_post = '#2ecc71'       # Green for post-adaptation
    color_pre = '#e74c3c'        # Red for pre-adaptation
    color_grad = '#9b59b6'       # Purple for gradients
    color_fid = '#27ae60'        # Green for fidelity

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    # ===== Panel (a): Training Loss =====
    ax1 = axes[0, 0]
    ax1.fill_between(iterations, meta_loss, alpha=0.15, color=color_train, linewidth=0)
    ax1.plot(iterations, smoothed_loss, color=color_train, linewidth=2,
             label='Training Loss')
    ax1.set_xlabel('Meta-Training Iteration')
    ax1.set_ylabel('Loss $(1 - \\mathcal{F})$')
    ax1.set_title('(a) Training Loss')
    ax1.set_yscale('log')
    ax1.set_xlim([0, n_iters])
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.legend(loc='upper right', framealpha=0.9)

    # ===== Panel (b): Validation Loss (Pre/Post) with Gap on Secondary Axis =====
    ax2 = axes[0, 1]
    if len(val_iterations) > 0:
        # Plot pre/post adaptation loss on primary axis
        ax2.plot(val_iterations, val_pre_adapt, 'o-', color=color_pre,
                 markersize=4, linewidth=1.5, label='Pre-adaptation', alpha=0.8)
        ax2.plot(val_iterations, val_post_adapt, 's-', color=color_post,
                 markersize=4, linewidth=1.5, label='Post-adaptation', alpha=0.8)

        ax2.set_xlabel('Meta-Training Iteration')
        ax2.set_ylabel('Validation Loss', color='black')
        ax2.set_title('(b) Validation Loss \\& Adaptation Gap')
        ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax2.set_xlim([0, n_iters])

        # Secondary y-axis for adaptation gap
        ax2_gap = ax2.twinx()
        adaptation_gap = val_pre_adapt - val_post_adapt
        ax2_gap.fill_between(val_iterations, 0, adaptation_gap,
                             alpha=0.35, color='#f39c12', linewidth=0)
        ax2_gap.plot(val_iterations, adaptation_gap, '-', color='#d35400',
                     linewidth=2, label='Adaptation Gap')
        ax2_gap.set_ylabel('Adaptation Gap', color='#d35400')
        ax2_gap.tick_params(axis='y', labelcolor='#d35400')
        ax2_gap.set_ylim(bottom=0)

        # Combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_gap.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.9)
    else:
        ax2.text(0.5, 0.5, 'No validation data', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=12)
        ax2.set_title('(b) Validation Loss')

    # ===== Panel (c): Gradient Norm =====
    ax3 = axes[1, 0]
    ax3.fill_between(iterations, grad_norms, alpha=0.15, color=color_grad, linewidth=0)
    ax3.plot(iterations, smoothed_grad, color=color_grad, linewidth=2,
             label='Gradient Norm')
    ax3.set_xlabel('Meta-Training Iteration')
    ax3.set_ylabel('$\\|\\nabla \\mathcal{L}\\|$')
    ax3.set_title('(c) Gradient Norm')
    ax3.set_xlim([0, n_iters])
    ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax3.legend(loc='upper right', framealpha=0.9)

    # ===== Panel (d): Validation Fidelity (no final fidelity label) =====
    ax4 = axes[1, 1]
    if len(val_iterations) > 0:
        ax4.errorbar(val_iterations, val_fidelity, yerr=val_fidelity_std,
                     fmt='o-', color=color_fid, markersize=8, linewidth=2,
                     capsize=4, capthick=2, label='Val Fidelity')
        ax4.set_xlabel('Meta-Training Iteration')
        ax4.set_ylabel('Fidelity $\\mathcal{F}$')
        ax4.set_title('(d) Validation Fidelity')
        ax4.set_xlim([0, n_iters])
        ax4.set_ylim([0, 1.05])
        ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax4.legend(loc='lower right', framealpha=0.9)
    else:
        ax4.text(0.5, 0.5, 'No validation data', ha='center', va='center',
                 transform=ax4.transAxes, fontsize=12)
        ax4.set_title('(d) Validation Fidelity')

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent
    save_path = str(output_dir / f"{args.output}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"  Total iterations: {n_iters}")
    print(f"  Final training loss: {meta_loss[-1]:.4f}")
    print(f"  Final training fidelity: {1 - meta_loss[-1]:.4f}")
    if len(val_fidelity) > 0:
        print(f"  Final validation fidelity: {val_fidelity[-1]:.4f} ± {val_fidelity_std[-1]:.4f}")
        print(f"  Final adaptation gap: {val_pre_adapt[-1] - val_post_adapt[-1]:.4f}")
    print(f"  Final gradient norm: {grad_norms[-1]:.4f}")


if __name__ == '__main__':
    main()
