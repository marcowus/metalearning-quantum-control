#!/usr/bin/env python3
"""
Plot training curves for CZ gate MAML training.

Usage:
    python plot_cz_training_curves.py --checkpoint checkpoints_cz_1000iter_fast/maml_cz_best.pt
 © 2025 The MITRE Corporation, All Rights Reserved  
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
from pathlib import Path

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})


def load_training_history(checkpoint_path: str):
    """Load training history from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print(f"Checkpoint keys: {list(ckpt.keys())}")
    print(f"Best iteration: {ckpt.get('iteration', 'unknown')}")

    history = ckpt.get('history', {})
    print(f"History keys: {list(history.keys())}")

    return history, ckpt


def plot_training_curves(history, output_prefix: str = 'cz_training_curves'):
    """Create training curve plots."""

    iters = np.array(history['iterations'])
    meta_loss = np.array(history['meta_loss'])
    val_pre = np.array(history.get('val_pre_adapt', []))
    val_post = np.array(history.get('val_post_adapt', []))

    print(f"\nData shapes:")
    print(f"  iterations: {len(iters)}")
    print(f"  meta_loss: {len(meta_loss)}")
    print(f"  val_pre_adapt: {len(val_pre)}")
    print(f"  val_post_adapt: {len(val_post)}")

    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: Meta Loss
    ax = axes[0]
    ax.plot(iters, meta_loss, 'b-', linewidth=1, alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Meta Loss')
    ax.set_title('(a) Training Meta Loss')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Panel 2: Validation Fidelity
    ax = axes[1]
    # val_pre and val_post are logged every val_interval iterations
    # They have fewer entries than meta_loss
    if len(val_pre) > 0 and len(val_post) > 0:
        # Assume validation every 20 iterations
        val_interval = max(1, len(iters) // len(val_pre)) if len(val_pre) > 0 else 20
        val_iters = iters[::val_interval][:len(val_pre)]

        ax.plot(val_iters, val_pre * 100, 'o-', color='#3498db',
                markersize=4, label='Pre-adaptation', linewidth=1.5)
        ax.plot(val_iters, val_post * 100, 's-', color='#2ecc71',
                markersize=4, label='Post-adaptation', linewidth=1.5)
        ax.legend(loc='lower right')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fidelity (%)')
    ax.set_title('(b) Validation Fidelity')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(40, 100)

    # Panel 3: Adaptation Gap
    ax = axes[2]
    if len(val_pre) > 0 and len(val_post) > 0:
        gap = (val_post - val_pre) * 100
        ax.plot(val_iters, gap, 'd-', color='#e74c3c', markersize=4, linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Adaptation Gap (%)')
    ax.set_title('(c) Adaptation Gap (Post - Pre)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Save figures
    png_path = f'{output_prefix}.png'
    pdf_path = f'{output_prefix}.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f'\nFigures saved to:')
    print(f'  {png_path}')
    print(f'  {pdf_path}')

    plt.close()

    return png_path, pdf_path


def print_summary(history, ckpt):
    """Print training summary statistics."""
    iters = history['iterations']
    meta_loss = history['meta_loss']
    val_pre = history.get('val_pre_adapt', [])
    val_post = history.get('val_post_adapt', [])

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Total iterations: {iters[-1] if iters else 0}")
    print(f"Best iteration: {ckpt.get('iteration', 'N/A')}")
    print(f"Best fidelity: {ckpt.get('fidelity', 'N/A')}")

    if meta_loss:
        print(f"\nMeta Loss:")
        print(f"  Initial: {meta_loss[0]:.4f}")
        print(f"  Final: {meta_loss[-1]:.4f}")
        print(f"  Min: {min(meta_loss):.4f}")

    if val_pre and val_post:
        print(f"\nValidation Fidelity:")
        print(f"  Pre-adaptation (final): {val_pre[-1]*100:.1f}%")
        print(f"  Post-adaptation (final): {val_post[-1]*100:.1f}%")
        print(f"  Adaptation Gap (final): {(val_post[-1] - val_pre[-1])*100:.1f}%")

    # Model info
    print(f"\nModel Architecture:")
    print(f"  hidden_dim: {ckpt.get('hidden_dim', 'N/A')}")
    print(f"  n_hidden_layers: {ckpt.get('n_hidden_layers', 'N/A')}")
    print(f"  n_segments: {ckpt.get('n_segments', 'N/A')}")
    print(f"  n_controls: {ckpt.get('n_controls', 'N/A')}")
    print(f"  gate_time: {ckpt.get('gate_time', 'N/A')}")


def save_history_json(history, ckpt, output_path: str):
    """Save training history to JSON."""
    data = {
        'iterations': list(history['iterations']),
        'meta_loss': list(history['meta_loss']),
        'val_pre_adapt': list(history.get('val_pre_adapt', [])),
        'val_post_adapt': list(history.get('val_post_adapt', [])),
        'best_iteration': ckpt.get('iteration'),
        'best_fidelity': ckpt.get('fidelity'),
        'model_config': {
            'hidden_dim': ckpt.get('hidden_dim'),
            'n_hidden_layers': ckpt.get('n_hidden_layers'),
            'n_segments': ckpt.get('n_segments'),
            'n_controls': ckpt.get('n_controls'),
            'gate_time': ckpt.get('gate_time'),
        }
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nHistory saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot CZ MAML training curves')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints_cz_1000iter_fast/maml_cz_best.pt',
                        help='Path to checkpoint file')
    parser.add_argument('--output', type=str, default='cz_training_curves_fast',
                        help='Output filename prefix')
    args = parser.parse_args()

    # Load data
    checkpoint_path = Path(__file__).parent / args.checkpoint
    if not checkpoint_path.exists():
        checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return

    history, ckpt = load_training_history(str(checkpoint_path))

    if not history:
        print("ERROR: No training history found in checkpoint")
        return

    # Plot curves
    output_dir = Path(__file__).parent
    output_prefix = str(output_dir / args.output)
    plot_training_curves(history, output_prefix)

    # Print summary
    print_summary(history, ckpt)

    # Save history to JSON
    save_history_json(history, ckpt, f"{output_prefix}_data.json")


if __name__ == '__main__':
    main()
