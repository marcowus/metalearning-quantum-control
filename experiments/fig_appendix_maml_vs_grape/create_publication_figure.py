"""
© 2025 The MITRE Corporation, All Rights Reserved 
"""
#!/usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
rcParams['mathtext.fontset'] = 'dejavuserif'
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 11
rcParams['legend.fontsize'] = 9
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['axes.linewidth'] = 1.0
rcParams['xtick.major.width'] = 1.0
rcParams['ytick.major.width'] = 1.0
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.top'] = True
rcParams['ytick.right'] = True


def create_figure(data_path: str, output_path: str):
    """Create publication-ready two-panel figure with all methods."""

    with open(data_path, 'r') as f:
        data = json.load(f)

    # Extract data
    means = data['means']
    stds = data['stds']
    grape_trajectory = np.array(data['avg_grape_trajectory'])

    # Find GRAPE iterations to reach 97% threshold
    idx_97 = np.where(grape_trajectory >= 97)[0]
    grape_steps_97 = idx_97[0] if len(idx_97) > 0 else 150

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # === Panel (a): Fidelity comparison bar chart ===
    ax = axes[0]

    method_labels = ['Robust\nGRAPE', 'FOMAML\n($K$=0)', 'FOMAML\n($K$=5)',
                     'FOMAML\n($K$=10)', 'FOMAML\n($K$=20)', 'Per-task\nGRAPE']

    # Colors: GRAPE variants red/orange, MAML blue
    colors = ['#D62728', '#1F77B4', '#1F77B4', '#1F77B4', '#1F77B4', '#FF7F0E']

    x = np.arange(len(method_labels))
    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors,
                  alpha=0.85, edgecolor='black', linewidth=0.8)


    bars[2].set_hatch('...')
    bars[3].set_hatch('///')
    bars[4].set_hatch('xxx')

    ax.set_ylabel('Gate Fidelity (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=8)
    ax.set_ylim(96, 100)
    ax.axhline(y=99, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='y', linewidth=0.5)
    ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.15,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax = axes[1]

    labels = ['FOMAML ($K$=10)', 'GRAPE (from scratch)']
    steps = [10, grape_steps_97]
    bar_colors = ['#1F77B4', '#D62728']

    x2 = np.arange(len(labels))
    bars2 = ax.bar(x2, steps, color=bar_colors, alpha=0.85,
                   edgecolor='black', linewidth=0.8, width=0.45)

    ax.set_ylabel('Optimization Steps')
    ax.set_xticks(x2)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, max(steps) * 1.35)
    ax.grid(True, alpha=0.3, axis='y', linewidth=0.5)
    ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    ax.text(bars2[0].get_x() + bars2[0].get_width()/2, steps[0]/2,
            f'{steps[0]}', ha='center', va='center', fontsize=16,
            fontweight='bold', color='white')
    ax.text(bars2[1].get_x() + bars2[1].get_width()/2, steps[1]/2,
            f'{steps[1]}', ha='center', va='center', fontsize=16,
            fontweight='bold', color='white')

    speedup = steps[1] / steps[0]
    ax.text(0.05, 0.92, f'{speedup:.1f}× faster',
            transform=ax.transAxes, ha='left', va='top', fontsize=14,
            fontweight='bold', color='#1F77B4',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F4FD',
                     edgecolor='#1F77B4', linewidth=2))

    # Subtitle
    ax.set_title('Steps to reach ~97% fidelity', fontsize=10, style='italic', pad=8)

    plt.tight_layout()

    # Save
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def create_figure_v2(data_path: str, output_path: str):
    """Alternative version with clearer computational cost comparison."""

    with open(data_path, 'r') as f:
        data = json.load(f)

    means = data['means']
    stds = data['stds']
    grape_trajectory = np.array(data['avg_grape_trajectory'])

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
 
    ax = axes[0]

    # Simplified: just compare key methods
    method_labels = ['Robust\nGRAPE', 'FOMAML\n($K$=0)', 'FOMAML\n($K$=10)', 'Per-task\nGRAPE']
    selected_idx = [0, 1, 3, 5]  # Robust, K=0, K=10, Per-task
    selected_means = [means[i] for i in selected_idx]
    selected_stds = [stds[i] for i in selected_idx]
    colors = ['#D62728', '#1F77B4', '#1F77B4', '#FF7F0E']

    x = np.arange(len(method_labels))
    bars = ax.bar(x, selected_means, yerr=selected_stds, capsize=3,
                  color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)

    # Add hatching to K=10 to distinguish from K=0
    bars[2].set_hatch('///')

    ax.set_ylabel('Gate Fidelity (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=9)
    ax.set_ylim(96, 100)
    ax.grid(True, alpha=0.3, axis='y', linewidth=0.5)
    ax.text(-0.18, 1.05, '(a)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    # Value labels - positioned clearly above error bars
    for bar, mean, std in zip(bars, selected_means, selected_stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.12,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax = axes[1]

    idx_97 = np.where(grape_trajectory >= 97)[0]
    grape_steps_97 = idx_97[0] if len(idx_97) > 0 else 150

    # Bar chart comparing optimization steps
    labels = ['FOMAML\n($K$=10)', 'GRAPE\n(from scratch)']
    steps = [10, grape_steps_97]
    colors = ['#1F77B4', '#D62728']

    x2 = np.arange(len(labels))
    bars2 = ax.bar(x2, steps, color=colors, alpha=0.85,
                   edgecolor='black', linewidth=0.8, width=0.5)

    ax.set_ylabel('Optimization Steps')
    ax.set_xticks(x2)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, max(steps) * 1.4)
    ax.grid(True, alpha=0.3, axis='y', linewidth=0.5)
    ax.text(-0.18, 1.05, '(b)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    # Clear value labels inside bars
    ax.text(bars2[0].get_x() + bars2[0].get_width()/2, steps[0]/2,
            f'{steps[0]}', ha='center', va='center', fontsize=14,
            fontweight='bold', color='white')
    ax.text(bars2[1].get_x() + bars2[1].get_width()/2, steps[1]/2,
            f'{steps[1]}', ha='center', va='center', fontsize=14,
            fontweight='bold', color='white')

    speedup = steps[1] / steps[0]
    ax.text(0.05, 0.92, f'{speedup:.1f}× faster\nwith FOMAML',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=12, fontweight='bold', color='#1F77B4',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F4FD',
                     edgecolor='#1F77B4', linewidth=1.5))

    ax.set_title('Steps to reach ~97% fidelity', fontsize=10, style='italic', pad=10)

    plt.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    output_dir = Path(__file__).parent
    data_path = str(output_dir / 'maml_vs_grape_data.json')

    # Create both versions
    create_figure(data_path, str(output_dir / 'fig_maml_vs_grape_v1.png'))
    create_figure_v2(data_path, str(output_dir / 'fig_maml_vs_grape_v2.png'))

    print("\nDone! Created two versions for comparison.")
