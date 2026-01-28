"""
Create publication-ready horizontal figure with pulse sequences and adaptation gap.
Layout: 1x3 horizontal with (a) K=0 pulses, (b) K=10 pulses, (c) adaptation gap
© 2025 The MITRE Corporation, All Rights Reserved 
"""

import sys
from pathlib import Path
import json

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
from copy import deepcopy
import argparse

from two_qubit_cz_maml_fast import (
    TwoQubitTaskParams,
    TwoQubitTaskDistribution,
    TwoQubitCZPolicy,
    compute_loss,
    CZ_IDEAL_GATE_TIME,
)

# Publication-quality settings
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
rcParams['font.size'] = 9
rcParams['axes.labelsize'] = 10
rcParams['axes.titlesize'] = 10
rcParams['legend.fontsize'] = 7
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['axes.linewidth'] = 0.8
rcParams['xtick.major.width'] = 0.8
rcParams['ytick.major.width'] = 0.8
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.top'] = True
rcParams['ytick.right'] = True
# Type 1 font settings for publication
rcParams['pdf.fonttype'] = 42  # TrueType fonts (Type 42) - accepted by journals
rcParams['ps.fonttype'] = 42   # TrueType fonts in PostScript
rcParams['text.usetex'] = True  # Use LaTeX for proper Type 1 fonts
rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'


def load_policy(checkpoint_path: str, device: str = 'cpu') -> TwoQubitCZPolicy:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden_dim = checkpoint.get('hidden_dim', 256)
    n_hidden_layers = checkpoint.get('n_hidden_layers', 4)
    n_segments = checkpoint.get('n_segments', 30)
    n_controls = checkpoint.get('n_controls', 6)
    task_feature_dim = checkpoint.get('task_feature_dim', 4)

    policy = TwoQubitCZPolicy(
        task_feature_dim=task_feature_dim,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
        n_segments=n_segments,
        n_controls=n_controls,
    ).to(device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    return policy


def get_pulse_sequence(policy, task, device='cpu'):
    task_features = torch.tensor(task.to_array(normalized=True), dtype=torch.float32, device=device)
    with torch.no_grad():
        controls = policy(task_features)
    return controls.cpu().numpy()


def adapt_policy(policy, task, K, inner_lr=2e-4, device='cpu'):
    adapted_policy = deepcopy(policy)
    adapted_policy.train()
    optimizer = optim.Adam(adapted_policy.parameters(), lr=inner_lr)
    for k in range(K):
        optimizer.zero_grad()
        loss = compute_loss(adapted_policy, task, device=device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapted_policy.parameters(), max_norm=1.0)
        optimizer.step()
    adapted_policy.eval()
    return adapted_policy


def smooth_pulse(time, pulse, n_smooth=200):
    time_smooth = np.linspace(time[0], time[-1], n_smooth)
    try:
        spline = make_interp_spline(time, pulse, k=3)
        pulse_smooth = spline(time_smooth)
    except:
        pulse_smooth = np.interp(time_smooth, time, pulse)
    pulse_smooth = gaussian_filter1d(pulse_smooth, sigma=2)
    return time_smooth, pulse_smooth


def exponential_saturation(K, c, beta):
    return c * (1 - np.exp(-beta * K))


def create_horizontal_figure(pulses_k0, pulses_k10, fid_k0, fid_k10, n_segments,
                              data_path, K_adapt=10, max_K=15, output_path=None):
    """Create horizontal 1x3 figure with panels (a)-(c)."""

    # Load adaptation data
    with open(data_path, 'r') as f:
        data = json.load(f)

    K_values = np.array(data['K_values'][:max_K+1])
    mean_gaps = np.array(data['mean_gaps'][:max_K+1])
    std_gaps = np.array(data['std_gaps'][:max_K+1])

    # Fit exponential
    try:
        popt, _ = curve_fit(exponential_saturation, K_values, mean_gaps,
                           p0=[0.4, 0.3], bounds=([0, 0.01], [2, 2]), maxfev=5000)
        c, beta = popt
    except:
        c, beta = None, None

    # Control labels and colors
    control_labels = ['$u_{X,1}$', '$u_{Y,1}$', '$u_{X,2}$', '$u_{Y,2}$', '$u_{Z,1}$', '$u_{Z,2}$']
    colors = ['#D62728', '#1F77B4', '#2CA02C', '#9467BD', '#FF7F0E', '#17BECF']

    T = CZ_IDEAL_GATE_TIME
    time = np.linspace(0, T, n_segments)

    # Create figure with 1x3 horizontal layout
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    # Panel (a): K=0 pulses
    ax_a = axes[0]
    for i, (label, color) in enumerate(zip(control_labels, colors)):
        time_smooth, pulse_smooth = smooth_pulse(time, pulses_k0[:, i])
        ax_a.plot(time_smooth, pulse_smooth, label=label, color=color, linewidth=1.2)
    ax_a.set_xlabel('Time (a.u.)')
    ax_a.set_ylabel('Control Amplitude (a.u.)')
    ax_a.axhline(y=0, color='#7F8C8D', linestyle='-', alpha=0.3, linewidth=0.5)
    ax_a.legend(loc='upper right', ncol=2, framealpha=0.95, edgecolor='none',
                columnspacing=0.5, handlelength=1.2, fontsize=6)
    ax_a.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax_a.text(-0.15, 1.05, '(a)', transform=ax_a.transAxes, fontsize=11, fontweight='bold')
    ax_a.text(0.97, 0.05, f'$K=0$\n$\\mathcal{{F}}={fid_k0*100:.1f}\\%$',
              transform=ax_a.transAxes, ha='right', va='bottom', fontsize=8,
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray', pad=0.2))

    # Panel (b): K=K_adapt pulses
    ax_b = axes[1]
    for i, (label, color) in enumerate(zip(control_labels, colors)):
        time_smooth, pulse_smooth = smooth_pulse(time, pulses_k10[:, i])
        ax_b.plot(time_smooth, pulse_smooth, label=label, color=color, linewidth=1.2)
    ax_b.set_xlabel('Time (a.u.)')
    ax_b.set_ylabel('Control Amplitude (a.u.)')
    ax_b.axhline(y=0, color='#7F8C8D', linestyle='-', alpha=0.3, linewidth=0.5)
    ax_b.legend(loc='upper right', ncol=2, framealpha=0.95, edgecolor='none',
                columnspacing=0.5, handlelength=1.2, fontsize=6)
    ax_b.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax_b.text(-0.15, 1.05, '(b)', transform=ax_b.transAxes, fontsize=11, fontweight='bold')
    ax_b.text(0.97, 0.05, f'$K={K_adapt}$\n$\\mathcal{{F}}={fid_k10*100:.1f}\\%$',
              transform=ax_b.transAxes, ha='right', va='bottom', fontsize=8,
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray', pad=0.2))

    # Panel (c): Adaptation Gap
    ax_c = axes[2]
    ax_c.errorbar(K_values, mean_gaps, yerr=std_gaps,
                  fmt='o', color='#2C3E50', markersize=4, capsize=2,
                  capthick=0.8, elinewidth=0.8, markerfacecolor='#3498DB',
                  markeredgecolor='#2C3E50', markeredgewidth=0.8,
                  label='FOMAML', zorder=3)
    if c is not None:
        K_fine = np.linspace(0, max_K, 100)
        fitted_fine = exponential_saturation(K_fine, c, beta)
        ax_c.plot(K_fine, fitted_fine, '-', color='#E74C3C', linewidth=1.2,
                  label='Fit', zorder=2)
    ax_c.set_xlabel('Adaptation Steps $K$')
    ax_c.set_ylabel('Adaptation Gap $G_K$')
    ax_c.set_xlim(-0.5, max_K + 0.5)
    ax_c.set_ylim(bottom=0)
    ax_c.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax_c.legend(loc='lower right', framealpha=0.95, edgecolor='none')
    ax_c.text(-0.15, 1.05, '(c)', transform=ax_c.transAxes, fontsize=11, fontweight='bold')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints_cz_1000iter_fast/maml_cz_best.pt')
    parser.add_argument('--data', type=str, default='cz_adaptation_gap_fast_data.json')
    parser.add_argument('--inner_lr', type=float, default=2e-4)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--max_K', type=int, default=15)
    parser.add_argument('--output', type=str, default='fig_cz_horizontal')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = 'cpu'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(__file__).parent

    print("Creating horizontal publication figure...")

    # Load policy
    checkpoint_path = output_dir / args.checkpoint
    if not checkpoint_path.exists():
        checkpoint_path = Path(args.checkpoint)
    policy = load_policy(str(checkpoint_path), device)
    n_segments = policy.n_segments
    print(f"  Loaded policy (n_segments={n_segments})")

    # Sample task
    task_dist = TwoQubitTaskDistribution(
        gamma_deph_range=(0.001, 0.01),
        gamma_relax_range=(0.0005, 0.005),
    )
    task = task_dist.sample(1)[0]

    # Get pulses
    print("  Computing K=0 and K=10 pulses...")
    pulses_k0 = get_pulse_sequence(policy, task, device)
    with torch.no_grad():
        fid_k0 = 1 - compute_loss(policy, task, device=device).item()

    adapted_policy = adapt_policy(policy, task, K=args.K, inner_lr=args.inner_lr, device=device)
    pulses_k10 = get_pulse_sequence(adapted_policy, task, device)
    with torch.no_grad():
        fid_k10 = 1 - compute_loss(adapted_policy, task, device=device).item()

    print(f"    K=0: {fid_k0*100:.1f}%, K={args.K}: {fid_k10*100:.1f}%")

    # Create horizontal figure
    data_path = str(output_dir / args.data)
    output_path = str(output_dir / f"{args.output}.png")

    create_horizontal_figure(pulses_k0, pulses_k10, fid_k0, fid_k10, n_segments,
                              data_path, K_adapt=args.K, max_K=args.max_K,
                              output_path=output_path)

if __name__ == '__main__':
    main()
