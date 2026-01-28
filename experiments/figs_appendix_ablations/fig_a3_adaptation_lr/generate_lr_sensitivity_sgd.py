"""
Inner Learning Rate Sensitivity Ablation - Using SGD Optimizer
Creates cleaner exponential saturation behavior compared to Adam.
© 2025 The MITRE Corporation, All Rights Reserved  
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from copy import deepcopy
import json
import argparse

from metaqctrl.meta_rl.policy_gamma import GammaPulsePolicy
from metaqctrl.quantum.lindblad_torch import DifferentiableLindbladSimulator

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'font.family': 'serif',
    # Type 1 font settings for publication
    'pdf.fonttype': 42,  # TrueType fonts (Type 42) - accepted by journals
    'ps.fonttype': 42,   # TrueType fonts in PostScript
    'text.usetex': True,  # Use LaTeX for proper Type 1 fonts
    'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',
})


def create_single_qubit_system(gamma_deph=0.05, gamma_relax=0.025, device='cpu'):
    """Create simulator with direct gamma rates."""
    sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
    sigma_p = torch.tensor([[0, 1], [0, 0]], dtype=torch.complex64, device=device)

    H0 = 0.0 * sigma_z
    H_controls = [sigma_x, sigma_y]

    L_operators = []
    if gamma_deph > 0:
        L_operators.append(np.sqrt(gamma_deph / 2.0) * sigma_z)
    if gamma_relax > 0:
        L_operators.append(np.sqrt(gamma_relax) * sigma_p)

    if not L_operators:
        L_operators.append(torch.zeros(2, 2, dtype=torch.complex64, device=device))

    sim = DifferentiableLindbladSimulator(
        H0=H0, H_controls=H_controls, L_operators=L_operators,
        dt=0.05, method='rk4', device=device
    )
    return sim


def compute_loss_gamma(policy, gamma_deph, gamma_relax, device='cpu'):
    """Compute loss using gamma-rate task features."""
    sim = create_single_qubit_system(gamma_deph, gamma_relax, device=device)

    task_features = torch.tensor([
        gamma_deph / 0.1,
        gamma_relax / 0.05,
        (gamma_deph + gamma_relax) / 0.15
    ], dtype=torch.float32, device=device)

    controls = policy(task_features)

    rho0 = torch.zeros(2, 2, dtype=torch.complex64, device=device)
    rho0[0, 0] = 1.0

    rho_final = sim.forward(rho0, controls, T=1.0)

    target = torch.zeros(2, 2, dtype=rho_final.dtype, device=device)
    target[1, 1] = 1.0

    fidelity = torch.abs(torch.trace(target @ rho_final)).real
    return 1.0 - fidelity


def load_pretrained_gamma_policy(checkpoint_path, device='cpu'):
    """Load pretrained gamma policy from checkpoint."""
    print(f"Loading pretrained policy from: {checkpoint_path}")

    policy = GammaPulsePolicy(
        task_feature_dim=3, hidden_dim=64, n_hidden_layers=2,
        n_segments=20, n_controls=2, output_scale=1.0
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'policy_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['policy_state_dict'])
    else:
        policy.load_state_dict(checkpoint)

    policy.eval()
    return policy


def sample_gamma_tasks(n_tasks, rng=None):
    """Sample gamma-rate tasks with uniform distribution matching training."""
    if rng is None:
        rng = np.random.default_rng(42)

    gamma_deph_range = (0.02, 0.15)
    gamma_relax_range = (0.01, 0.08)

    tasks = []
    for _ in range(n_tasks):
        gamma_deph = rng.uniform(gamma_deph_range[0], gamma_deph_range[1])
        gamma_relax = rng.uniform(gamma_relax_range[0], gamma_relax_range[1])
        tasks.append((gamma_deph, gamma_relax))

    return tasks


def exponential_saturation(K, c, beta):
    """Simplified exponential saturation: G_K = c * (1 - exp(-beta*K))"""
    return c * (1 - np.exp(-beta * K))


def compute_adaptation_curve(robust_policy, task_params_list, max_K=20, inner_lr=0.01, device='cpu'):
    """Compute adaptation gap curve using SGD optimizer."""
    n_tasks = len(task_params_list)
    all_gaps = np.zeros((n_tasks, max_K + 1))

    for task_idx, (gamma_deph, gamma_relax) in enumerate(task_params_list):
        adapted_policy = deepcopy(robust_policy)
        adapted_policy.train()
        inner_opt = optim.SGD(adapted_policy.parameters(), lr=inner_lr)

        with torch.no_grad():
            L_0 = compute_loss_gamma(adapted_policy, gamma_deph, gamma_relax, device).item()
            all_gaps[task_idx, 0] = 0.0

        for k in range(1, max_K + 1):
            inner_opt.zero_grad()
            loss = compute_loss_gamma(adapted_policy, gamma_deph, gamma_relax, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted_policy.parameters(), max_norm=1.0)
            inner_opt.step()

            with torch.no_grad():
                L_K = compute_loss_gamma(adapted_policy, gamma_deph, gamma_relax, device).item()
                all_gaps[task_idx, k] = L_0 - L_K

    K_values = np.arange(max_K + 1)
    mean_gaps = np.mean(all_gaps, axis=0)
    std_gaps = np.std(all_gaps, axis=0)

    return K_values, mean_gaps, std_gaps


def fit_exponential(K_values, mean_gaps):
    """Fit simplified exponential saturation model."""
    try:
        c_init = max(0.001, mean_gaps[-1] * 1.1)
        beta_init = 0.1

        popt, _ = curve_fit(
            exponential_saturation, K_values, mean_gaps,
            p0=[c_init, beta_init],
            bounds=([0, 0.001], [1, 10]),
            maxfev=5000
        )

        c, beta = popt
        fitted = exponential_saturation(K_values, c, beta)
        ss_res = np.sum((mean_gaps - fitted) ** 2)
        ss_tot = np.sum((mean_gaps - np.mean(mean_gaps)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return c, beta, r_squared
    except:
        return None, None, None


def main():
    parser = argparse.ArgumentParser(description='LR sensitivity with SGD optimizer')
    parser.add_argument('--n_tasks', type=int, default=50)
    parser.add_argument('--max_K', type=int, default=60)
    parser.add_argument('--checkpoint', type=str,
                        default='../../../checkpoints/checkpoints_gamma_v2/maml_gamma_pauli_x.pt')
    parser.add_argument('--output', type=str, default='lr_sensitivity_sgd')
    parser.add_argument('--plot_only', action='store_true',
                        help='Only plot from existing data, do not recompute')
    parser.add_argument('--data_file', type=str, default=None,
                        help='Path to existing data file for --plot_only mode')
    args = parser.parse_args()

    device = torch.device('cpu')
    print(f"Using device: {device}")

    output_dir = Path(__file__).parent

    # Plot-only mode: load existing data and replot
    if args.plot_only:
        data_file = args.data_file or str(output_dir / f"{args.output}_data.json")
        print(f"Loading existing data from: {data_file}")
        with open(data_file, 'r') as f:
            data = json.load(f)

        # Reconstruct results
        results = {}
        for lr_str, res in data['results'].items():
            lr = float(lr_str)
            results[lr] = {
                'K_values': np.array(res['K_values']),
                'mean_gaps': np.array(res['mean_gaps']),
                'c': res['c'],
                'beta': res['beta'],
                'r_squared': res['r_squared']
            }

        learning_rates = sorted(results.keys())
        max_K = data.get('max_K', args.max_K)
        colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(learning_rates)))

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

        for i, lr in enumerate(learning_rates):
            res = results[lr]
            axes[0].plot(res['K_values'], res['mean_gaps'], '-', color=colors[i],
                         linewidth=1.5, alpha=0.8)

        axes[0].set_xlabel('Adaptation Steps $K$')
        axes[0].set_ylabel('Adaptation Gap $G_K$')
        axes[0].set_title('(a) Adaptation Curves (SGD)')
        axes[0].set_xlim(-1, max_K + 1)
        axes[0].set_ylim(bottom=0)
        axes[0].grid(True, alpha=0.3)

        # Colorbar
        from matplotlib.colors import LogNorm
        sm = plt.cm.ScalarMappable(cmap='viridis',
                                    norm=LogNorm(vmin=min(learning_rates), vmax=max(learning_rates)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[0], pad=0.02)
        cbar.set_label('Learning Rate $\\eta$')

        # Plot beta vs LR (linear scale)
        valid_lrs = [lr for lr in learning_rates if results[lr]['beta'] is not None]
        valid_betas = [results[lr]['beta'] for lr in valid_lrs]

        ax2 = axes[1]
        ax2.plot(valid_lrs, valid_betas, 'o-', color='#E94F37', markersize=6, linewidth=2)
        ax2.set_xlabel('Learning Rate $\\eta$')
        ax2.set_ylabel('Adaptation Rate $\\beta$')
        ax2.set_title('(b) $\\beta$ vs Learning Rate')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = str(output_dir / f"{args.output}_publication.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
        plt.close()
        return

    # Load pretrained policy
    checkpoint_path = Path(__file__).parent / args.checkpoint
    if not checkpoint_path.exists():
        checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return

    robust_policy = load_pretrained_gamma_policy(str(checkpoint_path), device=device)

    # Learning rates to test - SGD needs higher LRs than Adam
    learning_rates = np.logspace(np.log10(0.001), np.log10(0.1), 12).tolist()
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(learning_rates)))

    rng = np.random.default_rng(42)
    task_params_list = sample_gamma_tasks(args.n_tasks, rng=rng)

    # Compute adaptation curves for each LR
    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    print("\n" + "=" * 60)
    print("Computing adaptation curves (SGD optimizer)")
    print("=" * 60)

    for i, lr in enumerate(learning_rates):
        print(f"  LR = {lr:.4f}...", end=" ")
        K_values, mean_gaps, std_gaps = compute_adaptation_curve(
            robust_policy, task_params_list, max_K=args.max_K,
            inner_lr=lr, device=device
        )

        c, beta, r_squared = fit_exponential(K_values, mean_gaps)

        results[lr] = {
            'K_values': K_values,
            'mean_gaps': mean_gaps,
            'std_gaps': std_gaps,
            'c': c,
            'beta': beta,
            'r_squared': r_squared
        }

        if beta is not None:
            print(f"beta={beta:.3f}, c={c:.4f}, R²={r_squared:.4f}")
        else:
            print("fit failed")

        # Plot curve
        axes[0].plot(K_values, mean_gaps, '-', color=colors[i],
                     linewidth=1.5, alpha=0.8)

    axes[0].set_xlabel('Adaptation Steps $K$')
    axes[0].set_ylabel('Adaptation Gap $G_K$')
    axes[0].set_title('(a) Adaptation Curves (SGD)')
    axes[0].set_xlim(-1, args.max_K + 1)
    axes[0].set_ylim(bottom=0)
    axes[0].grid(True, alpha=0.3)

    # Add colorbar
    from matplotlib.colors import LogNorm
    sm = plt.cm.ScalarMappable(cmap='viridis',
                                norm=LogNorm(vmin=min(learning_rates), vmax=max(learning_rates)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[0], pad=0.02)
    cbar.set_label('Learning Rate $\\eta$')

    # Plot beta vs LR
    valid_lrs = [lr for lr, res in results.items() if res['beta'] is not None]
    valid_betas = [results[lr]['beta'] for lr in valid_lrs]
    valid_r2 = [results[lr]['r_squared'] for lr in valid_lrs]

    ax2 = axes[1]
    ax2.semilogx(valid_lrs, valid_betas, 'o-', color='#E94F37', markersize=6, linewidth=2)
    ax2.set_xlabel('Learning Rate $\\eta$')
    ax2.set_ylabel('Adaptation Rate $\\beta$')
    ax2.set_title('(b) $\\beta$ vs Learning Rate')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([min(learning_rates) * 0.8, max(learning_rates) * 1.2])

    plt.tight_layout()

    output_dir = Path(__file__).parent
    save_path = str(output_dir / f"{args.output}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")
    plt.close()

    # Save data
    data_path = str(output_dir / f"{args.output}_data.json")
    data_to_save = {
        'optimizer': 'sgd',
        'max_K': args.max_K,
        'n_tasks': args.n_tasks,
        'results': {
            str(lr): {
                'K_values': res['K_values'].tolist(),
                'mean_gaps': res['mean_gaps'].tolist(),
                'c': float(res['c']) if res['c'] else None,
                'beta': float(res['beta']) if res['beta'] else None,
                'r_squared': float(res['r_squared']) if res['r_squared'] else None,
            }
            for lr, res in results.items()
        }
    }
    with open(data_path, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    print(f"Data saved to: {data_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary: LR Sensitivity (SGD)")
    print("=" * 60)
    print(f"{'LR':>10} {'beta':>8} {'c':>8} {'R²':>8}")
    print("-" * 36)
    for lr in sorted(valid_lrs):
        res = results[lr]
        print(f"{lr:>10.4f} {res['beta']:>8.3f} {res['c']:>8.4f} {res['r_squared']:>8.4f}")


if __name__ == '__main__':
    main()
