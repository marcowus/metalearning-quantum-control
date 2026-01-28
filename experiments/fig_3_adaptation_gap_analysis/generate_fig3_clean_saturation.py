"""
Make Figure 3   
© 2025 The MITRE Corporation, All Rights Reserved 
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
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
        H0=H0,
        H_controls=H_controls,
        L_operators=L_operators,
        dt=0.05,
        method='rk4',
        device=device
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


def load_pretrained_gamma_policy(checkpoint_path, device='cpu', hidden_dim=64, n_segments=20):
    """Load pretrained gamma policy from checkpoint."""
    print(f"Loading pretrained policy from: {checkpoint_path}")

    policy = GammaPulsePolicy(
        task_feature_dim=3,
        hidden_dim=hidden_dim,
        n_hidden_layers=2,
        n_segments=n_segments,
        n_controls=2,
        output_scale=1.0
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'policy_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['policy_state_dict'])
        print(f"  Loaded policy state dict (iteration {checkpoint.get('iteration', 'unknown')})")
    else:
        policy.load_state_dict(checkpoint)
        print("  Loaded policy weights directly")

    policy.eval()
    return policy


def compute_adaptation_gap_vs_K_gamma(robust_policy, task_params_list,
                                       max_K=20, inner_lr=0.01, device='cpu',
                                       optimizer_type='sgd'):
    """Compute adaptation gap G_K using gamma-rate tasks."""
    n_tasks = len(task_params_list)
    all_gaps = np.zeros((n_tasks, max_K + 1))
    initial_losses = np.zeros(n_tasks)

    for task_idx, (gamma_deph, gamma_relax) in enumerate(task_params_list):
        adapted_policy = deepcopy(robust_policy)
        adapted_policy.train()
        if optimizer_type == 'sgd':
            inner_opt = optim.SGD(adapted_policy.parameters(), lr=inner_lr)
        else:
            inner_opt = optim.Adam(adapted_policy.parameters(), lr=inner_lr)

        with torch.no_grad():
            L_0 = compute_loss_gamma(adapted_policy, gamma_deph, gamma_relax, device).item()
            initial_losses[task_idx] = L_0
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
    mean_initial_loss = np.mean(initial_losses)

    return K_values, mean_gaps, std_gaps, mean_initial_loss


def exponential_saturation(K, c, beta):
    """Simplified exponential saturation: G_K = c * (1 - exp(-beta*K))"""
    return c * (1 - np.exp(-beta * K))


def fit_exponential_saturation(K_values, mean_gaps):
    """Fit the simplified model."""
    c_init = max(0.05, mean_gaps[-1] * 1.1)
    beta_init = 0.3

    try:
        popt, _ = curve_fit(
            exponential_saturation,
            K_values,
            mean_gaps,
            p0=[c_init, beta_init],
            bounds=([0, 0.01], [1, 10]),
            maxfev=5000
        )
        c, beta = popt

        fitted = exponential_saturation(K_values, c, beta)
        ss_res = np.sum((mean_gaps - fitted) ** 2)
        ss_tot = np.sum((mean_gaps - np.mean(mean_gaps)) ** 2)
        R_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return c, beta, R_squared, fitted
    except Exception as e:
        print(f"  Fitting failed: {e}")
        return None, None, None, None


def sample_gamma_tasks(n_tasks, diversity_scale=1.0, rng=None):
    """Sample gamma-rate tasks with uniform distribution."""
    if rng is None:
        rng = np.random.default_rng(42)

    gamma_deph_range = (0.02, 0.15)
    gamma_relax_range = (0.01, 0.08)

    deph_center = (gamma_deph_range[0] + gamma_deph_range[1]) / 2
    relax_center = (gamma_relax_range[0] + gamma_relax_range[1]) / 2
    deph_half = (gamma_deph_range[1] - gamma_deph_range[0]) / 2 * diversity_scale
    relax_half = (gamma_relax_range[1] - gamma_relax_range[0]) / 2 * diversity_scale

    tasks = []
    for _ in range(n_tasks):
        gamma_deph = rng.uniform(deph_center - deph_half, deph_center + deph_half)
        gamma_relax = rng.uniform(relax_center - relax_half, relax_center + relax_half)
        gamma_deph = np.clip(gamma_deph, 0.001, 2.0)
        gamma_relax = np.clip(gamma_relax, 0.001, 1.0)
        tasks.append((gamma_deph, gamma_relax))

    return tasks


def generate_panel_a_data(robust_policy, n_tasks=60, max_K=30, inner_lr=0.00005, device='cpu',
                          optimizer_type='sgd'):
    """Generate panel (a): G_K vs K with smaller inner_lr for clean saturation."""
    print(f"Generating Panel (a) data with inner_lr={inner_lr}, optimizer={optimizer_type}...")

    rng = np.random.default_rng(42)
    task_params_list = sample_gamma_tasks(n_tasks, diversity_scale=1.0, rng=rng)

    K_values, mean_gaps, std_gaps, mean_initial_loss = compute_adaptation_gap_vs_K_gamma(
        robust_policy, task_params_list, max_K=max_K, inner_lr=inner_lr, device=device,
        optimizer_type=optimizer_type
    )

    c, beta, R_squared, fitted_curve = fit_exponential_saturation(K_values, mean_gaps)

    print(f"  Mean initial loss: {mean_initial_loss:.4f}")
    print(f"  Max gap at K={max_K}: {mean_gaps[-1]:.4f}")
    if c is not None:
        print(f"  Fit: c={c:.4f}, beta={beta:.4f}, R^2={R_squared:.4f}")

    return {
        'K_values': K_values,
        'mean_gaps': mean_gaps,
        'std_gaps': std_gaps,
        'c': c,
        'beta': beta,
        'R_squared': R_squared,
        'fitted_curve': fitted_curve,
        'mean_initial_loss': mean_initial_loss,
        'inner_lr': inner_lr,
        'optimizer_type': optimizer_type
    }


def load_panel_b_data(json_path):
    """Load existing panel (b) data from JSON file."""
    print(f"Loading panel (b) data from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    panel_b = data['panel_b']
    return {
        'sigma_squared_values': np.array(panel_b['actual_variance']),
        'G_inf_means': np.array(panel_b['G_inf_means']),
        'slope': panel_b['slope'],
        'intercept': panel_b['intercept'],
        'R_squared': panel_b['R_squared']
    }


def create_figure(panel_a_data, panel_b_data, save_path=None):
    """Create 2-panel publication-ready figure."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # Panel (a): G_K vs K with exponential fit
    ax = axes[0]
    K = panel_a_data['K_values']
    mean_gaps = panel_a_data['mean_gaps']
    c = panel_a_data['c']
    beta = panel_a_data['beta']
    R_sq_a = panel_a_data['R_squared']

    # Plot data points (no error bars)
    ax.plot(K, mean_gaps, 'o', color='#2E86AB', markersize=5,
            label='Data', zorder=3)

    if c is not None:
        K_fine = np.linspace(0, K[-1], 200)
        fitted_fine = exponential_saturation(K_fine, c, beta)
        ax.plot(K_fine, fitted_fine, '-', color='#E94F37', linewidth=2,
                label=f'$G_K = c(1-e^{{-\\beta K}})$')
        ax.axhline(y=c, color='gray', linestyle='--', alpha=0.6, linewidth=1)

    ax.set_xlabel('Adaptation Steps $K$')
    ax.set_ylabel('Adaptation Gap $G_K$')
    ax.set_title(f'(a) Exponential Saturation ($R^2 = {R_sq_a:.3f}$)')
    ax.set_xlim(-1, K[-1] + 1)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', framealpha=0.9)

    # Panel (b): G_inf vs actual task variance
    ax = axes[1]
    sigma_sq = panel_b_data['sigma_squared_values']
    G_inf = panel_b_data['G_inf_means']
    slope = panel_b_data['slope']
    intercept = panel_b_data['intercept']
    R_sq_b = panel_b_data['R_squared']

    ax.plot(sigma_sq, G_inf, 's', color='#2E86AB', markersize=6,
            label='Data', zorder=3)

    sigma_sq_fine = np.linspace(0, sigma_sq.max() * 1.05, 100)
    fitted_line = slope * sigma_sq_fine + intercept
    ax.plot(sigma_sq_fine, fitted_line, '-', color='#E94F37', linewidth=2,
            label=f'$G_\\infty \\propto \\sigma^2_\\tau$')

    ax.set_xlabel(r'Task Variance $\sigma^2_\tau$')
    ax.set_ylabel('Asymptotic Gap $G_\\infty$')
    ax.set_title(f'(b) Linear Scaling ($R^2 = {R_sq_b:.3f}$)')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', framealpha=0.9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Figure 3 with clean exponential saturation')
    parser.add_argument('--n_tasks', type=int, default=60,
                        help='Number of tasks for panel (a)')
    parser.add_argument('--max_K', type=int, default=30,
                        help='Maximum adaptation steps')
    parser.add_argument('--inner_lr', type=float, default=0.00005,
                        help='Inner loop learning rate (smaller for cleaner saturation)')
    parser.add_argument('--checkpoint', type=str,
                        default='../../checkpoints_gamma/maml_gamma_pauli_x.pt',
                        help='Path to pretrained gamma checkpoint')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--n_segments', type=int, default=20)
    parser.add_argument('--panel_b_data', type=str,
                        default='adaptation_gap_actual_variance_final_data.json',
                        help='JSON file with existing panel (b) data')
    parser.add_argument('--output', type=str, default='fig3_clean_saturation',
                        help='Output filename (without extension)')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'],
                        help='Optimizer type (sgd for cleaner saturation)')
    args = parser.parse_args()

    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load pretrained gamma policy
    checkpoint_path = Path(__file__).parent / args.checkpoint
    if not checkpoint_path.exists():
        checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return

    robust_policy = load_pretrained_gamma_policy(
        str(checkpoint_path), device=device,
        hidden_dim=args.hidden_dim, n_segments=args.n_segments
    )

    # Generate new panel (a) data with smaller learning rate
    print("\n" + "=" * 60)
    print(f"Running panel (a) with inner_lr={args.inner_lr}, optimizer={args.optimizer}")
    print("=" * 60)
    panel_a_data = generate_panel_a_data(
        robust_policy, n_tasks=args.n_tasks, max_K=args.max_K,
        inner_lr=args.inner_lr, device=device, optimizer_type=args.optimizer
    )

    # Load existing panel (b) data
    print("\n" + "=" * 60)
    panel_b_json = Path(__file__).parent / args.panel_b_data
    panel_b_data = load_panel_b_data(str(panel_b_json))

    # Create figure
    output_dir = Path(__file__).parent
    save_path = str(output_dir / f"{args.output}.png")
    create_figure(panel_a_data, panel_b_data, save_path=save_path)

    # Save new data
    data_path = str(output_dir / f"{args.output}_data.json")
    data_to_save = {
        'panel_a': {
            'K_values': panel_a_data['K_values'].tolist(),
            'mean_gaps': panel_a_data['mean_gaps'].tolist(),
            'std_gaps': panel_a_data['std_gaps'].tolist(),
            'c': float(panel_a_data['c']) if panel_a_data['c'] else None,
            'beta': float(panel_a_data['beta']) if panel_a_data['beta'] else None,
            'R_squared': float(panel_a_data['R_squared']) if panel_a_data['R_squared'] else None,
            'inner_lr': args.inner_lr,
            'optimizer': args.optimizer,
        },
        'panel_b': {
            'source': args.panel_b_data,
            'actual_variance': panel_b_data['sigma_squared_values'].tolist(),
            'G_inf_means': panel_b_data['G_inf_means'].tolist(),
            'slope': float(panel_b_data['slope']),
            'intercept': float(panel_b_data['intercept']),
            'R_squared': float(panel_b_data['R_squared']),
        }
    }
    with open(data_path, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    print(f"Data saved to: {data_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\nPanel (a) - Exponential Saturation: G_K = c*(1-exp(-beta*K))")
    print(f"  inner_lr = {args.inner_lr}")
    if panel_a_data['c'] is not None:
        print(f"  c = {panel_a_data['c']:.4f}")
        print(f"  beta = {panel_a_data['beta']:.4f}")
        print(f"  R^2 = {panel_a_data['R_squared']:.4f}")
    print(f"\nPanel (b) - Linear Scaling (from existing data):")
    print(f"  slope = {panel_b_data['slope']:.4f}")
    print(f"  R^2 = {panel_b_data['R_squared']:.4f}")


if __name__ == '__main__':
    main()
