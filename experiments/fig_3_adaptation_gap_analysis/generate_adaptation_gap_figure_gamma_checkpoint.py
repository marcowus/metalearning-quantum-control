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
from scipy.stats import linregress
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
})


def create_single_qubit_system(gamma_deph=0.05, gamma_relax=0.025, device='cpu'):
    """Create simulator with direct gamma rates."""
    sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
    sigma_p = torch.tensor([[0, 1], [0, 0]], dtype=torch.complex64, device=device)

    H0 = 0.0 * sigma_z  # No drift Hamiltonian
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

    # Gamma task features: normalized rates
    task_features = torch.tensor([
        gamma_deph / 0.1,
        gamma_relax / 0.05,
        (gamma_deph + gamma_relax) / 0.15
    ], dtype=torch.float32, device=device)

    controls = policy(task_features)

    rho0 = torch.zeros(2, 2, dtype=torch.complex64, device=device)
    rho0[0, 0] = 1.0

    rho_final = sim.forward(rho0, controls, T=1.0)

    # Target: |1><1| (Pauli-X gate on |0>)
    target = torch.zeros(2, 2, dtype=rho_final.dtype, device=device)
    target[1, 1] = 1.0

    fidelity = torch.abs(torch.trace(target @ rho_final)).real
    return 1.0 - fidelity


def load_pretrained_gamma_policy(checkpoint_path, device='cpu', hidden_dim=64, n_segments=20):
    """Load pretrained gamma policy from checkpoint."""
    print(f"Loading pretrained policy from: {checkpoint_path}")

    # Create policy with specified architecture
    policy = GammaPulsePolicy(
        task_feature_dim=3,
        hidden_dim=hidden_dim,
        n_hidden_layers=2,
        n_segments=n_segments,
        n_controls=2,
        output_scale=1.0
    ).to(device)

    # Load checkpoint
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
                                       max_K=20, inner_lr=0.01, device='cpu'):
    """Compute adaptation gap G_K using gamma-rate tasks."""
    n_tasks = len(task_params_list)
    all_gaps = np.zeros((n_tasks, max_K + 1))
    initial_losses = np.zeros(n_tasks)

    for task_idx, (gamma_deph, gamma_relax) in enumerate(task_params_list):
        adapted_policy = deepcopy(robust_policy)
        adapted_policy.train()
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
    """Exponential saturation without offset: G_K = c*(1 - exp(-beta*K))"""
    return c * (1 - np.exp(-beta * K))


def fit_exponential_saturation(K_values, mean_gaps):
    """Fit G_K = c*(1 - exp(-beta*K)) - no G0 offset term."""
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


def sample_gamma_tasks(n_tasks, diversity_scale=1.0, rng=None,
                       center_deph=None, center_relax=None):
    """Sample gamma-rate tasks with UNIFORM distribution.

    Training used uniform over [0.02, 0.15] x [0.01, 0.08].
    diversity_scale shrinks the range toward center for panel (b) analysis.

    If center_deph/center_relax provided, samples around that center point
    (for challenging/OOD task testing).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Base ranges (matching training distribution)
    gamma_deph_range = (0.02, 0.15)
    gamma_relax_range = (0.01, 0.08)

    # Use provided center or training distribution center
    if center_deph is not None and center_relax is not None:
        deph_center = center_deph
        relax_center = center_relax
        # Use ±15% spread around the center point
        deph_half = center_deph * 0.15 * diversity_scale
        relax_half = center_relax * 0.15 * diversity_scale
    else:
        # Default: training distribution center
        deph_center = (gamma_deph_range[0] + gamma_deph_range[1]) / 2
        relax_center = (gamma_relax_range[0] + gamma_relax_range[1]) / 2
        deph_half = (gamma_deph_range[1] - gamma_deph_range[0]) / 2 * diversity_scale
        relax_half = (gamma_relax_range[1] - gamma_relax_range[0]) / 2 * diversity_scale

    tasks = []
    for _ in range(n_tasks):
        gamma_deph = rng.uniform(deph_center - deph_half, deph_center + deph_half)
        gamma_relax = rng.uniform(relax_center - relax_half, relax_center + relax_half)
        # Clip to valid range (allow higher for challenging tasks)
        gamma_deph = np.clip(gamma_deph, 0.001, 2.0)
        gamma_relax = np.clip(gamma_relax, 0.001, 1.0)
        tasks.append((gamma_deph, gamma_relax))

    return tasks


def generate_panel_a_data(robust_policy, n_tasks=20, max_K=30, inner_lr=0.01, device='cpu',
                          center_deph=None, center_relax=None):
    """Generate panel (a): G_K vs K with exponential fit."""
    print("Generating Panel (a) data: Adaptation gap vs K...")
    if center_deph is not None:
        print(f"  Using challenging center: gamma_deph={center_deph}, gamma_relax={center_relax}")

    rng = np.random.default_rng(42)
    task_params_list = sample_gamma_tasks(n_tasks, diversity_scale=1.0, rng=rng,
                                          center_deph=center_deph, center_relax=center_relax)

    K_values, mean_gaps, std_gaps, mean_initial_loss = compute_adaptation_gap_vs_K_gamma(
        robust_policy, task_params_list, max_K=max_K, inner_lr=inner_lr, device=device
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
        'mean_initial_loss': mean_initial_loss
    }


def generate_panel_b_data(robust_policy, n_tasks_per_diversity=15, K_adapt=20,
                          inner_lr=0.01, device='cpu',
                          center_deph=None, center_relax=None):
    """Generate panel (b): G_inf vs task diversity."""
    print(f"Generating Panel (b) data: G_{K_adapt} vs task diversity...")
    if center_deph is not None:
        print(f"  Using challenging center: gamma_deph={center_deph}, gamma_relax={center_relax}")

    diversity_scales = list(np.linspace(0.1, 2.0, 15))

    G_inf_means = []
    G_inf_stds = []
    sigma_squared_values = []

    for ds in diversity_scales:
        print(f"  Processing diversity scale = {ds:.2f}...", end=" ")

        rng = np.random.default_rng(int(ds * 10000) + 42)
        task_params_list = sample_gamma_tasks(n_tasks_per_diversity, diversity_scale=ds, rng=rng,
                                              center_deph=center_deph, center_relax=center_relax)

        _, mean_gaps, std_gaps, _ = compute_adaptation_gap_vs_K_gamma(
            robust_policy, task_params_list, max_K=K_adapt, inner_lr=inner_lr, device=device
        )

        G_inf_means.append(mean_gaps[K_adapt])
        G_inf_stds.append(std_gaps[K_adapt])
        sigma_squared_values.append(ds ** 2)
        print(f"G_{K_adapt}={mean_gaps[K_adapt]:.4f}")

    slope, intercept, r_value, _, _ = linregress(sigma_squared_values, G_inf_means)

    print(f"  Linear fit: slope={slope:.4f}, intercept={intercept:.4f}, R^2={r_value**2:.4f}")

    return {
        'sigma_squared_values': np.array(sigma_squared_values),
        'G_inf_means': np.array(G_inf_means),
        'G_inf_stds': np.array(G_inf_stds),
        'slope': slope,
        'intercept': intercept,
        'R_squared': r_value ** 2
    }


def create_figure(panel_a_data, panel_b_data, save_path=None):
    """Create 2-panel figure."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel (a)
    ax = axes[0]
    K = panel_a_data['K_values']
    mean_gaps = panel_a_data['mean_gaps']
    c = panel_a_data['c']
    beta = panel_a_data['beta']

    ax.plot(K, mean_gaps, 'o', color='#3498db', markersize=5, label='Data')

    if c is not None:
        K_fine = np.linspace(0, K[-1], 100)
        fitted_fine = exponential_saturation(K_fine, c, beta)
        ax.plot(K_fine, fitted_fine, '-', color='#e74c3c', linewidth=2,
                label=f'Fit: $G_K = c(1-e^{{-\\beta K}})$')

    ax.set_xlabel('Inner-loop Steps $K$')
    ax.set_ylabel('Adaptation Gap $G_K$')
    ax.set_title('(a) Adaptation Gap vs Steps')
    ax.set_xlim(-0.5, K[-1] + 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    # Add R² in top-left (away from legend)
    if 'R_squared' in panel_a_data and panel_a_data['R_squared'] is not None:
        ax.text(0.05, 0.95, f'$R^2 = {panel_a_data["R_squared"]:.3f}$', transform=ax.transAxes,
                ha='left', va='top', fontsize=10)

    # Panel (b)
    ax = axes[1]
    sigma_sq = panel_b_data['sigma_squared_values']
    G_inf = panel_b_data['G_inf_means']
    slope = panel_b_data['slope']
    intercept = panel_b_data['intercept']
    R_sq = panel_b_data['R_squared']

    ax.plot(sigma_sq, G_inf, 's', color='#2ecc71', markersize=8, label='Data')

    sigma_sq_fine = np.linspace(sigma_sq.min(), sigma_sq.max(), 100)
    fitted_line = slope * sigma_sq_fine + intercept
    ax.plot(sigma_sq_fine, fitted_line, '-', color='#e74c3c', linewidth=2,
            label=f'Linear fit')

    ax.set_xlabel('Task Diversity $\\sigma^2_S$')
    ax.set_ylabel('Asymptotic Gap $G_\\infty$')
    ax.set_title(f'(b) Gap vs Task Diversity')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    # Add R² in top-left (away from legend)
    ax.text(0.05, 0.95, f'$R^2 = {R_sq:.3f}$', transform=ax.transAxes,
            ha='left', va='top', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_tasks', type=int, default=20)
    parser.add_argument('--max_K', type=int, default=30)
    parser.add_argument('--inner_lr', type=float, default=0.01)
    parser.add_argument('--checkpoint', type=str,
                        default='../../checkpoints_gamma/maml_gamma_pauli_x.pt',
                        help='Path to pretrained gamma checkpoint')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension of policy network (original=128, v2/v3/v4=64)')
    parser.add_argument('--n_segments', type=int, default=20,
                        help='Number of control segments (original=60, v2/v3/v4=20)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename (without extension). If not set, uses adaptation_gap_gamma_checkpoint')
    parser.add_argument('--gamma_deph', type=float, default=None,
                        help='Center gamma_deph for challenging tasks (default: None = in-distribution)')
    parser.add_argument('--gamma_relax', type=float, default=None,
                        help='Center gamma_relax for challenging tasks (default: None = in-distribution)')
    args = parser.parse_args()

    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load pretrained gamma policy
    checkpoint_path = Path(__file__).parent / args.checkpoint
    if not checkpoint_path.exists():
        # Try absolute path
        checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please train a gamma policy first using train_meta_gamma.py")
        return

    robust_policy = load_pretrained_gamma_policy(str(checkpoint_path), device=device,
                                                  hidden_dim=args.hidden_dim, n_segments=args.n_segments)

    # Log task center if specified
    if args.gamma_deph is not None and args.gamma_relax is not None:
        print(f"\nUsing challenging task center: gamma_deph={args.gamma_deph}, gamma_relax={args.gamma_relax}")
        print(f"  (Training range: gamma_deph=[0.02,0.15], gamma_relax=[0.01,0.08])")

    # Generate panel data
    print("\n" + "=" * 60)
    panel_a_data = generate_panel_a_data(
        robust_policy, n_tasks=args.n_tasks, max_K=args.max_K,
        inner_lr=args.inner_lr, device=device,
        center_deph=args.gamma_deph, center_relax=args.gamma_relax
    )

    print("\n" + "=" * 60)
    panel_b_data = generate_panel_b_data(
        robust_policy, n_tasks_per_diversity=args.n_tasks, K_adapt=20,
        inner_lr=args.inner_lr, device=device,
        center_deph=args.gamma_deph, center_relax=args.gamma_relax
    )

    # Create figure
    output_dir = Path(__file__).parent
    output_name = args.output if args.output else "adaptation_gap_gamma_checkpoint"
    save_path = str(output_dir / f"{output_name}.png")
    create_figure(panel_a_data, panel_b_data, save_path=save_path)

    # Save data
    data_path = str(output_dir / f"{output_name}_data.json")
    data_to_save = {
        'panel_a': {
            'K_values': panel_a_data['K_values'].tolist(),
            'mean_gaps': panel_a_data['mean_gaps'].tolist(),
            'std_gaps': panel_a_data['std_gaps'].tolist(),
            'c': float(panel_a_data['c']) if panel_a_data['c'] else None,
            'beta': float(panel_a_data['beta']) if panel_a_data['beta'] else None,
            'R_squared': float(panel_a_data['R_squared']) if panel_a_data['R_squared'] else None,
        },
        'panel_b': {
            'sigma_squared': panel_b_data['sigma_squared_values'].tolist(),
            'G_inf_means': panel_b_data['G_inf_means'].tolist(),
            'slope': float(panel_b_data['slope']),
            'R_squared': float(panel_b_data['R_squared']),
        }
    }
    with open(data_path, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    print(f"Data saved to: {data_path}")

    print("\n" + "=" * 60)
    print("Summary Statistics (GAMMA-RATE from checkpoint)")
    print("=" * 60)
    print(f"\nPanel (a) - Exponential Saturation: G_K = c*(1-exp(-beta*K))")
    if panel_a_data['c'] is not None:
        print(f"  c = {panel_a_data['c']:.4f}")
        print(f"  beta = {panel_a_data['beta']:.4f}")
        print(f"  R^2 = {panel_a_data['R_squared']:.4f}")
    print(f"\nPanel (b) - Linear Scaling:")
    print(f"  Slope = {panel_b_data['slope']:.4f}")
    print(f"  R^2 = {panel_b_data['R_squared']:.4f}")


if __name__ == '__main__':
    main()
