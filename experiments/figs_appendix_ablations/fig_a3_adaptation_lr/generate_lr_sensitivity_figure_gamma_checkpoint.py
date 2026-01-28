"""
Inner Learning Rate Sensitivity Ablation - Using Pretrained Gamma Checkpoint
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
    """Sample gamma-rate tasks with UNIFORM distribution matching training.

    Training distribution:
        gamma_deph: uniform [0.02, 0.15]
        gamma_relax: uniform [0.01, 0.08]
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Match the training distribution exactly
    gamma_deph_range = (0.02, 0.15)
    gamma_relax_range = (0.01, 0.08)

    tasks = []
    for _ in range(n_tasks):
        gamma_deph = rng.uniform(gamma_deph_range[0], gamma_deph_range[1])
        gamma_relax = rng.uniform(gamma_relax_range[0], gamma_relax_range[1])
        tasks.append((gamma_deph, gamma_relax))

    return tasks


def exponential_saturation(K, A_inf, beta, G0=0):
    return G0 + A_inf * (1 - np.exp(-beta * K))


def compute_adaptation_curve(robust_policy, task_params_list, max_K=20, inner_lr=0.01, device='cpu'):
    """Compute adaptation gap curve for given learning rate."""
    n_tasks = len(task_params_list)
    all_gaps = np.zeros((n_tasks, max_K + 1))

    for task_idx, (gamma_deph, gamma_relax) in enumerate(task_params_list):
        adapted_policy = deepcopy(robust_policy)
        adapted_policy.train()
        inner_opt = optim.Adam(adapted_policy.parameters(), lr=inner_lr)

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
    """Fit exponential saturation model."""
    try:
        G0_init = mean_gaps[0]
        A_inf_init = max(0.01, mean_gaps[-1] - mean_gaps[0])
        beta_init = 0.3

        popt, _ = curve_fit(
            exponential_saturation, K_values, mean_gaps,
            p0=[A_inf_init, beta_init, G0_init],
            bounds=([0, 0.01, -0.5], [1, 10, 0.5]),
            maxfev=5000
        )
        return popt[0], popt[1], popt[2]
    except:
        return None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_tasks', type=int, default=50)
    parser.add_argument('--max_K', type=int, default=25)
    parser.add_argument('--checkpoint', type=str,
                        default='../../../checkpoints_gamma/maml_gamma_pauli_x.pt')
    args = parser.parse_args()

    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load pretrained policy
    checkpoint_path = Path(__file__).parent / args.checkpoint
    if not checkpoint_path.exists():
        checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return

    robust_policy = load_pretrained_gamma_policy(str(checkpoint_path), device=device)

    # Learning rates to test - focus on stable monotonic regime
    # 15 points logarithmically spaced from 2e-05 to 3e-04
    learning_rates = np.logspace(np.log10(2e-05), np.log10(3e-04), 15).tolist()
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(learning_rates)))

    rng = np.random.default_rng(42)
    task_params_list = sample_gamma_tasks(args.n_tasks, rng=rng)

    # Compute adaptation curves for each LR
    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    print("\n" + "=" * 60)
    print("Computing adaptation curves for different learning rates...")
    print("=" * 60)

    beta_values = []
    A_inf_values = []

    for i, lr in enumerate(learning_rates):
        print(f"  LR = {lr}...", end=" ")
        K_values, mean_gaps, std_gaps = compute_adaptation_curve(
            robust_policy, task_params_list, max_K=args.max_K,
            inner_lr=lr, device=device
        )

        A_inf, beta, G0 = fit_exponential(K_values, mean_gaps)

        results[lr] = {
            'K_values': K_values,
            'mean_gaps': mean_gaps,
            'std_gaps': std_gaps,
            'A_inf': A_inf,
            'beta': beta
        }

        if beta is not None:
            beta_values.append(beta)
            A_inf_values.append(A_inf)
            print(f"beta={beta:.3f}, A_inf={A_inf:.4f}")
        else:
            print("fit failed")

        # Plot curve (no legend - will use colorbar)
        axes[0].plot(K_values, mean_gaps, 'o-', color=colors[i],
                     markersize=3, linewidth=1.5)

    axes[0].set_xlabel('Inner-loop Steps $K$')
    axes[0].set_ylabel('Adaptation Gap $G_K$')
    axes[0].set_title('(a) Adaptation Curves for Different LRs')
    axes[0].grid(True, alpha=0.3)

    # Add colorbar for panel (a) to show learning rate gradient
    from matplotlib.colors import LogNorm
    import matplotlib.cm as cm
    sm = plt.cm.ScalarMappable(cmap='viridis',
                                norm=LogNorm(vmin=min(learning_rates), vmax=max(learning_rates)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[0], pad=0.02)
    cbar.set_label('Learning Rate $\\eta$', fontsize=10)
    # Manually set colorbar ticks to span the full range
    cbar_ticks = [2e-05, 5e-05, 1e-04, 2e-04, 3e-04]
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f'{t:.0e}' for t in cbar_ticks])

    # Plot beta vs LR
    valid_lrs = [lr for lr, res in results.items() if res['beta'] is not None]
    valid_betas = [results[lr]['beta'] for lr in valid_lrs]

    axes[1].semilogx(valid_lrs, valid_betas, 'o-', color='#e74c3c', markersize=8, linewidth=2)
    axes[1].set_xlabel('Inner Learning Rate $\\eta$')
    axes[1].set_ylabel('Adaptation Rate $\\beta$')
    axes[1].set_title('(b) Adaptation Rate vs Learning Rate')
    axes[1].grid(True, alpha=0.3)

    # Fix x-axis ticks for panel (b) to show full range with manual ticks
    axes[1].set_xlim([min(learning_rates) * 0.8, max(learning_rates) * 1.2])
    x_ticks = [2e-05, 5e-05, 1e-04, 2e-04, 3e-04]
    axes[1].set_xticks(x_ticks)
    axes[1].set_xticklabels([f'{t:.0e}' for t in x_ticks])

    plt.tight_layout()

    output_dir = Path(__file__).parent
    save_path = str(output_dir / "lr_sensitivity_gamma_checkpoint.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")

    # Save data
    data_path = str(output_dir / "lr_sensitivity_gamma_checkpoint_data.json")
    data_to_save = {
        lr: {
            'K_values': res['K_values'].tolist(),
            'mean_gaps': res['mean_gaps'].tolist(),
            'beta': float(res['beta']) if res['beta'] else None,
            'A_inf': float(res['A_inf']) if res['A_inf'] else None,
        }
        for lr, res in results.items()
    }
    with open(data_path, 'w') as f:
        json.dump(data_to_save, f, indent=2)

    print("\n" + "=" * 60)
    print("Summary: Learning Rate Sensitivity")
    print("=" * 60)
    for lr in valid_lrs:
        print(f"  LR={lr}: beta={results[lr]['beta']:.3f}, A_inf={results[lr]['A_inf']:.4f}")


if __name__ == '__main__':
    main()
