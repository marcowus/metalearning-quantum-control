"""
MAML vs GRAPE Comparison for Single-Qubit Quantum Control
Version 2: Uses V2 checkpoint consistent with Figure 3 parameters.

Key changes from v1:
- Uses checkpoints_gamma_v2 (hidden_dim=64, n_segments=20)
- Uses SGD optimizer for MAML adaptation (consistent with Fig 3)
- Keeps same GRAPE implementation
© 2025 The MITRE Corporation, All Rights Reserved 
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from copy import deepcopy

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'font.family': 'serif',
})


# === Stable Single-Qubit Lindblad Simulator ===
def matrix_exp(A, n_terms=15):
    """Matrix exponential via Taylor series (differentiable)."""
    result = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    term = result.clone()
    for n in range(1, n_terms):
        term = term @ A / n
        result = result + term
    return result


class SingleQubitSimulator:
    """Differentiable Lindblad simulator using split-operator method."""

    def __init__(self, gamma_deph: float, gamma_relax: float, device='cpu'):
        self.device = device
        self.gamma_deph = gamma_deph
        self.gamma_relax = gamma_relax

        self.I = torch.eye(2, dtype=torch.complex64, device=device)
        self.sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
        self.sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
        self.sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
        self.sigma_m = torch.tensor([[0, 1], [0, 0]], dtype=torch.complex64, device=device)

    def evolve(self, rho0: torch.Tensor, controls: torch.Tensor, T: float) -> torch.Tensor:
        """Evolve density matrix with split-operator method."""
        n_segments = controls.shape[0]
        dt = T / n_segments
        n_substeps = 5
        sub_dt = dt / n_substeps

        rho = rho0.clone()

        for seg in range(n_segments):
            H = controls[seg, 0] * self.sigma_x + controls[seg, 1] * self.sigma_y

            for _ in range(n_substeps):
                # Unitary evolution via matrix exponential
                U = matrix_exp(-1j * H * sub_dt)
                rho = U @ rho @ U.conj().T

                # Dissipation (Euler step for Lindblad terms)
                drho_diss = torch.zeros_like(rho)

                # Relaxation
                if self.gamma_relax > 0:
                    L = self.sigma_m
                    LdL = L.conj().T @ L
                    drho_diss = drho_diss + self.gamma_relax * (
                        L @ rho @ L.conj().T - 0.5 * (LdL @ rho + rho @ LdL)
                    )

                # Dephasing
                if self.gamma_deph > 0:
                    L = self.sigma_z
                    drho_diss = drho_diss + (self.gamma_deph / 2) * (
                        L @ rho @ L - rho
                    )

                rho = rho + sub_dt * drho_diss
                rho = 0.5 * (rho + rho.conj().T)

        return rho


def state_fidelity(rho: torch.Tensor, target_dm: torch.Tensor) -> torch.Tensor:
    """Compute state fidelity."""
    fid = torch.real(torch.trace(rho @ target_dm))
    return torch.clamp(fid, 0.0, 1.0)


# === Policy (V2 architecture: hidden_dim=64, n_segments=20) ===
class GammaPulsePolicy(nn.Module):
    def __init__(self, task_feature_dim=3, hidden_dim=64, n_hidden_layers=2,
                 n_segments=20, n_controls=2, output_scale=1.0):
        super().__init__()
        self.task_feature_dim = task_feature_dim
        self.hidden_dim = hidden_dim
        self.n_segments = n_segments
        self.n_controls = n_controls
        self.output_scale = output_scale

        layers = [nn.Linear(task_feature_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, n_segments * n_controls))

        self.network = nn.Sequential(*layers)

    def forward(self, task_features: torch.Tensor) -> torch.Tensor:
        output = self.network(task_features)
        return self.output_scale * output.view(self.n_segments, self.n_controls)


# === GRAPE Optimizer (unchanged from v1) ===
class GRAPEOptimizer:
    def __init__(self, n_segments=20, n_controls=2, device='cpu'):
        self.n_segments = n_segments
        self.n_controls = n_controls
        self.device = device

    def optimize(self, gamma_deph, gamma_relax, target_dm, T, n_iters, lr=0.1,
                 init_controls=None, return_trajectory=False):
        sim = SingleQubitSimulator(gamma_deph, gamma_relax, self.device)
        rho0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64, device=self.device)

        if init_controls is not None:
            controls = init_controls.clone().detach().requires_grad_(True)
        else:
            controls = torch.randn(self.n_segments, self.n_controls,
                                   device=self.device) * 0.1
            controls.requires_grad_(True)

        optimizer = torch.optim.Adam([controls], lr=lr)
        trajectory = []

        for i in range(n_iters):
            optimizer.zero_grad()
            rho_final = sim.evolve(rho0, controls, T)
            fidelity = state_fidelity(rho_final, target_dm)

            if return_trajectory:
                trajectory.append(fidelity.item())

            loss = -fidelity
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            rho_final = sim.evolve(rho0, controls.detach(), T)
            final_fid = state_fidelity(rho_final, target_dm).item()

        if return_trajectory:
            trajectory.append(final_fid)
            return controls.detach(), final_fid, trajectory
        return controls.detach(), final_fid


def sample_gamma_tasks(n_tasks, rng=None):
    """Sample gamma-rate tasks with uniform distribution matching training."""
    if rng is None:
        rng = np.random.default_rng(42)

    gamma_deph_range = (0.02, 0.15)
    gamma_relax_range = (0.01, 0.08)

    tasks = []
    for _ in range(n_tasks):
        gamma_deph = rng.uniform(*gamma_deph_range)
        gamma_relax = rng.uniform(*gamma_relax_range)
        tasks.append({
            'gamma_deph': gamma_deph,
            'gamma_relax': gamma_relax
        })

    return tasks


def run_comparison(checkpoint_path, n_tasks=30, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cpu'

    T = 1.0
    gamma_deph_range = (0.02, 0.15)
    gamma_relax_range = (0.01, 0.08)

    # Target: X gate |0> -> |1>
    target_dm = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64, device=device)
    rho0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64, device=device)

    # Load MAML policy (V2 architecture)
    print("Loading MAML checkpoint (V2)...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    maml_policy = GammaPulsePolicy(
        task_feature_dim=3, hidden_dim=64, n_hidden_layers=2,
        n_segments=20, n_controls=2, output_scale=1.0
    )

    if 'policy_state_dict' in checkpoint:
        maml_policy.load_state_dict(checkpoint['policy_state_dict'])
        print(f"  Loaded from iteration {checkpoint.get('iteration', 'unknown')}")
    else:
        maml_policy.load_state_dict(checkpoint)
    maml_policy.eval()

    # Quick validation
    task_features = torch.tensor([0.05/0.1, 0.02/0.05, 0.07/0.15])
    with torch.no_grad():
        test_controls = maml_policy(task_features)
    sim_test = SingleQubitSimulator(0.05, 0.02, device)
    rho_test = sim_test.evolve(rho0, test_controls, T)
    fid_test = state_fidelity(rho_test, target_dm).item()
    print(f"  Validation fidelity: {fid_test*100:.2f}%")

    # Sample tasks
    rng = np.random.default_rng(seed)
    tasks = sample_gamma_tasks(n_tasks, rng=rng)

    mean_task = {
        'gamma_deph': np.mean(gamma_deph_range),
        'gamma_relax': np.mean(gamma_relax_range)
    }

    # Use n_segments=20 for GRAPE to match policy
    grape = GRAPEOptimizer(n_segments=20, n_controls=2, device=device)

    # 1. Robust GRAPE (trained on mean task)
    print("\n1. Training Robust GRAPE on mean task...")
    robust_controls, robust_fid_mean = grape.optimize(
        mean_task['gamma_deph'], mean_task['gamma_relax'],
        target_dm, T, n_iters=200, lr=0.1
    )
    print(f"   Robust GRAPE on mean task: {robust_fid_mean*100:.2f}%")

    robust_fidelities = []
    for task in tasks:
        sim = SingleQubitSimulator(task['gamma_deph'], task['gamma_relax'], device)
        with torch.no_grad():
            rho_final = sim.evolve(rho0, robust_controls, T)
            fid = state_fidelity(rho_final, target_dm).item()
        robust_fidelities.append(fid)

    # 2. Meta-init (K=0)
    print("\n2. Evaluating Meta-init (K=0)...")
    meta_init_fidelities = []
    for task in tasks:
        task_features = torch.tensor([
            task['gamma_deph'] / 0.1,
            task['gamma_relax'] / 0.05,
            (task['gamma_deph'] + task['gamma_relax']) / 0.15
        ], dtype=torch.float32, device=device)

        sim = SingleQubitSimulator(task['gamma_deph'], task['gamma_relax'], device)
        with torch.no_grad():
            controls = maml_policy(task_features)
            rho_final = sim.evolve(rho0, controls, T)
            fid = state_fidelity(rho_final, target_dm).item()
        meta_init_fidelities.append(fid)

    # 3. Meta-adapted (using SGD, consistent with Fig 3)
    K_values = [5, 10, 20]
    meta_adapted_fidelities = {K: [] for K in K_values}
    inner_lr = 0.01  # Same as Fig 3

    for K in K_values:
        print(f"\n3. Evaluating Meta-adapted (K={K}, SGD lr={inner_lr})...")
        for task in tasks:
            task_features = torch.tensor([
                task['gamma_deph'] / 0.1,
                task['gamma_relax'] / 0.05,
                (task['gamma_deph'] + task['gamma_relax']) / 0.15
            ], dtype=torch.float32, device=device)

            adapted_policy = deepcopy(maml_policy)
            adapted_policy.train()
            optimizer = optim.SGD(adapted_policy.parameters(), lr=inner_lr)

            sim = SingleQubitSimulator(task['gamma_deph'], task['gamma_relax'], device)

            for _ in range(K):
                optimizer.zero_grad()
                controls = adapted_policy(task_features)
                rho_final = sim.evolve(rho0, controls, T)
                loss = -state_fidelity(rho_final, target_dm)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapted_policy.parameters(), max_norm=1.0)
                optimizer.step()

            with torch.no_grad():
                controls = adapted_policy(task_features)
                rho_final = sim.evolve(rho0, controls, T)
                fid = state_fidelity(rho_final, target_dm).item()
            meta_adapted_fidelities[K].append(fid)

    # 4. Per-task GRAPE (warm-started from robust controls)
    print("\n4. Running Per-task GRAPE (200 iterations each, warm-started)...")
    per_task_fidelities = []
    per_task_iterations = 200

    for i, task in enumerate(tasks):
        _, fid = grape.optimize(
            task['gamma_deph'], task['gamma_relax'],
            target_dm, T, n_iters=per_task_iterations, lr=0.1,
            init_controls=robust_controls
        )
        per_task_fidelities.append(fid)
        if (i + 1) % 10 == 0:
            print(f"   Task {i+1}/{n_tasks}...")

    # 5. Computational cost - GRAPE from scratch vs MAML adaptation
    print("\n5. Analyzing computational cost (GRAPE from scratch)...")
    max_grape_iters = 150
    n_cost_tasks = min(10, n_tasks)

    grape_trajectories = []
    for i in range(n_cost_tasks):
        task = tasks[i]
        _, _, trajectory = grape.optimize(
            task['gamma_deph'], task['gamma_relax'],
            target_dm, T, n_iters=max_grape_iters, lr=0.1,
            init_controls=None,
            return_trajectory=True
        )
        grape_trajectories.append(trajectory)
        if (i + 1) % 5 == 0:
            print(f"   GRAPE trajectory {i+1}/{n_cost_tasks}...")

    # Create figure
    print("\nCreating figure...")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # Panel (a): Fidelity comparison (only K=0 and K=20)
    ax = axes[0]
    methods = ['Robust\nGRAPE', 'FOMAML\n($K$=0)', 'FOMAML\n($K$=20)', 'Per-task\nGRAPE']
    means = [
        np.mean(robust_fidelities) * 100,
        np.mean(meta_init_fidelities) * 100,
        np.mean(meta_adapted_fidelities[20]) * 100,
        np.mean(per_task_fidelities) * 100
    ]
    stds = [
        np.std(robust_fidelities) * 100,
        np.std(meta_init_fidelities) * 100,
        np.std(meta_adapted_fidelities[20]) * 100,
        np.std(per_task_fidelities) * 100
    ]
    colors = ['#D62728', '#1F77B4', '#1F77B4', '#FF7F0E']

    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=0.8)

    # Add hatching to K=20 to distinguish from K=0
    bars[2].set_hatch('///')

    ax.set_ylabel('Gate Fidelity (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylim(96, 100)
    ax.grid(True, alpha=0.3, axis='y')
    ax.text(-0.15, 1.05, '(a)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    # Add value labels above bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.1,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Panel (b): Computational steps comparison (bar chart)
    ax = axes[1]

    # Find GRAPE iterations to reach ~97% (comparable to FOMAML K=20)
    avg_grape_trajectory = np.mean(grape_trajectories, axis=0) * 100
    target_fid = np.mean(meta_adapted_fidelities[20]) * 100
    idx_match = np.where(avg_grape_trajectory >= target_fid)[0]
    grape_steps_to_match = idx_match[0] if len(idx_match) > 0 else max_grape_iters

    labels = ['FOMAML\n($K$=20)', 'GRAPE\n(from scratch)']
    steps = [20, grape_steps_to_match]
    bar_colors = ['#1F77B4', '#D62728']

    x2 = np.arange(len(labels))
    bars2 = ax.bar(x2, steps, color=bar_colors, alpha=0.85,
                   edgecolor='black', linewidth=0.8, width=0.5)

    ax.set_ylabel('Optimization Steps')
    ax.set_xticks(x2)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, max(steps) * 1.4)
    ax.grid(True, alpha=0.3, axis='y')
    ax.text(-0.15, 1.05, '(b)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    # Value labels inside bars
    ax.text(bars2[0].get_x() + bars2[0].get_width()/2, steps[0]/2,
            f'{steps[0]}', ha='center', va='center', fontsize=14,
            fontweight='bold', color='white')
    ax.text(bars2[1].get_x() + bars2[1].get_width()/2, steps[1]/2,
            f'{steps[1]}', ha='center', va='center', fontsize=14,
            fontweight='bold', color='white')

    # Speedup annotation
    if steps[1] > 0 and steps[0] > 0:
        speedup = steps[1] / steps[0]
        ax.text(0.05, 0.92, f'{speedup:.1f}$\\times$ faster',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=12, fontweight='bold', color='#1F77B4',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F4FD',
                         edgecolor='#1F77B4', linewidth=1.5))

    ax.set_title(f'Steps to reach {target_fid:.1f}% fidelity', fontsize=10, style='italic', pad=8)

    # For data saving
    maml_steps = [0, 5, 10, 20]
    maml_fids = [
        np.mean(meta_init_fidelities) * 100,
        np.mean(meta_adapted_fidelities[5]) * 100,
        np.mean(meta_adapted_fidelities[10]) * 100,
        np.mean(meta_adapted_fidelities[20]) * 100
    ]

    plt.tight_layout()

    output_dir = Path(__file__).parent
    plt.savefig(output_dir / 'maml_vs_grape_comparison_v2.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'maml_vs_grape_comparison_v2.pdf', bbox_inches='tight')
    print("Saved: maml_vs_grape_comparison_v2.png/pdf")

    # Save data
    data = {
        'version': 'v2',
        'checkpoint': str(checkpoint_path),
        'architecture': {
            'hidden_dim': 64,
            'n_segments': 20,
            'n_hidden_layers': 2
        },
        'adaptation': {
            'optimizer': 'sgd',
            'inner_lr': inner_lr
        },
        'methods': methods,
        'means': means,
        'stds': stds,
        'maml_steps': maml_steps,
        'maml_fids': maml_fids,
        'avg_grape_trajectory': avg_grape_trajectory.tolist(),
        'n_tasks': n_tasks,
        'n_cost_tasks': n_cost_tasks,
        'max_grape_iters': max_grape_iters
    }
    with open(output_dir / 'maml_vs_grape_data_v2.json', 'w') as f:
        json.dump(data, f, indent=2)
    print("Saved: maml_vs_grape_data_v2.json")

    # Summary
    print("\n" + "=" * 60)
    print("Summary (V2 - consistent with Fig 3)")
    print("=" * 60)
    print(f"Architecture: hidden_dim=64, n_segments=20")
    print(f"Adaptation: SGD, lr={inner_lr}")
    print(f"\nMethod              Mean Fid    Std")
    print("-" * 40)
    for m, mean, std in zip(methods, means, stds):
        print(f"{m.replace(chr(10), ' '):18} {mean:7.2f}%  {std:5.2f}%")

    return data


if __name__ == '__main__':
    checkpoint_path = Path(__file__).parent.parent.parent / 'checkpoints' / 'checkpoints_gamma_v2' / 'maml_gamma_pauli_x.pt'
    run_comparison(str(checkpoint_path), n_tasks=30, seed=42)
