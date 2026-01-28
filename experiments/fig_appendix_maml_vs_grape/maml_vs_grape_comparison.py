"""
MAML vs GRAPE Comparison for Single-Qubit Quantum Control
 © 2025 The MITRE Corporation, All Rights Reserved 
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


# Stable Single-Qubit Lindblad Simulator   
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
    """Compute state fidelity"""
    fid = torch.real(torch.trace(rho @ target_dm))
    return torch.clamp(fid, 0.0, 1.0)



# Policy  
class GammaPulsePolicy(nn.Module):
    def __init__(self, task_feature_dim=3, hidden_dim=128, n_segments=60, n_controls=2):
        super().__init__()
        self.task_feature_dim = task_feature_dim
        self.hidden_dim = hidden_dim
        self.n_segments = n_segments
        self.n_controls = n_controls

        self.network = nn.Sequential(
            nn.Linear(task_feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_segments * n_controls)
        )

    def forward(self, task_features: torch.Tensor) -> torch.Tensor:
        output = self.network(task_features)
        return output.view(self.n_segments, self.n_controls)



# GRAPE Optimizer   
class GRAPEOptimizer:
    def __init__(self, n_segments=60, n_controls=2, device='cpu'):
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



# Main Comparison
def run_comparison(checkpoint_path, n_tasks=30, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cpu'

    T = 1.0
    gamma_deph_range = (0.02, 0.15)
    gamma_relax_range = (0.01, 0.08)

    # Target: X gate |0⟩ → |1⟩
    target_dm = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64, device=device)
    rho0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64, device=device)

    # Load MAML
    print("Loading MAML checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    maml_policy = GammaPulsePolicy()
    maml_policy.load_state_dict(checkpoint)
    maml_policy.eval()

    # Quick validation
    task_features = torch.tensor([0.05/0.1, 0.02/0.05, 0.07/0.15])
    with torch.no_grad():
        test_controls = maml_policy(task_features)
    sim_test = SingleQubitSimulator(0.05, 0.02, device)
    rho_test = sim_test.evolve(rho0, test_controls, T)
    fid_test = state_fidelity(rho_test, target_dm).item()


    # Sample tasks 
    tasks = []
    for _ in range(n_tasks):
        tasks.append({
            'gamma_deph': np.random.uniform(*gamma_deph_range),
            'gamma_relax': np.random.uniform(*gamma_relax_range)
        })

    mean_task = {
        'gamma_deph': np.mean(gamma_deph_range),
        'gamma_relax': np.mean(gamma_relax_range)
    }

    grape = GRAPEOptimizer(n_segments=60, n_controls=2, device=device)

    # 1. Robust GRAPE
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

    # 3. Meta-adapted
    K_values = [5, 10, 20]
    meta_adapted_fidelities = {K: [] for K in K_values}

    for K in K_values:
        print(f"\n3. Evaluating Meta-adapted (K={K})...")
        for task in tasks:
            task_features = torch.tensor([
                task['gamma_deph'] / 0.1,
                task['gamma_relax'] / 0.05,
                (task['gamma_deph'] + task['gamma_relax']) / 0.15
            ], dtype=torch.float32, device=device)

            adapted_policy = GammaPulsePolicy()
            adapted_policy.load_state_dict(maml_policy.state_dict())

            sim = SingleQubitSimulator(task['gamma_deph'], task['gamma_relax'], device)
            optimizer = torch.optim.SGD(adapted_policy.parameters(), lr=0.01)

            for _ in range(K):
                optimizer.zero_grad()
                controls = adapted_policy(task_features)
                rho_final = sim.evolve(rho0, controls, T)
                loss = -state_fidelity(rho_final, target_dm)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                controls = adapted_policy(task_features)
                rho_final = sim.evolve(rho0, controls, T)
                fid = state_fidelity(rho_final, target_dm).item()
            meta_adapted_fidelities[K].append(fid)

    # 4. Per-task GRAPE (with sufficient iterations for fair comparison)
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
    n_cost_tasks = min(10, n_tasks)  # Use subset for cost analysis

    # For subset of tasks, run GRAPE from scratch and track trajectory
    grape_trajectories = []
    for i in range(n_cost_tasks):
        task = tasks[i]
        _, _, trajectory = grape.optimize(
            task['gamma_deph'], task['gamma_relax'],
            target_dm, T, n_iters=max_grape_iters, lr=0.1,
            init_controls=None,  # Start from scratch
            return_trajectory=True
        )
        grape_trajectories.append(trajectory)
        if (i + 1) % 5 == 0:
            print(f"   GRAPE trajectory {i+1}/{n_cost_tasks}...")

    # For each K, find how many GRAPE iterations needed to match MAML fidelity
    K_values_for_cost = [0, 5, 10, 20]
    grape_iters_per_K = {K: [] for K in K_values_for_cost}

    for i in range(n_cost_tasks):
        for K in K_values_for_cost:
            if K == 0:
                target_fid = meta_init_fidelities[i]
            else:
                target_fid = meta_adapted_fidelities[K][i]

            trajectory = grape_trajectories[i]
            matched_iter = next((j for j, fid in enumerate(trajectory) if fid >= target_fid), max_grape_iters)
            grape_iters_per_K[K].append(matched_iter)

    for K in K_values_for_cost:
        mean_iters = np.mean(grape_iters_per_K[K])
        std_iters = np.std(grape_iters_per_K[K])
        print(f"   GRAPE iters to match MAML K={K}: {mean_iters:.1f} ± {std_iters:.1f}")

    # Create figure
    print("\nCreating figure...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel (a)
    ax = axes[0]
    methods = ['Robust\nGRAPE', 'Meta-init\n(K=0)', 'Meta-adapted\n(K=5)',
               'Meta-adapted\n(K=10)', 'Meta-adapted\n(K=20)', 'Per-task\nGRAPE']
    means = [
        np.mean(robust_fidelities) * 100,
        np.mean(meta_init_fidelities) * 100,
        np.mean(meta_adapted_fidelities[5]) * 100,
        np.mean(meta_adapted_fidelities[10]) * 100,
        np.mean(meta_adapted_fidelities[20]) * 100,
        np.mean(per_task_fidelities) * 100
    ]
    stds = [
        np.std(robust_fidelities) * 100,
        np.std(meta_init_fidelities) * 100,
        np.std(meta_adapted_fidelities[5]) * 100,
        np.std(meta_adapted_fidelities[10]) * 100,
        np.std(meta_adapted_fidelities[20]) * 100,
        np.std(per_task_fidelities) * 100
    ]
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#2ca02c', '#2ca02c', '#ff7f0e']

    x = np.arange(len(methods))
    ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Fidelity (%)', fontsize=12)
    ax.set_title('(a) Fidelity Comparison Across Methods', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylim(0, 105)
    ax.axhline(y=99, color='gray', linestyle='--', alpha=0.5, label='99% threshold')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel (b) - Fidelity vs computational steps
    ax = axes[1]

    # MAML fidelity trajectory (K=0, 5, 10, 20)
    maml_steps = [0, 5, 10, 20]
    maml_fids = [
        np.mean(meta_init_fidelities) * 100,
        np.mean(meta_adapted_fidelities[5]) * 100,
        np.mean(meta_adapted_fidelities[10]) * 100,
        np.mean(meta_adapted_fidelities[20]) * 100
    ]

    # Average GRAPE trajectory across tasks
    avg_grape_trajectory = np.mean([traj for traj in grape_trajectories], axis=0) * 100
    grape_steps = np.arange(len(avg_grape_trajectory))

    # Plot GRAPE trajectory
    ax.plot(grape_steps, avg_grape_trajectory, 'r-', linewidth=2, alpha=0.8,
            label='GRAPE (from scratch)')

    # Plot MAML points
    ax.plot(maml_steps, maml_fids, 'b-o', linewidth=2, markersize=10,
            label='MAML adaptation', zorder=5)

    ax.set_xlabel('Optimization Steps', fontsize=12)
    ax.set_ylabel('Fidelity (%)', fontsize=12)
    ax.set_title('(b) Convergence: MAML vs GRAPE', fontsize=14)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_grape_iters)
    ax.set_ylim(50, 100)

    # Annotate the speedup
    maml_k10_fid = maml_fids[2]
    ax.axhline(y=maml_k10_fid, color='blue', linestyle='--', alpha=0.3)
    ax.annotate(f'MAML K=10\n({maml_k10_fid:.1f}%)',
               xy=(10, maml_k10_fid), xytext=(50, maml_k10_fid - 8),
               fontsize=9, ha='left',
               arrowprops=dict(arrowstyle='->', color='blue', lw=1))

    plt.tight_layout()
    output_dir = Path(__file__).parent
    plt.savefig(output_dir / 'maml_vs_grape_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'maml_vs_grape_comparison.pdf', bbox_inches='tight')
    print("Saved: maml_vs_grape_comparison.png/pdf")

    data = {
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
    with open(output_dir / 'maml_vs_grape_data.json', 'w') as f:
        json.dump(data, f, indent=2)
    print("Saved: maml_vs_grape_data.json")

    return data


if __name__ == '__main__':
    checkpoint_path = '/Users/nimalec/Documents/metarl_project/meta-quantum-control/checkpoints/checkpoints_gamma/maml_gamma_pauli_x_best_policy.pt'
    run_comparison(checkpoint_path, n_tasks=15, seed=42)
