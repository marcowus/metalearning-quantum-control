"""
Two-Qubit CZ Gate: Varying Coupling Strength J
  © 2025 The MITRE Corporation, All Rights Reserved 
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import json
from datetime import datetime
from typing import List, Tuple
from dataclasses import dataclass

# Import base components
from two_qubit_cz_maml_fast import (
    TwoQubitLindbladSimulator, TwoQubitCZPolicy,
    average_gate_fidelity_cz,
    X1, Y1, X2, Y2, Z1, Z2, ZZ, Sm1, Sm2,
)

torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
J_RANGE = (1.0, 4.0)  # Coupling strength range (4x variation)
GAMMA_DEPH_RANGE = (0.001, 0.05)  # Dephasing range
GAMMA_RELAX_RANGE = (0.0005, 0.025)  # Relaxation range

MAML_ITERATIONS = 500
MAML_INNER_STEPS = 5
MAML_INNER_LR = 0.05
MAML_META_LR = 0.001

FIXED_AVG_ITERATIONS = 500
EVAL_N_TASKS = 20
ADAPT_K = 30
ADAPT_LR = 0.001

device = 'cpu'

# ============================================================================
# EXTENDED TASK PARAMETERS (includes J)
# ============================================================================
@dataclass
class CZTaskParams:
    """Task parameters including coupling strength J."""
    J: float  # Coupling strength
    gamma_deph_1: float
    gamma_relax_1: float
    gamma_deph_2: float
    gamma_relax_2: float

    @property
    def gate_time(self) -> float:
        """Ideal gate time for CZ: T = π/(4J)"""
        return np.pi / (4 * self.J)

    def to_array(self, normalized: bool = True) -> np.ndarray:
        """Convert to feature array for policy input."""
        if normalized:
            # Normalize each parameter to roughly [0, 1] range
            J_norm = (self.J - J_RANGE[0]) / (J_RANGE[1] - J_RANGE[0])
            gd_norm = self.gamma_deph_1 / 0.05
            gr_norm = self.gamma_relax_1 / 0.025
            return np.array([J_norm, gd_norm, gr_norm, gd_norm, gr_norm])
        return np.array([self.J, self.gamma_deph_1, self.gamma_relax_1,
                        self.gamma_deph_2, self.gamma_relax_2])


# ============================================================================
# EXTENDED POLICY (5 input features: J + 4 noise params)
# ============================================================================
class CZPolicyWithJ(nn.Module):
    """Policy that takes J as input along with noise parameters."""

    def __init__(
        self,
        task_feature_dim: int = 5,  # J + gamma_deph_1, gamma_relax_1, gamma_deph_2, gamma_relax_2
        hidden_dim: int = 256,
        n_hidden_layers: int = 4,
        n_segments: int = 30,
        n_controls: int = 6,
    ):
        super().__init__()
        self.n_segments = n_segments
        self.n_controls = n_controls
        self.output_dim = n_segments * n_controls

        layers = []
        layers.append(nn.Linear(task_feature_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, self.output_dim))
        self.network = nn.Sequential(*layers)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, task_features: torch.Tensor) -> torch.Tensor:
        output = self.network(task_features)
        output = output.view(self.n_segments, self.n_controls)
        controls = torch.tanh(output) * np.pi
        return controls


# ============================================================================
# SIMULATOR WITH VARIABLE J
# ============================================================================
def create_simulator_with_J(task: CZTaskParams, device='cpu') -> TwoQubitLindbladSimulator:
    """Create simulator with variable coupling strength J."""

    # Static Hamiltonian with variable J
    H0 = torch.tensor(task.J * ZZ, dtype=torch.complex64, device=device)

    # Control Hamiltonians
    H_controls = [
        torch.tensor(X1, dtype=torch.complex64, device=device),
        torch.tensor(Y1, dtype=torch.complex64, device=device),
        torch.tensor(X2, dtype=torch.complex64, device=device),
        torch.tensor(Y2, dtype=torch.complex64, device=device),
        torch.tensor(Z1, dtype=torch.complex64, device=device),
        torch.tensor(Z2, dtype=torch.complex64, device=device),
    ]

    # Lindblad operators
    L_operators = [
        torch.tensor(Z1, dtype=torch.complex64, device=device),  # Dephasing qubit 1
        torch.tensor(Sm1, dtype=torch.complex64, device=device),  # Relaxation qubit 1
        torch.tensor(Z2, dtype=torch.complex64, device=device),  # Dephasing qubit 2
        torch.tensor(Sm2, dtype=torch.complex64, device=device),  # Relaxation qubit 2
    ]

    gamma_rates = torch.tensor([
        task.gamma_deph_1 / 2,
        task.gamma_relax_1,
        task.gamma_deph_2 / 2,
        task.gamma_relax_2,
    ], dtype=torch.float32, device=device)

    return TwoQubitLindbladSimulator(
        H0=H0,
        H_controls=H_controls,
        L_operators=L_operators,
        gamma_rates=gamma_rates,
        device=device,
    )


def compute_loss_with_J(policy, task: CZTaskParams, device='cpu'):
    """Compute loss for task with variable J."""
    simulator = create_simulator_with_J(task, device)

    task_features = torch.tensor(
        task.to_array(normalized=True),
        dtype=torch.float32, device=device
    )

    controls = policy(task_features)

    # Use task-specific gate time T = π/(4J)
    fidelity = average_gate_fidelity_cz(simulator, controls, task.gate_time, device)

    return 1 - fidelity, fidelity


# ============================================================================
# TASK DISTRIBUTION
# ============================================================================
class TaskDistributionWithJ:
    def __init__(self):
        self.J_range = J_RANGE
        self.gamma_deph_range = GAMMA_DEPH_RANGE
        self.gamma_relax_range = GAMMA_RELAX_RANGE

    def sample(self, n_tasks: int) -> List[CZTaskParams]:
        tasks = []
        for _ in range(n_tasks):
            J = np.random.uniform(*self.J_range)
            gamma_deph = np.random.uniform(*self.gamma_deph_range)
            gamma_relax = np.random.uniform(*self.gamma_relax_range)

            tasks.append(CZTaskParams(
                J=J,
                gamma_deph_1=gamma_deph,
                gamma_relax_1=gamma_relax,
                gamma_deph_2=gamma_deph * np.random.uniform(0.8, 1.2),
                gamma_relax_2=gamma_relax * np.random.uniform(0.8, 1.2),
            ))
        return tasks

    def sample_grid(self) -> List[CZTaskParams]:
        """Sample tasks on a grid of J and noise values."""
        tasks = []
        J_values = [1.0, 2.0, 3.0, 4.0]
        noise_levels = ['low', 'high']

        for J in J_values:
            for noise in noise_levels:
                if noise == 'low':
                    gd, gr = 0.005, 0.0025
                else:
                    gd, gr = 0.04, 0.02
                tasks.append(CZTaskParams(J, gd, gr, gd, gr))

        return tasks


# ============================================================================
# MAML TRAINING
# ============================================================================
def maml_inner_loop_J(policy, task: CZTaskParams, K, inner_lr, device):
    """MAML inner loop for task with J."""
    adapted = deepcopy(policy)
    adapted.train()

    for _ in range(K):
        loss, _ = compute_loss_with_J(adapted, task, device)
        grads = torch.autograd.grad(loss, adapted.parameters(), create_graph=False)

        with torch.no_grad():
            for param, grad in zip(adapted.parameters(), grads):
                param.sub_(inner_lr * grad.clamp(-1.0, 1.0))

    return adapted


def train_maml_with_J(n_iterations=MAML_ITERATIONS, device='cpu'):
    """Train MAML with variable J."""
    print("\n" + "=" * 70)
    print("TRAINING MAML WITH VARIABLE J")
    print(f"J range: {J_RANGE}, γ_deph: {GAMMA_DEPH_RANGE}, γ_relax: {GAMMA_RELAX_RANGE}")
    print("=" * 70)

    policy = CZPolicyWithJ(task_feature_dim=5).to(device)
    optimizer = optim.AdamW(policy.parameters(), lr=MAML_META_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iterations, eta_min=1e-5)

    task_dist = TaskDistributionWithJ()
    history = {'iterations': [], 'meta_loss': [], 'val_pre': [], 'val_post': []}

    for iteration in range(n_iterations):
        task_batch = task_dist.sample(4)

        optimizer.zero_grad()
        total_loss = 0.0

        for task in task_batch:
            adapted = maml_inner_loop_J(policy, task, MAML_INNER_STEPS, MAML_INNER_LR, device)
            loss, _ = compute_loss_with_J(adapted, task, device)
            total_loss += loss

        meta_loss = total_loss / len(task_batch)
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        history['iterations'].append(iteration)
        history['meta_loss'].append(meta_loss.item())

        if iteration % 50 == 0 or iteration == n_iterations - 1:
            val_tasks = task_dist.sample(5)
            pre_fids, post_fids = [], []

            for task in val_tasks:
                with torch.no_grad():
                    _, pre_fid = compute_loss_with_J(policy, task, device)
                    pre_fids.append(pre_fid.item())

                adapted = maml_inner_loop_J(policy, task, MAML_INNER_STEPS, MAML_INNER_LR, device)
                with torch.no_grad():
                    _, post_fid = compute_loss_with_J(adapted, task, device)
                    post_fids.append(post_fid.item())

            history['val_pre'].append(np.mean(pre_fids))
            history['val_post'].append(np.mean(post_fids))

            print(f"Iter {iteration:4d} | Loss: {meta_loss.item():.4f} | "
                  f"Pre: {np.mean(pre_fids):.4f} | Post: {np.mean(post_fids):.4f}")

    return policy, history


def train_fixed_average_with_J(device='cpu'):
    """Train Fixed Average on center of distribution."""
    print("\n" + "=" * 70)
    print("TRAINING FIXED AVERAGE WITH VARIABLE J")
    print("=" * 70)

    # Use center of distributions
    avg_J = (J_RANGE[0] + J_RANGE[1]) / 2
    avg_gd = (GAMMA_DEPH_RANGE[0] + GAMMA_DEPH_RANGE[1]) / 2
    avg_gr = (GAMMA_RELAX_RANGE[0] + GAMMA_RELAX_RANGE[1]) / 2

    avg_task = CZTaskParams(avg_J, avg_gd, avg_gr, avg_gd, avg_gr)
    print(f"Training on: J={avg_J}, γ_deph={avg_gd}, γ_relax={avg_gr}")
    print(f"Gate time for avg task: T={avg_task.gate_time:.4f}")

    policy = CZPolicyWithJ(task_feature_dim=5).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=0.001)

    for i in range(FIXED_AVG_ITERATIONS):
        optimizer.zero_grad()
        loss, fid = compute_loss_with_J(policy, avg_task, device)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Iter {i:4d} | Fidelity: {fid.item():.4f}")

    return policy


# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_policies_with_J(maml_policy, fixed_policy, device='cpu'):
    """Evaluate both policies on diverse tasks."""
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    task_dist = TaskDistributionWithJ()

    # Random tasks + grid tasks
    random_tasks = task_dist.sample(EVAL_N_TASKS)
    grid_tasks = task_dist.sample_grid()
    all_tasks = random_tasks + grid_tasks

    results = {
        'maml_k0': [], 'maml_k30': [],
        'fixed_k0': [], 'fixed_k30': [],
        'task_info': []
    }

    print(f"\nEvaluating on {len(all_tasks)} tasks...")

    for i, task in enumerate(all_tasks):
        results['task_info'].append({
            'J': task.J,
            'gamma_deph': task.gamma_deph_1,
            'gamma_relax': task.gamma_relax_1,
            'gate_time': task.gate_time,
        })

        task_features = torch.tensor(task.to_array(normalized=True),
                                      dtype=torch.float32, device=device)
        simulator = create_simulator_with_J(task, device)

        # MAML K=0
        with torch.no_grad():
            controls = maml_policy(task_features)
            fid = average_gate_fidelity_cz(simulator, controls, task.gate_time, device).item()
        results['maml_k0'].append(fid)

        # MAML K=30
        adapted = deepcopy(maml_policy)
        adapted.train()
        opt = optim.Adam(adapted.parameters(), lr=ADAPT_LR)
        for _ in range(ADAPT_K):
            opt.zero_grad()
            loss, _ = compute_loss_with_J(adapted, task, device)
            loss.backward()
            opt.step()
        adapted.eval()
        with torch.no_grad():
            controls = adapted(task_features)
            fid = average_gate_fidelity_cz(simulator, controls, task.gate_time, device).item()
        results['maml_k30'].append(fid)

        # Fixed K=0
        with torch.no_grad():
            controls = fixed_policy(task_features)
            fid = average_gate_fidelity_cz(simulator, controls, task.gate_time, device).item()
        results['fixed_k0'].append(fid)

        # Fixed K=30
        adapted = deepcopy(fixed_policy)
        adapted.train()
        opt = optim.Adam(adapted.parameters(), lr=ADAPT_LR)
        for _ in range(ADAPT_K):
            opt.zero_grad()
            loss, _ = compute_loss_with_J(adapted, task, device)
            loss.backward()
            opt.step()
        adapted.eval()
        with torch.no_grad():
            controls = adapted(task_features)
            fid = average_gate_fidelity_cz(simulator, controls, task.gate_time, device).item()
        results['fixed_k30'].append(fid)

        if (i + 1) % 5 == 0:
            print(f"  Task {i+1}/{len(all_tasks)} (J={task.J:.1f}) | "
                  f"MAML: {results['maml_k0'][-1]:.3f}->{results['maml_k30'][-1]:.3f} | "
                  f"Fixed: {results['fixed_k0'][-1]:.3f}->{results['fixed_k30'][-1]:.3f}")

    return results


def analyze_by_J(results):
    """Analyze results grouped by J value."""
    print("\n" + "=" * 70)
    print("ANALYSIS BY J VALUE")
    print("=" * 70)

    # Group by J
    J_bins = [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]

    for J_lo, J_hi in J_bins:
        indices = [i for i, t in enumerate(results['task_info'])
                   if J_lo <= t['J'] < J_hi]

        if not indices:
            continue

        maml_k0 = np.mean([results['maml_k0'][i] for i in indices])
        maml_k30 = np.mean([results['maml_k30'][i] for i in indices])
        fixed_k0 = np.mean([results['fixed_k0'][i] for i in indices])
        fixed_k30 = np.mean([results['fixed_k30'][i] for i in indices])

        print(f"\nJ ∈ [{J_lo}, {J_hi})  (n={len(indices)} tasks)")
        print(f"  MAML:  {maml_k0:.4f} -> {maml_k30:.4f} (Δ={maml_k30-maml_k0:+.4f})")
        print(f"  Fixed: {fixed_k0:.4f} -> {fixed_k30:.4f} (Δ={fixed_k30-fixed_k0:+.4f})")
        print(f"  MAML advantage at K=30: {maml_k30 - fixed_k30:+.4f}")


def main():
    start_time = datetime.now()
    print(f"\nStarted: {start_time}")
    print("=" * 70)
    print("TWO-QUBIT CZ: VARYING COUPLING STRENGTH J")
    print("=" * 70)
    print(f"J range: {J_RANGE} (gate time T = π/(4J) varies accordingly)")
    print(f"Noise: γ_deph={GAMMA_DEPH_RANGE}, γ_relax={GAMMA_RELAX_RANGE}")

    # Train both policies
    maml_policy, maml_history = train_maml_with_J(device=device)
    fixed_policy = train_fixed_average_with_J(device=device)

    # Evaluate
    results = evaluate_policies_with_J(maml_policy, fixed_policy, device=device)

    # Analysis by J
    analyze_by_J(results)

    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    summary = {
        'maml_k0_mean': np.mean(results['maml_k0']),
        'maml_k0_min': np.min(results['maml_k0']),
        'maml_k30_mean': np.mean(results['maml_k30']),
        'maml_k30_min': np.min(results['maml_k30']),
        'fixed_k0_mean': np.mean(results['fixed_k0']),
        'fixed_k0_min': np.min(results['fixed_k0']),
        'fixed_k30_mean': np.mean(results['fixed_k30']),
        'fixed_k30_min': np.min(results['fixed_k30']),
    }

    print(f"\n{'Method':<20} {'K=0 Mean':>12} {'K=0 Min':>12} {'K=30 Mean':>12} {'K=30 Min':>12}")
    print("-" * 70)
    print(f"{'MAML':<20} {summary['maml_k0_mean']:>12.4f} {summary['maml_k0_min']:>12.4f} "
          f"{summary['maml_k30_mean']:>12.4f} {summary['maml_k30_min']:>12.4f}")
    print(f"{'Fixed Average':<20} {summary['fixed_k0_mean']:>12.4f} {summary['fixed_k0_min']:>12.4f} "
          f"{summary['fixed_k30_mean']:>12.4f} {summary['fixed_k30_min']:>12.4f}")

    maml_adv_k0 = summary['maml_k0_mean'] - summary['fixed_k0_mean']
    maml_adv_k30 = summary['maml_k30_mean'] - summary['fixed_k30_mean']

    print(f"\nMAML advantage at K=0:  {maml_adv_k0:+.4f}")
    print(f"MAML advantage at K=30: {maml_adv_k30:+.4f}")

    print("\n" + "=" * 70)
    if maml_adv_k30 > 0.02:
        print("✓ MAML WINS! Varying J creates genuine task diversity.")
    elif maml_adv_k30 > 0:
        print("≈ MAML slightly better. J variation helps but may need more range.")
    else:
        print("✗ Fixed Average still competitive. May need different variation.")
    print("=" * 70)

    # Save results
    output = {
        'config': {
            'J_range': J_RANGE,
            'gamma_deph_range': GAMMA_DEPH_RANGE,
            'gamma_relax_range': GAMMA_RELAX_RANGE,
            'maml_iterations': MAML_ITERATIONS,
            'adapt_K': ADAPT_K,
        },
        'summary': summary,
        'detailed_results': results,
        'maml_history': maml_history,
        'timestamp': str(datetime.now()),
        'runtime_minutes': (datetime.now() - start_time).total_seconds() / 60,
    }

    output_path = Path(__file__).parent / 'two_qubit_vary_J_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Total runtime: {output['runtime_minutes']:.1f} minutes")


if __name__ == "__main__":
    main()
