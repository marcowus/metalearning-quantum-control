"""
Two-Qubit CZ Gate: Wide Distribution Experiment  
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

from two_qubit_cz_maml_fast import (
    TwoQubitTaskParams, TwoQubitCZPolicy,
    create_two_qubit_simulator, average_gate_fidelity_cz,
    CZ_IDEAL_GATE_TIME, compute_loss
)

torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# CONFIGURATION - WIDER DISTRIBUTION
# ============================================================================
WIDE_GAMMA_DEPH_RANGE = (0.001, 0.3)    # 300x range (was 0.001-0.01)
WIDE_GAMMA_RELAX_RANGE = (0.0005, 0.15)  # 300x range (was 0.0005-0.005)

MAML_ITERATIONS = 500
MAML_INNER_STEPS = 5
MAML_INNER_LR = 0.05
MAML_META_LR = 0.001
TASKS_PER_BATCH = 4

FIXED_AVG_ITERATIONS = 500
EVAL_N_TASKS = 20
ADAPT_K = 30
ADAPT_LR = 0.001

device = 'cpu'

# ============================================================================
# WIDE TASK DISTRIBUTION
# ============================================================================
class WideTaskDistribution:
    def __init__(self):
        self.gamma_deph_range = WIDE_GAMMA_DEPH_RANGE
        self.gamma_relax_range = WIDE_GAMMA_RELAX_RANGE

    def sample(self, n_tasks: int) -> List[TwoQubitTaskParams]:
        tasks = []
        for _ in range(n_tasks):
            # Log-uniform sampling for better coverage of wide range
            log_gd = np.random.uniform(np.log(self.gamma_deph_range[0]),
                                        np.log(self.gamma_deph_range[1]))
            log_gr = np.random.uniform(np.log(self.gamma_relax_range[0]),
                                        np.log(self.gamma_relax_range[1]))
            gamma_deph = np.exp(log_gd)
            gamma_relax = np.exp(log_gr)

            # Correlated noise on both qubits
            tasks.append(TwoQubitTaskParams(
                gamma_deph_1=gamma_deph,
                gamma_relax_1=gamma_relax,
                gamma_deph_2=gamma_deph * np.random.uniform(0.8, 1.2),
                gamma_relax_2=gamma_relax * np.random.uniform(0.8, 1.2),
            ))
        return tasks

    def sample_extreme_corners(self) -> List[TwoQubitTaskParams]:
        """Sample tasks at extreme corners of the distribution."""
        gd_lo, gd_hi = self.gamma_deph_range
        gr_lo, gr_hi = self.gamma_relax_range
        return [
            TwoQubitTaskParams(gd_lo, gr_lo, gd_lo, gr_lo),  # Low noise
            TwoQubitTaskParams(gd_hi, gr_hi, gd_hi, gr_hi),  # High noise
            TwoQubitTaskParams(gd_lo, gr_hi, gd_lo, gr_hi),  # Low deph, high relax
            TwoQubitTaskParams(gd_hi, gr_lo, gd_hi, gr_lo),  # High deph, low relax
            TwoQubitTaskParams(0.05, 0.025, 0.05, 0.025),    # Medium
            TwoQubitTaskParams(0.15, 0.075, 0.15, 0.075),    # Medium-high
        ]


# ============================================================================
# MAML TRAINING (simplified for this experiment)
# ============================================================================
def maml_inner_loop(policy, task, K, inner_lr, device):
    """Perform K steps of gradient descent for task adaptation."""
    adapted = deepcopy(policy)
    adapted.train()

    for _ in range(K):
        loss = compute_loss(adapted, task, device=device)
        grads = torch.autograd.grad(loss, adapted.parameters(), create_graph=False)

        with torch.no_grad():
            for param, grad in zip(adapted.parameters(), grads):
                param.sub_(inner_lr * grad.clamp(-1.0, 1.0))

    return adapted


def train_maml_wide(n_iterations=MAML_ITERATIONS, device='cpu'):
    """Train MAML on wide distribution."""
    print("\n" + "=" * 70)
    print("TRAINING MAML ON WIDE DISTRIBUTION")
    print(f"γ_deph: {WIDE_GAMMA_DEPH_RANGE}, γ_relax: {WIDE_GAMMA_RELAX_RANGE}")
    print("=" * 70)

    policy = TwoQubitCZPolicy(
        task_feature_dim=4, hidden_dim=256, n_hidden_layers=4,
        n_segments=30, n_controls=6
    ).to(device)

    optimizer = optim.AdamW(policy.parameters(), lr=MAML_META_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iterations, eta_min=1e-5)

    task_dist = WideTaskDistribution()

    history = {'iterations': [], 'meta_loss': [], 'val_pre': [], 'val_post': []}

    for iteration in range(n_iterations):
        task_batch = task_dist.sample(TASKS_PER_BATCH)

        optimizer.zero_grad()
        total_loss = 0.0

        for task in task_batch:
            # Inner loop adaptation
            adapted = maml_inner_loop(policy, task, MAML_INNER_STEPS, MAML_INNER_LR, device)

            # Compute loss on adapted policy
            loss = compute_loss(adapted, task, device=device)
            total_loss += loss

        meta_loss = total_loss / len(task_batch)
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        history['iterations'].append(iteration)
        history['meta_loss'].append(meta_loss.item())

        # Validation every 50 iterations
        if iteration % 50 == 0 or iteration == n_iterations - 1:
            val_tasks = task_dist.sample(5)
            pre_fids, post_fids = [], []

            for task in val_tasks:
                with torch.no_grad():
                    pre_loss = compute_loss(policy, task, device=device)
                    pre_fids.append(1 - pre_loss.item())

                adapted = maml_inner_loop(policy, task, MAML_INNER_STEPS, MAML_INNER_LR, device)
                with torch.no_grad():
                    post_loss = compute_loss(adapted, task, device=device)
                    post_fids.append(1 - post_loss.item())

            history['val_pre'].append(np.mean(pre_fids))
            history['val_post'].append(np.mean(post_fids))

            print(f"Iter {iteration:4d} | Loss: {meta_loss.item():.4f} | "
                  f"Pre: {np.mean(pre_fids):.4f} | Post: {np.mean(post_fids):.4f}")

    return policy, history


def train_fixed_average_wide(device='cpu'):
    """Train Fixed Average policy on geometric mean of wide distribution."""
    print("\n" + "=" * 70)
    print("TRAINING FIXED AVERAGE ON WIDE DISTRIBUTION")
    print("=" * 70)

    # Use geometric mean for log-uniform distribution
    avg_gamma_deph = np.sqrt(WIDE_GAMMA_DEPH_RANGE[0] * WIDE_GAMMA_DEPH_RANGE[1])
    avg_gamma_relax = np.sqrt(WIDE_GAMMA_RELAX_RANGE[0] * WIDE_GAMMA_RELAX_RANGE[1])

    avg_task = TwoQubitTaskParams(avg_gamma_deph, avg_gamma_relax,
                                   avg_gamma_deph, avg_gamma_relax)

    print(f"Training on average task: γ_deph={avg_gamma_deph:.4f}, γ_relax={avg_gamma_relax:.4f}")

    policy = TwoQubitCZPolicy(
        task_feature_dim=4, hidden_dim=256, n_hidden_layers=4,
        n_segments=30, n_controls=6
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=0.001)

    for i in range(FIXED_AVG_ITERATIONS):
        optimizer.zero_grad()
        loss = compute_loss(policy, avg_task, device=device)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            fid = 1 - loss.item()
            print(f"Iter {i:4d} | Fidelity on avg task: {fid:.4f}")

    return policy


def evaluate_policies(maml_policy, fixed_policy, device='cpu'):
    """Comprehensive evaluation of both policies."""
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    task_dist = WideTaskDistribution()

    # Test on random tasks
    test_tasks = task_dist.sample(EVAL_N_TASKS)

    # Also include extreme corners
    corner_tasks = task_dist.sample_extreme_corners()
    all_tasks = test_tasks + corner_tasks

    results = {
        'maml_k0': [], 'maml_k30': [],
        'fixed_k0': [], 'fixed_k30': [],
        'task_info': []
    }

    print(f"\nEvaluating on {len(all_tasks)} tasks...")

    for i, task in enumerate(all_tasks):
        task_info = {
            'gamma_deph': task.gamma_deph_1,
            'gamma_relax': task.gamma_relax_1,
        }
        results['task_info'].append(task_info)

        sim = create_two_qubit_simulator(task, device=device)
        task_features = torch.tensor(task.to_array(normalized=True),
                                      dtype=torch.float32, device=device)

        # MAML K=0
        with torch.no_grad():
            controls = maml_policy(task_features)
            fid = average_gate_fidelity_cz(sim, controls, CZ_IDEAL_GATE_TIME, device).item()
        results['maml_k0'].append(fid)

        # MAML K=30 (with Adam)
        adapted = deepcopy(maml_policy)
        adapted.train()
        opt = optim.Adam(adapted.parameters(), lr=ADAPT_LR)
        for _ in range(ADAPT_K):
            opt.zero_grad()
            controls = adapted(task_features)
            fid_curr = average_gate_fidelity_cz(sim, controls, CZ_IDEAL_GATE_TIME, device)
            loss = 1 - fid_curr
            loss.backward()
            opt.step()
        adapted.eval()
        with torch.no_grad():
            controls = adapted(task_features)
            fid = average_gate_fidelity_cz(sim, controls, CZ_IDEAL_GATE_TIME, device).item()
        results['maml_k30'].append(fid)

        # Fixed K=0
        with torch.no_grad():
            controls = fixed_policy(task_features)
            fid = average_gate_fidelity_cz(sim, controls, CZ_IDEAL_GATE_TIME, device).item()
        results['fixed_k0'].append(fid)

        # Fixed K=30
        adapted = deepcopy(fixed_policy)
        adapted.train()
        opt = optim.Adam(adapted.parameters(), lr=ADAPT_LR)
        for _ in range(ADAPT_K):
            opt.zero_grad()
            controls = adapted(task_features)
            fid_curr = average_gate_fidelity_cz(sim, controls, CZ_IDEAL_GATE_TIME, device)
            loss = 1 - fid_curr
            loss.backward()
            opt.step()
        adapted.eval()
        with torch.no_grad():
            controls = adapted(task_features)
            fid = average_gate_fidelity_cz(sim, controls, CZ_IDEAL_GATE_TIME, device).item()
        results['fixed_k30'].append(fid)

        if (i + 1) % 5 == 0:
            print(f"  Task {i+1}/{len(all_tasks)} | "
                  f"MAML: {results['maml_k0'][-1]:.3f}->{results['maml_k30'][-1]:.3f} | "
                  f"Fixed: {results['fixed_k0'][-1]:.3f}->{results['fixed_k30'][-1]:.3f}")

    return results


def main():
    start_time = datetime.now()
    print(f"\nStarted: {start_time}")
    print("=" * 70)
    print("TWO-QUBIT CZ: WIDE DISTRIBUTION EXPERIMENT")
    print("=" * 70)
    print(f"Goal: Show where MAML provides genuine advantage")
    print(f"Wide noise range: γ_deph={WIDE_GAMMA_DEPH_RANGE}, γ_relax={WIDE_GAMMA_RELAX_RANGE}")

    # Train both policies
    maml_policy, maml_history = train_maml_wide(device=device)
    fixed_policy = train_fixed_average_wide(device=device)

    # Evaluate
    results = evaluate_policies(maml_policy, fixed_policy, device=device)

    # Summary statistics
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    summary = {
        'maml_k0_mean': np.mean(results['maml_k0']),
        'maml_k0_std': np.std(results['maml_k0']),
        'maml_k0_min': np.min(results['maml_k0']),
        'maml_k30_mean': np.mean(results['maml_k30']),
        'maml_k30_std': np.std(results['maml_k30']),
        'maml_k30_min': np.min(results['maml_k30']),
        'fixed_k0_mean': np.mean(results['fixed_k0']),
        'fixed_k0_std': np.std(results['fixed_k0']),
        'fixed_k0_min': np.min(results['fixed_k0']),
        'fixed_k30_mean': np.mean(results['fixed_k30']),
        'fixed_k30_std': np.std(results['fixed_k30']),
        'fixed_k30_min': np.min(results['fixed_k30']),
    }

    print(f"\n{'Method':<20} {'K=0 Mean':>12} {'K=0 Min':>12} {'K=30 Mean':>12} {'K=30 Min':>12}")
    print("-" * 70)
    print(f"{'MAML':<20} {summary['maml_k0_mean']:>12.4f} {summary['maml_k0_min']:>12.4f} "
          f"{summary['maml_k30_mean']:>12.4f} {summary['maml_k30_min']:>12.4f}")
    print(f"{'Fixed Average':<20} {summary['fixed_k0_mean']:>12.4f} {summary['fixed_k0_min']:>12.4f} "
          f"{summary['fixed_k30_mean']:>12.4f} {summary['fixed_k30_min']:>12.4f}")

    print(f"\nMAML advantage at K=0:  {summary['maml_k0_mean'] - summary['fixed_k0_mean']:+.4f}")
    print(f"MAML advantage at K=30: {summary['maml_k30_mean'] - summary['fixed_k30_mean']:+.4f}")

    # Determine winner
    print("\n" + "=" * 70)
    if summary['maml_k30_mean'] > summary['fixed_k30_mean'] + 0.01:
        print("✓ MAML WINS! Shows genuine advantage with wide distribution.")
    elif summary['fixed_k30_mean'] > summary['maml_k30_mean'] + 0.01:
        print("✗ Fixed Average still wins. May need even wider distribution.")
    else:
        print("≈ Results are comparable. Tasks may still be too similar.")
    print("=" * 70)

    # Save results
    output = {
        'config': {
            'gamma_deph_range': WIDE_GAMMA_DEPH_RANGE,
            'gamma_relax_range': WIDE_GAMMA_RELAX_RANGE,
            'maml_iterations': MAML_ITERATIONS,
            'adapt_K': ADAPT_K,
            'adapt_lr': ADAPT_LR,
        },
        'summary': summary,
        'detailed_results': {
            'maml_k0': results['maml_k0'],
            'maml_k30': results['maml_k30'],
            'fixed_k0': results['fixed_k0'],
            'fixed_k30': results['fixed_k30'],
            'task_info': results['task_info'],
        },
        'maml_training_history': maml_history,
        'timestamp': str(datetime.now()),
        'runtime_minutes': (datetime.now() - start_time).total_seconds() / 60,
    }

    output_path = Path(__file__).parent / 'two_qubit_wide_distribution_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Total runtime: {output['runtime_minutes']:.1f} minutes")
    print(f"Finished: {datetime.now()}")


if __name__ == "__main__":
    main()
