"""
Two-Qubit CZ Gate with PULSED Coupling - EXTREME DISTRIBUTION 
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
from typing import List
from dataclasses import dataclass

from two_qubit_cz_maml_fast import (
    X1, Y1, X2, Y2, Z1, Z2, ZZ, Sm1, Sm2,
    CZ_GATE, ket_0, ket_1, ket_p, ket_m, ket_pi, ket_mi,
)

torch.manual_seed(42)
np.random.seed(42)

 
J_TARGET_RANGE = (0.5, 10.0)  # 20x range! (was 4x)
GAMMA_DEPH_RANGE = (0.0005, 0.05)  # 100x range
GAMMA_RELAX_RANGE = (0.00025, 0.025)  # 100x range
GATE_TIME = 1.0

# Training params
MAML_ITERATIONS = 250
MAML_INNER_STEPS = 3
MAML_INNER_LR = 0.05
MAML_META_LR = 0.002
TASKS_PER_BATCH = 2
N_SEGMENTS = 15
DT = 0.02

FIXED_AVG_ITERATIONS = 400
EVAL_N_TASKS = 16
ADAPT_K = 25
ADAPT_LR = 0.002

device = 'cpu'


@dataclass
class PulsedCZTask:
    J_target: float
    gamma_deph_1: float
    gamma_relax_1: float
    gamma_deph_2: float
    gamma_relax_2: float

    def to_array(self, normalized: bool = True) -> np.ndarray:
        if normalized:
            # Log-scale normalization for wide ranges
            J_norm = (np.log(self.J_target) - np.log(J_TARGET_RANGE[0])) / \
                     (np.log(J_TARGET_RANGE[1]) - np.log(J_TARGET_RANGE[0]))
            gd_norm = (np.log(self.gamma_deph_1 + 1e-6) - np.log(GAMMA_DEPH_RANGE[0])) / \
                      (np.log(GAMMA_DEPH_RANGE[1]) - np.log(GAMMA_DEPH_RANGE[0]))
            gr_norm = (np.log(self.gamma_relax_1 + 1e-6) - np.log(GAMMA_RELAX_RANGE[0])) / \
                      (np.log(GAMMA_RELAX_RANGE[1]) - np.log(GAMMA_RELAX_RANGE[0]))
            return np.array([J_norm, gd_norm, gr_norm, gd_norm, gr_norm])
        return np.array([self.J_target, self.gamma_deph_1, self.gamma_relax_1,
                        self.gamma_deph_2, self.gamma_relax_2])


class PulsedCouplingSimulator:
    """Lindblad simulator with pulsed ZZ coupling."""

    def __init__(self, task: PulsedCZTask, device='cpu'):
        self.device = device
        self.task = task

        self.H0 = torch.zeros(4, 4, dtype=torch.complex64, device=device)

        self.H_controls = [
            torch.tensor(X1, dtype=torch.complex64, device=device),
            torch.tensor(Y1, dtype=torch.complex64, device=device),
            torch.tensor(X2, dtype=torch.complex64, device=device),
            torch.tensor(Y2, dtype=torch.complex64, device=device),
            torch.tensor(Z1, dtype=torch.complex64, device=device),
            torch.tensor(Z2, dtype=torch.complex64, device=device),
            torch.tensor(ZZ, dtype=torch.complex64, device=device),  # PULSED
        ]

        self.L_operators = [
            torch.tensor(Z1, dtype=torch.complex64, device=device),
            torch.tensor(Sm1, dtype=torch.complex64, device=device),
            torch.tensor(Z2, dtype=torch.complex64, device=device),
            torch.tensor(Sm2, dtype=torch.complex64, device=device),
        ]

        self.gamma_rates = torch.tensor([
            task.gamma_deph_1 / 2, task.gamma_relax_1,
            task.gamma_deph_2 / 2, task.gamma_relax_2,
        ], dtype=torch.float32, device=device)

        self.dt = DT

    def _lindbladian(self, rho, H):
        comm = -1j * (H @ rho - rho @ H)
        dissipator = torch.zeros_like(rho)
        for L, gamma in zip(self.L_operators, self.gamma_rates):
            L_dag = L.conj().T
            L_dag_L = L_dag @ L
            dissipator += gamma * (L @ rho @ L_dag - 0.5 * (L_dag_L @ rho + rho @ L_dag_L))
        return comm + dissipator

    def forward(self, rho0, control_sequence, T):
        n_segments = control_sequence.shape[0]
        segment_duration = T / n_segments
        n_steps = max(1, int(segment_duration / self.dt))
        dt = segment_duration / n_steps

        rho = rho0.clone()

        for seg in range(n_segments):
            H = self.H0.clone()
            for c, H_c in enumerate(self.H_controls):
                H = H + control_sequence[seg, c] * H_c

            for _ in range(n_steps):
                k1 = self._lindbladian(rho, H)
                k2 = self._lindbladian(rho + 0.5 * dt * k1, H)
                k3 = self._lindbladian(rho + 0.5 * dt * k2, H)
                k4 = self._lindbladian(rho + dt * k3, H)
                rho = rho + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

        return rho


def compute_cz_fidelity(simulator, control_sequence, T, device='cpu'):
    input_states = [
        np.kron(ket_p, ket_p), np.kron(ket_m, ket_m),
        np.kron(ket_1, ket_p), np.kron(ket_p, ket_1),
        np.kron(ket_0, ket_0), np.kron(ket_1, ket_1),
    ]
    target_states = [CZ_GATE @ state for state in input_states]

    total_fidelity = torch.tensor(0.0, device=device)

    for psi, psi_target in zip(input_states, target_states):
        psi_t = torch.tensor(psi, dtype=torch.complex64, device=device)
        rho0 = torch.outer(psi_t, psi_t.conj())
        rho_final = simulator.forward(rho0, control_sequence, T)
        psi_target_t = torch.tensor(psi_target, dtype=torch.complex64, device=device)
        fidelity = torch.real(psi_target_t.conj() @ rho_final @ psi_target_t)
        total_fidelity = total_fidelity + fidelity

    return total_fidelity / len(input_states)


class PulsedCouplingPolicy(nn.Module):
    def __init__(self, task_feature_dim: int = 5, hidden_dim: int = 128,
                 n_hidden_layers: int = 3, n_segments: int = N_SEGMENTS, n_controls: int = 7):
        super().__init__()
        self.n_segments = n_segments
        self.n_controls = n_controls
        self.output_dim = n_segments * n_controls

        layers = [nn.Linear(task_feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
        for _ in range(n_hidden_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, self.output_dim))
        self.network = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, task_features):
        output = self.network(task_features)
        output = output.view(self.n_segments, self.n_controls)
        # Wider ZZ coupling range for extreme J values
        controls = torch.tanh(output) * torch.tensor(
            [np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, 12.0], device=output.device)
        return controls


def compute_loss(policy, task: PulsedCZTask, device='cpu'):
    simulator = PulsedCouplingSimulator(task, device)
    task_features = torch.tensor(task.to_array(normalized=True), dtype=torch.float32, device=device)
    controls = policy(task_features)
    fidelity = compute_cz_fidelity(simulator, controls, GATE_TIME, device)
    return 1 - fidelity, fidelity


class PulsedCZTaskDistribution:
    def sample(self, n_tasks: int) -> List[PulsedCZTask]:
        tasks = []
        for _ in range(n_tasks):
            # Log-uniform sampling for wider coverage
            J = np.exp(np.random.uniform(np.log(J_TARGET_RANGE[0]), np.log(J_TARGET_RANGE[1])))
            gd = np.exp(np.random.uniform(np.log(GAMMA_DEPH_RANGE[0]), np.log(GAMMA_DEPH_RANGE[1])))
            gr = np.exp(np.random.uniform(np.log(GAMMA_RELAX_RANGE[0]), np.log(GAMMA_RELAX_RANGE[1])))
            tasks.append(PulsedCZTask(J, gd, gr,
                                      gd * np.random.uniform(0.7, 1.3),
                                      gr * np.random.uniform(0.7, 1.3)))
        return tasks

    def sample_grid(self) -> List[PulsedCZTask]:
        """Sample at extreme grid points."""
        tasks = []
        for J in [0.5, 1.5, 3.0, 6.0, 10.0]:  # Wide J range
            for gd, gr in [(0.001, 0.0005), (0.03, 0.015)]:  # Low and high noise
                tasks.append(PulsedCZTask(J, gd, gr, gd, gr))
        return tasks


def maml_inner_loop(policy, task, K, inner_lr, device):
    adapted = deepcopy(policy)
    adapted.train()

    for _ in range(K):
        loss, _ = compute_loss(adapted, task, device)
        grads = torch.autograd.grad(loss, adapted.parameters(), create_graph=False)
        with torch.no_grad():
            for param, grad in zip(adapted.parameters(), grads):
                param.sub_(inner_lr * grad.clamp(-1.0, 1.0))

    return adapted


def train_maml(n_iterations=MAML_ITERATIONS, device='cpu'):
    print("\n" + "=" * 70)
    print("TRAINING MAML - EXTREME DISTRIBUTION")
    print(f"J_target: {J_TARGET_RANGE} (20x range!)")
    print(f"gamma_deph: {GAMMA_DEPH_RANGE} (100x range)")
    print("=" * 70)
    sys.stdout.flush()

    policy = PulsedCouplingPolicy().to(device)
    optimizer = optim.AdamW(policy.parameters(), lr=MAML_META_LR, weight_decay=1e-4)

    task_dist = PulsedCZTaskDistribution()
    history = {'iterations': [], 'meta_loss': [], 'val_pre': [], 'val_post': []}

    for iteration in range(n_iterations):
        task_batch = task_dist.sample(TASKS_PER_BATCH)

        optimizer.zero_grad()
        total_loss = 0.0

        for task in task_batch:
            adapted = maml_inner_loop(policy, task, MAML_INNER_STEPS, MAML_INNER_LR, device)
            loss, _ = compute_loss(adapted, task, device)
            total_loss += loss

        meta_loss = total_loss / len(task_batch)
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        history['iterations'].append(iteration)
        history['meta_loss'].append(meta_loss.item())

        if iteration % 25 == 0 or iteration == n_iterations - 1:
            val_tasks = task_dist.sample(3)
            pre_fids, post_fids = [], []

            for task in val_tasks:
                with torch.no_grad():
                    _, pre_fid = compute_loss(policy, task, device)
                    pre_fids.append(pre_fid.item())

                adapted = maml_inner_loop(policy, task, MAML_INNER_STEPS, MAML_INNER_LR, device)
                with torch.no_grad():
                    _, post_fid = compute_loss(adapted, task, device)
                    post_fids.append(post_fid.item())

            history['val_pre'].append(np.mean(pre_fids))
            history['val_post'].append(np.mean(post_fids))

            print(f"Iter {iteration:4d} | Loss: {meta_loss.item():.4f} | "
                  f"Pre: {np.mean(pre_fids):.4f} | Post: {np.mean(post_fids):.4f}")
            sys.stdout.flush()

    return policy, history


def train_fixed_average(device='cpu'):
    print("\n" + "=" * 70)
    print("TRAINING FIXED AVERAGE")
    print("=" * 70)
    sys.stdout.flush()

    # Geometric mean for log-uniform distribution
    avg_J = np.sqrt(J_TARGET_RANGE[0] * J_TARGET_RANGE[1])
    avg_gd = np.sqrt(GAMMA_DEPH_RANGE[0] * GAMMA_DEPH_RANGE[1])
    avg_gr = np.sqrt(GAMMA_RELAX_RANGE[0] * GAMMA_RELAX_RANGE[1])
    avg_task = PulsedCZTask(avg_J, avg_gd, avg_gr, avg_gd, avg_gr)

    print(f"Training on geometric mean: J={avg_J:.2f}, gamma_deph={avg_gd:.4f}")
    sys.stdout.flush()

    policy = PulsedCouplingPolicy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=0.002)

    for i in range(FIXED_AVG_ITERATIONS):
        optimizer.zero_grad()
        loss, fid = compute_loss(policy, avg_task, device)
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"Iter {i:4d} | Fidelity: {fid.item():.4f}")
            sys.stdout.flush()

    return policy


def evaluate(maml_policy, fixed_policy, device='cpu'):
    print("\n" + "=" * 70)
    print("EVALUATION ON EXTREME DISTRIBUTION")
    print("=" * 70)
    sys.stdout.flush()

    task_dist = PulsedCZTaskDistribution()
    all_tasks = task_dist.sample(EVAL_N_TASKS) + task_dist.sample_grid()

    results = {'maml_k0': [], 'maml_k': [], 'fixed_k0': [], 'fixed_k': [], 'task_info': []}

    for i, task in enumerate(all_tasks):
        results['task_info'].append({'J': task.J_target, 'gamma_deph': task.gamma_deph_1})

        # MAML K=0
        with torch.no_grad():
            _, fid = compute_loss(maml_policy, task, device)
        results['maml_k0'].append(fid.item())

        # MAML K=ADAPT_K
        adapted = deepcopy(maml_policy)
        adapted.train()
        opt = optim.Adam(adapted.parameters(), lr=ADAPT_LR)
        for _ in range(ADAPT_K):
            opt.zero_grad()
            loss, _ = compute_loss(adapted, task, device)
            loss.backward()
            opt.step()
        with torch.no_grad():
            _, fid = compute_loss(adapted, task, device)
        results['maml_k'].append(fid.item())

        # Fixed K=0
        with torch.no_grad():
            _, fid = compute_loss(fixed_policy, task, device)
        results['fixed_k0'].append(fid.item())

        # Fixed K=ADAPT_K
        adapted = deepcopy(fixed_policy)
        adapted.train()
        opt = optim.Adam(adapted.parameters(), lr=ADAPT_LR)
        for _ in range(ADAPT_K):
            opt.zero_grad()
            loss, _ = compute_loss(adapted, task, device)
            loss.backward()
            opt.step()
        with torch.no_grad():
            _, fid = compute_loss(adapted, task, device)
        results['fixed_k'].append(fid.item())

        if (i + 1) % 4 == 0:
            print(f"Task {i+1}/{len(all_tasks)} (J={task.J_target:.1f}) | "
                  f"MAML: {results['maml_k0'][-1]:.3f}->{results['maml_k'][-1]:.3f} | "
                  f"Fixed: {results['fixed_k0'][-1]:.3f}->{results['fixed_k'][-1]:.3f}")
            sys.stdout.flush()

    return results


def main():
    start_time = datetime.now()
    print(f"\nStarted: {start_time}")
    print("=" * 70)
    print("TWO-QUBIT CZ - EXTREME TASK DISTRIBUTION")
    print("=" * 70)
    print(f"J_target range: {J_TARGET_RANGE} (20x range)")
    print(f"Noise range: {GAMMA_DEPH_RANGE} (100x range)")
    print("Goal: Find where MAML beats Fixed Average")
    sys.stdout.flush()

    maml_policy, maml_history = train_maml(device=device)
    fixed_policy = train_fixed_average(device=device)
    results = evaluate(maml_policy, fixed_policy, device=device)

    print("\n" + "=" * 70)
    print("FINAL RESULTS - EXTREME DISTRIBUTION")
    print("=" * 70)

    summary = {
        'maml_k0_mean': np.mean(results['maml_k0']),
        'maml_k0_min': np.min(results['maml_k0']),
        'maml_k_mean': np.mean(results['maml_k']),
        'maml_k_min': np.min(results['maml_k']),
        'fixed_k0_mean': np.mean(results['fixed_k0']),
        'fixed_k0_min': np.min(results['fixed_k0']),
        'fixed_k_mean': np.mean(results['fixed_k']),
        'fixed_k_min': np.min(results['fixed_k']),
    }

    print(f"\n{'Method':<20} {'K=0 Mean':>12} {'K=0 Min':>12} {f'K={ADAPT_K} Mean':>12} {f'K={ADAPT_K} Min':>12}")
    print("-" * 72)
    print(f"{'MAML':<20} {summary['maml_k0_mean']:>12.4f} {summary['maml_k0_min']:>12.4f} "
          f"{summary['maml_k_mean']:>12.4f} {summary['maml_k_min']:>12.4f}")
    print(f"{'Fixed Average':<20} {summary['fixed_k0_mean']:>12.4f} {summary['fixed_k0_min']:>12.4f} "
          f"{summary['fixed_k_mean']:>12.4f} {summary['fixed_k_min']:>12.4f}")

    maml_adv_mean = summary['maml_k_mean'] - summary['fixed_k_mean']
    maml_adv_min = summary['maml_k_min'] - summary['fixed_k_min']

    print(f"\nMAML advantage (mean): {maml_adv_mean:+.4f}")
    print(f"MAML advantage (min):  {maml_adv_min:+.4f}")

    if maml_adv_mean > 0.02:
        print("\n[SUCCESS] MAML WINS on extreme distribution!")
    elif maml_adv_mean > 0:
        print("\n[~] MAML slightly better.")
    else:
        print("\n[X] Fixed Average still competitive.")

    # Per-J analysis
    print("\n" + "=" * 70)
    print("PER-J ANALYSIS")
    print("=" * 70)

    task_info = results['task_info']
    for J_bin in [(0.5, 1.5), (1.5, 4.0), (4.0, 10.0)]:
        indices = [i for i, t in enumerate(task_info) if J_bin[0] <= t['J'] < J_bin[1]]
        if indices:
            maml_k = np.mean([results['maml_k'][i] for i in indices])
            fixed_k = np.mean([results['fixed_k'][i] for i in indices])
            print(f"J in [{J_bin[0]:.1f}, {J_bin[1]:.1f}): MAML={maml_k:.4f}, Fixed={fixed_k:.4f}, Diff={maml_k-fixed_k:+.4f}")

    output = {
        'config': {'J_range': J_TARGET_RANGE, 'gate_time': GATE_TIME,
                   'gamma_deph_range': GAMMA_DEPH_RANGE, 'gamma_relax_range': GAMMA_RELAX_RANGE,
                   'n_segments': N_SEGMENTS, 'adapt_k': ADAPT_K},
        'summary': summary,
        'results': results,
        'runtime_minutes': (datetime.now() - start_time).total_seconds() / 60,
    }

    output_path = Path(__file__).parent / 'two_qubit_pulsed_extreme_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to: {output_path}")
    print(f"Runtime: {output['runtime_minutes']:.1f} min")
    print(f"Finished: {datetime.now()}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
