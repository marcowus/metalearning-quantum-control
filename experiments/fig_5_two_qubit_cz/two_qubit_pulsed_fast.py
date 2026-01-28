"""
Two-Qubit CZ Gate with PULSED Coupling   
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

# ============================================================================
# FAST CONFIG
# ============================================================================
J_TARGET_RANGE = (1.0, 4.0)
GAMMA_DEPH_RANGE = (0.001, 0.03)
GAMMA_RELAX_RANGE = (0.0005, 0.015)
GATE_TIME = 1.0

# REDUCED for speed
MAML_ITERATIONS = 200
MAML_INNER_STEPS = 3
MAML_INNER_LR = 0.05
MAML_META_LR = 0.002
TASKS_PER_BATCH = 2
N_SEGMENTS = 15
DT = 0.02

FIXED_AVG_ITERATIONS = 300
EVAL_N_TASKS = 12
ADAPT_K = 20
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
            J_norm = (self.J_target - J_TARGET_RANGE[0]) / (J_TARGET_RANGE[1] - J_TARGET_RANGE[0])
            return np.array([
                J_norm,
                self.gamma_deph_1 / 0.03,
                self.gamma_relax_1 / 0.015,
                self.gamma_deph_2 / 0.03,
                self.gamma_relax_2 / 0.015,
            ])
        return np.array([self.J_target, self.gamma_deph_1, self.gamma_relax_1,
                        self.gamma_deph_2, self.gamma_relax_2])


class PulsedCouplingSimulator:
    """Lindblad simulator with pulsed ZZ coupling."""

    def __init__(self, task: PulsedCZTask, device='cpu'):
        self.device = device
        self.task = task

        # NO static Hamiltonian
        self.H0 = torch.zeros(4, 4, dtype=torch.complex64, device=device)

        # 7 controls including pulsed ZZ
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
    """Use fewer input states for speed."""
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
        controls = torch.tanh(output) * torch.tensor(
            [np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, 5.0], device=output.device)
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
            J = np.random.uniform(*J_TARGET_RANGE)
            gd = np.random.uniform(*GAMMA_DEPH_RANGE)
            gr = np.random.uniform(*GAMMA_RELAX_RANGE)
            tasks.append(PulsedCZTask(J, gd, gr,
                                      gd * np.random.uniform(0.8, 1.2),
                                      gr * np.random.uniform(0.8, 1.2)))
        return tasks

    def sample_grid(self) -> List[PulsedCZTask]:
        tasks = []
        for J in [1.0, 2.5, 4.0]:
            for gd, gr in [(0.005, 0.0025), (0.025, 0.0125)]:
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
    print("TRAINING MAML WITH PULSED ZZ COUPLING (FAST)")
    print(f"J_target: {J_TARGET_RANGE}, Gate time: {GATE_TIME}")
    print(f"n_segments: {N_SEGMENTS}, tasks_per_batch: {TASKS_PER_BATCH}")
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

    avg_J = (J_TARGET_RANGE[0] + J_TARGET_RANGE[1]) / 2
    avg_gd = (GAMMA_DEPH_RANGE[0] + GAMMA_DEPH_RANGE[1]) / 2
    avg_gr = (GAMMA_RELAX_RANGE[0] + GAMMA_RELAX_RANGE[1]) / 2
    avg_task = PulsedCZTask(avg_J, avg_gd, avg_gr, avg_gd, avg_gr)

    print(f"Training on: J={avg_J}, gamma_deph={avg_gd:.4f}")
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
    print("EVALUATION")
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
    print("TWO-QUBIT CZ WITH PULSED ZZ COUPLING (FAST VERSION)")
    print("=" * 70)
    print("ZZ coupling is now a CONTROL (not static Hamiltonian)")
    print(f"Policy outputs 7 controls: [X1, Y1, X2, Y2, Z1, Z2, ZZ]")
    print(f"J_target range: {J_TARGET_RANGE}, Gate time: {GATE_TIME}")
    sys.stdout.flush()

    maml_policy, maml_history = train_maml(device=device)
    fixed_policy = train_fixed_average(device=device)
    results = evaluate(maml_policy, fixed_policy, device=device)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    summary = {
        'maml_k0_mean': np.mean(results['maml_k0']),
        'maml_k_mean': np.mean(results['maml_k']),
        'fixed_k0_mean': np.mean(results['fixed_k0']),
        'fixed_k_mean': np.mean(results['fixed_k']),
    }

    print(f"\n{'Method':<20} {'K=0':>12} {f'K={ADAPT_K}':>12} {'Delta':>12}")
    print("-" * 60)
    print(f"{'MAML':<20} {summary['maml_k0_mean']:>12.4f} {summary['maml_k_mean']:>12.4f} "
          f"{summary['maml_k_mean']-summary['maml_k0_mean']:>+12.4f}")
    print(f"{'Fixed Average':<20} {summary['fixed_k0_mean']:>12.4f} {summary['fixed_k_mean']:>12.4f} "
          f"{summary['fixed_k_mean']-summary['fixed_k0_mean']:>+12.4f}")

    maml_adv = summary['maml_k_mean'] - summary['fixed_k_mean']
    print(f"\nMAML advantage at K={ADAPT_K}: {maml_adv:+.4f}")

    if maml_adv > 0.02:
        print("\n[SUCCESS] MAML WINS with pulsed coupling!")
    elif maml_adv > 0:
        print("\n[~] MAML slightly better.")
    else:
        print("\n[X] Fixed Average still competitive.")

    output = {
        'config': {'J_range': J_TARGET_RANGE, 'gate_time': GATE_TIME,
                   'gamma_deph_range': GAMMA_DEPH_RANGE, 'gamma_relax_range': GAMMA_RELAX_RANGE,
                   'n_segments': N_SEGMENTS, 'adapt_k': ADAPT_K},
        'summary': summary,
        'results': results,
        'runtime_minutes': (datetime.now() - start_time).total_seconds() / 60,
    }

    output_path = Path(__file__).parent / 'two_qubit_pulsed_fast_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to: {output_path}")
    print(f"Runtime: {output['runtime_minutes']:.1f} min")
    print(f"Finished: {datetime.now()}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
