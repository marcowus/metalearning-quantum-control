"""
Test MAML vs Fixed Average on OUT-OF-DISTRIBUTION tasks.
Tasks with J values OUTSIDE the training range.
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
from dataclasses import dataclass

from two_qubit_cz_maml_fast import (
    X1, Y1, X2, Y2, Z1, Z2, ZZ, Sm1, Sm2,
    CZ_GATE, ket_0, ket_1, ket_p, ket_m,
)

torch.manual_seed(42)
np.random.seed(42)

# Training range
J_TRAIN_RANGE = (0.5, 10.0)
GAMMA_DEPH_RANGE = (0.0005, 0.05)
GAMMA_RELAX_RANGE = (0.00025, 0.025)
GATE_TIME = 1.0
N_SEGMENTS = 15
DT = 0.02
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
            # Clamp to training range for normalization
            J_clamped = np.clip(self.J_target, J_TRAIN_RANGE[0], J_TRAIN_RANGE[1])
            J_norm = (np.log(J_clamped) - np.log(J_TRAIN_RANGE[0])) / \
                     (np.log(J_TRAIN_RANGE[1]) - np.log(J_TRAIN_RANGE[0]))
            gd_norm = (np.log(self.gamma_deph_1 + 1e-6) - np.log(GAMMA_DEPH_RANGE[0])) / \
                      (np.log(GAMMA_DEPH_RANGE[1]) - np.log(GAMMA_DEPH_RANGE[0]))
            gr_norm = (np.log(self.gamma_relax_1 + 1e-6) - np.log(GAMMA_RELAX_RANGE[0])) / \
                      (np.log(GAMMA_RELAX_RANGE[1]) - np.log(GAMMA_RELAX_RANGE[0]))
            return np.array([J_norm, gd_norm, gr_norm, gd_norm, gr_norm])
        return np.array([self.J_target, self.gamma_deph_1, self.gamma_relax_1,
                        self.gamma_deph_2, self.gamma_relax_2])


class PulsedCouplingSimulator:
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
            torch.tensor(ZZ, dtype=torch.complex64, device=device),
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
        controls = torch.tanh(output) * torch.tensor(
            [np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, 12.0], device=output.device)
        return controls


def compute_loss(policy, task, device='cpu'):
    simulator = PulsedCouplingSimulator(task, device)
    task_features = torch.tensor(task.to_array(normalized=True), dtype=torch.float32, device=device)
    controls = policy(task_features)
    fidelity = compute_cz_fidelity(simulator, controls, GATE_TIME, device)
    return 1 - fidelity, fidelity


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


def train_maml(n_iterations=150, device='cpu'):
    print("Training MAML on J in [0.5, 10.0]...")
    policy = PulsedCouplingPolicy().to(device)
    optimizer = optim.AdamW(policy.parameters(), lr=0.002, weight_decay=1e-4)

    for iteration in range(n_iterations):
        tasks = [PulsedCZTask(
            np.exp(np.random.uniform(np.log(0.5), np.log(10.0))),
            np.exp(np.random.uniform(np.log(0.0005), np.log(0.05))),
            np.exp(np.random.uniform(np.log(0.00025), np.log(0.025))),
            np.exp(np.random.uniform(np.log(0.0005), np.log(0.05))),
            np.exp(np.random.uniform(np.log(0.00025), np.log(0.025)))
        ) for _ in range(2)]

        optimizer.zero_grad()
        total_loss = 0.0
        for task in tasks:
            adapted = maml_inner_loop(policy, task, 3, 0.05, device)
            loss, _ = compute_loss(adapted, task, device)
            total_loss += loss
        meta_loss = total_loss / len(tasks)
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        if iteration % 50 == 0:
            print(f"  Iter {iteration}: loss={meta_loss.item():.4f}")

    return policy


def train_fixed_average(n_iterations=300, device='cpu'):
    print("Training Fixed Average on J=2.24...")
    avg_J = np.sqrt(0.5 * 10.0)  # 2.24
    avg_gd = np.sqrt(0.0005 * 0.05)
    avg_gr = np.sqrt(0.00025 * 0.025)
    avg_task = PulsedCZTask(avg_J, avg_gd, avg_gr, avg_gd, avg_gr)

    policy = PulsedCouplingPolicy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=0.002)

    for i in range(n_iterations):
        optimizer.zero_grad()
        loss, fid = compute_loss(policy, avg_task, device)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"  Iter {i}: fid={fid.item():.4f}")

    return policy


def adapt_and_evaluate(policy, task, K=25, lr=0.002, device='cpu'):
    """Adapt and return K=0 and K=final fidelity."""
    with torch.no_grad():
        _, fid_k0 = compute_loss(policy, task, device)

    adapted = deepcopy(policy)
    adapted.train()
    opt = optim.Adam(adapted.parameters(), lr=lr)
    for _ in range(K):
        opt.zero_grad()
        loss, _ = compute_loss(adapted, task, device)
        loss.backward()
        opt.step()

    with torch.no_grad():
        _, fid_k = compute_loss(adapted, task, device)

    return fid_k0.item(), fid_k.item()


def main():
    print("=" * 70)
    print("TESTING OUT-OF-DISTRIBUTION TASKS")
    print("=" * 70)
    print(f"\nTraining range: J in [{J_TRAIN_RANGE[0]}, {J_TRAIN_RANGE[1]}]")
    print(f"Fixed Average trained on: J = {np.sqrt(0.5 * 10.0):.2f}")

    maml_policy = train_maml(device=device)
    fixed_policy = train_fixed_average(device=device)

    # Test on IN-DISTRIBUTION and OUT-OF-DISTRIBUTION tasks
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    gd, gr = 0.005, 0.0025  # Fixed noise

    test_scenarios = [
        ("IN-DIST (J=0.7)", 0.7),
        ("IN-DIST (J=2.2)", 2.2),
        ("IN-DIST (J=5.0)", 5.0),
        ("IN-DIST (J=9.0)", 9.0),
        ("OUT-DIST (J=0.2)", 0.2),   # Below training range
        ("OUT-DIST (J=0.3)", 0.3),   # Below training range
        ("OUT-DIST (J=15.0)", 15.0), # Above training range
        ("OUT-DIST (J=20.0)", 20.0), # Above training range
    ]

    print(f"\n{'Task':<22} {'MAML K=0':>10} {'MAML K=25':>10} {'Fixed K=0':>10} {'Fixed K=25':>10} {'Winner':>10}")
    print("-" * 82)

    for name, J in test_scenarios:
        task = PulsedCZTask(J, gd, gr, gd, gr)

        maml_k0, maml_k25 = adapt_and_evaluate(maml_policy, task, K=25, device=device)
        fixed_k0, fixed_k25 = adapt_and_evaluate(fixed_policy, task, K=25, device=device)

        winner = "MAML" if maml_k25 > fixed_k25 + 0.01 else ("Fixed" if fixed_k25 > maml_k25 + 0.01 else "Tie")

        print(f"{name:<22} {maml_k0:>10.4f} {maml_k25:>10.4f} {fixed_k0:>10.4f} {fixed_k25:>10.4f} {winner:>10}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("IN-DIST tasks: J values within training range [0.5, 10.0]")
    print("OUT-DIST tasks: J values OUTSIDE training range")
    print("\nKey question: Does MAML's meta-learned initialization help more")
    print("when adapting to truly novel tasks outside the training distribution?")


if __name__ == "__main__":
    main()
