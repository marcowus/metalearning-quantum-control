"""
Two-Qubit CZ Gate MAML Experiment

This script demonstrates MAML for quantum control on a 2-qubit system implementing
a Controlled-Z (CZ) gate under varying noise conditions.

CZ gate: |00⟩ ==> |00⟩, |01⟩ --> 01⟩, |10⟩ --> |10⟩, |11⟩ -- >   -|11⟩
© 2025 The MITRE Corporation, All Rights Reserved  
"""

import os
import sys
import argparse
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import curve_fit
from scipy.linalg import expm
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)

# Two-Qubit Quantum Operators 
# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
sigma_p = np.array([[0, 1], [0, 0]], dtype=np.complex128) 
sigma_m = np.array([[0, 0], [1, 0]], dtype=np.complex128)   
I_2 = np.eye(2, dtype=np.complex128)
I_4 = np.eye(4, dtype=np.complex128)

# Two-qubit operators via tensor products
# Qubit 1 operators 
X1 = np.kron(sigma_x, I_2)
Y1 = np.kron(sigma_y, I_2)
Z1 = np.kron(sigma_z, I_2)
Sp1 = np.kron(sigma_p, I_2)
Sm1 = np.kron(sigma_m, I_2)

# Qubit 2 operators  
X2 = np.kron(I_2, sigma_x)
Y2 = np.kron(I_2, sigma_y)
Z2 = np.kron(I_2, sigma_z)
Sp2 = np.kron(I_2, sigma_p)
Sm2 = np.kron(I_2, sigma_m)

# Two-qubit coupling
ZZ = np.kron(sigma_z, sigma_z)
XX = np.kron(sigma_x, sigma_x)
YY = np.kron(sigma_y, sigma_y)

# CZ gate
CZ_GATE = np.diag([1, 1, 1, -1]).astype(np.complex128)

# Ideal gate time for CZ with ZZ coupling: T = pi/(2*J_coupling)
# For J=2.0: T_ideal = pi/4 ≈ 0.785
J_COUPLING = 2.0  # Standard coupling
CZ_IDEAL_GATE_TIME = np.pi / 4  # T = pi/4 for J=2.0   
 
ket_00 = np.array([1, 0, 0, 0], dtype=np.complex128)
ket_01 = np.array([0, 1, 0, 0], dtype=np.complex128)
ket_10 = np.array([0, 0, 1, 0], dtype=np.complex128)
ket_11 = np.array([0, 0, 0, 1], dtype=np.complex128)


# Two-Qubit Task Distribution 
@dataclass
class TwoQubitTaskParams:
    """Parameters for a two-qubit noise task."""
    gamma_deph_1: float  # Dephasing rate for qubit 1
    gamma_relax_1: float  # Relaxation rate for qubit 1
    gamma_deph_2: float  # Dephasing rate for qubit 2
    gamma_relax_2: float  # Relaxation rate for qubit 2

    def to_array(self, normalized: bool = True) -> np.ndarray:
        """Convert to normalized feature array."""
        if normalized: 
            return np.array([
                self.gamma_deph_1 / 0.1,
                self.gamma_relax_1 / 0.05,
                self.gamma_deph_2 / 0.1,
                self.gamma_relax_2 / 0.05,
            ])
        return np.array([
            self.gamma_deph_1, self.gamma_relax_1,
            self.gamma_deph_2, self.gamma_relax_2
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray, normalized: bool = True):
        if normalized:
            return cls(
                gamma_deph_1=arr[0] * 0.1,
                gamma_relax_1=arr[1] * 0.05,
                gamma_deph_2=arr[2] * 0.1,
                gamma_relax_2=arr[3] * 0.05,
            )
        return cls(*arr)


class TwoQubitTaskDistribution:
    def __init__(
        self,
        gamma_deph_range: Tuple[float, float] = (0.0001, 0.001),  
        gamma_relax_range: Tuple[float, float] = (0.00005, 0.0005),
        correlated: bool = True,  
        diversity_scale: float = 1.0,
    ):
        self.gamma_deph_range = gamma_deph_range
        self.gamma_relax_range = gamma_relax_range
        self.correlated = correlated
        self.diversity_scale = diversity_scale

        # Compute effective ranges based on diversity scale
        deph_center = (gamma_deph_range[0] + gamma_deph_range[1]) / 2
        deph_width = (gamma_deph_range[1] - gamma_deph_range[0]) / 2 * diversity_scale
        self.effective_deph_range = (
            max(0.001, deph_center - deph_width),
            deph_center + deph_width
        )

        relax_center = (gamma_relax_range[0] + gamma_relax_range[1]) / 2
        relax_width = (gamma_relax_range[1] - gamma_relax_range[0]) / 2 * diversity_scale
        self.effective_relax_range = (
            max(0.001, relax_center - relax_width),
            relax_center + relax_width
        )

    def sample(self, n_tasks: int = 1) -> List[TwoQubitTaskParams]:
        """Sample noise parameters for n_tasks."""
        tasks = []
        for _ in range(n_tasks):
            gamma_deph_1 = np.random.uniform(*self.effective_deph_range)
            gamma_relax_1 = np.random.uniform(*self.effective_relax_range)

            if self.correlated: 
                gamma_deph_2 = gamma_deph_1 * np.random.uniform(0.8, 1.2)
                gamma_relax_2 = gamma_relax_1 * np.random.uniform(0.8, 1.2)
            else: 
                gamma_deph_2 = np.random.uniform(*self.effective_deph_range)
                gamma_relax_2 = np.random.uniform(*self.effective_relax_range)

            tasks.append(TwoQubitTaskParams(
                gamma_deph_1=gamma_deph_1,
                gamma_relax_1=gamma_relax_1,
                gamma_deph_2=gamma_deph_2,
                gamma_relax_2=gamma_relax_2,
            ))
        return tasks

class TwoQubitLindbladSimulator:
    """
    Lindblad simulation for two qubit case...  
    """

    def __init__(
        self,
        H0: torch.Tensor,  # Static Hamiltonian (4x4)
        H_controls: List[torch.Tensor],  # Control Hamiltonians
        L_operators: List[torch.Tensor],  # Lindblad operators
        gamma_rates: torch.Tensor,  # Dissipation rates
        dt: float = 0.01,
        device: str = 'cpu',
    ):
        self.device = device
        self.H0 = H0.to(device)
        self.H_controls = [H.to(device) for H in H_controls]
        self.L_operators = [L.to(device) for L in L_operators]
        self.gamma_rates = gamma_rates.to(device)
        self.dt = dt
        self.dim = 4  # 2-qubit Hilbert space

    def _lindbladian(self, rho: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Compute Lindbladian superoperator: L(rho) = -i[H, rho] + dissipator.""" 
        comm = -1j * (H @ rho - rho @ H)  
        dissipator = torch.zeros_like(rho)
        for k, (L, gamma) in enumerate(zip(self.L_operators, self.gamma_rates)):
            L_dag = L.conj().T
            L_dag_L = L_dag @ L
            dissipator += gamma * (
                L @ rho @ L_dag
                - 0.5 * (L_dag_L @ rho + rho @ L_dag_L)
            ) 
        return comm + dissipator

    def forward(
        self,
        rho0: torch.Tensor,
        control_sequence: torch.Tensor,   
        T: float,
    ) -> torch.Tensor:
        """
        Evolve initial state under piecewise-constant controls.

        Args:
            rho0: Initial density matrix (4x4)
            control_sequence: Control amplitudes for each segment
            T: Total evolution time

        Returns:
            Final density matrix
        """
        n_segments = control_sequence.shape[0]
        segment_duration = T / n_segments
        n_steps_per_segment = max(1, int(segment_duration / self.dt))
        dt = segment_duration / n_steps_per_segment

        rho = rho0.clone()

        for seg in range(n_segments): 
            H = self.H0.clone()
            for c, H_c in enumerate(self.H_controls):
                H = H + control_sequence[seg, c] * H_c
 
            for _ in range(n_steps_per_segment):
                k1 = self._lindbladian(rho, H)
                k2 = self._lindbladian(rho + 0.5 * dt * k1, H)
                k3 = self._lindbladian(rho + 0.5 * dt * k2, H)
                k4 = self._lindbladian(rho + dt * k3, H)
                rho = rho + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

        return rho


def create_two_qubit_simulator(
    task_params: TwoQubitTaskParams,
    J_coupling: float = J_COUPLING,  # ZZ coupling strength (uses global)
    device: str = 'cpu',
    include_z_controls: bool = True,  # Add Z controls for phase control
) -> TwoQubitLindbladSimulator:
    """Create a two-qubit Lindblad simulator with given noise parameters.

    For CZ gate with H0 = J*ZZ:
    - The ZZ term naturally gives a phase to |11⟩ vs other computational states
    - X/Y drives allow single-qubit rotations for preparation/correction
    - Z drives allow dynamic phase corrections  
    """

    # Static Hamiltonian: ZZ coupling for entanglement  
    H0 = torch.tensor(J_coupling * ZZ, dtype=torch.complex64, device=device)

    # Control Hamiltonians: X, Y, and optionally Z drives on each qubit
    H_controls = [
        torch.tensor(X1, dtype=torch.complex64, device=device),  # u_x1
        torch.tensor(Y1, dtype=torch.complex64, device=device),  # u_y1
        torch.tensor(X2, dtype=torch.complex64, device=device),  # u_x2
        torch.tensor(Y2, dtype=torch.complex64, device=device),  # u_y2
    ]

    if include_z_controls: 
        H_controls.extend([
            torch.tensor(Z1, dtype=torch.complex64, device=device),  # u_z1
            torch.tensor(Z2, dtype=torch.complex64, device=device),  # u_z2
        ])

    L_deph_1 = torch.tensor(Z1, dtype=torch.complex64, device=device)
    L_relax_1 = torch.tensor(Sm1, dtype=torch.complex64, device=device)

    L_deph_2 = torch.tensor(Z2, dtype=torch.complex64, device=device)
    L_relax_2 = torch.tensor(Sm2, dtype=torch.complex64, device=device)

    L_operators = [L_deph_1, L_relax_1, L_deph_2, L_relax_2]

    gamma_rates = torch.tensor([
        task_params.gamma_deph_1 / 2,  # Pure dephasing (factor of 1/2 for Z operator)
        task_params.gamma_relax_1,
        task_params.gamma_deph_2 / 2,  # Pure dephasing
        task_params.gamma_relax_2,
    ], dtype=torch.float32, device=device)

    return TwoQubitLindbladSimulator(
        H0=H0,
        H_controls=H_controls,
        L_operators=L_operators,
        gamma_rates=gamma_rates,
        device=device,
    )


ket_0 = np.array([1, 0], dtype=np.complex128)
ket_1 = np.array([0, 1], dtype=np.complex128)
ket_p = (ket_0 + ket_1) / np.sqrt(2)   # |+⟩
ket_m = (ket_0 - ket_1) / np.sqrt(2)   # |−⟩
ket_pi = (ket_0 + 1j * ket_1) / np.sqrt(2)  # |+i⟩
ket_mi = (ket_0 - 1j * ket_1) / np.sqrt(2)  # |−i⟩


def average_gate_fidelity_cz(
    simulator: TwoQubitLindbladSimulator,
    control_sequence: torch.Tensor,
    T: float,
    device: str = 'cpu',
) -> torch.Tensor:
    """
    Compute average gate fidelity for CZ gate using 12 input states.
    """
    # Build 12 input states
    input_states = [
        np.kron(ket_p, ket_p),   # |++⟩
        np.kron(ket_p, ket_m),   # |+−⟩
        np.kron(ket_m, ket_p),   # |−+⟩
        np.kron(ket_m, ket_m),   # |−−⟩
        np.kron(ket_pi, ket_pi), # |+i,+i⟩
        np.kron(ket_pi, ket_mi), # |+i,−i⟩
        np.kron(ket_1, ket_p),   # |1+⟩
        np.kron(ket_1, ket_m),   # |1−⟩
        np.kron(ket_p, ket_1),   # |+1⟩
        np.kron(ket_m, ket_1),   # |−1⟩
        np.kron(ket_0, ket_0),   # |00⟩
        np.kron(ket_1, ket_1),   # |11⟩
    ]


    target_states = [CZ_GATE @ state for state in input_states]

    total_fidelity = torch.tensor(0.0, device=device)

    for psi, psi_target in zip(input_states, target_states):

        psi_t = torch.tensor(psi, dtype=torch.complex64, device=device)
        rho0 = torch.outer(psi_t, psi_t.conj())

        # Evolve
        rho_final = simulator.forward(rho0, control_sequence, T)

        # Target state
        psi_target_t = torch.tensor(psi_target, dtype=torch.complex64, device=device)

        # State fidelity: ⟨psi_target|rho|psi_target⟩
        fidelity = torch.real(psi_target_t.conj() @ rho_final @ psi_target_t)
        total_fidelity = total_fidelity + fidelity

    # Average over 12 states
    return total_fidelity / 12.0


# Policy Network

class TwoQubitCZPolicy(nn.Module):
    """
    Task-conditioned MLP policy that outputs control pulse sequence. 

    Input: task_features tensor (4-dim: gamma_deph_1, gamma_relax_1, gamma_deph_2, gamma_relax_2)
    Output: Control sequence (n_segments x n_controls) for [u_x1, u_y1, u_x2, u_y2, u_z1, u_z2]
    """

    def __init__(
        self,
        task_feature_dim: int = 4,  # gamma_deph_1, gamma_relax_1, gamma_deph_2, gamma_relax_2
        hidden_dim: int = 256,  # Increased for 2-qubit complexity
        n_hidden_layers: int = 4,  # Deeper network
        n_segments: int = 20,  # Reduced for faster training
        n_controls: int = 6,  # X1, Y1, X2, Y2, Z1, Z2  
        output_scale: float = 1.0,
    ):
        super().__init__()
        self.task_feature_dim = task_feature_dim
        self.n_segments = n_segments
        self.n_controls = n_controls
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.output_dim = n_segments * n_controls
        self.output_scale = output_scale
 
        layers = []

        # Input layer with layer normalization for stability
        layers.append(nn.Linear(task_feature_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Tanh())

        # Hidden layers with residual-like structure
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Tanh())

        # Output layer
        layers.append(nn.Linear(hidden_dim, self.output_dim))

        self.network = nn.Sequential(*layers)
 
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == self.output_dim:
                    nn.init.orthogonal_(m.weight, gain=0.1)
                else:
                    nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, task_features: torch.Tensor) -> torch.Tensor:
        """
        Generate control sequence from task features.

        Args:
            task_features: (task_feature_dim,) normalized gamma rates

        Returns:
            control_sequence: (n_segments, n_controls) pulse amplitudes
        """
        output = self.network(task_features)   
        output = output.view(self.n_segments, self.n_controls)

        # Apply tanh to bound controls to [-π, π]
        controls = torch.tanh(output) * np.pi * self.output_scale
        return controls


# MAML Training 
def compute_loss(
    policy: TwoQubitCZPolicy,
    task_params: TwoQubitTaskParams,
    T: float = None,  # Gate time (default: CZ_IDEAL_GATE_TIME = pi/4 for J=2.0)
    device: str = 'cpu',
    include_z_controls: bool = True,
) -> torch.Tensor:
    """Compute loss = 1 - fidelity for a given task."""
    if T is None:
        T = CZ_IDEAL_GATE_TIME  # Use the correct gate time for CZ

    # Create simulator for this task
    simulator = create_two_qubit_simulator(
        task_params, device=device, include_z_controls=include_z_controls
    )

    # Get normalized task features for policy input
    task_features = torch.tensor(
        task_params.to_array(normalized=True),
        dtype=torch.float32,
        device=device,
    )

    # Get control sequence from policy with task features
    control_sequence = policy(task_features)

    # Compute fidelity
    fidelity = average_gate_fidelity_cz(simulator, control_sequence, T, device)

    return 1.0 - fidelity


def maml_inner_loop(
    policy: TwoQubitCZPolicy,
    task_params: TwoQubitTaskParams,
    n_steps: int,
    inner_lr: float,
    T: float = None,  # Gate time (default: CZ_IDEAL_GATE_TIME)
    device: str = 'cpu',
    include_z_controls: bool = True,
) -> Tuple[TwoQubitCZPolicy, List[float]]:
    """
    Perform MAML inner loop adaptation.

    Returns:
        Adapted policy (with cloned parameters)
        List of losses during adaptation
    """
    if T is None:
        T = CZ_IDEAL_GATE_TIME

    # Clone policy for adaptation
    adapted_policy = TwoQubitCZPolicy(
        task_feature_dim=policy.task_feature_dim,
        hidden_dim=policy.hidden_dim,
        n_hidden_layers=policy.n_hidden_layers,
        n_segments=policy.n_segments,
        n_controls=policy.n_controls,
    ).to(device)
    adapted_policy.load_state_dict(policy.state_dict())

    losses = []

    for _ in range(n_steps):
        loss = compute_loss(adapted_policy, task_params, T, device, include_z_controls)
        losses.append(loss.item())

        # Manual gradient step (FOMAML)
        grads = torch.autograd.grad(loss, adapted_policy.parameters())
        with torch.no_grad():
            for param, grad in zip(adapted_policy.parameters(), grads):
                # Gradient clipping for stability
                grad_clipped = torch.clamp(grad, -1.0, 1.0)
                param.sub_(inner_lr * grad_clipped)

    return adapted_policy, losses


def fomaml_meta_step(
    policy: TwoQubitCZPolicy,
    task_batch: List[TwoQubitTaskParams],
    inner_steps: int,
    inner_lr: float,
    meta_lr: float,
    optimizer: torch.optim.Optimizer,
    T: float = None,  # Gate time (default: CZ_IDEAL_GATE_TIME)
    device: str = 'cpu',
    include_z_controls: bool = True,
) -> float:
    """
    Perform one FOMAML meta-update step.

    Returns:
        Mean meta-loss across tasks
    """
    if T is None:
        T = CZ_IDEAL_GATE_TIME

    optimizer.zero_grad()

    # Accumulate gradients across tasks
    accumulated_grads = [torch.zeros_like(p) for p in policy.parameters()]
    total_loss = 0.0

    for task_params in task_batch:
        # Inner loop adaptation
        adapted_policy, _ = maml_inner_loop(
            policy, task_params, inner_steps, inner_lr, T, device, include_z_controls
        )

        # Compute post-adaptation loss
        post_loss = compute_loss(adapted_policy, task_params, T, device, include_z_controls)
        total_loss += post_loss.item()

        # Compute gradients w.r.t. adapted policy parameters
        grads = torch.autograd.grad(post_loss, adapted_policy.parameters())

        # FOMAML: Accumulate these gradients for the meta-policy
        for i, grad in enumerate(grads):
            accumulated_grads[i] += grad

    # Average gradients across tasks
    n_tasks = len(task_batch)
    for i in range(len(accumulated_grads)):
        accumulated_grads[i] /= n_tasks

    # Copy accumulated gradients to meta-policy
    for param, grad in zip(policy.parameters(), accumulated_grads):
        param.grad = grad.clone()

    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)

    optimizer.step()

    return total_loss / n_tasks


def train_maml(
    n_iterations: int = 1000,  # More iterations for convergence
    n_tasks_per_batch: int = 4,  # Reduced batch for faster training
    inner_steps: int = 3,  # Reduced for faster training
    inner_lr: float = 0.05,  # Higher LR for better adaptation
    meta_lr: float = 0.001,  # Meta learning rate
    T: float = None,  # Gate time (default: CZ_IDEAL_GATE_TIME = pi/4)
    val_interval: int = 20,
    save_dir: str = None,
    device: str = 'cpu',
    include_z_controls: bool = True,
) -> Tuple[TwoQubitCZPolicy, dict]:  
    if T is None:
        T = CZ_IDEAL_GATE_TIME
 
    # Initialize task-conditioned policy with improved architecture
    n_controls = 6 if include_z_controls else 4
    policy = TwoQubitCZPolicy(
        task_feature_dim=4,  # gamma_deph_1, gamma_relax_1, gamma_deph_2, gamma_relax_2
        hidden_dim=256,  # Larger network for 2-qubit complexity
        n_hidden_layers=4,  # Deeper network
        n_segments=30,  # Reduced for faster training
        n_controls=n_controls,
    ).to(device)

    # Use AdamW with weight decay for regularization
    optimizer = torch.optim.AdamW(policy.parameters(), lr=meta_lr, weight_decay=1e-4)

    # Task distribution with wider noise range for robustness
    task_dist = TwoQubitTaskDistribution(
        gamma_deph_range=(0.001, 0.01),   
        gamma_relax_range=(0.0005, 0.005),
    )

    # Learning rate scheduler for better convergence 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iterations, eta_min=1e-5)

    # Training history
    history = {
        'iterations': [],
        'meta_loss': [],
        'val_fidelity': [],
        'val_pre_adapt': [],
        'val_post_adapt': [],
    }

    best_fidelity = 0.0

    for iteration in range(n_iterations):  
        task_batch = task_dist.sample(n_tasks_per_batch)

        # FOMAML update
        meta_loss = fomaml_meta_step(
            policy, task_batch, inner_steps, inner_lr, meta_lr, optimizer, T, device, include_z_controls
        )

        history['iterations'].append(iteration)
        history['meta_loss'].append(meta_loss)

        # Step scheduler
        scheduler.step()

        # Validation
        if iteration % val_interval == 0 or iteration == n_iterations - 1:
            val_tasks = task_dist.sample(10)
            pre_fids, post_fids = [], []

            for task in val_tasks:
                # Pre-adaptation fidelity
                with torch.no_grad():
                    pre_loss = compute_loss(policy, task, T, device, include_z_controls)
                    pre_fids.append(1.0 - pre_loss.item())

                # Post-adaptation fidelity
                adapted_policy, _ = maml_inner_loop(
                    policy, task, inner_steps, inner_lr, T, device, include_z_controls
                )
                with torch.no_grad():
                    post_loss = compute_loss(adapted_policy, task, T, device, include_z_controls)
                    post_fids.append(1.0 - post_loss.item())

            mean_pre = np.mean(pre_fids)
            mean_post = np.mean(post_fids)
            gap = mean_post - mean_pre

            history['val_fidelity'].append(mean_post)
            history['val_pre_adapt'].append(mean_pre)
            history['val_post_adapt'].append(mean_post)

            current_lr = scheduler.get_last_lr()[0]
            print(f"Iter {iteration:4d} | Meta Loss: {meta_loss:.4f} | "
                  f"Val Pre: {mean_pre*100:.1f}% | Val Post: {mean_post*100:.1f}% | "
                  f"Gap: {gap*100:+.1f}% | LR: {current_lr:.2e}")

            # Save best
            if mean_post > best_fidelity and save_dir:
                best_fidelity = mean_post
                os.makedirs(save_dir, exist_ok=True)
                torch.save({
                    'policy_state_dict': policy.state_dict(),
                    'iteration': iteration,
                    'fidelity': mean_post,
                    'history': history,
                    # Architecture info for loading
                    'task_feature_dim': policy.task_feature_dim,
                    'hidden_dim': policy.hidden_dim,
                    'n_hidden_layers': policy.n_hidden_layers,
                    'n_segments': policy.n_segments,
                    'n_controls': policy.n_controls,
                    'gate_time': T,
                    'include_z_controls': include_z_controls,
                }, os.path.join(save_dir, 'maml_cz_best.pt'))

    # Save final
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'policy_state_dict': policy.state_dict(),
            'iteration': n_iterations - 1,
            'history': history,
            # Architecture info for loading
            'task_feature_dim': policy.task_feature_dim,
            'hidden_dim': policy.hidden_dim,
            'n_hidden_layers': policy.n_hidden_layers,
            'n_segments': policy.n_segments,
            'n_controls': policy.n_controls,
            'gate_time': T,
            'include_z_controls': include_z_controls,
        }, os.path.join(save_dir, 'maml_cz_final.pt'))

        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best fidelity: {best_fidelity*100:.2f}%")

    return policy, history



# Analysis and Figure Generation
 

def exponential_saturation(K, c, beta):
    """G_K = c * (1 - exp(-beta * K))"""
    return c * (1 - np.exp(-beta * K))


def generate_adaptation_gap_figure(
    policy: TwoQubitCZPolicy,
    n_tasks: int = 30,
    max_K: int = 30,
    inner_lr: float = 0.01,
    T: float = None,
    output_name: str = 'adaptation_gap_cz',
    device: str = 'cpu',
    include_z_controls: bool = True,
): 
    if T is None:
        T = CZ_IDEAL_GATE_TIME

  
    print("\nPanel (a): Computing G_K vs K...")
    task_dist = TwoQubitTaskDistribution(
        gamma_deph_range=(0.001, 0.01),
        gamma_relax_range=(0.0005, 0.005),
    )
    tasks = task_dist.sample(n_tasks)

    K_values = list(range(0, max_K + 1, max(1, max_K // 15)))
    G_K_means = []
    G_K_stds = []

    for K in K_values:
        gaps = []
        for task in tqdm(tasks, desc=f"K={K:2d}", leave=False):
            # Pre-adaptation fidelity
            with torch.no_grad():
                pre_loss = compute_loss(policy, task, T, device, include_z_controls)
                pre_fid = 1.0 - pre_loss.item()

            if K == 0:
                gaps.append(0.0)
            else:
                # Adapt and get post-adaptation fidelity
                adapted, _ = maml_inner_loop(policy, task, K, inner_lr, T, device, include_z_controls)
                with torch.no_grad():
                    post_loss = compute_loss(adapted, task, T, device, include_z_controls)
                    post_fid = 1.0 - post_loss.item()
                gaps.append(post_fid - pre_fid)

        G_K_means.append(np.mean(gaps))
        G_K_stds.append(np.std(gaps))
        print(f"  K={K:2d}: G_K = {np.mean(gaps):.4f} +/- {np.std(gaps):.4f}")

    # Fit exponential
    try:
        K_arr = np.array(K_values)
        G_arr = np.array(G_K_means)
        popt, _ = curve_fit(exponential_saturation, K_arr, G_arr, p0=[0.01, 0.1], maxfev=5000)
        c_fit, beta_fit = popt
        G_fit = exponential_saturation(K_arr, c_fit, beta_fit)
        ss_res = np.sum((G_arr - G_fit) ** 2)
        ss_tot = np.sum((G_arr - np.mean(G_arr)) ** 2)
        R2_a = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f"\n  Fit: c={c_fit:.4f}, beta={beta_fit:.4f}, R²={R2_a:.4f}")
    except Exception as e:
        print(f"  Fit failed: {e}")
        c_fit, beta_fit, R2_a = 0, 0, 0

    # Panel (b) 
    print("\nPanel (b): Computing G_∞ vs task diversity...")
    diversity_scales = np.linspace(0.2, 2.0, 8)
    G_inf_values = []
    variances = []
    K_adapt = min(10, max_K)

    for div_scale in diversity_scales:
        div_dist = TwoQubitTaskDistribution(
            gamma_deph_range=(0.001, 0.01),
            gamma_relax_range=(0.0005, 0.005),
            diversity_scale=div_scale
        )
        div_tasks = div_dist.sample(n_tasks // 2)

        # Compute variance of task parameters
        all_params = np.array([t.to_array(normalized=False) for t in div_tasks])
        task_var = np.var(all_params[:, 0]) + np.var(all_params[:, 1])  # Just qubit 1
        variances.append(task_var)

        gaps = []
        for task in div_tasks:
            with torch.no_grad():
                pre_loss = compute_loss(policy, task, T, device, include_z_controls)
                pre_fid = 1.0 - pre_loss.item()

            adapted, _ = maml_inner_loop(policy, task, K_adapt, inner_lr, T, device, include_z_controls)
            with torch.no_grad():
                post_loss = compute_loss(adapted, task, T, device, include_z_controls)
                post_fid = 1.0 - post_loss.item()

            gaps.append(post_fid - pre_fid)

        G_inf = np.mean(gaps)
        G_inf_values.append(G_inf)
        print(f"  diversity={div_scale:.2f}: var={task_var:.4f}, G_{K_adapt}={G_inf:.4f}")

    # Linear fit for panel (b)
    var_arr = np.array(variances)
    G_inf_arr = np.array(G_inf_values)
    if len(var_arr) > 1 and np.std(var_arr) > 0:
        slope, intercept = np.polyfit(var_arr, G_inf_arr, 1)
        G_inf_fit = slope * var_arr + intercept
        ss_res_b = np.sum((G_inf_arr - G_inf_fit) ** 2)
        ss_tot_b = np.sum((G_inf_arr - np.mean(G_inf_arr)) ** 2)
        R2_b = 1 - ss_res_b / ss_tot_b if ss_tot_b > 0 else 0
        print(f"\n  Linear fit: slope={slope:.4f}, intercept={intercept:.4f}, R²={R2_b:.4f}")
    else:
        slope, intercept, R2_b = 0, 0, 0

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel (a)
    ax = axes[0]
    ax.errorbar(K_values, G_K_means, yerr=G_K_stds, fmt='o', capsize=3,
                label='Data', color='#3498db')
    if R2_a > 0:
        K_smooth = np.linspace(0, max_K, 100)
        ax.plot(K_smooth, exponential_saturation(K_smooth, c_fit, beta_fit),
                '--', color='#e74c3c', linewidth=2,
                label=f'$G_K = c(1-e^{{-\\beta K}})$')
    ax.set_xlabel('Adaptation Steps (K)', fontsize=12)
    ax.set_ylabel('Adaptation Gap $G_K$', fontsize=12)
    ax.set_title('(a) Adaptation Gap vs Steps', fontsize=14)
    ax.legend(loc='lower right')
    ax.text(0.05, 0.95, f'$R^2 = {R2_a:.4f}$\n$\\beta = {beta_fit:.4f}$',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.grid(True, alpha=0.3)

    # Panel (b)
    ax = axes[1]
    ax.scatter(variances, G_inf_values, s=80, c='#2ecc71', edgecolors='black', zorder=3)
    if R2_b > 0:
        var_smooth = np.linspace(min(variances), max(variances), 100)
        ax.plot(var_smooth, slope * var_smooth + intercept,
                '--', color='#e74c3c', linewidth=2, label='Linear fit')
    ax.set_xlabel('Task Variance $\\sigma_\\tau^2$', fontsize=12)
    ax.set_ylabel(f'$G_{{\\infty}}$ (K={K_adapt})', fontsize=12)
    ax.set_title('(b) Adaptation Gap vs Task Diversity', fontsize=14)
    ax.text(0.05, 0.95, f'$R^2 = {R2_b:.4f}$',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, f'{output_name}.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(script_dir, f'{output_name}.pdf'), bbox_inches='tight')

    # Save data
    data = {
        'panel_a': {
            'K_values': K_values,
            'G_K_means': G_K_means,
            'G_K_stds': G_K_stds,
            'c': float(c_fit),
            'beta': float(beta_fit),
            'R2': float(R2_a),
        },
        'panel_b': {
            'variances': list(variances),
            'G_inf_values': list(G_inf_values),
            'slope': float(slope),
            'intercept': float(intercept),
            'R2': float(R2_b),
        }
    }
    with open(os.path.join(script_dir, f'{output_name}_data.json'), 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nFigure saved to: {os.path.join(script_dir, output_name)}.png")

    return data


def generate_adaptation_dynamics_figure(
    policy: TwoQubitCZPolicy,
    n_tasks: int = 30,
    max_K: int = 30,
    inner_lr: float = 0.01,
    T: float = None,
    output_name: str = 'adaptation_dynamics_cz',
    device: str = 'cpu',
    include_z_controls: bool = True,
):
    """Generate Fig 4 style adaptation dynamics figure for CZ gate."""
    if T is None:
        T = CZ_IDEAL_GATE_TIME

    print("\n" + "=" * 60)
    print("Generating Adaptation Dynamics Figure for CZ Gate")
    print("=" * 60)
    print(f"Gate time T = {T:.4f}")

    task_dist = TwoQubitTaskDistribution(
        gamma_deph_range=(0.001, 0.01),
        gamma_relax_range=(0.0005, 0.005),
    )

    # Panel (a): Loss curves during adaptation
    print("\nPanel (a): Adaptation dynamics...")
    K_values = list(range(0, max_K + 1))

    # MAML policy
    maml_fids = {k: [] for k in K_values}
    tasks = task_dist.sample(n_tasks)

    for task in tqdm(tasks, desc="MAML"):
        for K in K_values:
            if K == 0:
                with torch.no_grad():
                    loss = compute_loss(policy, task, T, device, include_z_controls)
                    maml_fids[K].append(1.0 - loss.item())
            else:
                adapted, _ = maml_inner_loop(policy, task, K, inner_lr, T, device, include_z_controls)
                with torch.no_grad():
                    loss = compute_loss(adapted, task, T, device, include_z_controls)
                    maml_fids[K].append(1.0 - loss.item())

    maml_means = [np.mean(maml_fids[k]) for k in K_values]
    maml_stds = [np.std(maml_fids[k]) for k in K_values]

    # Random initialization baseline
    n_controls = 6 if include_z_controls else 4
    random_policy = TwoQubitCZPolicy(
        n_controls=n_controls,
        n_segments=policy.n_segments,
        hidden_dim=policy.hidden_dim,
        n_hidden_layers=policy.n_hidden_layers,
    ).to(device)
    random_fids = {k: [] for k in K_values}

    for task in tqdm(tasks, desc="Random"):
        for K in K_values:
            if K == 0:
                with torch.no_grad():
                    loss = compute_loss(random_policy, task, T, device, include_z_controls)
                    random_fids[K].append(1.0 - loss.item())
            else:
                adapted, _ = maml_inner_loop(random_policy, task, K, inner_lr, T, device, include_z_controls)
                with torch.no_grad():
                    loss = compute_loss(adapted, task, T, device, include_z_controls)
                    random_fids[K].append(1.0 - loss.item())

    random_means = [np.mean(random_fids[k]) for k in K_values]
    random_stds = [np.std(random_fids[k]) for k in K_values]

    print(f"  MAML: {maml_means[0]*100:.1f}% -> {maml_means[-1]*100:.1f}%")
    print(f"  Random: {random_means[0]*100:.1f}% -> {random_means[-1]*100:.1f}%")

    print("\nPanel (b): Fidelity distributions...")
    final_maml = maml_fids[max_K]
    final_random = random_fids[max_K]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel (a)
    ax = axes[0]
    ax.fill_between(K_values,
                    np.array(maml_means) - np.array(maml_stds),
                    np.array(maml_means) + np.array(maml_stds),
                    alpha=0.2, color='#2ecc71')
    ax.plot(K_values, maml_means, '-', color='#2ecc71', linewidth=2, label='MAML')

    ax.fill_between(K_values,
                    np.array(random_means) - np.array(random_stds),
                    np.array(random_means) + np.array(random_stds),
                    alpha=0.2, color='#e67e22')
    ax.plot(K_values, random_means, '-', color='#e67e22', linewidth=2, label='Random Init')

    ax.set_xlabel('Adaptation Steps (K)', fontsize=12)
    ax.set_ylabel('Average Fidelity', fontsize=12)
    ax.set_title('(a) Adaptation Dynamics', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # Panel (b)
    ax = axes[1]
    positions = [1, 2]
    data = [final_maml, final_random]
    colors = ['#2ecc71', '#e67e22']
    labels = ['MAML', 'Random']

    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f'Final Fidelity (K={max_K})', fontsize=12)
    ax.set_title('(b) Fidelity Distribution', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
 
    for i, (d, pos) in enumerate(zip(data, positions)):
        mean_val = np.mean(d)
        std_val = np.std(d)
        ax.annotate(f'{mean_val*100:.1f}%±{std_val*100:.1f}%',
                    xy=(pos, mean_val), xytext=(pos + 0.3, mean_val),
                    fontsize=9, ha='left', va='center')

    plt.tight_layout()

    # Save
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, f'{output_name}.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(script_dir, f'{output_name}.pdf'), bbox_inches='tight')

    print(f"\nFigure saved to: {os.path.join(script_dir, output_name)}.png")


def main():
    parser = argparse.ArgumentParser(description='Two-Qubit CZ Gate MAML Experiment')
    parser.add_argument('--train', action='store_true', help='Train MAML policy')
    parser.add_argument('--analyze', action='store_true', help='Generate figures')
    parser.add_argument('--n_iterations', type=int, default=1000, help='Training iterations')
    parser.add_argument('--n_tasks', type=int, default=30, help='Tasks for analysis')
    parser.add_argument('--max_K', type=int, default=30, help='Max adaptation steps')
    parser.add_argument('--inner_lr', type=float, default=0.05, help='Inner loop LR')
    parser.add_argument('--meta_lr', type=float, default=0.001, help='Meta learning rate')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save checkpoints')

    args = parser.parse_args()

    device = 'cpu'  # Use CPU for stability
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = args.save_dir or os.path.join(script_dir, 'checkpoints_cz')

    if args.train:
        policy, history = train_maml(
            n_iterations=args.n_iterations,
            n_tasks_per_batch=4,  
            inner_steps=3,  
            inner_lr=args.inner_lr, 
            meta_lr=args.meta_lr,  
            T=CZ_IDEAL_GATE_TIME,  
            val_interval=20,
            save_dir=save_dir,
            device=device,
            include_z_controls=True,  # Use Z controls for phase correction
        )

    if args.analyze:
        # Load checkpoint
        checkpoint_path = args.checkpoint or os.path.join(save_dir, 'maml_cz_best.pt')

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            print("Please train first with --train flag")
            return


        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Load with saved architecture
        n_controls = checkpoint.get('n_controls', 6)
        policy = TwoQubitCZPolicy(
            task_feature_dim=checkpoint.get('task_feature_dim', 4),
            hidden_dim=checkpoint.get('hidden_dim', 256),
            n_hidden_layers=checkpoint.get('n_hidden_layers', 4),
            n_segments=checkpoint.get('n_segments', 60),
            n_controls=n_controls,
        ).to(device)
        policy.load_state_dict(checkpoint['policy_state_dict'])

        T = checkpoint.get('gate_time', CZ_IDEAL_GATE_TIME)

        # Generate figures
        generate_adaptation_gap_figure(
            policy,
            n_tasks=args.n_tasks,
            max_K=args.max_K,
            inner_lr=args.inner_lr,
            T=T,
            output_name='adaptation_gap_cz',
            device=device,
        )

        generate_adaptation_dynamics_figure(
            policy,
            n_tasks=args.n_tasks,
            max_K=args.max_K,
            inner_lr=args.inner_lr,
            T=T,
            output_name='adaptation_dynamics_cz',
            device=device,
        )


if __name__ == '__main__':
    main()
