"""
Figure 5: Two-Qubit CZ Gate with Pulsed Coupling

Multi-panel figure showing:
(a) Adaptation dynamics with exponential fit
(b-e) Adapted pulse sequences for different coupling strengths J
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
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline
from dataclasses import dataclass

from two_qubit_cz_maml_fast import (
    X1, Y1, X2, Y2, Z1, Z2, ZZ, Sm1, Sm2,
    CZ_GATE, ket_0, ket_1, ket_p, ket_m,
)

torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# PUBLICATION STYLE
# ============================================================================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Times']
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.size'] = 10
rcParams['axes.linewidth'] = 1.0
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 11
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'

# Colors
COLORS = {
    'data': '#0072B2',
    'fit': '#D55E00',
    'pulse_colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
}

# ============================================================================
# CONFIG
# ============================================================================
J_TARGET_RANGE = (0.5, 10.0)
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


def train_maml(n_iterations=5, device='cpu'):
    print("Training FOMAML...")
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


def get_adaptation_curve(policy, task, max_K=30, lr=0.002, device='cpu'):
    fidelities = []
    with torch.no_grad():
        _, fid = compute_loss(policy, task, device)
        fidelities.append(fid.item())

    adapted = deepcopy(policy)
    adapted.train()
    opt = optim.Adam(adapted.parameters(), lr=lr)

    for k in range(max_K):
        opt.zero_grad()
        loss, _ = compute_loss(adapted, task, device)
        loss.backward()
        opt.step()
        with torch.no_grad():
            _, fid = compute_loss(adapted, task, device)
            fidelities.append(fid.item())

    return fidelities


def adapt_policy(policy, task, K=20, lr=0.002, device='cpu'):
    """Adapt policy to task and return adapted policy."""
    adapted = deepcopy(policy)
    adapted.train()
    opt = optim.Adam(adapted.parameters(), lr=lr)

    for _ in range(K):
        opt.zero_grad()
        loss, _ = compute_loss(adapted, task, device)
        loss.backward()
        opt.step()

    return adapted


def get_pulse_sequence(policy, task, device='cpu'):
    task_features = torch.tensor(task.to_array(normalized=True), dtype=torch.float32, device=device)
    with torch.no_grad():
        controls = policy(task_features)
    return controls.cpu().numpy()


def smooth_pulse(time_points, pulse_values, n_smooth=200):
    """Smooth pulse using cubic spline interpolation."""
    t_extended = np.concatenate([[time_points[0] - 0.01], time_points, [time_points[-1] + 0.01]])
    p_extended = np.concatenate([[pulse_values[0]], pulse_values, [pulse_values[-1]]])
    try:
        spline = make_interp_spline(t_extended, p_extended, k=3)
        t_smooth = np.linspace(time_points[0], time_points[-1], n_smooth)
        p_smooth = spline(t_smooth)
        return t_smooth, p_smooth
    except:
        t_smooth = np.linspace(time_points[0], time_points[-1], n_smooth)
        p_smooth = np.interp(t_smooth, time_points, pulse_values)
        return t_smooth, p_smooth


def exponential_saturation(K, F_0, F_inf, tau):
    """Exponential saturation: F(K) = F_∞ - (F_∞ - F_0) * exp(-K/τ)"""
    return F_inf - (F_inf - F_0) * np.exp(-K / tau)


def main():
    print("=" * 70)
    print("GENERATING FIGURE 5: TWO-QUBIT CZ WITH PULSED COUPLING")
    print("=" * 70)

    # Train MAML
    maml_policy = train_maml(device=device)

    # Collect adaptation curves
    print("\nCollecting adaptation curves...")
    max_K = 5
    K_values = np.arange(max_K + 1)
    n_tasks = 2

    all_curves = []
    for i in range(n_tasks):
        task = PulsedCZTask(
            np.exp(np.random.uniform(np.log(0.5), np.log(10.0))),
            0.005, 0.0025, 0.005, 0.0025
        )
        curve = get_adaptation_curve(maml_policy, task, max_K, device=device)
        all_curves.append(curve)

    mean_curve = np.mean(all_curves, axis=0)
    std_curve = np.std(all_curves, axis=0)

    # Fit exponential saturation
    print("\nFitting exponential saturation...")
    p0 = [mean_curve[0], 0.99, 5.0]
    bounds = ([0, 0.8, 0.1], [1, 1.0, 50])
    popt, pcov = curve_fit(exponential_saturation, K_values, mean_curve, p0=p0, bounds=bounds)
    F_0_fit, F_inf_fit, tau_fit = popt
    perr = np.sqrt(np.diag(pcov))
    print(f"  F_0 = {F_0_fit:.3f} ± {perr[0]:.3f}")
    print(f"  F_∞ = {F_inf_fit:.3f} ± {perr[1]:.3f}")
    print(f"  τ = {tau_fit:.2f} ± {perr[2]:.2f}")

    K_smooth = np.linspace(0, max_K, 200)
    fit_curve = exponential_saturation(K_smooth, *popt)

    # Define test tasks for pulse visualization
    test_tasks = [
        PulsedCZTask(1.0, 0.005, 0.0025, 0.005, 0.0025),
        PulsedCZTask(3.0, 0.005, 0.0025, 0.005, 0.0025),
        PulsedCZTask(6.0, 0.005, 0.0025, 0.005, 0.0025),
        PulsedCZTask(9.0, 0.005, 0.0025, 0.005, 0.0025),
    ]
    task_J_values = [1.0, 3.0, 6.0, 9.0]

    # Get adapted policies and pulses
    print("\nAdapting policies for pulse visualization...")
    adapted_policies = []
    for task in test_tasks:
        adapted = adapt_policy(maml_policy, task, K=20, device=device)
        adapted_policies.append(adapted)

    time_points = np.linspace(0, GATE_TIME, N_SEGMENTS)
    control_labels = [r'$u_{X_1}$', r'$u_{Y_1}$', r'$u_{X_2}$', r'$u_{Y_2}$',
                      r'$u_{Z_1}$', r'$u_{Z_2}$', r'$u_{ZZ}$']

    # ========================================================================
    # CREATE FIGURE 5
    # ========================================================================
    print("\nGenerating Figure 5...")

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.2, 1], hspace=0.35, wspace=0.3)

    # -------------------------------------------------------------------------
    # Panel (a): Adaptation Dynamics - spans first row, first 2 columns
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, :2])

    # Plot individual curves (light)
    for curve in all_curves:
        ax_a.plot(K_values, curve, '-', color=COLORS['data'], alpha=0.12, linewidth=0.8)

    # Plot mean with shaded region
    ax_a.fill_between(K_values, mean_curve - std_curve, mean_curve + std_curve,
                      color=COLORS['data'], alpha=0.25)
    ax_a.plot(K_values, mean_curve, 'o', color=COLORS['data'], markersize=5,
              markeredgecolor='white', markeredgewidth=0.5, label='FOMAML')

    # Plot fit
    ax_a.plot(K_smooth, fit_curve, '-', color=COLORS['fit'], linewidth=2.5,
              label='Exponential fit')

    # Horizontal lines for F_0 and F_∞
    ax_a.axhline(y=F_0_fit, color='gray', linestyle=':', alpha=0.6, linewidth=1)
    ax_a.axhline(y=F_inf_fit, color='gray', linestyle=':', alpha=0.6, linewidth=1)

    # Annotations
    ax_a.text(max_K + 1, F_0_fit, r'$\mathcal{F}_0$', fontsize=10, va='center', ha='left')
    ax_a.text(max_K + 1, F_inf_fit, r'$\mathcal{F}_\infty$', fontsize=10, va='center', ha='left')

    # Vertical line at τ
    ax_a.axvline(x=tau_fit, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax_a.text(tau_fit, 0.27, r'$\tau$', fontsize=10, ha='center')

    # Adaptation gap arrow
    F_at_tau = exponential_saturation(tau_fit, *popt)
    ax_a.annotate('', xy=(tau_fit - 0.5, F_at_tau), xytext=(tau_fit - 0.5, F_0_fit),
                  arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))

    # Model equation box
    eq_text = r'$\mathcal{F}(K) = \mathcal{F}_\infty - (\mathcal{F}_\infty - \mathcal{F}_0)\, e^{-K/\tau}$'
    ax_a.text(0.98, 0.15, eq_text, transform=ax_a.transAxes, fontsize=10,
              ha='right', va='bottom',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))

    # Fitted parameters box
    param_text = (f'$\\mathcal{{F}}_0 = {F_0_fit:.2f}$\n'
                  f'$\\mathcal{{F}}_\\infty = {F_inf_fit:.2f}$\n'
                  f'$\\tau = {tau_fit:.1f}$')
    ax_a.text(0.98, 0.38, param_text, transform=ax_a.transAxes, fontsize=9,
              ha='right', va='bottom',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF8DC', edgecolor='gray', alpha=0.9))

    ax_a.set_xlabel(r'Adaptation Steps ($K$)')
    ax_a.set_ylabel(r'Gate Fidelity ($\mathcal{F}$)')
    ax_a.set_xlim([0, max_K])
    ax_a.set_ylim([0.25, 1.03])
    ax_a.legend(loc='center right', frameon=True, fancybox=False, edgecolor='gray')
    ax_a.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax_a.set_title(r'$\mathbf{(a)}$ Adaptation Dynamics', loc='left', fontsize=11)

    # -------------------------------------------------------------------------
    # Panel (b): ZZ Coupling pulses comparison - spans first row, last 2 columns
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 2:])

    linestyles = ['-', '--', '-.', ':']
    for idx, (task, J, ls) in enumerate(zip(test_tasks, task_J_values, linestyles)):
        adapted_pulses = get_pulse_sequence(adapted_policies[idx], task, device=device)
        t_smooth, p_smooth = smooth_pulse(time_points, adapted_pulses[:, 6])  # ZZ coupling
        ax_b.plot(t_smooth, p_smooth, linestyle=ls, color=COLORS['data'], linewidth=2,
                  label=f'$J = {J}$')
        # Mark target J
        ax_b.axhline(y=J, color=COLORS['pulse_colors'][idx], linestyle=ls, alpha=0.4, linewidth=1)

    ax_b.set_xlabel(r'Time ($t/T$)')
    ax_b.set_ylabel(r'$u_{ZZ}$ Amplitude')
    ax_b.set_xlim([0, GATE_TIME])
    ax_b.set_ylim([-13, 13])
    ax_b.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='gray', ncol=2)
    ax_b.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax_b.set_title(r'$\mathbf{(b)}$ Adapted $ZZ$ Coupling Pulses', loc='left', fontsize=11)

    # -------------------------------------------------------------------------
    # Panels (c-f): Full pulse sequences for each J value
    # -------------------------------------------------------------------------
    panel_labels = ['(c)', '(d)', '(e)', '(f)']

    for idx, (task, J, panel_label) in enumerate(zip(test_tasks, task_J_values, panel_labels)):
        ax = fig.add_subplot(gs[1, idx])

        adapted_pulses = get_pulse_sequence(adapted_policies[idx], task, device=device)

        for c in range(7):
            t_smooth, p_smooth = smooth_pulse(time_points, adapted_pulses[:, c])
            ax.plot(t_smooth, p_smooth, color=COLORS['pulse_colors'][c], linewidth=1.2,
                    label=control_labels[c] if idx == 0 else None)

        ax.set_xlabel(r'Time ($t/T$)')
        if idx == 0:
            ax.set_ylabel('Amplitude')
        ax.set_xlim([0, GATE_TIME])
        ax.set_ylim([-13, 13])
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_title(f'$\\mathbf{{{panel_label}}}$ $J = {J}$', loc='left', fontsize=10)

        if idx == 0:
            ax.legend(loc='upper right', fontsize=7, ncol=2, frameon=True,
                      fancybox=False, edgecolor='gray')

    plt.tight_layout()

    # Save
    fig_path = Path(__file__).parent / 'figure5_two_qubit_cz.pdf'
    plt.savefig(fig_path)
    plt.savefig(fig_path.with_suffix('.png'), dpi=300)
    print(f"\nSaved: {fig_path}")
    print(f"Saved: {fig_path.with_suffix('.png')}")
    plt.close()

    # ========================================================================
    # ALTERNATIVE LAYOUT: Vertical
    # ========================================================================
    print("\nGenerating alternative vertical layout...")

    fig2, axes = plt.subplots(3, 2, figsize=(10, 11))

    # Panel (a): Adaptation dynamics (top left, larger)
    ax_a2 = axes[0, 0]
    for curve in all_curves:
        ax_a2.plot(K_values, curve, '-', color=COLORS['data'], alpha=0.12, linewidth=0.8)
    ax_a2.fill_between(K_values, mean_curve - std_curve, mean_curve + std_curve,
                       color=COLORS['data'], alpha=0.25)
    ax_a2.plot(K_values, mean_curve, 'o', color=COLORS['data'], markersize=5,
               markeredgecolor='white', markeredgewidth=0.5, label='FOMAML')
    ax_a2.plot(K_smooth, fit_curve, '-', color=COLORS['fit'], linewidth=2.5, label='Fit')
    ax_a2.axhline(y=F_0_fit, color='gray', linestyle=':', alpha=0.6)
    ax_a2.axhline(y=F_inf_fit, color='gray', linestyle=':', alpha=0.6)
    ax_a2.text(max_K + 0.5, F_0_fit, r'$\mathcal{F}_0$', fontsize=9, va='center')
    ax_a2.text(max_K + 0.5, F_inf_fit, r'$\mathcal{F}_\infty$', fontsize=9, va='center')

    param_text = (f'$\\mathcal{{F}}_0 = {F_0_fit:.2f}$\n'
                  f'$\\mathcal{{F}}_\\infty = {F_inf_fit:.2f}$\n'
                  f'$\\tau = {tau_fit:.1f}$')
    ax_a2.text(0.97, 0.25, param_text, transform=ax_a2.transAxes, fontsize=9,
               ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='#FFF8DC', edgecolor='gray', alpha=0.9))

    ax_a2.set_xlabel(r'Adaptation Steps ($K$)')
    ax_a2.set_ylabel(r'Gate Fidelity ($\mathcal{F}$)')
    ax_a2.set_xlim([0, max_K])
    ax_a2.set_ylim([0.25, 1.03])
    ax_a2.legend(loc='center right', frameon=True)
    ax_a2.grid(True, alpha=0.3)
    ax_a2.set_title(r'$\mathbf{(a)}$ Adaptation Dynamics', loc='left')

    # Panel (b): ZZ coupling comparison (top right)
    ax_b2 = axes[0, 1]
    for idx, (task, J, ls) in enumerate(zip(test_tasks, task_J_values, linestyles)):
        adapted_pulses = get_pulse_sequence(adapted_policies[idx], task, device=device)
        t_smooth, p_smooth = smooth_pulse(time_points, adapted_pulses[:, 6])
        ax_b2.plot(t_smooth, p_smooth, linestyle=ls, color=COLORS['data'], linewidth=2,
                   label=f'$J = {J}$')
    ax_b2.set_xlabel(r'Time ($t/T$)')
    ax_b2.set_ylabel(r'$u_{ZZ}$ Amplitude')
    ax_b2.set_xlim([0, GATE_TIME])
    ax_b2.set_ylim([-13, 13])
    ax_b2.legend(loc='upper right', ncol=2)
    ax_b2.grid(True, alpha=0.3)
    ax_b2.set_title(r'$\mathbf{(b)}$ $ZZ$ Coupling Pulses', loc='left')

    # Panels (c-f): Individual pulse sequences
    panel_axes = [axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1]]
    panel_labels = ['(c)', '(d)', '(e)', '(f)']

    for idx, (ax, task, J, label) in enumerate(zip(panel_axes, test_tasks, task_J_values, panel_labels)):
        adapted_pulses = get_pulse_sequence(adapted_policies[idx], task, device=device)
        for c in range(7):
            t_smooth, p_smooth = smooth_pulse(time_points, adapted_pulses[:, c])
            ax.plot(t_smooth, p_smooth, color=COLORS['pulse_colors'][c], linewidth=1.2,
                    label=control_labels[c] if idx == 0 else None)
        ax.set_xlabel(r'Time ($t/T$)')
        ax.set_ylabel('Amplitude')
        ax.set_xlim([0, GATE_TIME])
        ax.set_ylim([-13, 13])
        ax.grid(True, alpha=0.3)
        ax.set_title(f'$\\mathbf{{{label}}}$ Adapted Pulse ($J = {J}$)', loc='left')
        if idx == 0:
            ax.legend(loc='upper right', fontsize=7, ncol=2)

    plt.tight_layout()
    fig_path2 = Path(__file__).parent / 'figure5_two_qubit_cz_vertical.pdf'
    plt.savefig(fig_path2)
    plt.savefig(fig_path2.with_suffix('.png'), dpi=300)
    print(f"Saved: {fig_path2}")
    plt.close()

    print("\n" + "=" * 70)
    print("Figure 5 generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
