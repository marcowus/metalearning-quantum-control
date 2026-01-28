"""
Figure 5: Two-Qubit CZ Gate with Pulsed Coupling
Publication-quality plotting script.

Panel (a): Adaptation GAP G(K) = F(K) - F(0) fitted to A(1 - e^{-βK})
Panel (b): ZZ coupling pulses with distinct colors
Panels (c-f): Full adapted pulse sequences

This script can be re-run to reproduce figures from saved data.
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
import json

from two_qubit_cz_maml_fast import (
    X1, Y1, X2, Y2, Z1, Z2, ZZ, Sm1, Sm2,
    CZ_GATE, ket_0, ket_1, ket_p, ket_m,
)

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

# Colors - distinct colors for panel (b)
COLORS = {
    'data': '#0072B2',
    'fit': '#D55E00',
    'J_colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],  # Blue, Orange, Green, Red
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

DATA_FILE = Path(__file__).parent / 'fig5_data.json'


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


def train_maml(n_iterations=150, device='cpu'):
    print("Training FOMAML...")
    torch.manual_seed(42)
    np.random.seed(42)

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


def adaptation_gap_model(K, A, beta):
    """Adaptation gap model: G(K) = A(1 - e^{-βK})"""
    return A * (1 - np.exp(-beta * K))


def generate_data():
    """Generate all data needed for plotting."""
    print("=" * 70)
    print("GENERATING DATA FOR FIGURE 5")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)

    # Train MAML
    maml_policy = train_maml(device=device)

    # Collect adaptation curves
    print("\nCollecting adaptation curves...")
    max_K = 30
    n_tasks = 12

    np.random.seed(123)  # Different seed for test tasks
    all_curves = []
    for i in range(n_tasks):
        task = PulsedCZTask(
            np.exp(np.random.uniform(np.log(0.5), np.log(10.0))),
            0.005, 0.0025, 0.005, 0.0025
        )
        curve = get_adaptation_curve(maml_policy, task, max_K, device=device)
        all_curves.append(curve)

    # Test tasks for pulse visualization
    test_tasks = [
        PulsedCZTask(1.0, 0.005, 0.0025, 0.005, 0.0025),
        PulsedCZTask(3.0, 0.005, 0.0025, 0.005, 0.0025),
        PulsedCZTask(6.0, 0.005, 0.0025, 0.005, 0.0025),
        PulsedCZTask(9.0, 0.005, 0.0025, 0.005, 0.0025),
    ]

    # Get adapted pulses
    print("\nAdapting policies...")
    adapted_pulses = []
    time_points = np.linspace(0, GATE_TIME, N_SEGMENTS)

    for task in test_tasks:
        adapted = adapt_policy(maml_policy, task, K=20, device=device)
        pulses = get_pulse_sequence(adapted, task, device=device)
        adapted_pulses.append(pulses.tolist())

    # Save data
    data = {
        'all_curves': [c for c in all_curves],
        'adapted_pulses': adapted_pulses,
        'test_J_values': [1.0, 3.0, 6.0, 9.0],
        'time_points': time_points.tolist(),
        'max_K': max_K,
    }

    with open(DATA_FILE, 'w') as f:
        json.dump(data, f)

    print(f"\nData saved to: {DATA_FILE}")
    return data


def load_data():
    """Load saved data."""
    if DATA_FILE.exists():
        print(f"Loading data from: {DATA_FILE}")
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    else:
        print("No saved data found. Generating new data...")
        return generate_data()


def create_figure(data):
    """Create Figure 5 from saved data."""
    print("\n" + "=" * 70)
    print("CREATING FIGURE 5")
    print("=" * 70)

    all_curves = np.array(data['all_curves'])
    adapted_pulses = [np.array(p) for p in data['adapted_pulses']]
    test_J_values = data['test_J_values']
    time_points = np.array(data['time_points'])
    max_K = data['max_K']

    K_values = np.arange(max_K + 1)

    # Compute adaptation gaps: G(K) = F(K) - F(0)
    # Using meta-initialized model as baseline (K=0)
    all_gaps = []
    for curve in all_curves:
        F_0 = curve[0]  # Meta-initialized fidelity
        gap = np.array(curve) - F_0
        all_gaps.append(gap)

    all_gaps = np.array(all_gaps)
    mean_gap = np.mean(all_gaps, axis=0)
    std_gap = np.std(all_gaps, axis=0)

    # Also compute mean fidelity for reference
    mean_fidelity = np.mean(all_curves, axis=0)
    F_0_mean = mean_fidelity[0]

    # Fit adaptation gap model: G(K) = A(1 - e^{-βK})
    print("\nFitting adaptation gap model: G(K) = A(1 - e^{-βK})")
    try:
        # Initial guesses
        A_guess = mean_gap[-1]  # Asymptotic gap
        beta_guess = 0.2
        p0 = [A_guess, beta_guess]
        bounds = ([0, 0.01], [1, 2])

        popt, pcov = curve_fit(adaptation_gap_model, K_values, mean_gap, p0=p0, bounds=bounds)
        A_fit, beta_fit = popt
        perr = np.sqrt(np.diag(pcov))

        print(f"  A = {A_fit:.3f} ± {perr[0]:.3f}")
        print(f"  β = {beta_fit:.3f} ± {perr[1]:.3f}")
        print(f"  τ = 1/β = {1/beta_fit:.2f}")

        K_smooth = np.linspace(0, max_K, 200)
        fit_gap = adaptation_gap_model(K_smooth, *popt)
        fit_success = True
    except Exception as e:
        print(f"  Fitting failed: {e}")
        fit_success = False

    # Control labels
    control_labels = [r'$u_{X_1}$', r'$u_{Y_1}$', r'$u_{X_2}$', r'$u_{Y_2}$',
                      r'$u_{Z_1}$', r'$u_{Z_2}$', r'$u_{ZZ}$']

    # ========================================================================
    # CREATE FIGURE
    # ========================================================================
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.2, 1], hspace=0.35, wspace=0.3)

    # -------------------------------------------------------------------------
    # Panel (a): Adaptation Gap
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, :2])

    # Plot individual gap curves (light)
    for gap in all_gaps:
        ax_a.plot(K_values, gap, '-', color=COLORS['data'], alpha=0.12, linewidth=0.8)

    # Plot mean gap with shaded region
    ax_a.fill_between(K_values, mean_gap - std_gap, mean_gap + std_gap,
                      color=COLORS['data'], alpha=0.25)
    ax_a.plot(K_values, mean_gap, 'o', color=COLORS['data'], markersize=5,
              markeredgecolor='white', markeredgewidth=0.5, label='FOMAML')

    # Plot fit
    if fit_success:
        ax_a.plot(K_smooth, fit_gap, '-', color=COLORS['fit'], linewidth=2.5,
                  label='Exponential fit')

    # Asymptotic line
    if fit_success:
        ax_a.axhline(y=A_fit, color='gray', linestyle=':', alpha=0.6, linewidth=1)
        ax_a.text(max_K + 0.5, A_fit, r'$A$', fontsize=10, va='center', ha='left')

    # Model equation and parameters removed for cleaner figure
    # (fit parameters: A, β, τ are printed to console during generation)

    # Formatting
    ax_a.set_xlabel(r'Adaptation Steps ($K$)')
    ax_a.set_ylabel(r'Adaptation Gap ($G_K = \mathcal{F}_K - \mathcal{F}_0$)')
    ax_a.set_xlim([0, max_K])
    ax_a.set_ylim([-0.02, 0.75])
    ax_a.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='gray')
    ax_a.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax_a.set_title(r'$\mathbf{(a)}$ Adaptation Gap', loc='left', fontsize=11)

    # Add F_0 annotation in corner
    ax_a.text(0.02, 0.98, f'$\\mathcal{{F}}_0 = {F_0_mean:.2f}$', transform=ax_a.transAxes,
              fontsize=9, va='top', ha='left',
              bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.8))

    # -------------------------------------------------------------------------
    # Panel (b): ZZ Coupling pulses - distinct colors, no horizontal lines
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 2:])

    linestyles = ['-', '-', '-', '-']  # All solid for clarity with different colors

    for idx, (pulses, J) in enumerate(zip(adapted_pulses, test_J_values)):
        t_smooth, p_smooth = smooth_pulse(time_points, pulses[:, 6])  # ZZ coupling
        ax_b.plot(t_smooth, p_smooth, linestyle=linestyles[idx],
                  color=COLORS['J_colors'][idx], linewidth=2.2,
                  label=f'$J = {J}$')

    ax_b.set_xlabel(r'Time ($t/T$)')
    ax_b.set_ylabel(r'$u_{ZZ}$ Amplitude')
    ax_b.set_xlim([0, GATE_TIME])
    ax_b.set_ylim([-13, 13])
    ax_b.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='gray', ncol=2)
    ax_b.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax_b.set_title(r'$\mathbf{(b)}$ Adapted $ZZ$ Coupling Pulses', loc='left', fontsize=11)

    # -------------------------------------------------------------------------
    # Panels (c-f): Full pulse sequences
    # -------------------------------------------------------------------------
    panel_labels = ['(c)', '(d)', '(e)', '(f)']

    for idx, (pulses, J, panel_label) in enumerate(zip(adapted_pulses, test_J_values, panel_labels)):
        ax = fig.add_subplot(gs[1, idx])

        for c in range(7):
            t_smooth, p_smooth = smooth_pulse(time_points, pulses[:, c])
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
    fig_path = Path(__file__).parent / 'figure5_final.pdf'
    plt.savefig(fig_path)
    plt.savefig(fig_path.with_suffix('.png'), dpi=300)
    print(f"\nSaved: {fig_path}")
    print(f"Saved: {fig_path.with_suffix('.png')}")
    plt.close()

    print("\n" + "=" * 70)
    print("Figure 5 complete!")
    print("=" * 70)


def main():
    """Main function - load or generate data, then create figure."""
    import argparse
    parser = argparse.ArgumentParser(description='Generate Figure 5')
    parser.add_argument('--regenerate', action='store_true',
                        help='Regenerate data even if saved data exists')
    args = parser.parse_args()

    if args.regenerate or not DATA_FILE.exists():
        data = generate_data()
    else:
        data = load_data()

    create_figure(data)


if __name__ == "__main__":
    main()
