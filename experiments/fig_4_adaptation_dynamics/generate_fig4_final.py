"""
Figure 4: Adaptation Dynamics  
© 2025 The MITRE Corporation, All Rights Reserved  
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from copy import deepcopy
import argparse
import json

from metaqctrl.meta_rl.policy_gamma import GammaPulsePolicy
from metaqctrl.quantum.lindblad_torch import DifferentiableLindbladSimulator

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'font.family': 'serif',
    'figure.dpi': 150, 
    'pdf.fonttype': 42,   
    'ps.fonttype': 42,    
    'text.usetex': True,   
    'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',
})


def create_single_qubit_system(gamma_deph=0.05, gamma_relax=0.025, device='cpu'):
    """Create simulator with direct gamma rates."""
    sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
    sigma_p = torch.tensor([[0, 1], [0, 0]], dtype=torch.complex64, device=device)

    H0 = 0.0 * sigma_z
    H_controls = [sigma_x, sigma_y]

    L_operators = []
    if gamma_deph > 0:
        L_operators.append(np.sqrt(gamma_deph / 2.0) * sigma_z)
    if gamma_relax > 0:
        L_operators.append(np.sqrt(gamma_relax) * sigma_p)

    if not L_operators:
        L_operators.append(torch.zeros(2, 2, dtype=torch.complex64, device=device))

    sim = DifferentiableLindbladSimulator(
        H0=H0,
        H_controls=H_controls,
        L_operators=L_operators,
        dt=0.05,
        method='rk4',
        device=device
    )
    return sim


def compute_loss_gamma(policy, gamma_deph, gamma_relax, device='cpu', return_fidelity=False, return_controls=False):
    """Compute loss using gamma-rate task features."""
    sim = create_single_qubit_system(gamma_deph, gamma_relax, device=device)

    task_features = torch.tensor([
        gamma_deph / 0.1,
        gamma_relax / 0.05,
        (gamma_deph + gamma_relax) / 0.15
    ], dtype=torch.float32, device=device)

    controls = policy(task_features)

    rho0 = torch.zeros(2, 2, dtype=torch.complex64, device=device)
    rho0[0, 0] = 1.0

    rho_final = sim.forward(rho0, controls, T=1.0)

    target = torch.zeros(2, 2, dtype=rho_final.dtype, device=device)
    target[1, 1] = 1.0

    fidelity = torch.abs(torch.trace(target @ rho_final)).real
    loss = 1.0 - fidelity

    if return_controls and return_fidelity:
        return loss, fidelity, controls
    elif return_fidelity:
        return loss, fidelity
    elif return_controls:
        return loss, controls
    return loss


def load_pretrained_gamma_policy(checkpoint_path, device='cpu'):
    """Load pretrained gamma policy from checkpoint (V2 architecture)."""
    print(f"Loading pretrained policy from: {checkpoint_path}")

    policy = GammaPulsePolicy(
        task_feature_dim=3,
        hidden_dim=64,
        n_hidden_layers=2,
        n_segments=20,
        n_controls=2,
        output_scale=1.0
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'policy_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['policy_state_dict'])
        print(f"  Loaded policy state dict (iteration {checkpoint.get('iteration', 'unknown')})")
    else:
        policy.load_state_dict(checkpoint)
        print("  Loaded policy weights directly")

    return policy


def train_robust_policy_gamma(n_iterations=500, train_lr=0.001, device='cpu'):
    """Train a robust policy on a fixed average gamma level using Adam (for training from scratch)."""
    print("Training robust baseline on FIXED average gamma rates (Adam)...")

    policy = GammaPulsePolicy(
        task_feature_dim=3,
        hidden_dim=64,
        n_hidden_layers=2,
        n_segments=20,
        n_controls=2,
        output_scale=1.0
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=train_lr)

    avg_gamma_deph = 0.085  # (0.02 + 0.15) / 2
    avg_gamma_relax = 0.045  # (0.01 + 0.08) / 2

    for iteration in range(n_iterations):
        optimizer.zero_grad()
        loss = compute_loss_gamma(policy, avg_gamma_deph, avg_gamma_relax, device)
        loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            fid = 1 - loss.item()
            print(f"  Iter {iteration}: Loss={loss.item():.4f}, Fidelity={fid:.4f}")

    final_fid = 1 - loss.item()
    print(f"Robust training complete: Fidelity={final_fid:.4f}")
    return policy


def generate_panel_a_data(meta_policy, robust_policy, max_K=50, inner_lr=0.01, device='cpu',
                          ood_gamma_deph=0.35, ood_gamma_relax=0.15, use_distribution=False,
                          use_boundary=False):
    """
    Panel (a): Loss vs K for different initializations.
    Uses SGD optimizer consistent with Figure 3.
    If use_distribution=True, averages over multiple tasks from training distribution.
    If use_boundary=True, uses tasks at the boundary of training distribution.
    """
    print("Generating Panel (a) data: Loss vs K (SGD optimizer)...")

    torch.manual_seed(42)
    np.random.seed(42)

    if use_boundary:
        # Use tasks at the boundary of the training distribution (harder but still in-domain)
        n_panel_a_tasks = 20
        # Sample near the max values of training range
        gamma_deph_range = (0.12, 0.15)  # Upper end of (0.02, 0.15)
        gamma_relax_range = (0.06, 0.08)  # Upper end of (0.01, 0.08)
        tasks = [(np.random.uniform(*gamma_deph_range), np.random.uniform(*gamma_relax_range))
                 for _ in range(n_panel_a_tasks)]
        print(f"  Averaging over {n_panel_a_tasks} BOUNDARY tasks (γ_deph~0.12-0.15, γ_relax~0.06-0.08)")
    elif use_distribution:
        # Sample multiple tasks from training distribution
        n_panel_a_tasks = 20
        gamma_deph_range = (0.02, 0.15)
        gamma_relax_range = (0.01, 0.08)
        tasks = [(np.random.uniform(*gamma_deph_range), np.random.uniform(*gamma_relax_range))
                 for _ in range(n_panel_a_tasks)]
        print(f"  Averaging over {n_panel_a_tasks} in-distribution tasks")
    else:
        tasks = [(ood_gamma_deph, ood_gamma_relax)]
        print(f"  Single task: gamma_deph={ood_gamma_deph}, gamma_relax={ood_gamma_relax}")

    # Collect losses for each initialization across all tasks
    all_meta_losses = []
    all_robust_losses = []
    all_fresh_losses = []

    for task_idx, (test_gamma_deph, test_gamma_relax) in enumerate(tasks):
        # 1. Meta-learned initialization
        meta_adapted = deepcopy(meta_policy)
        meta_adapted.train()
        opt = optim.SGD(meta_adapted.parameters(), lr=inner_lr)
        meta_losses = []

        with torch.no_grad():
            loss_val = compute_loss_gamma(meta_adapted, test_gamma_deph, test_gamma_relax, device).item()
            meta_losses.append(loss_val)

        for k in range(max_K):
            opt.zero_grad()
            loss = compute_loss_gamma(meta_adapted, test_gamma_deph, test_gamma_relax, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(meta_adapted.parameters(), max_norm=1.0)
            opt.step()
            with torch.no_grad():
                loss_val = compute_loss_gamma(meta_adapted, test_gamma_deph, test_gamma_relax, device).item()
                meta_losses.append(loss_val)

        all_meta_losses.append(meta_losses)

        # 2. Robust (fixed average) initialization
        robust_adapted = deepcopy(robust_policy)
        robust_adapted.train()
        opt = optim.SGD(robust_adapted.parameters(), lr=inner_lr)
        robust_losses = []

        with torch.no_grad():
            loss_val = compute_loss_gamma(robust_adapted, test_gamma_deph, test_gamma_relax, device).item()
            robust_losses.append(loss_val)

        for k in range(max_K):
            opt.zero_grad()
            loss = compute_loss_gamma(robust_adapted, test_gamma_deph, test_gamma_relax, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(robust_adapted.parameters(), max_norm=1.0)
            opt.step()
            with torch.no_grad():
                loss_val = compute_loss_gamma(robust_adapted, test_gamma_deph, test_gamma_relax, device).item()
                robust_losses.append(loss_val)

        all_robust_losses.append(robust_losses)

        # 3. Random initialization (use Adam with higher lr since training from scratch)
        fresh_policy = GammaPulsePolicy(
            task_feature_dim=3,
            hidden_dim=64,
            n_hidden_layers=2,
            n_segments=20,
            n_controls=2,
            output_scale=1.0
        ).to(device)
        fresh_policy.train()
        opt = optim.Adam(fresh_policy.parameters(), lr=0.01)
        fresh_losses = []

        with torch.no_grad():
            loss_val = compute_loss_gamma(fresh_policy, test_gamma_deph, test_gamma_relax, device).item()
            fresh_losses.append(loss_val)

        for k in range(max_K):
            opt.zero_grad()
            loss = compute_loss_gamma(fresh_policy, test_gamma_deph, test_gamma_relax, device)
            loss.backward()
            opt.step()
            with torch.no_grad():
                loss_val = compute_loss_gamma(fresh_policy, test_gamma_deph, test_gamma_relax, device).item()
                fresh_losses.append(loss_val)

        all_fresh_losses.append(fresh_losses)

    # Average across tasks (excluding Random Init for cleaner visualization)
    losses_by_init = {}
    losses_by_init[0] = {'losses': np.mean(all_meta_losses, axis=0).tolist(), 'label': 'FOMAML'}
    losses_by_init[1] = {'losses': np.mean(all_robust_losses, axis=0).tolist(), 'label': 'Fixed Average'}
    # Random Init excluded from panel (a) for cleaner comparison
    # losses_by_init[2] = {'losses': np.mean(all_fresh_losses, axis=0).tolist(), 'label': 'Random Init'}

    return losses_by_init


def generate_panel_b_data(meta_policy, robust_policy, n_tasks=50, K_adapt=20, inner_lr=0.01, device='cpu',
                          ood_gamma_deph=None, ood_gamma_relax=None):
    """
    Panel (b): Fidelity distributions across tasks.
    Uses SGD optimizer consistent with Figure 3.
    """
    print("Generating Panel (b) data: Fidelity distributions (SGD optimizer)...")

    np.random.seed(123)

    if ood_gamma_deph is not None and ood_gamma_relax is not None:
        spread_deph = 0.15
        spread_relax = 0.15
        gamma_deph_vals = np.random.uniform(
            ood_gamma_deph * (1 - spread_deph),
            ood_gamma_deph * (1 + spread_deph),
            n_tasks
        )
        gamma_relax_vals = np.random.uniform(
            ood_gamma_relax * (1 - spread_relax),
            ood_gamma_relax * (1 + spread_relax),
            n_tasks
        )
        print(f"  Using challenging tasks: gamma_deph ~ {ood_gamma_deph:.2f}, gamma_relax ~ {ood_gamma_relax:.2f}")
    else:
        gamma_deph_range = (0.02, 0.15)
        gamma_relax_range = (0.01, 0.08)
        gamma_deph_vals = np.random.uniform(*gamma_deph_range, n_tasks)
        gamma_relax_vals = np.random.uniform(*gamma_relax_range, n_tasks)

    fidelities = {
        'robust': [],
        'robust_adapted': [],
        'meta_init': [],
        'adapted': []
    }

    for i, (gamma_deph, gamma_relax) in enumerate(zip(gamma_deph_vals, gamma_relax_vals)):
        if (i + 1) % 10 == 0:
            print(f"  Processing task {i+1}/{n_tasks}...")

        # Robust baseline (K=0)
        with torch.no_grad():
            _, fid_robust = compute_loss_gamma(robust_policy, gamma_deph, gamma_relax, device, return_fidelity=True)
            fidelities['robust'].append(fid_robust.item())

        # Robust adapted (after K steps with SGD)
        robust_adapted = deepcopy(robust_policy)
        robust_adapted.train()
        robust_opt = optim.SGD(robust_adapted.parameters(), lr=inner_lr)

        for _ in range(K_adapt):
            robust_opt.zero_grad()
            loss = compute_loss_gamma(robust_adapted, gamma_deph, gamma_relax, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(robust_adapted.parameters(), max_norm=1.0)
            robust_opt.step()

        with torch.no_grad():
            _, fid_robust_adapted = compute_loss_gamma(robust_adapted, gamma_deph, gamma_relax, device, return_fidelity=True)
            fidelities['robust_adapted'].append(fid_robust_adapted.item())

        # Meta-init (before adaptation)
        with torch.no_grad():
            _, fid_meta = compute_loss_gamma(meta_policy, gamma_deph, gamma_relax, device, return_fidelity=True)
            fidelities['meta_init'].append(fid_meta.item())

        # Adapted (after K steps with SGD)
        adapted_policy = deepcopy(meta_policy)
        adapted_policy.train()
        inner_opt = optim.SGD(adapted_policy.parameters(), lr=inner_lr)

        for _ in range(K_adapt):
            inner_opt.zero_grad()
            loss = compute_loss_gamma(adapted_policy, gamma_deph, gamma_relax, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted_policy.parameters(), max_norm=1.0)
            inner_opt.step()

        with torch.no_grad():
            _, fid_adapted = compute_loss_gamma(adapted_policy, gamma_deph, gamma_relax, device, return_fidelity=True)
            fidelities['adapted'].append(fid_adapted.item())

    return fidelities


def generate_panel_c_data(meta_policy, robust_policy, K_adapt=20, inner_lr=0.01, device='cpu',
                          ood_gamma_deph=None, ood_gamma_relax=None):
    """
    Panel (c): Pulse sequences before and after adaptation.
    Uses SGD optimizer consistent with Figure 3.
    """
    print("Generating Panel (c) data: Pulse sequences (SGD optimizer)...")

    torch.manual_seed(42)

    if ood_gamma_deph is not None and ood_gamma_relax is not None:
        gamma_deph = ood_gamma_deph
        gamma_relax = ood_gamma_relax
        print(f"  Using task: gamma_deph={gamma_deph:.2f}, gamma_relax={gamma_relax:.2f}")
    else:
        gamma_deph = 0.08
        gamma_relax = 0.04

    pulses = {}

    # 1. Meta-policy (K=0)
    with torch.no_grad():
        _, fid_meta, controls = compute_loss_gamma(meta_policy, gamma_deph, gamma_relax, device,
                                            return_fidelity=True, return_controls=True)
    pulses[0] = {
        'controls': controls.detach().cpu().numpy(),
        'label': 'FOMAML ($K$=0)',
        'fidelity': fid_meta.item()
    }

    # 2. Meta-policy after adaptation (K=K_adapt) with SGD
    adapted_policy = deepcopy(meta_policy)
    adapted_policy.train()
    opt = optim.SGD(adapted_policy.parameters(), lr=inner_lr)

    for _ in range(K_adapt):
        opt.zero_grad()
        loss = compute_loss_gamma(adapted_policy, gamma_deph, gamma_relax, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapted_policy.parameters(), max_norm=1.0)
        opt.step()

    with torch.no_grad():
        _, fid_adapted, controls = compute_loss_gamma(adapted_policy, gamma_deph, gamma_relax, device,
                                            return_fidelity=True, return_controls=True)
    pulses[1] = {
        'controls': controls.detach().cpu().numpy(),
        'label': f'FOMAML ($K$={K_adapt})',
        'fidelity': fid_adapted.item()
    }

    # 3. Fixed Average (robust) policy
    with torch.no_grad():
        _, fid_robust, controls = compute_loss_gamma(robust_policy, gamma_deph, gamma_relax, device,
                                            return_fidelity=True, return_controls=True)
    pulses[2] = {
        'controls': controls.detach().cpu().numpy(),
        'label': 'Fixed Average',
        'fidelity': fid_robust.item()
    }

    return pulses


def create_figure(losses_data, fidelity_data, pulse_data, K_adapt, save_path=None):
    """Create the 3-panel figure."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    colors_init = ['#2E86AB', '#E94F37']  # FOMAML=blue, Fixed=red

    # --- Panel (a): Fidelity vs K (linear scale) ---
    ax = axes[0]

    for i, (init_id, data) in enumerate(losses_data.items()):
        losses = data['losses']
        fidelities = [1 - l for l in losses]  # Convert loss to fidelity
        K_vals = np.arange(len(fidelities))
        ax.plot(K_vals, fidelities, 'o-', color=colors_init[i], markersize=4,
                linewidth=1.5, alpha=0.8, label=data['label'])

    ax.set_xlabel('Adaptation Steps $K$')
    ax.set_ylabel('Fidelity $\\mathcal{F}$')
    ax.set_xlim(-0.5, len(losses_data[0]['losses']) - 0.5)

    # Set y-axis to show relevant range
    all_fidelities = []
    for data in losses_data.values():
        all_fidelities.extend([1 - l for l in data['losses']])
    y_min = max(0, min(all_fidelities) - 0.05)
    y_max = min(1.02, max(all_fidelities) + 0.02)
    ax.set_ylim(y_min, y_max)

    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)
    ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    # --- Panel (b): Fidelity Distribution ---
    ax = axes[1]

    positions = [1, 2, 3, 4]
    labels = ['Fixed Avg\n($K$=0)', f'Fixed Avg\n($K$={K_adapt})', 'FOMAML\n($K$=0)', f'FOMAML\n($K$={K_adapt})']
    data_lists = [fidelity_data['robust'], fidelity_data['robust_adapted'], fidelity_data['meta_init'], fidelity_data['adapted']]
    colors_violin = ['#E94F37', '#c0392b', '#3498db', '#2E86AB']

    parts = ax.violinplot(data_lists, positions=positions, showmeans=True, showmedians=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_violin[i])
        pc.set_alpha(0.7)

    parts['cmeans'].set_color('black')
    parts['cmedians'].set_color('white')

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Fidelity $\\mathcal{F}$')

    # Zoom y-axis to show differences better
    all_fids = fidelity_data['robust'] + fidelity_data['robust_adapted'] + fidelity_data['meta_init'] + fidelity_data['adapted']
    y_min_b = max(0.5, min(all_fids) - 0.05)
    ax.set_ylim(y_min_b, 1.02)
    ax.grid(True, alpha=0.3, axis='y')
    ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    # Add statistics
    for i, (pos, data) in enumerate(zip(positions, data_lists)):
        mean = np.mean(data)
        std = np.std(data)
        ax.text(pos, y_min_b + 0.02, f'{mean:.3f}',
                ha='center', fontsize=8, color=colors_violin[i], fontweight='bold')

    # --- Panel (c): Pulse Sequences ---
    ax = axes[2]

    n_segments = list(pulse_data.values())[0]['controls'].shape[0]
    t = np.linspace(0, 1, n_segments)

    # Distinct colors: FOMAML K=0 (light blue), FOMAML K=20 (dark green), Fixed Average (red)
    colors_pulse = ['#3498db', '#27ae60', '#E94F37']

    from scipy.interpolate import make_interp_spline
    t_smooth = np.linspace(0, 1, 200)

    for i, (pulse_id, data) in enumerate(pulse_data.items()):
        controls = data['controls']
        label = data['label']

        try:
            spl_x = make_interp_spline(t, controls[:, 0], k=3)
            spl_y = make_interp_spline(t, controls[:, 1], k=3)
            controls_x_smooth = spl_x(t_smooth)
            controls_y_smooth = spl_y(t_smooth)
        except:
            t_smooth = t
            controls_x_smooth = controls[:, 0]
            controls_y_smooth = controls[:, 1]

        ax.plot(t_smooth, controls_x_smooth, linestyle='-', color=colors_pulse[i],
                linewidth=1.2, label=f'{label}: $u_x$', alpha=0.9)
        ax.plot(t_smooth, controls_y_smooth, linestyle='--', color=colors_pulse[i],
                linewidth=1.0, label=f'{label}: $u_y$', alpha=0.7)

    ax.set_xlabel('Time $t/T$')
    ax.set_ylabel('Control Amplitude')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=6, ncol=2)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.text(-0.12, 1.05, '(c)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.close()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Generate Figure 4: Adaptation Dynamics (Final Version)')
    parser.add_argument('--checkpoint', type=str,
                        default='../../checkpoints/checkpoints_gamma_v2/maml_gamma_pauli_x.pt',
                        help='Path to gamma checkpoint (V2)')
    parser.add_argument('--output', type=str, default='fig4_adaptation_dynamics_final',
                        help='Output filename prefix')
    parser.add_argument('--max_K', type=int, default=10, help='Max adaptation steps for panel (a)')
    parser.add_argument('--n_tasks', type=int, default=50, help='Number of tasks for panel (b)')
    parser.add_argument('--K_adapt', type=int, default=10, help='Adaptation steps for panels (b) and (c)')
    parser.add_argument('--inner_lr', type=float, default=0.01, help='Inner learning rate (SGD)')
    parser.add_argument('--ood_gamma_deph', type=float, default=0.17,
                        help='Mild OOD gamma_deph (training max=0.15)')
    parser.add_argument('--ood_gamma_relax', type=float, default=0.085,
                        help='Mild OOD gamma_relax (training max=0.08)')
    parser.add_argument('--use_distribution', action='store_true',
                        help='Use in-distribution tasks (average over training distribution)')
    parser.add_argument('--use_boundary', action='store_true',
                        help='Use boundary tasks (at edge of training distribution)')
    args = parser.parse_args()

    device = torch.device('cpu')
    print(f"Using device: {device}")
    print(f"\nParameters (consistent with Figure 3):")
    print(f"  Optimizer: SGD")
    print(f"  inner_lr: {args.inner_lr}")
    print(f"  Gradient clipping: max_norm=1.0")
    print(f"  max_K: {args.max_K}, n_tasks: {args.n_tasks}, K_adapt: {args.K_adapt}")
    print(f"  OOD task: gamma_deph={args.ood_gamma_deph}, gamma_relax={args.ood_gamma_relax}")

    # Load meta-policy (V2 checkpoint)
    checkpoint_path = Path(__file__).parent / args.checkpoint
    if not checkpoint_path.exists():
        checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return

    meta_policy = load_pretrained_gamma_policy(str(checkpoint_path), device)
    meta_policy.eval()

    # Train robust baseline with Adam (for training from scratch)
    robust_policy = train_robust_policy_gamma(n_iterations=500, train_lr=0.001, device=device)
    robust_policy.eval()

    # Generate data for each panel
    print("\n" + "-" * 50)
    losses_data = generate_panel_a_data(meta_policy, robust_policy, max_K=args.max_K,
                                        inner_lr=args.inner_lr, device=device,
                                        ood_gamma_deph=args.ood_gamma_deph,
                                        ood_gamma_relax=args.ood_gamma_relax,
                                        use_distribution=args.use_distribution,
                                        use_boundary=args.use_boundary)

    print("-" * 50)
    # Use appropriate tasks for panel (b)
    if args.use_distribution:
        fidelity_data = generate_panel_b_data(meta_policy, robust_policy, n_tasks=args.n_tasks,
                                              K_adapt=args.K_adapt, inner_lr=args.inner_lr, device=device,
                                              ood_gamma_deph=None, ood_gamma_relax=None)
    elif args.use_boundary:
        # Use boundary tasks (at edge of training distribution)
        fidelity_data = generate_panel_b_data(meta_policy, robust_policy, n_tasks=args.n_tasks,
                                              K_adapt=args.K_adapt, inner_lr=args.inner_lr, device=device,
                                              ood_gamma_deph=0.14, ood_gamma_relax=0.07)
    else:
        fidelity_data = generate_panel_b_data(meta_policy, robust_policy, n_tasks=args.n_tasks,
                                              K_adapt=args.K_adapt, inner_lr=args.inner_lr, device=device,
                                              ood_gamma_deph=args.ood_gamma_deph, ood_gamma_relax=args.ood_gamma_relax)

    print("-" * 50)
    # Use a representative task for panel (c) pulses
    if args.use_distribution:
        pulse_data = generate_panel_c_data(meta_policy, robust_policy, K_adapt=args.K_adapt,
                                           inner_lr=args.inner_lr, device=device,
                                           ood_gamma_deph=0.10, ood_gamma_relax=0.05)
    elif args.use_boundary:
        pulse_data = generate_panel_c_data(meta_policy, robust_policy, K_adapt=args.K_adapt,
                                           inner_lr=args.inner_lr, device=device,
                                           ood_gamma_deph=0.14, ood_gamma_relax=0.07)
    else:
        pulse_data = generate_panel_c_data(meta_policy, robust_policy, K_adapt=args.K_adapt,
                                           inner_lr=args.inner_lr, device=device,
                                           ood_gamma_deph=args.ood_gamma_deph, ood_gamma_relax=args.ood_gamma_relax)

    # Create figure
    print("\n" + "-" * 50)
    print("Creating figure...")

    output_dir = Path(__file__).parent
    save_path = str(output_dir / f"{args.output}.png")

    create_figure(losses_data, fidelity_data, pulse_data, args.K_adapt, save_path=save_path)

    # Save data
    json_path = str(output_dir / f"{args.output}_data.json")
    results = {
        'panel_a': {str(k): {'losses': v['losses'], 'label': v['label']} for k, v in losses_data.items()},
        'panel_b': {k: v for k, v in fidelity_data.items()},
        'panel_c': {str(k): {'label': v['label']} for k, v in pulse_data.items()},
        'params': {
            'checkpoint': args.checkpoint,
            'optimizer': 'sgd',
            'inner_lr': args.inner_lr,
            'grad_clip': 1.0,
            'max_K': args.max_K,
            'n_tasks': args.n_tasks,
            'K_adapt': args.K_adapt,
            'ood_gamma_deph': args.ood_gamma_deph,
            'ood_gamma_relax': args.ood_gamma_relax
        }
    }
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Data saved to: {json_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    print("\nPanel (a) - Adaptation dynamics:")
    for init_id, data in losses_data.items():
        initial_fid = 1 - data['losses'][0]
        final_fid = 1 - data['losses'][-1]
        improvement = final_fid - initial_fid
        print(f"  {data['label']:20s}: {initial_fid:.4f} -> {final_fid:.4f} (+{improvement:.4f})")

    print("\nPanel (b) - Fidelity distributions:")
    for name, fids in fidelity_data.items():
        print(f"  {name:12s}: {np.mean(fids):.4f} +/- {np.std(fids):.4f}")


if __name__ == "__main__":
    main()
