"""
Figure 4: Adaptation Dynamics.
Shows per-task adaptation dynamics, fidelity distributions, and pulse sequences. 
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
    """Load pretrained gamma policy from checkpoint."""
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


def train_robust_policy_gamma(n_iterations=300, inner_lr=0.001, device='cpu'):
    """Train a robust policy on a fixed average gamma level."""
    print("Training robust baseline on FIXED average gamma rates...")

    policy = GammaPulsePolicy(
        task_feature_dim=3,
        hidden_dim=64,
        n_hidden_layers=2,
        n_segments=20,
        n_controls=2,
        output_scale=1.0
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=inner_lr)

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

    print(f"Robust training complete (fixed gamma_deph={avg_gamma_deph:.3f}, gamma_relax={avg_gamma_relax:.3f})")
    return policy


def generate_panel_a_data(meta_policy, robust_policy, max_K=50, inner_lr=0.001, device='cpu',
                          ood_gamma_deph=0.35, ood_gamma_relax=0.15):
    """
    Panel (a): Loss vs K for different initializations on a challenging task.
    Shows exponential convergence with different starting points.
    """
    print("Generating Panel (a) data: Loss vs K...")
    print(f"  Challenging task: gamma_deph={ood_gamma_deph}, gamma_relax={ood_gamma_relax}")
    print(f"  (Training range: gamma_deph=[0.02,0.15], gamma_relax=[0.01,0.08])")

    torch.manual_seed(42)
    np.random.seed(42)

    test_gamma_deph = ood_gamma_deph
    test_gamma_relax = ood_gamma_relax

    losses_by_init = {}

    meta_adapted = deepcopy(meta_policy)
    meta_adapted.train()
    opt = optim.Adam(meta_adapted.parameters(), lr=inner_lr)
    meta_losses = []

    with torch.no_grad():
        loss_val = compute_loss_gamma(meta_adapted, test_gamma_deph, test_gamma_relax, device).item()
        meta_losses.append(loss_val)

    for k in range(max_K):
        opt.zero_grad()
        loss = compute_loss_gamma(meta_adapted, test_gamma_deph, test_gamma_relax, device)
        loss.backward()
        opt.step()
        with torch.no_grad():
            loss_val = compute_loss_gamma(meta_adapted, test_gamma_deph, test_gamma_relax, device).item()
            meta_losses.append(loss_val)

    losses_by_init[0] = {'losses': meta_losses, 'label': 'Meta-learned (MAML)'}

    robust_adapted = deepcopy(robust_policy)
    robust_adapted.train()
    opt = optim.Adam(robust_adapted.parameters(), lr=inner_lr)
    robust_losses = []

    with torch.no_grad():
        loss_val = compute_loss_gamma(robust_adapted, test_gamma_deph, test_gamma_relax, device).item()
        robust_losses.append(loss_val)

    for k in range(max_K):
        opt.zero_grad()
        loss = compute_loss_gamma(robust_adapted, test_gamma_deph, test_gamma_relax, device)
        loss.backward()
        opt.step()
        with torch.no_grad():
            loss_val = compute_loss_gamma(robust_adapted, test_gamma_deph, test_gamma_relax, device).item()
            robust_losses.append(loss_val)

    losses_by_init[1] = {'losses': robust_losses, 'label': 'Fixed Average'}

    # 3. Random initialization
    fresh_policy = GammaPulsePolicy(
        task_feature_dim=3,
        hidden_dim=64,
        n_hidden_layers=2,
        n_segments=20,
        n_controls=2,
        output_scale=1.0
    ).to(device)
    fresh_policy.train()
    opt = optim.Adam(fresh_policy.parameters(), lr=inner_lr)
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

    losses_by_init[2] = {'losses': fresh_losses, 'label': 'Random Init'}

    return losses_by_init


def generate_panel_b_data(meta_policy, robust_policy, n_tasks=50, K_adapt=10, inner_lr=0.001, device='cpu',
                          ood_gamma_deph=None, ood_gamma_relax=None):
    """
    Panel (b): Fidelity distributions across tasks.
    Compares: Robust baseline, Meta-init (K=0), Adapted (K=K_adapt)

    If ood_gamma_deph/relax provided, samples around those values (challenging regime)
    """
    print("Generating Panel (b) data: Fidelity distributions...")

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
        # Sample from training distribution (in-distribution)
        gamma_deph_range = (0.02, 0.15)
        gamma_relax_range = (0.01, 0.08)
        gamma_deph_vals = np.random.uniform(*gamma_deph_range, n_tasks)
        gamma_relax_vals = np.random.uniform(*gamma_relax_range, n_tasks)

    fidelities = {
        'robust': [],
        'meta_init': [],
        'adapted': []
    }

    for i, (gamma_deph, gamma_relax) in enumerate(zip(gamma_deph_vals, gamma_relax_vals)):
        if (i + 1) % 10 == 0:
            print(f"  Processing task {i+1}/{n_tasks}...")

        
        with torch.no_grad():
            _, fid_robust = compute_loss_gamma(robust_policy, gamma_deph, gamma_relax, device, return_fidelity=True)
            fidelities['robust'].append(fid_robust.item())

        # Meta-init (before adaptation)
        with torch.no_grad():
            _, fid_meta = compute_loss_gamma(meta_policy, gamma_deph, gamma_relax, device, return_fidelity=True)
            fidelities['meta_init'].append(fid_meta.item())

        # Adapted (after K steps)
        adapted_policy = deepcopy(meta_policy)
        adapted_policy.train()
        inner_opt = optim.Adam(adapted_policy.parameters(), lr=inner_lr)

        for _ in range(K_adapt):
            inner_opt.zero_grad()
            loss = compute_loss_gamma(adapted_policy, gamma_deph, gamma_relax, device)
            loss.backward()
            inner_opt.step()

        with torch.no_grad():
            _, fid_adapted = compute_loss_gamma(adapted_policy, gamma_deph, gamma_relax, device, return_fidelity=True)
            fidelities['adapted'].append(fid_adapted.item())

    return fidelities


def generate_panel_c_data(meta_policy, K_adapt=20, inner_lr=0.001, device='cpu',
                          ood_gamma_deph=None, ood_gamma_relax=None):
    """
    Panel (c): Pulse sequences from different initializations.
    Shows how meta vs random init lead to different control sequences.
    """
    print("Generating Panel (c) data: Pulse sequences...")

    torch.manual_seed(42)


    if ood_gamma_deph is not None and ood_gamma_relax is not None:
        gamma_deph = ood_gamma_deph
        gamma_relax = ood_gamma_relax
        print(f"  Using challenging task: gamma_deph={gamma_deph:.2f}, gamma_relax={gamma_relax:.2f}")
    else:
        gamma_deph = 0.08
        gamma_relax = 0.04

    pulses = {}

    # 1. Meta-policy (K=0)
    with torch.no_grad():
        _, _, controls = compute_loss_gamma(meta_policy, gamma_deph, gamma_relax, device,
                                            return_fidelity=True, return_controls=True)
    pulses[0] = {
        'controls': controls.detach().cpu().numpy(),
        'label': 'Meta-init (K=0)'
    }

    # 2. Meta-policy after adaptation (K=K_adapt)
    adapted_policy = deepcopy(meta_policy)
    adapted_policy.train()
    opt = optim.Adam(adapted_policy.parameters(), lr=inner_lr)

    for _ in range(K_adapt):
        opt.zero_grad()
        loss = compute_loss_gamma(adapted_policy, gamma_deph, gamma_relax, device)
        loss.backward()
        opt.step()

    with torch.no_grad():
        _, _, controls = compute_loss_gamma(adapted_policy, gamma_deph, gamma_relax, device,
                                            return_fidelity=True, return_controls=True)
    pulses[1] = {
        'controls': controls.detach().cpu().numpy(),
        'label': f'Meta-adapted (K={K_adapt})'
    }

    # 3. Random init after adaptation
    fresh_policy = GammaPulsePolicy(
        task_feature_dim=3,
        hidden_dim=64,
        n_hidden_layers=2,
        n_segments=20,
        n_controls=2,
        output_scale=1.0
    ).to(device)
    fresh_policy.train()
    opt = optim.Adam(fresh_policy.parameters(), lr=inner_lr)

    for _ in range(K_adapt):
        opt.zero_grad()
        loss = compute_loss_gamma(fresh_policy, gamma_deph, gamma_relax, device)
        loss.backward()
        opt.step()

    with torch.no_grad():
        _, _, controls = compute_loss_gamma(fresh_policy, gamma_deph, gamma_relax, device,
                                            return_fidelity=True, return_controls=True)
    pulses[2] = {
        'controls': controls.detach().cpu().numpy(),
        'label': f'Random-adapted (K={K_adapt})'
    }

    return pulses


def create_figure(losses_data, fidelity_data, pulse_data, save_path=None):
    """Create the 3-panel figure."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    colors_init = ['#2ecc71', '#e74c3c', '#e67e22']  # Meta=green, Fixed=red, Random=orange

    # --- Panel (a): Loss vs K (log scale) ---
    ax = axes[0]

    for i, (init_id, data) in enumerate(losses_data.items()):
        losses = data['losses']
        K_vals = np.arange(len(losses))
        ax.semilogy(K_vals, losses, 'o-', color=colors_init[i], markersize=3,
                    linewidth=1.5, alpha=0.8, label=data['label'])

    ax.set_xlabel('Adaptation Steps $K$')
    ax.set_ylabel('Loss $\\mathcal{L}$ (log scale)')
    ax.set_title('(a) Adaptation Dynamics')
    ax.set_xlim(-0.5, len(losses_data[0]['losses']) - 0.5)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper right', fontsize=8)

    # --- Panel (b): Fidelity Distribution ---
    ax = axes[1]

    positions = [1, 2, 3]
    labels = ['Fixed Avg', 'Meta-init', 'Adapted']
    data_lists = [fidelity_data['robust'], fidelity_data['meta_init'], fidelity_data['adapted']]
    colors_violin = ['#e74c3c', '#3498db', '#2ecc71']

    parts = ax.violinplot(data_lists, positions=positions, showmeans=True, showmedians=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_violin[i])
        pc.set_alpha(0.7)

    parts['cmeans'].set_color('black')
    parts['cmedians'].set_color('white')

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Fidelity $\\mathcal{F}$')
    ax.set_title('(b) Fidelity Distribution')
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    # Add statistics
    for i, (pos, data) in enumerate(zip(positions, data_lists)):
        mean = np.mean(data)
        std = np.std(data)
        ax.text(pos, 0.55, f'{mean:.2f}$\\pm${std:.2f}',
                ha='center', fontsize=8, color=colors_violin[i])

    # --- Panel (c): Pulse Sequences ---
    ax = axes[2]

    n_segments = list(pulse_data.values())[0]['controls'].shape[0]
    t = np.linspace(0, 1, n_segments)

    colors_pulse = ['#3498db', '#2ecc71']   
    K_adapt_val = 10  # Default
    if '1' in pulse_data and 'label' in pulse_data['1']: 
        import re
        match = re.search(r'K=(\d+)', pulse_data['1'].get('label', ''))
        if match:
            K_adapt_val = int(match.group(1))
    labels_pulse = ['Meta-init (K=0)', f'Meta-adapted (K={K_adapt_val})']

    for i, (pulse_id, data) in enumerate(pulse_data.items()):
        if i >= 2:  # Skip random-adapted
            continue
        controls = data['controls']
        label = labels_pulse[i]

        from scipy.interpolate import make_interp_spline
        t_smooth = np.linspace(0, 1, 200)
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
                linewidth=2.0, label=f'{label}: $u_x$', alpha=0.9)
        ax.plot(t_smooth, controls_y_smooth, linestyle='--', color=colors_pulse[i],
                linewidth=1.5, label=f'{label}: $u_y$', alpha=0.9)

    ax.set_xlabel('Time $t/T$')
    ax.set_ylabel('Control Amplitude')
    ax.set_title('(c) Learned Pulse Sequences')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=7, ncol=1)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.close()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints_gamma/maml_gamma_pauli_x_best.pt',
                        help='Path to gamma checkpoint')
    parser.add_argument('--output', type=str, default='adaptation_dynamics_gamma_checkpoint',
                        help='Output filename prefix')
    parser.add_argument('--max_K', type=int, default=50, help='Max adaptation steps for panel (a)')
    parser.add_argument('--n_tasks', type=int, default=50, help='Number of tasks for panel (b)')
    parser.add_argument('--K_adapt', type=int, default=10, help='Adaptation steps for panel (b)')
    parser.add_argument('--inner_lr', type=float, default=0.001, help='Inner learning rate')
    parser.add_argument('--ood_gamma_deph', type=float, default=1.05,
                        help='Challenging gamma_deph (training max=0.15, default=1.05 is 7×)')
    parser.add_argument('--ood_gamma_relax', type=float, default=0.56,
                        help='Challenging gamma_relax (training max=0.08, default=0.56 is 7×)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Parameters: max_K={args.max_K}, n_tasks={args.n_tasks}, K_adapt={args.K_adapt}, inner_lr={args.inner_lr}")
    print(f"OOD task: gamma_deph={args.ood_gamma_deph} (training max=0.15), gamma_relax={args.ood_gamma_relax} (training max=0.08)")

    # Load meta-policy
    checkpoint_path = project_root / args.checkpoint
    meta_policy = load_pretrained_gamma_policy(checkpoint_path, device)
    meta_policy.eval()

    # Train robust baseline
    robust_policy = train_robust_policy_gamma(n_iterations=300, inner_lr=args.inner_lr, device=device)
    robust_policy.eval()

    # Generate data for each panel
    print("\n" + "-" * 50)
    losses_data = generate_panel_a_data(meta_policy, robust_policy, max_K=args.max_K,
                                        inner_lr=args.inner_lr, device=device,
                                        ood_gamma_deph=args.ood_gamma_deph,
                                        ood_gamma_relax=args.ood_gamma_relax)

    print("-" * 50)
    fidelity_data = generate_panel_b_data(meta_policy, robust_policy, n_tasks=args.n_tasks,
                                          K_adapt=args.K_adapt, inner_lr=args.inner_lr, device=device,
                                          ood_gamma_deph=args.ood_gamma_deph, ood_gamma_relax=args.ood_gamma_relax)

    print("-" * 50)
    pulse_data = generate_panel_c_data(meta_policy, K_adapt=args.K_adapt, inner_lr=args.inner_lr, device=device,
                                       ood_gamma_deph=args.ood_gamma_deph, ood_gamma_relax=args.ood_gamma_relax)

    # Create figure
    print("\n" + "-" * 50)
    print("Creating figure...")

    output_dir = Path(__file__).parent
    save_path = str(output_dir / f"{args.output}.png")

    create_figure(losses_data, fidelity_data, pulse_data, save_path=save_path)

    json_path = str(output_dir / f"{args.output}_data.json")
    results = {
        'panel_a': {k: {'losses': v['losses'], 'label': v['label']} for k, v in losses_data.items()},
        'panel_b': {k: v for k, v in fidelity_data.items()},
        'panel_c': {k: {'label': v['label']} for k, v in pulse_data.items()},  # Skip numpy arrays
        'params': {
            'checkpoint': args.checkpoint,
            'max_K': args.max_K,
            'n_tasks': args.n_tasks,
            'K_adapt': args.K_adapt,
            'inner_lr': args.inner_lr,
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

    for init_id, data in losses_data.items():
        initial_fid = 1 - data['losses'][0]
        final_fid = 1 - data['losses'][-1]
        improvement = final_fid - initial_fid
        print(f"  {data['label']:20s}: {initial_fid:.4f} -> {final_fid:.4f} (+{improvement:.4f})")

    for name, fids in fidelity_data.items():
        print(f"  {name:12s}: {np.mean(fids):.4f} +/- {np.std(fids):.4f}")




if __name__ == "__main__":
    main()
