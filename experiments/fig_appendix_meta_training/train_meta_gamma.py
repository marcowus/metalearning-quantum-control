"""
Meta-Training Script for Gamma-Parameterized Quantum Control
© 2025 The MITRE Corporation, All Rights Reserved  
"""

import torch
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
import argparse

from metaqctrl.quantum.noise_models_gamma import GammaNoiseParameters, GammaTaskDistribution
from metaqctrl.quantum.lindblad_torch import DifferentiableLindbladSimulator
from metaqctrl.quantum.gates import TargetGates
from metaqctrl.meta_rl.policy_gamma import GammaPulsePolicy
from metaqctrl.meta_rl.maml import MAML, MAMLTrainer


def create_gamma_lindblad_system(gamma_deph: float, gamma_relax: float, device='cpu'):
    """
    Args:
        gamma_deph: Pure dephasing rate 
        gamma_relax: Relaxation rate  
        device: 'cpu' or 'cuda'

    Returns:
        sim: DifferentiableLindbladSimulator
    """
    # Pauli matrices
    sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)

    L_relax = torch.sqrt(torch.tensor(gamma_relax, dtype=torch.float32, device=device)) * \
              torch.tensor([[0, 1], [0, 0]], dtype=torch.complex64, device=device)
    L_deph = torch.sqrt(torch.tensor(gamma_deph / 2.0, dtype=torch.float32, device=device)) * \
             sigma_z.to(torch.complex64)

    L_operators = [L_relax, L_deph]

    # Hamiltonians
    H0 = torch.zeros((2, 2), dtype=torch.complex64, device=device)
    H_controls = [sigma_x, sigma_y]

    sim = DifferentiableLindbladSimulator(
        H0=H0,
        H_controls=H_controls,
        L_operators=L_operators,
        dt=0.05,
        device=device
    )

    return sim


def create_gamma_task_distribution(config: dict) -> GammaTaskDistribution:
    """Create task distribution over gamma parameters."""
    return GammaTaskDistribution(
        dist_type=config.get('task_dist_type', 'uniform'),
        gamma_deph_range=tuple(config.get('gamma_deph_range', [0.02, 0.15])),
        gamma_relax_range=tuple(config.get('gamma_relax_range', [0.01, 0.08])),
        diversity_scale=config.get('diversity_scale', 1.0)
    )


def gamma_task_sampler(n_tasks: int, split: str, task_dist: GammaTaskDistribution, rng: np.random.Generator):
    """Sample tasks from gamma distribution."""
    if split == 'train':
        seed_offset = 0
    elif split == 'val':
        seed_offset = 100000
    else:  # test
        seed_offset = 200000

    local_rng = np.random.default_rng(rng.integers(0, 1000000) + seed_offset)
    return task_dist.sample(n_tasks, local_rng)


def gamma_data_generator(
    task_params: GammaNoiseParameters,
    n_trajectories: int,
    split: str,
    device: torch.device
):
    """Generate data for a gamma task."""
    task_features = torch.tensor(
        task_params.to_array(normalized=True),
        dtype=torch.float32,
        device=device
    )

    # Repeat for batch
    task_features_batch = task_features.unsqueeze(0).repeat(n_trajectories, 1)

    return {
        'task_features': task_features_batch,
        'task_params': task_params
    }


def create_gamma_loss_function(target_state: np.ndarray, device, config: dict):
    """
    Create loss function for gamma-parameterized tasks.
    """
    target_state_torch = torch.tensor(target_state, dtype=torch.complex64, device=device)
    T = config.get('gate_time', 1.0)

    def loss_fn(policy: torch.nn.Module, data: dict):
        task_params = data['task_params']

        # Create simulator for this task
        sim = create_gamma_lindblad_system(
            task_params.gamma_deph,
            task_params.gamma_relax,
            device=str(device)
        )

        # Get task features
        task_features = data['task_features'][0]  # Take first from batch

        # Generate control sequence
        controls = policy(task_features)

        rho0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64, device=device)

        # Evolve quantum state
        rho_final = sim.forward(rho0, controls, T=T)

        # Compute fidelity
        fidelity = torch.real(torch.trace(rho_final @ target_state_torch))

        # Loss = 1 - Fidelity
        loss = 1.0 - fidelity

        return loss

    return loss_fn


def main(config_path: str):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set random seeds
    seed = config.get('seed', 42)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed: {seed}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Target gate
    target_gate_name = config.get('target_gate', 'pauli_x')
    if target_gate_name == 'hadamard':
        U_target = TargetGates.hadamard()
    elif target_gate_name == 'pauli_x':
        U_target = TargetGates.pauli_x()
    else:
        raise ValueError(f"Unknown target gate: {target_gate_name}")

    # Target state: U|0⟩
    ket_0 = np.array([1, 0], dtype=complex)
    target_state = np.outer(U_target @ ket_0, (U_target @ ket_0).conj())
    print(f"Target gate: {target_gate_name}")

    # Create gamma task distribution
    print("\nCreating gamma task distribution...")
    task_dist = create_gamma_task_distribution(config)
    print(f"  Gamma dephasing range: {task_dist.gamma_deph_range}")
    print(f"  Gamma relaxation range: {task_dist.gamma_relax_range}")
    print(f"  Distribution variance: {task_dist.compute_variance():.6f}")

    # Create policy (gamma features: 3D)
    print("\nCreating gamma policy network...")
    policy = GammaPulsePolicy(
        task_feature_dim=config.get('task_feature_dim', 3),
        hidden_dim=config.get('hidden_dim', 128),
        n_hidden_layers=config.get('n_hidden_layers', 2),
        n_segments=config.get('n_segments', 20),
        n_controls=config.get('n_controls', 2),
        output_scale=config.get('output_scale', 1.0),
        activation=config.get('activation', 'tanh')
    )
    policy = policy.to(device)
    print(f"  Parameters: {policy.count_parameters():,}")
    print(f"  Lipschitz constant: {policy.get_lipschitz_constant():.2f}")

    # Create MAML
    print("\nInitializing MAML...")
    maml = MAML(
        policy=policy,
        inner_lr=config.get('inner_lr', 0.01),
        inner_steps=config.get('inner_steps', 5),
        meta_lr=config.get('meta_lr', 0.001),
        first_order=config.get('first_order', False),
        device=device
    )
    print(f"  Inner: {maml.inner_steps} steps @ lr={maml.inner_lr}")
    print(f"  Meta lr: {maml.meta_lr}")
    print(f"  Second-order: {not maml.first_order}")

    # Create loss function
    loss_fn = create_gamma_loss_function(target_state, device, config)

    # Data generator wrapper
    def data_generator_wrapper(task_params, n_trajectories, split):
        return gamma_data_generator(task_params, n_trajectories, split, device)

    # Create trainer
    print("\nSetting up trainer...")
    trainer = MAMLTrainer(
        maml=maml,
        task_sampler=lambda n, split: gamma_task_sampler(n, split, task_dist, rng),
        data_generator=data_generator_wrapper,
        loss_fn=loss_fn,
        n_support=config.get('n_support', 1),
        n_query=config.get('n_query', 1),
        log_interval=config.get('log_interval', 10),
        val_interval=config.get('val_interval', 50)
    )

    # Create save directory
    save_dir = Path(config.get('save_dir', 'checkpoints'))
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"maml_gamma_{target_gate_name}.pt"

    print(f"\nCheckpoints will be saved to: {save_path}")

    # Train
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70 + "\n")

    trainer.train(
        n_iterations=config.get('n_iterations', 1000),
        tasks_per_batch=config.get('tasks_per_batch', 4),
        val_tasks=config.get('val_tasks', 20),
        save_path=str(save_path)
    )

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"\nFinal model saved to: {save_path}")
    print(f"Best model saved to: {str(save_path).replace('.pt', '_best.pt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train meta-learned gamma quantum controller')
    parser.add_argument(
        '--config',
        type=str,
        default='../../configs/experiment_config_gamma.yaml',
        help='Path to gamma config file'
    )

    args = parser.parse_args()

    main(args.config)
