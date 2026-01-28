"""
Main Meta-Training Script

Train meta-learned initialization for quantum control.
© 2025 The MITRE Corporation, All Rights Reserved  
"""

import torch
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
import argparse

from metaqctrl.quantum.lindblad import LindbladSimulator
from metaqctrl.quantum.noise_adapter import (
    TaskDistribution, NoisePSDModel, PSDToLindblad2, NoiseParameters,
    estimate_qubit_frequency_from_hamiltonian  
)
from metaqctrl.quantum.gates import GateFidelityComputer, TargetGates
from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.meta_rl.maml import MAML, MAMLTrainer

def create_quantum_system():
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    # System Hamiltonians
    H0 = 0.0 * sigma_z   
    H_controls = [sigma_x, sigma_y]

    # PSD model for noise
    psd_model = NoisePSDModel(model_type='one_over_f')
    omega_sample = np.array([1.0, 5.0, 10.0])

    psd_to_lindblad = PSDToLindblad(
        basis_operators=[sigma_x, sigma_y, sigma_z],
        sampling_freqs=omega_sample,
        psd_model=psd_model
    )

    return H0, H_controls, psd_to_lindblad

def create_task_distribution(config: dict):
    model_types = config.get('model_types')
    model_probs = config.get('model_probs')

    return TaskDistribution(
        dist_type=config.get('task_dist_type'),
        ranges={
            'alpha': tuple(config.get('alpha_range')),
            'A': tuple(config.get('A_range')),
            'omega_c': tuple(config.get('omega_c_range'))
        },
        model_types=model_types,   
        model_probs=model_probs   
    )


def task_sampler(n_tasks: int, split: str, task_dist: TaskDistribution, rng: np.random.Generator):
    ## Sample tasks
    """Sample tasks from distribution."""

    if split == 'train':
        seed_offset = 0
    elif split == 'val':
        seed_offset = 100000
    else:  # test
        seed_offset = 200000

    local_rng = np.random.default_rng(rng.integers(0, 1000000) + seed_offset)
    return task_dist.sample(n_tasks, local_rng)


def data_generator(
    task_params: NoiseParameters,
    n_trajectories: int,
    split: str,
    quantum_system: dict,
    config: dict,
    device: torch.device
):

    """Generate data for a task."""
    task_features = torch.tensor(
        task_params.to_array(), 
        dtype=torch.float32,
        device=device
    )

    # Repeat for batch
    task_features_batch = task_features.unsqueeze(0).repeat(n_trajectories, 1)

    return {
        'task_features': task_features_batch,
        'task_params': task_params,
        'quantum_system': quantum_system
    }


def create_loss_function(env, device, config):

    dt = config.get('dt_training') 
    use_rk4 = config.get('use_rk4_training')   


    def loss_fn(policy: torch.nn.Module, data: dict):
        """
        Loss = 1 - Fidelity(ρ_final, ρ_target)

        Args:
            policy: Policy network
            data: Dictionary with task_features and task_params

        Returns:
            loss: Scalar tensor
        """
        task_params = data['task_params']
 
        loss = env.compute_loss_differentiable(
            policy,
            task_params,
            device,
            use_rk4=use_rk4,
            dt=dt
        )

        return loss

    return loss_fn


def main(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 70) 
    print("Meta-RL for Quantum Control - Training")
    print("=" * 70)
    print(f"Config: {config_path}\n")

    seed = config.get('seed', 42)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    target_gate_name = config.get('target_gate') 
  
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
    
    from metaqctrl.quantum.quantum_environment import create_quantum_environment

    
    env = create_quantum_environment(config, target_state)
    print(f"  Environment created: {env.get_cache_stats()}")

    print("\nCreating task distribution...")
    task_dist = create_task_distribution(config)
    
    
    # Create policy
    print("Creating policy network...")
    policy = PulsePolicy(
        task_feature_dim=config.get('task_feature_dim'),
        hidden_dim=config.get('hidden_dim'),
        n_hidden_layers=config.get('n_hidden_layers'),
        n_segments=config.get('n_segments'),
        n_controls=config.get('n_controls'),
        output_scale=config.get('output_scale'),
        activation=config.get('activation')
    )
    policy = policy.to(device)
    print(f"  Parameters: {policy.count_parameters():,}")
    print(f"  Lipschitz constant: {policy.get_lipschitz_constant():.2f}")
    
    # Create MAML
    print("\nInitializing MAML...")
    maml = MAML(
        policy=policy,
        inner_lr=config.get('inner_lr'),
        inner_steps=config.get('inner_steps'),
        meta_lr=config.get('meta_lr'),
        first_order=config.get('first_order'),
        device=device
    )
    print(f"  Inner: {maml.inner_steps} steps @ lr={maml.inner_lr}")
    print(f"  Meta lr: {maml.meta_lr}")
    print(f"  Second-order: {not maml.first_order}")
    
    # Create loss function with GPU-optimized settings
    loss_fn = create_loss_function(env, device, config)

    # Print integration settings
    print(f"\nIntegration settings:")
    print(f"  dt: {config.get('dt_training', 0.01)}")
    print(f"  method: {'RK4' if config.get('use_rk4_training', True) else 'Euler'}")
    
    # Modified data generator to work with environment
    def data_generator_env(task_params, n_trajectories, split):
        """Generate data compatible with environment."""
        task_features = torch.tensor(
            task_params.to_array(),  
            dtype=torch.float32,
            device=device
        )

        task_features_batch = task_features.unsqueeze(0).repeat(n_trajectories, 1)
        return {
            'task_features': task_features_batch,
            'task_params': task_params  # Single task params
        }
    
    print("\nSetting up trainer...")
    trainer = MAMLTrainer(
        maml=maml,
        task_sampler=lambda n, split: task_sampler(n, split, task_dist, rng),
        data_generator=data_generator_env,
        loss_fn=loss_fn,
        n_support=config.get('n_support'),
        n_query=config.get('n_query'),
        log_interval=config.get('log_interval'),
        val_interval=config.get('val_interval')
    )
    
    save_dir = Path(config.get('save_dir', 'checkpoints'))
    save_dir.mkdir(parents=True, exist_ok=True) 
    save_path = save_dir / f"maml_best_pauli_x.pt"
    
    print(f"\nCheckpoints will be saved to: {save_path}")
    
    
    trainer.train(
        n_iterations=config.get('n_iterations'),
        tasks_per_batch=config.get('tasks_per_batch'),
        val_tasks=config.get('val_tasks'),
        save_path=str(save_path)
    )
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train meta-learned quantum controller')
    parser.add_argument(
        '--config',
        type=str,
        default='../../configs/experiment_config.yaml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    main(args.config)
