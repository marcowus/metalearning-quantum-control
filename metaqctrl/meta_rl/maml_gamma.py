"""
Model-Agnostic Meta-Learning (MAML) for Gamma-Parameterized Quantum Control
© 2025 The MITRE Corporation, All Rights Reserved    
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
from copy import deepcopy

from metaqctrl.meta_rl.maml import MAML, MAMLTrainer
from metaqctrl.quantum.noise_models_gamma import GammaNoiseParameters, GammaTaskDistribution
from metaqctrl.quantum.lindblad_torch import DifferentiableLindbladSimulator


def create_gamma_lindblad_simulator(
    gamma_deph: float,
    gamma_relax: float,
    device: str = 'cpu'
) -> DifferentiableLindbladSimulator:
    """
    Create a DifferentiableLindbladSimulator with gamma-rate Lindblad operators.

    This directly uses gamma rates - no PSD-to-Lindblad conversion needed.

    Args:
        gamma_deph: Pure dephasing rate [1/s]
        gamma_relax: Relaxation rate [1/s]
        device: 'cpu' or 'cuda'

    Returns:
        sim: DifferentiableLindbladSimulator ready for quantum evolution
    """

    sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
    L_relax = torch.sqrt(torch.tensor(gamma_relax, dtype=torch.float32, device=device)) * \
              torch.tensor([[0, 1], [0, 0]], dtype=torch.complex64, device=device)

    L_deph = torch.sqrt(torch.tensor(gamma_deph / 2.0, dtype=torch.float32, device=device)) * \
             sigma_z.to(torch.complex64)

    L_operators = [L_relax, L_deph]
    H0 = torch.zeros((2, 2), dtype=torch.complex64, device=device)  # No drift
    H_controls = [sigma_x, sigma_y]

    sim = DifferentiableLindbladSimulator(
        H0=H0,
        H_controls=H_controls,
        L_operators=L_operators,
        dt=0.05,
        device=device
    )

    return sim


def compute_gamma_loss(
    policy: nn.Module,
    task_params: GammaNoiseParameters,
    target_state: torch.Tensor,
    device: str = 'cpu',
    T: float = 1.0
) -> torch.Tensor:
    """
    Compute loss for a gamma-parameterized task.

    Loss = 1 - Fidelity(ρ_final, ρ_target)

    Args:
        policy: GammaPulsePolicy network
        task_params: GammaNoiseParameters with gamma_deph, gamma_relax
        target_state: Target density matrix (2x2 complex tensor)
        device: 'cpu' or 'cuda'
        T: Total evolution time

    Returns:
        loss: Scalar tensor (differentiable)
    """

    sim = create_gamma_lindblad_simulator(
        task_params.gamma_deph,
        task_params.gamma_relax,
        device=device
    ) 
    task_features = torch.tensor(
        task_params.to_array(normalized=True),
        dtype=torch.float32,
        device=device
    )
    controls = policy(task_features)
    rho0 = torch.tensor(
        [[1, 0], [0, 0]],
        dtype=torch.complex64,
        device=device
    )

    rho_final = sim.forward(rho0, controls, T=T) 
    fidelity = torch.real(torch.trace(rho_final @ target_state.to(device))) ##state fidelity calculated ...  
    loss = 1.0 - fidelity
    return loss


class GammaMAMLTrainer(MAMLTrainer):
    """
    High-level trainer for MAML with gamma-parameterized tasks.
    """

    def __init__(
        self,
        maml: MAML,
        task_distribution: GammaTaskDistribution,
        target_state: torch.Tensor,
        device: torch.device = torch.device('cpu'),
        n_support: int = 1,
        n_query: int = 1,
        log_interval: int = 10,
        val_interval: int = 50
    ):
        """
        Args:
            maml: MAML instance with GammaPulsePolicy
            task_distribution: GammaTaskDistribution for sampling tasks
            target_state: Target quantum state (2x2 density matrix)
            device: torch device
            n_support: Number of support evaluations per task
            n_query: Number of query evaluations per task
            log_interval: Log every N iterations
            val_interval: Validate every N iterations
        """
        self.task_distribution = task_distribution
        self.target_state = target_state
        self.device = device

        # Create loss function
        def gamma_loss_fn(policy, data):
            task_params = data['task_params']
            return compute_gamma_loss(
                policy, task_params, self.target_state, str(self.device)
            )

        def gamma_task_sampler(n_tasks, split='train'):
            return self.task_distribution.sample(n_tasks)

        def gamma_data_generator(task_params, n_trajectories, split):
            task_features = torch.tensor(
                task_params.to_array(normalized=True),
                dtype=torch.float32,
                device=self.device
            )
            return {
                'task_features': task_features.unsqueeze(0).repeat(n_trajectories, 1),
                'task_params': task_params
            }

        super().__init__(
            maml=maml,
            task_sampler=gamma_task_sampler,
            data_generator=gamma_data_generator,
            loss_fn=gamma_loss_fn,
            n_support=n_support,
            n_query=n_query,
            log_interval=log_interval,
            val_interval=val_interval
        )


__all__ = [
    'MAML',
    'MAMLTrainer',
    'GammaMAMLTrainer',
    'GammaNoiseParameters',
    'GammaTaskDistribution',
    'create_gamma_lindblad_simulator',
    'compute_gamma_loss'
] 
