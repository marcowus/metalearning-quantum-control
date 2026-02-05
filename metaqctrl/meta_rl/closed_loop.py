"""
Closed-Loop Meta-Quantum Control
Integrates Calibration, Hardware Constraints, and Robust Control.
© 2025 The MITRE Corporation, All Rights Reserved
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Callable, Optional

from metaqctrl.meta_rl.maml import MAML, MAMLTrainer
from metaqctrl.meta_rl.maml_gamma import create_gamma_lindblad_simulator
from metaqctrl.quantum.noise_models_gamma import GammaNoiseParameters, GammaTaskDistribution
from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.meta_rl.estimators import NeuralEstimator
from metaqctrl.quantum.hardware import PulseShaper
from metaqctrl.quantum.measurement import measure_probs, simulate_counts
from metaqctrl.meta_rl.robust import cvar_loss, entropic_risk_measure

class ClosedLoopPolicy(nn.Module):
    """
    Integrated policy for closed-loop control.
    Contains:
      - Diagnostic pulses (learnable or fixed)
      - Estimator (Measurement -> Parameters)
      - Control Policy (Parameters -> Pulses)
      - Pulse Shaper (Pulses -> Hardware Pulses)
    """
    def __init__(
        self,
        estimator: NeuralEstimator,
        control_policy: PulsePolicy,
        pulse_shaper: PulseShaper,
        diagnostic_pulses: torch.Tensor, # (n_diag, n_segments, n_controls)
        target_state: torch.Tensor
    ):
        super().__init__()
        self.estimator = estimator
        self.control_policy = control_policy
        self.pulse_shaper = pulse_shaper
        self.target_state = target_state

        # Diagnostic pulses can be parameters (learnable) or buffers (fixed)
        self.register_parameter('diagnostic_pulses', nn.Parameter(diagnostic_pulses))

    def forward(self, task_params_real: GammaNoiseParameters, device: str = 'cpu') -> Dict:
        """
        Full closed-loop simulation forward pass.

        Args:
            task_params_real: The TRUE environment parameters.
            device: torch device.

        Returns:
            Dict containing loss, fidelity, estimated_params, etc.
        """
        # 1. Environment Simulation (Real World)
        # We need a simulator grounded in the REAL parameters
        sim_real = create_gamma_lindblad_simulator(
            task_params_real.gamma_deph,
            task_params_real.gamma_relax,
            device=device
        )

        rho0 = torch.zeros((2, 2), dtype=torch.complex64, device=device)
        rho0[0, 0] = 1.0

        # 2. Diagnostic Phase
        # Apply diagnostic pulses to the real system
        # Note: Diagnostic pulses are also subject to hardware shaping!
        diag_pulses_phys = self.pulse_shaper(self.diagnostic_pulses)

        measurements = []
        for i in range(diag_pulses_phys.shape[0]):
            u_diag = diag_pulses_phys[i]
            rho_final, _ = sim_real.evolve(rho0, u_diag, T=1.0) # Assume T=1.0 for diag

            # Measure
            prob = measure_probs(rho_final) # Project to |1>
            # Use soft counts for differentiability
            count = simulate_counts(prob, n_shots=1000, method='soft')
            measurements.append(count)

        measurements_tensor = torch.stack(measurements).unsqueeze(0) # (1, n_diag)

        # 3. Estimation Phase
        # Estimate parameters from measurements
        # Output is [gamma_deph, gamma_relax]
        estimated_params_raw = self.estimator(measurements_tensor)

        # Normalize for policy input (as expected by PulsePolicy)
        # Policy expects [gamma_deph/0.1, gamma_relax/0.05, sum/0.15]
        # We need to compute this normalization manually here
        gamma_deph_est = estimated_params_raw[:, 0]
        gamma_relax_est = estimated_params_raw[:, 1]

        # Enforce positive rates (already Softplus in estimator, but good to be safe)
        gamma_deph_est = torch.abs(gamma_deph_est)
        gamma_relax_est = torch.abs(gamma_relax_est)

        norm_deph = gamma_deph_est / 0.1
        norm_relax = gamma_relax_est / 0.05
        norm_sum = (gamma_deph_est + gamma_relax_est) / 0.15

        task_features = torch.stack([norm_deph, norm_relax, norm_sum], dim=1) # (1, 3)

        # 4. Control Phase
        # Generate gate pulses based on ESTIMATED parameters
        controls_ideal = self.control_policy(task_features) # (1, n_seg, n_ctrl)

        # Apply hardware shaping
        controls_phys = self.pulse_shaper(controls_ideal)

        # 5. Execution Phase
        # Apply shaped pulses to the REAL system
        rho_gate_final, _ = sim_real.evolve(rho0, controls_phys[0], T=1.0) # Assume T=1.0

        # 6. Loss Computation
        # Fidelity
        fidelity = torch.real(torch.trace(rho_gate_final @ self.target_state.to(device)))

        # Hardware penalties
        slew_loss = self.pulse_shaper.compute_slew_penalty(controls_phys)

        # Total loss
        # We can weight the slew loss
        loss = (1.0 - fidelity) + 0.001 * slew_loss

        return {
            'loss': loss,
            'fidelity': fidelity,
            'estimated_params': estimated_params_raw,
            'real_params': torch.tensor([task_params_real.gamma_deph, task_params_real.gamma_relax], device=device),
            'slew_loss': slew_loss
        }


class ClosedLoopMAML(MAML):
    """
    MAML adapted for Closed-Loop Robust Control.
    Overrides meta_train_step to support CVaR aggregation.
    """
    def __init__(self, robust_alpha: float = 0.9, **kwargs):
        super().__init__(**kwargs)
        self.robust_alpha = robust_alpha

    def meta_train_step(
        self,
        task_batch: List[Dict],
        loss_fn: Callable,
        use_higher: bool = False # Force False for Route B (no inner loop)
    ) -> Dict[str, float]:
        """
        Meta-training step with Robust Aggregation.
        """
        self.meta_optimizer.zero_grad()

        task_losses = []
        fidelities = []

        # Accumulate gradients manually?
        # Or compute total loss and backward once?
        # CVaR requires collecting all losses first.

        # Forward pass for all tasks
        for task_data in task_batch:
            # Route B: "inner_steps" should be 0, so 'inner_loop' just returns policy.
            # But we can just skip inner loop and call loss_fn directly if we know we are in Route B.

            # loss_fn here is expected to be 'compute_closed_loop_loss_wrapper'
            # It returns the dictionary from ClosedLoopPolicy

            result = loss_fn(self.policy, task_data)
            loss = result['loss']

            task_losses.append(loss)
            fidelities.append(result['fidelity'].item())

        # Stack losses
        if len(task_losses) == 0:
            return {'error': 'no_tasks'}

        losses_tensor = torch.stack(task_losses)

        # Robust Aggregation (CVaR)
        if self.robust_alpha < 1.0:
            meta_loss = cvar_loss(losses_tensor, alpha=self.robust_alpha)
        else:
            meta_loss = torch.mean(losses_tensor)

        # Backward
        meta_loss.backward()

        # Clip grad
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)

        # Step
        self.meta_optimizer.step()

        return {
            'meta_loss': meta_loss.item(),
            'mean_fidelity': np.mean(fidelities),
            'min_fidelity': np.min(fidelities),
            'cvar_fidelity': 1.0 - meta_loss.item(), # Approx
            'grad_norm': grad_norm.item()
        }

    def meta_validate(
        self,
        val_tasks: List[Dict],
        loss_fn: Callable
    ) -> Dict[str, float]:
        """
        Validate Closed-Loop Policy.
        No adaptation needed for Route B.
        """
        self.policy.eval()

        val_losses = []
        fidelities = []

        with torch.no_grad():
            for task_data in val_tasks:
                # loss_fn returns dict
                result = loss_fn(self.policy, task_data)
                val_losses.append(result['loss'].item())
                fidelities.append(result['fidelity'].item())

        self.policy.train()

        mean_loss = np.mean(val_losses)
        mean_fid = np.mean(fidelities)

        return {
            'val_loss_pre_adapt': mean_loss,   # "Pre" and "Post" are same
            'val_loss_post_adapt': mean_loss,
            'adaptation_gain': 0.0,
            'std_post_adapt': np.std(val_losses),
            'mean_fidelity': mean_fid
        }

class ClosedLoopMAMLTrainer(MAMLTrainer):
    """
    Trainer for Closed-Loop system.
    """
    def __init__(
        self,
        maml: ClosedLoopMAML,
        task_distribution: GammaTaskDistribution,
        device: torch.device,
        **kwargs
    ):
        self.task_distribution = task_distribution
        self.device = device

        # Define the specialized loss function
        def closed_loop_loss_fn(policy, data):
            # data is just wrapper around task_params
            task_params = data['task_params']
            return policy(task_params, device=str(device))

        # Sampler
        def gamma_task_sampler(n_tasks, split='train'):
            return self.task_distribution.sample(n_tasks)

        # Data Generator (simplified for Route B)
        def gamma_data_generator(task_params, n_trajectories, split):
            # We don't need trajectories in the traditional sense for Route B
            # Just pass the task params
            return {'task_params': task_params}

        super().__init__(
            maml=maml,
            task_sampler=gamma_task_sampler,
            data_generator=gamma_data_generator,
            loss_fn=closed_loop_loss_fn,
            n_support=1, # Dummy
            n_query=1,   # Dummy
            **kwargs
        )
