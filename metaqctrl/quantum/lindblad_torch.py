"""
Differentiable Lindblad Master Equation Simulator (PyTorch)

This module provides a fully differentiable implementation of the Lindblad
master equation using PyTorch.   
© 2025 The MITRE Corporation, All Rights Reserved   
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class DifferentiableLindbladSimulator(nn.Module):
    """
    PyTorch-native Lindblad simulator with full gradient support. 
    """

    def __init__(
        self,
        H0: torch.Tensor,
        H_controls: List[torch.Tensor],
        L_operators: List[torch.Tensor],
        dt: float = 0.005,   
        method: str = 'rk4',
        device: torch.device = torch.device('cpu'),
        max_control_amp: float = 10.0   
    ):
        """
        Args:
            H0: Drift Hamiltonian (d x d) complex tensor
            H_controls: List of control Hamiltonians, each (d x d) complex
            L_operators: List of Lindblad operators, each (d x d) complex
            dt: Time step for integration (smaller = more accurate)
            method: Integration method ('euler', 'rk4')
            device: torch device
            max_control_amp: Maximum allowed control amplitude (for stability)
        """
        super().__init__()

        self.device = device
        self.method = method
        self.dt = dt
        self.max_control_amp = max_control_amp
        self.H0 = H0.to(device).to(torch.complex64)
        self.H_controls = [H.to(device).to(torch.complex64) for H in H_controls]
        self.L_operators = [L.to(device).to(torch.complex64) for L in L_operators]

        self.d = H0.shape[0]
        self.n_controls = len(H_controls)
        self.n_lindblad = len(L_operators)

        self.anti_commutators = [
            L.conj().T @ L for L in self.L_operators
        ]

    def lindbladian(
        self,
        rho: torch.Tensor,
        u: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Lindblad time derivative: ρ̇ = L[ρ].

        Args:
            rho: Density matrix (d x d) complex tensor
            u: Control amplitudes (n_controls,) real tensor

        Returns:
            drho_dt: Time derivative (d x d) complex tensor
        """

        u = torch.clamp(u, -self.max_control_amp, self.max_control_amp)

        H_total = self.H0 + sum(u_k * H_k for u_k, H_k in zip(u, self.H_controls))

        hamiltonian_term = -1j * (H_total @ rho - rho @ H_total)

        dissipation_term = torch.zeros_like(rho)
        for j, L_j in enumerate(self.L_operators):
            dissipation_term = dissipation_term + L_j @ rho @ L_j.conj().T 
            dissipation_term = dissipation_term - 0.5 * (
                self.anti_commutators[j] @ rho + rho @ self.anti_commutators[j]
            )

        return hamiltonian_term + dissipation_term

    def step_euler(
        self,
        rho: torch.Tensor,
        u: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """
        Single Euler integration step: ρ(t+dt) = ρ(t) + dt * ρ̇(t).

        Args:
            rho: Current density matrix
            u: Control amplitudes
            dt: Time step

        Returns:
            rho_next: Density matrix at next time step
        """
        drho = self.lindbladian(rho, u)
        return rho + dt * drho

    def step_rk4(
        self,
        rho: torch.Tensor,
        u: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """
        Single RK4 integration step (4th order Runge-Kutta).

        More accurate than Euler, still fully differentiable.

        Args:
            rho: Current density matrix
            u: Control amplitudes
            dt: Time step

        Returns:
            rho_next: Density matrix at next time step
        """
        k1 = self.lindbladian(rho, u)
        k2 = self.lindbladian(rho + 0.5 * dt * k1, u)
        k3 = self.lindbladian(rho + 0.5 * dt * k2, u)
        k4 = self.lindbladian(rho + dt * k3, u)

        rho_next = rho + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return rho_next

    def evolve(
        self,
        rho0: torch.Tensor,
        control_sequence: torch.Tensor,
        T: float,
        return_trajectory: bool = False,
        normalize: bool = True   
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Evolve state from t=0 to t=T under piecewise-constant controls.

        FULLY DIFFERENTIABLE - gradients flow through entire simulation!

        Args:
            rho0: Initial state (d x d) complex tensor
            control_sequence: Control pulses (n_segments, n_controls) real tensor
            T: Total evolution time
            return_trajectory: If True, return full trajectory
            normalize: If True, renormalize density matrix to maintain trace=1

        Returns:
            rho_final: Final state (d x d)
            trajectory: Optional trajectory tensor (n_segments+1, d, d)
        """
        n_segments = control_sequence.shape[0]
        t_segment = T / n_segments

        # Ensure rho0 is complex
        rho = rho0.to(torch.complex64).to(self.device)
        trace0 = torch.trace(rho).real
        # Renormalize initial state if needed
        if torch.abs(trace0 - 1.0) > 0.01 and trace0 > 1e-10:
            rho = rho / trace0

        if return_trajectory:
            trajectory = [rho.clone()]
        for seg_idx in range(n_segments):
            u_seg = control_sequence[seg_idx]
            n_substeps = max(1, int(t_segment / self.dt))
            dt_substep = t_segment / n_substeps

            for substep in range(n_substeps):
                if self.method == 'euler':
                    rho = self.step_euler(rho, u_seg, dt_substep)
                elif self.method == 'rk4':
                    rho = self.step_rk4(rho, u_seg, dt_substep)
                else:
                    raise ValueError(f"Unknown method: {self.method}")

                if normalize:
                    trace = torch.trace(rho).real
                    if trace > 1e-10: 
                        rho = rho / trace
                    else:
                        rho = rho0.clone()
                        if return_trajectory:
                            trajectory.append(rho.clone())
                        return rho, (torch.stack(trajectory) if return_trajectory else None)

                    if torch.isnan(rho).any() or torch.isinf(rho).any():
                        rho = rho0.clone()


            if return_trajectory:
                trajectory.append(rho.clone())
        trace_final = torch.trace(rho).real
        if trace_final > 1e-10:
            rho = rho / trace_final

        if return_trajectory:
            trajectory_tensor = torch.stack(trajectory)
            return rho, trajectory_tensor
        else:
            return rho, None

    def forward(
        self,
        rho0: torch.Tensor,
        control_sequence: torch.Tensor,
        T: float
    ) -> torch.Tensor:
        """
        Forward pass for nn.Module compatibility.

        Args:
            rho0: Initial state (d x d)
            control_sequence: Controls (n_segments, n_controls)
            T: Evolution time

        Returns:
            rho_final: Final state (d x d)
        """
        rho_final, _ = self.evolve(rho0, control_sequence, T, return_trajectory=False)
        return rho_final


def numpy_to_torch_complex(array, device='cpu'):
    """
    Convert NumPy complex array to PyTorch complex tensor.

    Args:
        array: NumPy array (complex)
        device: Target device

    Returns:
        tensor: PyTorch complex tensor
    """
    import numpy as np

    if np.iscomplexobj(array):
        real_part = torch.from_numpy(array.real).float()
        imag_part = torch.from_numpy(array.imag).float()
        tensor = torch.complex(real_part, imag_part)
    else:
        tensor = torch.from_numpy(array).float()
        tensor = tensor.to(torch.complex64)

    return tensor.to(device)


def torch_to_numpy_complex(tensor):
    """
    Convert PyTorch complex tensor to NumPy complex array.

    Args:
        tensor: PyTorch complex tensor

    Returns:
        array: NumPy complex array
    """
    import numpy as np

    tensor_cpu = tensor.cpu()
    real_part = tensor_cpu.real.numpy()
    imag_part = tensor_cpu.imag.numpy()

    return real_part + 1j * imag_part 
