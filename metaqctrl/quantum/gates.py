"""
Gate Fidelity Computation
© 2025 The MITRE Corporation, All Rights Reserved   
"""

import numpy as np
from scipy.linalg import sqrtm
from typing import Tuple, Optional
try:
    import jax.numpy as jnp
    from jax import jit
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = None
    jit = lambda f: f  # No-op decorator if JAX not available


def state_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Compute quantum state fidelity. 
    
    Args:
        rho: Density matrix 1
        sigma: Density matrix 2
        
    Returns:
        fidelity: Value in [0, 1]
    """
    sqrt_rho = sqrtm(rho)
    M = sqrt_rho @ sigma @ sqrt_rho
    sqrt_M = sqrtm(M)
    return np.real(np.trace(sqrt_M)) ** 2


def process_fidelity_choi(Phi_choi: np.ndarray, Psi_choi: np.ndarray) -> float:
    """ 
    Process fidelity via Choi matrices:
    
    Args:
        Phi_choi: Choi matrix of channel phi 
        Psi_choi: Choi matrix of channel psi 
 
    Returns:
        fidelity: Value in [0, 1]
    """
    d2 = Phi_choi.shape[0]
    d = int(np.sqrt(d2))
    return np.real(np.trace(Phi_choi.conj().T @ Psi_choi)) / (d ** 2)


def average_gate_fidelity(rho_final: np.ndarray, rho_target: np.ndarray) -> float:
    """
    Average gate fidelity (averaged over pure input states on Bloch sphere).
    
    For single qubit: F_avg = (2F + 1) / 3 where F is state fidelity
    
    Args:
        rho_final: Achieved final state
        rho_target: Target state
        
    Returns:
        avg_fidelity: Value in [0, 1]
    """
    F_state = state_fidelity(rho_final, rho_target)
    d = rho_final.shape[0] 
    F_avg = (d * F_state + 1) / (d + 1)
    return F_avg


def gate_infidelity(rho_final: np.ndarray, rho_target: np.ndarray) -> float:
    """ 
    Gate infidelity: 1 - F(ρ_final, ρ_target).
    This is what we minimize in training.
    """
    return 1.0 - state_fidelity(rho_final, rho_target)


def diamond_norm_distance(Phi_choi: np.ndarray, Psi_choi: np.ndarray) -> float:
    """ 
    Diamond norm distance.  
    """
    diff = Phi_choi - Psi_choi
    eigenvalues = np.linalg.eigvalsh(diff.conj().T @ diff)
    return np.sqrt(np.max(eigenvalues))


class GateFidelityComputer:
    """ Unified interface for computing various fidelity measures."""
    
    def __init__(
        self,
        target_gate: np.ndarray,
        fidelity_type: str = 'state',
        d: int = 2
    ):
        """
        Args:
            target_gate: Target unitary or target state
            fidelity_type: 'state', 'process', 'average'
            d: Hilbert space dimension
        """
        self.target_gate = target_gate
        self.fidelity_type = fidelity_type
        self.d = d
        
        if fidelity_type == 'state':
            self.rho_target = target_gate  # Assume already a density matrix
        elif fidelity_type == 'process': 
            self.target_choi = self._unitary_to_choi(target_gate)
    
    def compute(self, rho_final: np.ndarray) -> float:
        """ 
        Compute fidelity between achieved state and target.
        
        Args:
            rho_final: Final density matrix achieved
            
        Returns:
            fidelity: Value in [0, 1]
        """
        if self.fidelity_type == 'state':
            return state_fidelity(rho_final, self.rho_target)
        elif self.fidelity_type == 'average':
            return average_gate_fidelity(rho_final, self.rho_target)
        else:
            raise NotImplementedError(f"Fidelity type {self.fidelity_type} not yet supported")
    
    def compute_from_unitary(self, U_achieved: np.ndarray, rho_init: np.ndarray) -> float:
        """ 
        Compute fidelity given achieved unitary and initial state.
        
        Args:
            U_achieved: Unitary approximation
            rho_init: Initial state
            
        Returns:
            fidelity
        """
        rho_final = U_achieved @ rho_init @ U_achieved.conj().T
        return self.compute(rho_final)
    
    def _unitary_to_choi(self, U: np.ndarray) -> np.ndarray:
        """ 
        Convert unitary to Choi matrix."""
        d = U.shape[0]
        choi = np.zeros((d**2, d**2), dtype=complex)
        for i in range(d):
            for j in range(d):
                ket_i = np.zeros(d)
                ket_i[i] = 1
                ket_j = np.zeros(d)
                ket_j[j] = 1
                
                input_op = np.outer(ket_i, ket_j.conj())
                output_op = U @ input_op @ U.conj().T
                
                choi += np.kron(input_op, output_op)
        
        return choi / d


class TargetGates:
    """ 
    Standard quantum gates for benchmarking."""
    @staticmethod
    def pauli_x() -> np.ndarray:
        """Pauli X gate (NOT gate)."""
        return np.array([[0, 1], [1, 0]], dtype=complex)
    
    @staticmethod
    def pauli_y() -> np.ndarray:
        """Pauli Y gate."""
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    
    @staticmethod
    def pauli_z() -> np.ndarray:
        """Pauli Z gate."""
        return np.array([[1, 0], [0, -1]], dtype=complex)
    
    @staticmethod
    def hadamard() -> np.ndarray:
        """Hadamard gate."""
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    
    @staticmethod
    def phase(phi: float) -> np.ndarray:
        """Phase gate R_φ."""
        return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)
    
    @staticmethod
    def rotation_x(theta: float) -> np.ndarray:
        """Rotation around X-axis."""
        return np.array([
            [np.cos(theta/2), -1j * np.sin(theta/2)],
            [-1j * np.sin(theta/2), np.cos(theta/2)]
        ], dtype=complex)
    
    @staticmethod
    def rotation_y(theta: float) -> np.ndarray:
        """Rotation around Y-axis."""
        return np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ], dtype=complex)
    
    @staticmethod
    def rotation_z(theta: float) -> np.ndarray:
        """Rotation around Z-axis."""
        return np.array([
            [np.exp(-1j * theta/2), 0],
            [0, np.exp(1j * theta/2)]
        ], dtype=complex)
    
    @staticmethod
    def arbitrary_unitary(alpha: float, beta: float, gamma: float) -> np.ndarray:
        """ 
        Arbitrary single-qubit unitary via Euler angles.
        """
        Rz_alpha = TargetGates.rotation_z(alpha)
        Ry_beta = TargetGates.rotation_y(beta)
        Rz_gamma = TargetGates.rotation_z(gamma)
        return Rz_gamma @ Ry_beta @ Rz_alpha

if JAX_AVAILABLE:
    @jit
    def state_fidelity_jax(rho, sigma) -> float:
        """JAX implementation of state fidelity.""" 
        return jnp.abs(jnp.trace(rho @ sigma)) ** 2

    @jit
    def gate_infidelity_jax(rho_final, rho_target) -> float:
        """JAX gate infidelity for gradient computation."""
        fidelity = state_fidelity_jax(rho_final, rho_target)
        return 1.0 - fidelity
else:
    def state_fidelity_jax(rho, sigma) -> float:
        """JAX implementation not available. Use numpy version."""
        raise ImportError("JAX is not installed. Install with: pip install jax jaxlib")

    def gate_infidelity_jax(rho_final, rho_target) -> float:
        """JAX implementation not available. Use numpy version."""
        raise ImportError("JAX is not installed. Install with: pip install jax jaxlib")
