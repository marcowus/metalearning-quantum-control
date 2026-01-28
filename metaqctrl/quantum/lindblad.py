"""
Lindblad Master Equation Simulator
© 2025 The MITRE Corporation, All Rights Reserved  

"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from typing import List, Tuple, Callable, Optional

# Optional JAX support
try:
    import jax.numpy as jnp
    from jax import jit, vmap
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = None
    jit = lambda f: f  # No-op decorator if JAX not available
    vmap = None


class LindbladSimulator:
    """Simulate Lindblad dynamics with time-dependent controls."""
    
    def __init__(
        self,
        H0: np.ndarray,
        H_controls: List[np.ndarray],
        L_operators: List[np.ndarray],
        dt: float = 0.05,
        method: str = 'RK45'
    ):
        """
        Args:
            H0: Drift Hamiltonian (d x d)
            H_controls: List of control Hamiltonians [H₁, H₂, ...] each (d x d)
            L_operators: List of Lindblad operators [L₁, L₂, ...] parameterized by task
            dt: Time step for discretization
            method: Integration method ('RK45', 'magnus', 'expm')
        """
        self.H0 = H0
        self.H_controls = H_controls
        self.L_operators = L_operators
        self.dt = dt
        self.method = method
        
        self.d = H0.shape[0]  # Hilbert space dimension
        self.n_controls = len(H_controls)
        self.n_lindblad = len(L_operators)
        
        self.anti_commutators = [
            L.conj().T @ L for L in L_operators
        ]
        
    def lindbladian(self, rho: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        """
        Compute ρ̇ = L[ρ] at time t with controls u.
        
        Args:
            rho: Density matrix (d x d)
            u: Control amplitudes (n_controls,)
            t: Current time
            
        Returns:
            drho_dt: Time derivative (d x d)
        """
        # Hamiltonian evolution: -i[H, ρ]
        H_total = self.H0.copy()
        for k, u_k in enumerate(u):
            H_total += u_k * self.H_controls[k]
        
        hamiltonian_term = -1j * (H_total @ rho - rho @ H_total)
        dissipation_term = np.zeros_like(rho)
        for j, L_j in enumerate(self.L_operators):
            dissipation_term += (
                L_j @ rho @ L_j.conj().T
                - 0.5 * (self.anti_commutators[j] @ rho + rho @ self.anti_commutators[j])
            )
        
        return hamiltonian_term + dissipation_term
    
    def evolve(
        self,
        rho0: np.ndarray,
        control_sequence: np.ndarray,
        T: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evolve state from t=0 to t=T under piecewise-constant controls.
        
        Args:
            rho0: Initial state (d x d)
            control_sequence: Control pulses (n_segments, n_controls)
            T: Total evolution time
            
        Returns:
            rho_final: Final state (d x d)
            trajectory: State at each time step for logging
        """
        n_segments = control_sequence.shape[0]
        t_segment = T / n_segments
        
        rho_t = rho0.copy()
        trajectory = [rho_t.copy()]
        
        for seg_idx in range(n_segments):
            u_seg = control_sequence[seg_idx]
            
            if self.method == 'expm':
                # Use matrix exponential (exact for constant controls)
                rho_t = self._expm_step(rho_t, u_seg, t_segment)
            else:
                # Use scipy ODE solver
                def rhs(t, rho_vec):
                    rho = rho_vec.reshape(self.d, self.d)
                    drho = self.lindbladian(rho, u_seg, t)
                    return drho.flatten()
                
                sol = solve_ivp(
                    rhs,
                    [0, t_segment],
                    rho_t.flatten(), 
                    rtol=1e-8,
                    atol=1e-10
                )
                rho_t = sol.y[:, -1].reshape(self.d, self.d)
            
            trajectory.append(rho_t.copy())
        
        return rho_t, np.array(trajectory)
    
    def _expm_step(self, rho: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """Single step using matrix exponential (Magnus expansion)."""
        # Build superoperator for this control 
        L_super = self._build_superoperator(u)
        rho_vec = rho.flatten()
        rho_vec_new = expm(L_super * dt) @ rho_vec
        return rho_vec_new.reshape(self.d, self.d)
    
    def _build_superoperator(self, u: np.ndarray) -> np.ndarray:
        """
        Build superoperator. 
        """
        d2 = self.d ** 2
        L_super = np.zeros((d2, d2), dtype=complex)
        
        # Hamiltonian part: -i[H, ρ]
        H_total = self.H0.copy()
        for k, u_k in enumerate(u):
            H_total += u_k * self.H_controls[k]
        
        I = np.eye(self.d)
        L_super += -1j * (np.kron(I, H_total) - np.kron(H_total.T, I))
        
        # Dissipation part
        for j, L_j in enumerate(self.L_operators):
            L_super += np.kron(L_j.conj(), L_j)
            L_super -= 0.5 * np.kron(I, self.anti_commutators[j])
            L_super -= 0.5 * np.kron(self.anti_commutators[j].T, I)
        
        return L_super
    
    def get_process_matrix(
        self,
        control_sequence: np.ndarray,
        T: float
    ) -> np.ndarray:
        """
        Compute Choi matrix of the quantum channel Φ.
        
        Returns:
            choi: Choi matrix (d² x d²)
        """ 
        d = self.d
        phi_plus = np.eye(d).flatten() / np.sqrt(d)
        rho_entangled = np.outer(phi_plus, phi_plus.conj())
         
        choi = np.zeros((d**2, d**2), dtype=complex)
        for i in range(d):
            for j in range(d):
                basis_state = np.outer(
                    self._basis_vec(i, d),
                    self._basis_vec(j, d).conj()
                )
                evolved, _ = self.evolve(basis_state, control_sequence, T)
                choi += np.kron(
                    np.outer(self._basis_vec(i, d), self._basis_vec(j, d).conj()),
                    evolved
                )
        
        return choi / d
    
    @staticmethod
    def _basis_vec(i: int, d: int) -> np.ndarray:
        """Computational basis vector |i⟩."""
        v = np.zeros(d, dtype=complex)
        v[i] = 1.0
        return v


class LindbladJAX:
    """JAX implementation for fast batched simulations and autodiff."""

    def __init__(
        self,
        H0,
        H_controls: List,
        n_segments: int,
        T: float
    ):
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for LindbladJAX but is not installed. "
                            "Install with: pip install jax jaxlib")

        self.H0 = jnp.asarray(H0) if jnp is not None else H0
        self.H_controls = [jnp.asarray(H) if jnp is not None else H for H in H_controls]
        self.n_segments = n_segments
        self.dt = T / n_segments
        self.d = H0.shape[0]

    @jit
    def evolve_step(self, rho, u, L_ops: List):
        """Single time step evolution with controls u and Lindblad operators L_ops."""
        H_total = self.H0
        for k, H_k in enumerate(self.H_controls):
            H_total = H_total + u[k] * H_k
        
        ham_term = -1j * (H_total @ rho - rho @ H_total)
        
        diss_term = jnp.zeros_like(rho)
        for L in L_ops:
            diss_term = diss_term + (
                L @ rho @ L.conj().T
                - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)
            )
        
        drho = ham_term + diss_term
        return rho + self.dt * drho
    
    def evolve_trajectory(
        self,
        rho0,
        control_sequence,
        L_ops: List
    ):
        """Full trajectory evolution."""
        rho = rho0
        for u in control_sequence:
            rho = self.evolve_step(rho, u, L_ops)
        return rho


# Example usage
if __name__ == "__main__":
    ##Takeaway --> T = 1 
    #  Note H0 = 0.5 * sigma_z 
    #N_segments = 100  
    # 1-qubit system
    from scipy.linalg import expm
    from metaqctrl.quantum.gates import *
    
    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = (1.0 / np.sqrt(2.0)) * np.array([[1, 0], [0, -1]], dtype=complex)
    sig_p = np.array([[0, 1], [0, 0]], dtype=complex)
    
    H0 = 0.5 * sigma_z  # Drift
    H_controls = [sigma_x, sigma_y]  # Control Hamiltonians
    
    # Example Lindblad operator (dephasing)
    T_op = 1 
    gamma_1 = 30         
    gamma_2 = 50        
   # L_ops = [np.sqrt(gamma) * sigma_z]
    L_ops = [np.sqrt(gamma_1) * sig_p, np.sqrt(gamma_2) * sigma_z ]
    
    sim = LindbladSimulator(H0, H_controls, L_ops, method='RK45')
    
    # Initial state |0⟩
    rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
    
    # Random control sequence
    n_segments = 20  
    controls = np.ones((n_segments, 2))*20  
    
    rho_final, traj = sim.evolve(rho0, controls, T=T_op)
    
    ket_0 = np.array([1, 0], dtype=complex)
    ket_1 = np.array([0, 1], dtype=complex)  
    rho_0 = np.outer(ket_0, ket_0.conj())
    rho_1 = np.outer(ket_1, ket_1.conj())

   
    gates = TargetGates()
    X = gates.pauli_x()
    rho_X = X @ rho_0 @ X.conj().T    
    fid = state_fidelity(rho_X,  rho_final)   
    print(fid)
    

