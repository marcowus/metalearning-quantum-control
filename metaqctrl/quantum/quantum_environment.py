"""
Quantum Environment Bridge 
© 2025 The MITRE Corporation, All Rights Reserved   
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
from functools import lru_cache
from metaqctrl.quantum.lindblad import LindbladSimulator
from metaqctrl.quantum.lindblad_torch import DifferentiableLindbladSimulator, numpy_to_torch_complex  
from metaqctrl.quantum.noise_adapter import NoiseParameters
from metaqctrl.quantum.noise_models_v2 import *  
from metaqctrl.quantum.gates import state_fidelity


class QuantumEnvironment:
    """
    Unified environment for quantum control. 
    """
    
    def __init__(
        self,
        H0: np.ndarray,
        H_controls: list,
        psd_to_lindblad,
        target_state: np.ndarray,
        T: float = 1,
        method: str = 'RK45',
        target_unitary: np.ndarray = None
    ):
        """
        Args:
            H0: Drift Hamiltonian
            H_controls: List of control Hamiltonians
            psd_to_lindblad: PSDToLindblad instance
            target_state: Target density matrix
            T: Evolution time
            method: Integration method
            target_unitary: Target unitary gate (optional, for process fidelity)
        """
        self.H0 = H0
        self.H_controls = H_controls
        self.psd_to_lindblad = psd_to_lindblad
        self.target_state = target_state
        self.target_unitary = target_unitary  # Store for process fidelity
        self.T = T
        self.method = method
        self.sequence = None  
        self.omega0 = None 


        self.d = H0.shape[0]
        self.n_controls = len(H_controls)

        self._L_cache = {}

        self._sim_cache = {}

        self._torch_sim_cache = {}

        self.rho0 = np.zeros((self.d, self.d), dtype=complex)
        self.rho0[0, 0] = 1.0 
    @property
    def num_controls(self) -> int:
        """Alias for n_controls for backward compatibility."""
        return self.n_controls

    @property
    def evolution_time(self) -> float:
        """Alias for T for backward compatibility."""
        return self.T

    @property
    def omega_control(self) -> np.ndarray:
        """Control frequencies - placeholder for compatibility."""
        return np.array([1.0, 5.0, 10.0])

    @property
    def control_susceptibility(self) -> np.ndarray:
        """Control susceptibility matrix - placeholder for compatibility."""
        return np.eye(len(self.H_controls))

    def compute_fidelity(self, controls: np.ndarray, task_params: NoiseParameters) -> float:
        """Alias for evaluate_controls for backward compatibility."""
        return self.evaluate_controls(controls, task_params, return_trajectory=False)

    def _task_hash(self, task_params: NoiseParameters) -> tuple:
        """Create hashable key for task (including model type)."""
        return (
            round(task_params.alpha, 6),
            round(task_params.A, 6),
            round(task_params.omega_c, 6), 
            task_params.model_type 
        )
    
    def get_lindblad_operators(self, task_params: NoiseParameters) -> list:
        """ 
        Get Lindblad operators for task with caching.
        
        Args:
            task_params: Task noise parameters
            
        Returns:
            L_ops: List of Lindblad operators
        """
        key = self._task_hash(task_params)

        if key not in self._L_cache:
            L_ops = self.psd_to_lindblad.get_lindblad_operators(task_params)
            self._L_cache[key] = L_ops


        return self._L_cache[key]
    
    def get_simulator(self, task_params: NoiseParameters) -> LindbladSimulator:
        """ 
        Get simulator for task with caching.

        Args:
            task_params: Task noise parameters

        Returns:
            sim: LindbladSimulator instance
        """
        key = self._task_hash(task_params)


        if key not in self._sim_cache:
            L_ops = self.get_lindblad_operators(task_params)
         

            sim = LindbladSimulator(
                H0=self.H0,
                H_controls=self.H_controls,
                L_operators=L_ops,
                method=self.method
            )

            self._sim_cache[key] = sim

        return self._sim_cache[key]

    def get_torch_simulator(
        self,
        task_params: NoiseParameters,
        device: torch.device,
        dt: float = 0.01,
        use_rk4: bool = True
    ) -> DifferentiableLindbladSimulator:
        """
        Get cached differentiable PyTorch simulator for task.

        Args:
            task_params: Task noise parameters
            device: torch device
            dt: Integration time step
            use_rk4: If True, use RK4 integration

        Returns:
            sim: Cached DifferentiableLindbladSimulator instance
        """
        key = (self._task_hash(task_params), str(device), dt, use_rk4)

        if key not in self._torch_sim_cache:

            L_ops_numpy = self.psd_to_lindblad.get_lindblad_operators(task_params)

            H0_torch = numpy_to_torch_complex(self.H0, device)
            H_controls_torch = [numpy_to_torch_complex(H, device) for H in self.H_controls]
            L_ops_torch = [numpy_to_torch_complex(L, device) for L in L_ops_numpy]

            sim = DifferentiableLindbladSimulator(
                H0=H0_torch,
                H_controls=H_controls_torch,
                L_operators=L_ops_torch,
                dt=dt,
                method='rk4' if use_rk4 else 'euler',
                device=device
            )

            self._torch_sim_cache[key] = sim

        return self._torch_sim_cache[key]
    
    def evaluate_controls(
        self,
        controls: np.ndarray,
        task_params: NoiseParameters,
        return_trajectory: bool = False,
        use_process_fidelity: bool = False
    ) -> float:
        """  
        Simulate and compute fidelity.

        Args:
            controls: Control sequence (n_segments, n_controls)
            task_params: Task parameters
            return_trajectory: If True, return (fidelity, trajectory)
            use_process_fidelity: If True, use average gate fidelity over all input states
                                  (important for multi-qubit gates like CNOT!)

        Returns:
            fidelity: Achieved fidelity (float)
            or (fidelity, trajectory) if return_trajectory=True
        """
    
        sim = self.get_simulator(task_params)

        if use_process_fidelity and self.d > 2:
            fidelity = self._compute_average_gate_fidelity(sim, controls)

            if return_trajectory:
    
                rho_final, trajectory = sim.evolve(self.rho0, controls, self.T)
                return fidelity, trajectory
            return fidelity
        else:

            rho_final, trajectory = sim.evolve(self.rho0, controls, self.T)
            fidelity = state_fidelity(rho_final, self.target_state)

            if return_trajectory:
                return fidelity, trajectory
            return fidelity

    def _compute_average_gate_fidelity(
        self,
        sim: LindbladSimulator,
        controls: np.ndarray
    ) -> float:
        """
        Compute average gate fidelity over all computational basis states.

        This is the proper fidelity measure for multi-qubit gates!

        For 2-qubits: Average over |00⟩, |01⟩, |10⟩, |11⟩

        Args:
            sim: LindbladSimulator instance
            controls: Control sequence

        Returns:
            avg_fidelity: Average fidelity over all basis states
        """
        from metaqctrl.quantum.gates import state_fidelity

        if self.target_unitary is None:
 
            print("WARNING: target_unitary not provided. Using approximate fidelity.")
            print("         Set target_unitary in QuantumEnvironment for accurate process fidelity.")

            rho_final, _ = sim.evolve(self.rho0, controls, self.T)
            return state_fidelity(rho_final, self.target_state)
        fidelities = []

        for i in range(self.d):
     
            ket_i = np.zeros(self.d, dtype=complex)
            ket_i[i] = 1.0
            rho_i = np.outer(ket_i, ket_i.conj())

            # Evolve under controls
            rho_final, _ = sim.evolve(rho_i, controls, self.T)

            # Target output: U_target |i⟩
            ket_target = self.target_unitary @ ket_i
            rho_target_i = np.outer(ket_target, ket_target.conj())

            # Compute fidelity
            fid = state_fidelity(rho_final, rho_target_i)
            fidelities.append(fid)

        # Average over all input states
        return float(np.mean(fidelities))
    
    def evaluate_policy(
        self,
        policy: torch.nn.Module,
        task_params: NoiseParameters,
        device: torch.device = torch.device('cpu')
    ) -> float:
        """ 
        Evaluate policy on task.

        Args:
            policy: Policy network
            task_params: Task parameters
            device: torch device

        Returns:
            fidelity: Achieved fidelity
        """
        policy.eval()

        with torch.no_grad():
            # Task features
            task_features = torch.tensor(
                task_params.to_array(),
                dtype=torch.float32,
                device=device
            )

            # Generate controls
            controls = policy(task_features)
            controls_np = controls.cpu().numpy()

        # Evaluate
        fidelity = self.evaluate_controls(controls_np, task_params)

        return fidelity
    
    def compute_loss(
        self,
        policy: torch.nn.Module,
        task_params: NoiseParameters,
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """ 

        Args:
            policy: Policy network
            task_params: Task parameters
            device: torch device

        Returns:
            loss: Loss tensor (gradients only partial)
        """
        # Task features
        task_features = torch.tensor(
            task_params.to_array(),
            dtype=torch.float32,
            device=device
        )

        # Generate controls
        controls = policy(task_features)

        controls_np = controls.detach().cpu().numpy()


        fidelity = self.evaluate_controls(controls_np, task_params)

        loss = torch.tensor(
            1.0 - fidelity,
            dtype=torch.float32,
            device=device,
            requires_grad=False
        )

        return loss

    def compute_loss_differentiable(
        self,
        policy: torch.nn.Module,
        task_params: NoiseParameters,
        device: torch.device = torch.device('cpu'),
        use_rk4: bool = True,
        dt: float = 0.01
    ) -> torch.Tensor:
        """ 

        Args:
            policy: Policy network
            task_params: Task parameters
            device: torch device
            use_rk4: If True, use RK4 integration (more accurate but slower)
            dt: Integration time step (larger = faster but less accurate)
        """


        task_params_array = task_params.to_array(normalized=True)
        task_features = torch.as_tensor(
            task_params_array,
            dtype=torch.float32,
            device=device
        )


        controls = policy(task_features)  # (n_segments, n_controls)



        sim = self.get_torch_simulator(task_params, device, dt=dt, use_rk4=use_rk4)
        
        rho0 = torch.zeros((self.d, self.d), dtype=torch.complex64, device=device)
        rho0[0, 0] = 1.0

        rho_final = sim(rho0, controls, self.T)

        # Target state (convert to torch)
        target_state_torch = numpy_to_torch_complex(self.target_state, device)

        # Compute fidelity (differentiable)
        fidelity = self._torch_state_fidelity(rho_final, target_state_torch)

        # Loss = infidelity (differentiable!)
        loss = 1.0 - fidelity

        return loss

    def _torch_state_fidelity(
        self,
        rho: torch.Tensor,
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """Proper quantum fidelity for density matrices (differentiable).

        Args:
            rho: Density matrix (d x d) complex tensor
            sigma: Density matrix (d x d) complex tensor

        Returns:
            fidelity: Real-valued fidelity in [0, 1]
        """

        trace_prod = torch.trace(rho @ sigma)


        fidelity = torch.abs(trace_prod) ** 2

        fidelity = torch.clamp(fidelity, 0.0, 1.0)

        return fidelity

    def clear_cache(self):
        """Clear all caches."""
        self._L_cache.clear()
        self._sim_cache.clear()
        self._torch_sim_cache.clear()

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'n_cached_operators': len(self._L_cache),
            'n_cached_simulators': len(self._sim_cache),
            'n_cached_torch_simulators': len(self._torch_sim_cache),
            'cache_size_mb': (
                len(str(self._L_cache)) + len(str(self._sim_cache)) + len(str(self._torch_sim_cache))
            ) / 1e6
        }


class BatchedQuantumEnvironment(QuantumEnvironment):
    """
    Batched version for parallel task evaluation.
    Uses JAX for vectorization.
    """
    
    def __init__(self, *args, use_jax: bool = True, **kwargs):
        ## This uses Jax 
        super().__init__(*args, **kwargs)
        self.use_jax = use_jax
        
        if use_jax:
            try:
                from metaqctrl.quantum.lindblad import LindbladJAX
                self.jax_sim = LindbladJAX(
                    self.H0,
                    self.H_controls,
                    n_segments=20,  # From config
                    T=self.T
                )
                print("JAX batching enabled")
            except ImportError:
                print("JAX not available, falling back to serial")
                self.use_jax = False
    
    def evaluate_controls_batch(
        self,
        controls_batch: np.ndarray,
        task_params_batch: list
    ) -> np.ndarray:
        """
        Evaluate multiple control sequences in parallel.
        
        Args:
            controls_batch: (batch_size, n_segments, n_controls)
            task_params_batch: List of NoiseParameters
            
        Returns:
            fidelities: (batch_size,) array of fidelities
        """
        if self.use_jax:

            pass
        
        fidelities = []
        for controls, task_params in zip(controls_batch, task_params_batch):
            fid = self.evaluate_controls(controls, task_params)
            fidelities.append(fid)
        
        return np.array(fidelities)


# Helper functions
def get_target_state_from_config(config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get target density matrix and unitary from config.

    Args:
        config: Configuration dictionary with 'target_gate' and 'num_qubits' keys

    Returns:
        target_state: Target density matrix (d x d)
        target_unitary: Target unitary gate (d x d)
    """
    from metaqctrl.quantum.gates import TargetGates

    target_gate_name = config.get('target_gate')
    num_qubits = config.get('num_qubits')

    # Get target unitary
    if target_gate_name == 'hadamard':
        U_target = TargetGates.hadamard()
    elif target_gate_name == 'pauli_x':
        U_target = TargetGates.pauli_x()
    elif target_gate_name == 'pauli_y':
        U_target = TargetGates.pauli_y()
    elif target_gate_name == 'pauli_z':
        U_target = TargetGates.pauli_z()
    elif target_gate_name == 'cnot':
        U_target = TargetGates.cnot()
    else:
        raise ValueError(f"Unknown target gate: {target_gate_name}")

    # Initial state (|0...0⟩)
    d = 2 ** num_qubits
    ket_0 = np.zeros(d, dtype=complex)
    ket_0[0] = 1.0

    # Target state: U|0...0⟩
    target_ket = U_target @ ket_0
    target_state = np.outer(target_ket, target_ket.conj())

    return target_state, U_target


def create_quantum_environment(config: dict, target_state: np.ndarray = None, target_unitary: np.ndarray = None) -> QuantumEnvironment:
    """
    Create quantum environment from config.

    Args:
        config: Configuration dictionary
        target_state: Target density matrix. If None, will be created from config['target_gate']
        target_unitary: Target unitary gate. If None, will be created from config['target_gate']

    Returns:
        env: QuantumEnvironment instance
    """
    from metaqctrl.quantum.noise_adapter import PSDToLindblad2, estimate_qubit_frequency_from_hamiltonian    

    # Get number of qubits from config
    num_qubits = config.get('num_qubits')
 
    # Get target state and unitary if not provided
    if target_state is None or target_unitary is None:
        target_state, target_unitary = get_target_state_from_config(config)

    if num_qubits == 1:
        # 1-qubit system (original code)
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        sigma_p = np.array([[0, 1], [0, 0]], dtype=complex)

        # System Hamiltonians
        drift_strength = config.get('drift_strength')
        H0 = drift_strength * sigma_z
        H_controls = [sigma_x, sigma_y]

        # Noise basis operators
        basis_operators = [sigma_p, sigma_z]


    else:
        raise ValueError(f"num_qubits={num_qubits} not supported. Use 1 or 2.")


    model_types = config.get('model_types')
    if model_types is None:
        psd_model = NoisePSDModel(model_type=config.get('psd_model')) 
    else:

        psd_model = None 
        print(f"INFO: Mixed model mode enabled with models: {model_types}")

    # Sampling frequencies (control bandwidth)
    n_segments = config.get('n_segments')
    T = config.get('horizon')
    omega_max = n_segments / T
    omega_sample = np.linspace(0, omega_max, 1000)
    omega0 = config.get('omega0')
    if omega0 is None:
        omega0 = estimate_qubit_frequency_from_hamiltonian(H0) 
    noise_type = config.get('noise_type')   

    sequence = config.get('sequence') 
 
    Gamma_h = config.get('Gamma_h')
 
    
    psd_to_lindblad = PSDToLindblad2(
        basis_operators=basis_operators,
        sampling_freqs=omega_sample,
        psd_model=psd_model,  # Can be None for dynamic model selection
        T=T,
        sequence=sequence,
        omega0=omega0,  
        Gamma_h=Gamma_h
    )
 
    # Create environment
    env = QuantumEnvironment(
        H0=H0,
        H_controls=H_controls,
        psd_to_lindblad=psd_to_lindblad,
        target_state=target_state,
        T=T,
        method=config.get('integration_method'),
        target_unitary=target_unitary
    )

    return env
