"""
© 2025 The MITRE Corporation, All Rights Reserved   
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from metaqctrl.quantum.noise_models_v2 import *  

TWO_PI = 2.0 * np.pi

class PSDToLindblad2:
    """
    Backward-compatible wrapper around noise_models_v2.PSDToLindblad.

    Provides the old API:
        converter = PSDToLindblad(basis_operators, sampling_freqs, psd_model)
        L_ops = converter.get_lindblad_operators(theta)

    """

    def __init__(
        self,
        basis_operators: List[np.ndarray],
        sampling_freqs: np.ndarray,
        psd_model: NoisePSDModel,  
        T: float = 0.1,   
        sequence: str = 'ramsey',
        omega0: Optional[float] = None,  
        Gamma_h: float = 100,  
        integration_method: str = 'trapz'):
        """
        Args:
            basis_operators: List of Pauli operators [σx, σy, σz] (or 2-qubit equivalent)
            sampling_freqs: Control bandwidth frequencies (used to estimate omega0)
            psd_model: NoisePSDModelV2 instance
            T: Gate evolution time [seconds]
            sequence: 'ramsey', 'echo', or 'cpmg_N' for dephasing filter function
            omega0: Qubit transition frequency [rad/s]. If None, estimated from bandwidth  
            Gamma_h: Homogeneous linewidth [rad/s]. 0 = sharp transition
            integration_method: Kept for API compatibility (always uses v2 integral)
 
        """
        self.basis_ops = basis_operators
        self.sampling_freqs = np.asarray(sampling_freqs)
        self.psd_model = psd_model

        # Physics parameters
        self.T = float(T)
        self.sequence = sequence 
        self.Gamma_h = float(Gamma_h)

        # Estimate omega0 from control bandwidth if not provided
        if omega0 is None: 
            omega_max = np.max(self.sampling_freqs) if len(self.sampling_freqs) > 0 else 20.0
            self.omega0 = omega_max / 2.0  # Transition typically below control bandwidth
            self._omega0_estimated = True
        else:
            self.omega0 = float(omega0)
            self._omega0_estimated = False
        self.converter = PSDToLindblad(psd_model=self.psd_model)

    def get_lindblad_operators(self, theta: NoiseParameters) -> List[np.ndarray]:
        """
        Backward-compatible API: returns Lindblad operators using v2 physics.

        Args:
            theta: NoiseParameters (alpha, A, omega_c)

        Returns:
            L_ops: List of Lindblad operators  
        """

        ops, rates = self.converter.qubit_lindblad_ops(
                theta,
                T=self.T,
                sequence=self.sequence,
                omega0=self.omega0,  
                Gamma_h=self.Gamma_h
        )
 
        return ops

    def get_effective_rates(self, theta: NoiseParameters) -> np.ndarray:
        """
        Get effective decay rates. 

        Returns:
            rates: Array of decay rates  
        """
        _, rates = self.converter.qubit_lindblad_ops(theta, T=self.T, sequence=self.sequence, omega0=self.omega0,  Gamma_h=self.Gamma_h)
        return rates


    def get_rates_dict(self, theta: NoiseParameters) -> Dict[str, float]:
        """
        Get decay rates as labeled dictionary.

        Returns:
            rates: {'Gamma_down': float, 'Gamma_up': float, 'gamma_phi': float}
        """
        rates = self.get_effective_rates(theta)
        return {
            'relax_rate': float(rates[0]),
            'dephase_rate': float(rates[1]) 
        }

    def update_physics_parameters(
        self,
        T: Optional[float] = None,
        sequence: Optional[str] = None,
        omega0: Optional[float] = None, 
        Gamma_h: Optional[float] = None
    ):
        """
        Update physics parameters after construction.
        """
        if T is not None:
            self.T = float(T)
        if sequence is not None:
            self.sequence = sequence
        if omega0 is not None:
            self.omega0 = float(omega0)
            self._omega0_estimated = False 
        if Gamma_h is not None:
            self.Gamma_h = float(Gamma_h)


def estimate_qubit_frequency_from_hamiltonian(H0: np.ndarray) -> float:
    """
    Estimate qubit transition frequency from drift Hamiltonian.

    Args:
        H0: Drift Hamiltonian (d×d complex matrix)

    Returns:
        omega0: Transition frequency [rad/s]
    """
    eigenvalues = np.linalg.eigvalsh(H0)
    eigenvalues = np.sort(np.real(eigenvalues))

    if len(eigenvalues) >= 2:
        omega0 = float(np.abs(eigenvalues[1] - eigenvalues[0]))
    else:
        omega0 = TWO_PI   

    return omega0
  
