#© 2025 The MITRE Corporation, All Rights Reserved    
from __future__ import annotations
import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass

TWO_PI = 2.0 * np.pi

@dataclass
class NoiseParameters:
    alpha: float   # spectral exponent  
    A: float       # amplitude/strength 
    omega_c: float # cutoff 
    model_type: str = 'one_over_f'  # PSD model type

    ALPHA_RANGE = (0.1, 2.0)
    A_RANGE = (0.01, 10.0)  # Note: spans 3 orders of magnitude
    OMEGA_C_RANGE = (1.0, 300.0)

    def to_array(self, include_model: bool = False, normalized: bool = False) -> np.ndarray:
        """
        Convert to array representation.

        Args:
            include_model: If True, encode model_type as 4th dimension.
                          Default False for backward compatibility.
            normalized: If True, normalize features to ~[0, 1] range.
                       Uses log-scale for A (amplitude) since it spans orders of magnitude.
                       This improves gradient flow and optimizer stability.

        Returns:
            arr: Array [alpha, A, omega_c] or [alpha, A, omega_c, model_encoding]
                 If normalized=True, values are scaled to roughly [0, 1].
        """
        if normalized:
            alpha_norm = (self.alpha - self.ALPHA_RANGE[0]) / (self.ALPHA_RANGE[1] - self.ALPHA_RANGE[0])
            log_A = np.log10(np.clip(self.A, self.A_RANGE[0], self.A_RANGE[1]))
            log_A_min, log_A_max = np.log10(self.A_RANGE[0]), np.log10(self.A_RANGE[1])
            A_norm = (log_A - log_A_min) / (log_A_max - log_A_min) 
            omega_c_norm = (self.omega_c - self.OMEGA_C_RANGE[0]) / (self.OMEGA_C_RANGE[1] - self.OMEGA_C_RANGE[0])

            if include_model:
                model_encoding = self._encode_model_type(self.model_type) / 2.0  # normalize to [0, 1]
                return np.array([alpha_norm, A_norm, omega_c_norm, model_encoding], dtype=float)
            else:
                return np.array([alpha_norm, A_norm, omega_c_norm], dtype=float)
        else:
            if include_model: 
                model_encoding = self._encode_model_type(self.model_type)
                return np.array([self.alpha, self.A, self.omega_c, model_encoding], dtype=float)
            else:
                return np.array([self.alpha, self.A, self.omega_c], dtype=float)

    @staticmethod
    def _encode_model_type(model_type: str) -> float:
        """Encode model type as numeric value for neural network input."""
        encoding = {
            'one_over_f': 0.0, 
            'lorentzian': 1.0,
            'double_exp': 2.0
        }
        if model_type not in encoding:
            raise ValueError(f"Unknown model_type '{model_type}'. Must be one of {list(encoding.keys())}")
        return encoding[model_type]

    @staticmethod
    def _decode_model_type(encoding: float) -> str:
        """Decode numeric value back to model type string."""
        rounded = int(round(encoding))
        decoding = {
            0: 'one_over_f',
            1: 'lorentzian',
            2: 'double_exp'
        }
        if rounded not in decoding:
            raise ValueError(f"Invalid model encoding {encoding}. Must round to 0, 1, or 2")
        return decoding[rounded]

    @classmethod
    def from_array(cls, arr: np.ndarray, has_model: bool = None) -> "NoiseParameters":
        """
        Create from array representation.

        Args:
            arr: Array of parameters
            has_model: If True, arr has 4 elements [alpha, A, omega_c, model].
                      If None, inferred from array length.

        Returns:
            params: NoiseParameters instance
        """
        if has_model is None:
            has_model = len(arr) >= 4

        if has_model:
            model_type = cls._decode_model_type(float(arr[3]))
            return cls(alpha=float(arr[0]), A=float(arr[1]), omega_c=float(arr[2]), model_type=model_type)
        else:
            return cls(alpha=float(arr[0]), A=float(arr[1]), omega_c=float(arr[2]), model_type='one_over_f')

# ---------- PSD models ----------
class NoisePSDModel:
    """
    Power spectral density models S(ω; θ). 
    """

    def __init__(self, model_type: str = "one_over_f"):
        self.model_type = model_type

    def psd(self, omega: np.ndarray, theta: NoiseParameters) -> np.ndarray:
        omega = np.asarray(omega, dtype=float)
        if self.model_type == "one_over_f":
            return self._one_over_f_psd(omega, theta)
        elif self.model_type == "lorentzian":
            return self._lorentzian_psd(omega, theta)
        elif self.model_type == "double_exp":
            return self._double_exp_psd(omega, theta)
        else:
            raise ValueError(f"Unknown model_type='{self.model_type}'")

    def _one_over_f_psd(self, omega: np.ndarray, theta: NoiseParameters) -> np.ndarray:
        eps = 1e-30
        if theta.alpha > 0:
            denom = np.power(np.abs(omega), theta.alpha) + (theta.omega_c ** theta.alpha)
        else:
            denom = np.ones_like(omega)
        return theta.A / (denom + eps)

    def _lorentzian_psd(self, omega: np.ndarray, theta: NoiseParameters) -> np.ndarray:
        eps = 1e-30
        return theta.A / (omega**2 + theta.omega_c**2 + eps)

    def _double_exp_psd(self, omega: np.ndarray, theta: NoiseParameters) -> np.ndarray: 
        eps = 1e-30
        omega_c1 = theta.omega_c
        omega_c2 = theta.omega_c * (1.0 + max(theta.alpha, 0.0))
        A1 = theta.A * (1 - min(max(theta.alpha, 0.0), 5.0) / 5.0)
        A2 = theta.A - A1
        return A1 / (omega**2 + omega_c1**2 + eps) + A2 / (omega**2 + omega_c2**2 + eps)

    def correlation_function(self, tau: np.ndarray, theta: NoiseParameters) -> np.ndarray:
        """ Inverse FT of PSD """
        tau = np.asarray(tau, dtype=float)
        if self.model_type == "lorentzian":
            return (theta.A / (2.0 * theta.omega_c)) * np.exp(-theta.omega_c * np.abs(tau))
        w = np.linspace(-1e3*theta.omega_c - 1.0, 1e3*theta.omega_c + 1.0, 20001)
        S_w = self.psd(w, theta)
        C_tau = np.trapz(S_w[:, None] * np.exp(1j * w[:, None] * tau), w, axis=0) / (2.0 * np.pi)
        return C_tau.real


class PSDToLindblad:
    """
    Map a colored PSD to Lindblad jump operators (qubit).


    Inputs:
      psd_model: returns two-sided angular PSD S(ω) with units [xi^2 * s]
                 Can be None if using per-task model_type from NoiseParameters  
    """

    def __init__(self, psd_model: "NoisePSDModel" = None):
        self.psd_model = psd_model  
        self._model_cache = {}

    def _get_psd_model(self, theta: "NoiseParameters") -> "NoisePSDModel":
        """
        Get PSD model for task, either from init or created dynamically.

        Args:
            theta: Task parameters with model_type attribute

        Returns:
            psd_model: NoisePSDModel instance
        """
        if self.psd_model is not None:

            return self.psd_model

        model_type = theta.model_type  
        if model_type not in self._model_cache:
            self._model_cache[model_type] = NoisePSDModel(model_type=model_type)

        return self._model_cache[model_type]


    @staticmethod
    def _F_abs(omega_T: np.ndarray, sequence: str) -> np.ndarray:
        seq = sequence.lower()
        if seq == "ramsey":
            return 4.0 * np.sin(0.5 * omega_T) ** 2
        if seq == "echo":
            return 8.0 * np.sin(0.25 * omega_T) ** 4
        if seq.startswith("cpmg_"):
            n = int(seq.split("_")[1])
            num = np.sin(n * omega_T / 2.0)
            den = np.maximum(np.sin(omega_T / 2.0), 1e-30)
            return 8.0 * np.sin(omega_T / (4.0 * n)) ** 4 * (num / den) ** 2
        raise ValueError(f"Unknown sequence '{sequence}'")

    def dephasing_rate(self, theta: "NoiseParameters", T: float, sequence: str = "ramsey",
                       omega_max_factor: float = 1000.0, n_w: int = 4000) -> float:

        w_max = omega_max_factor / T
        w = np.linspace(0.0, w_max, int(n_w))
        psd_model = self._get_psd_model(theta)   
        S_w = psd_model.psd(w, theta)                                # [xi^2 * s]
        F = self._F_abs(w * T, sequence)

        integrand =  S_w * (F / np.maximum(w, 1e-30) ** 2)  # J^2 * s
        chi = (1.0 / np.pi) * np.trapz(integrand, w)
        gamma_phi = max(chi / T, 0.0)                                        # 1/s
        return gamma_phi
                           
    @staticmethod
    def _lorentzian_L(w: np.ndarray, w0: float, Gamma_h: float) -> np.ndarray:
        """Normalized Lorentzian line shape centered at w0, FWHM=Gamma_h."""
        return (Gamma_h / 2.0) / np.pi / ((w - w0) ** 2 + (Gamma_h / 2.0) ** 2)
    
    def relaxation_rates(self, theta: "NoiseParameters", omega0: float, Gamma_h: float = 1.0, span_factor: float = 50.0, n_w: int = 4001) -> Tuple[float, float]:
        psd_model = self._get_psd_model(theta) 
        if Gamma_h > 0.0:
            w_span = max(span_factor * max(Gamma_h, 1e-12), 5.0 * abs(omega0))
            w = np.linspace(omega0 - w_span, omega0 + w_span, int(n_w))
            L = self._lorentzian_L(w, omega0, Gamma_h)
            S_w = psd_model.psd(w, theta)
            S_eff = np.trapz(S_w * L, w) 
        else:
            S_eff = float(psd_model.psd(np.array([abs(omega0)]), theta)[0])

        Gamma_down = max( S_eff, 0.0)
        return Gamma_down 

    def qubit_lindblad_ops(self, theta: "NoiseParameters", *,
                           T: float, sequence: str,
                           omega0: float, 
                           Gamma_h: float = 0.0):

        # dephasing
        gamma_phi = self.dephasing_rate(theta, T=T, sequence=sequence)
        L_phi = (1.0 / np.sqrt(2.0)) * np.array([[1, 0], [0, -1]], dtype=complex)

        # relaxation
        Gamma_down = self.relaxation_rates(theta, omega0, Gamma_h)
        L_minus = np.array([[0, 1], [0, 0]], dtype=complex)  # |g><e|
 
        ops = [np.sqrt(Gamma_down) * L_minus,
               np.sqrt(gamma_phi)  * L_phi]
        rates = np.array([Gamma_down, gamma_phi], dtype=float)
        return ops, rates

    def get_effective_rates(self, theta: "NoiseParameters", *,
                            T: float, sequence: str,
                            omega0: float,
                            Gamma_h: float = 1.0) -> Dict[str, float]:
        """Convenience: return {'Gamma_down','Gamma_up','gamma_phi'} in 1/s."""
        gamma_phi = self.dephasing_rate(theta, T=T, sequence=sequence)
        Gamma_down = self.relaxation_rates(theta, omega0, Gamma_h)
        return {"relax_rate": Gamma_down, "dephase_rate": gamma_phi}

class TaskDistribution:
    """
    Distribution P over θ = (alpha, A, omega_c, [model_type]).
    """

    def __init__(self,
                 dist_type: str = "uniform",
                 ranges: Dict[str, Tuple[float, float]] | None = None,
                 mean: np.ndarray | None = None,
                 cov: np.ndarray | None = None,
                 model_types: List[str] | None = None,
                 model_probs: List[float] | None = None):
        """
        Args:
            dist_type: 'uniform' or 'gaussian'
            ranges: Parameter ranges for uniform sampling
            mean: Mean for gaussian sampling
            cov: Covariance for gaussian sampling
            model_types: List of PSD model types to sample from.
                        Examples: ['one_over_f'], ['lorentzian'], or ['one_over_f', 'lorentzian']
                        If None, defaults to ['one_over_f']
            model_probs: Probability of sampling each model type. Must sum to 1.
                        If None, uniform distribution over model_types.
        """
        self.dist_type = dist_type
        self.ranges = ranges or {
            "alpha":   (0.5, 4.0),
            "A":       (10, 1e5),
            "omega_c": (1.0,80),
        }
        self.mean = mean
        self.cov = cov

        self.model_types = model_types or ['one_over_f']
        if model_probs is None:
            self.model_probs = [1.0 / len(self.model_types)] * len(self.model_types)
        else:
            if len(model_probs) != len(self.model_types):
                raise ValueError("model_probs must have same length as model_types")
            if not np.isclose(sum(model_probs), 1.0):
                raise ValueError("model_probs must sum to 1.0")
            self.model_probs = model_probs

    def sample(self, n_tasks: int, rng: np.random.Generator | None = None) -> List[NoiseParameters]:
        rng = rng or np.random.default_rng()
        if self.dist_type == "uniform":
            return self._sample_uniform(n_tasks, rng)
        elif self.dist_type == "gaussian":
            return self._sample_gaussian(n_tasks, rng)
        else:
            raise ValueError(f"Unknown distribution type '{self.dist_type}'")
  
    def _sample_uniform(self, n: int, rng: np.random.Generator) -> List[NoiseParameters]:
        tasks: List[NoiseParameters] = []
        for _ in range(n):
            alpha   = rng.uniform(*self.ranges["alpha"])
            A       = rng.uniform(*self.ranges["A"])
            omega_c = rng.uniform(*self.ranges["omega_c"])
            model_type = rng.choice(self.model_types, p=self.model_probs) 

            tasks.append(NoiseParameters(
                alpha,
                A,
                omega_c,
               model_type
            ))
        return tasks

    def _sample_gaussian(self, n: int, rng: np.random.Generator) -> List[NoiseParameters]:
        if self.mean is None or self.cov is None:
            raise ValueError("Gaussian sampling requires 'mean' and 'cov'")
        samples = rng.multivariate_normal(self.mean, self.cov, size=n)

        tasks = []
        for sample in samples:
            model_type = rng.choice(self.model_types, p=self.model_probs)
            task = NoiseParameters.from_array(sample, has_model=False)
            task.model_type = model_type
            tasks.append(task) 
        return tasks

    def compute_variance(self) -> float:
        """
        Compute variance of task distribution.

 
        """
        if self.dist_type == "uniform":
            var_alpha = ((self.ranges["alpha"][1]   - self.ranges["alpha"][0])   ** 2) / 12.0
            var_A     = ((self.ranges["A"][1]       - self.ranges["A"][0])       ** 2) / 12.0
            var_omega = ((self.ranges["omega_c"][1] - self.ranges["omega_c"][0]) ** 2) / 12.0 
            if len(self.model_types) > 1:
    
                model_variance = sum([p * (1 - p) for p in self.model_probs])
                print(f"INFO: Mixed model variance contribution: {model_variance:.4f}")
                return float(var_alpha + var_A + var_omega + model_variance)

            return float(var_alpha + var_A + var_omega)
        elif self.dist_type == "gaussian":

            base_var = float(np.trace(self.cov))

            if len(self.model_types) > 1:
                model_variance = sum([p * (1 - p) for p in self.model_probs])
                return base_var + model_variance

            return base_var

def psd_distance(psd_model: NoisePSDModel,
                 theta1: NoiseParameters,
                 theta2: NoiseParameters,
                 omega_grid: np.ndarray) -> float:
    """d(θ,θ') = sup_ω |S(ω;θ) - S(ω;θ')| over provided grid"""
    S1 = psd_model.psd(omega_grid, theta1)
    S2 = psd_model.psd(omega_grid, theta2)
    return float(np.max(np.abs(S1 - S2)))
