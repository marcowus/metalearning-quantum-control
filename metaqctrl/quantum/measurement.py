"""
Quantum Measurement Simulation
© 2025 The MITRE Corporation, All Rights Reserved
"""

import torch
import numpy as np

def measure_probs(rho: torch.Tensor, projector: torch.Tensor = None) -> torch.Tensor:
    """
    Compute measurement probabilities Tr(M * rho).

    Args:
        rho: Density matrix (batch, d, d) or (d, d)
        projector: Measurement operator (d, d). Defaults to |1><1| (excited state).

    Returns:
        prob: Probability of measuring 1 (batch,) or scalar.
    """
    if projector is None:
        # Default to |1><1| for qubit
        projector = torch.zeros_like(rho)
        projector[..., 0, 0] = 0.0 # |0><0|
        projector[..., 1, 1] = 1.0 # |1><1|
        # Or more simply, if we just want prob of |1>:
        # It's rho[1,1] (real part)

    if rho.ndim == 2:
        val = torch.trace(projector @ rho)
    else:
        # Batch trace: sum over last two dims of element-wise product (or matmul)
        val = torch.einsum('bij,bij->b', projector.unsqueeze(0), rho) # This assumes projector is constant for batch
        # If projector is (d,d) and rho is (b,d,d):
        # projector @ rho is (b,d,d)

        # Simpler:
        # val = (projector @ rho).diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)

    return torch.real(val)

def simulate_counts(
    probs: torch.Tensor,
    n_shots: int = 1000,
    method: str = 'soft'
) -> torch.Tensor:
    """
    Simulate measurement counts.

    Args:
        probs: Probabilities of outcome 1.
        n_shots: Number of measurement shots.
        method:
            'soft': Returns n_shots * prob (differentiable).
            'hard': Returns sampled binomial counts (non-differentiable).
            'gumbel': Gumbel-Softmax approximation (not implemented yet, but good for future).

    Returns:
        counts: Tensor of counts (normalized or raw).
    """
    if method == 'soft':
        return probs * n_shots

    elif method == 'hard':
        # Binomial sampling
        # Detach grad because sampling is discrete
        probs_np = probs.detach().cpu().numpy()
        counts = np.random.binomial(n_shots, probs_np)
        return torch.tensor(counts, device=probs.device, dtype=torch.float32)

    else:
        raise ValueError(f"Unknown measurement method: {method}")

class QuantumMeasurement(torch.nn.Module):
    """Module wrapper for measurement."""
    def __init__(self, n_shots=1000, method='soft'):
        super().__init__()
        self.n_shots = n_shots
        self.method = method
        # Define projectors
        self.register_buffer('proj_1', torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64))

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """Measure rho and return counts of |1> state."""
        p = measure_probs(rho, self.proj_1)
        c = simulate_counts(p, self.n_shots, self.method)
        return c
