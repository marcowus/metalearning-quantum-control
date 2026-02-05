"""
Parameter Estimators for Closed-Loop Quantum Control
© 2025 The MITRE Corporation, All Rights Reserved
"""

import torch
import torch.nn as nn

class NeuralEstimator(nn.Module):
    """
    Neural network that estimates task parameters from measurement statistics.
    Maps (n_diagnostic_pulses, 1) -> (gamma_deph, gamma_relax).
    """

    def __init__(
        self,
        n_diagnostic_pulses: int,
        hidden_dim: int = 64,
        output_dim: int = 2, # gamma_deph, gamma_relax
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_diagnostic_pulses, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus() # Ensure positive rates
        )

    def forward(self, measurement_stats: torch.Tensor) -> torch.Tensor:
        """
        Estimate parameters.

        Args:
            measurement_stats: (batch, n_diagnostic_pulses) or (n_diagnostic_pulses,)

        Returns:
            estimates: (batch, 2) [gamma_deph, gamma_relax]
        """
        return self.net(measurement_stats)
