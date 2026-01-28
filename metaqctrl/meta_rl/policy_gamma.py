"""
Policy Networks for Quantum Control (Gamma Parameterization)

Maps gamma-rate task features to control pulse sequences.

Key differences from policy.py:
- task_feature_dim=3 for gamma features [γ_deph/0.1, γ_relax/0.05, sum/0.15]
- Input normalization designed for gamma rates
- Use with GammaNoiseParameters from noise_models_gamma.py
© 2025 The MITRE Corporation, All Rights Reserved    
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class GammaPulsePolicy(nn.Module):
    """
    Neural network policy for gamma-parameterized quantum control.

    Input features (3D):
        [gamma_deph/0.1, gamma_relax/0.05, (gamma_deph + gamma_relax)/0.15]

    This normalization ensures:
        - Features are roughly in [0, 2] range for typical gamma values
        - Network sees balanced feature magnitudes
        - Compatible with GammaNoiseParameters.to_array(normalized=True)
    """

    def __init__(
        self,
        task_feature_dim: int = 3,
        hidden_dim: int = 128,
        n_hidden_layers: int = 2,
        n_segments: int = 20,
        n_controls: int = 2,  
        output_scale: float = 1.0,
        activation: str = 'tanh'
    ):
        """
        Args:
            task_feature_dim: Dimension of gamma task encoding (default 3)
            hidden_dim: Hidden layer width
            n_hidden_layers: Number of hidden layers
            n_segments: Number of pulse segments
            n_controls: Number of control channels
            output_scale: Scale factor for control amplitudes
            activation: 'tanh', 'relu', 'elu'
        """
        super().__init__()

        self.task_feature_dim = task_feature_dim
        self.hidden_dim = hidden_dim
        self.n_segments = n_segments
        self.n_controls = n_controls
        self.output_dim = n_segments * n_controls
        self.output_scale = output_scale

        # Build MLP
        layers = []

        # Input layer
        layers.append(nn.Linear(task_feature_dim, hidden_dim))
        layers.append(self._get_activation(activation))

        # Hidden layers
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self._get_activation(activation))

        # Output layer
        layers.append(nn.Linear(hidden_dim, self.output_dim))

        self.network = nn.Sequential(*layers)

        self._init_weights()

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function."""
        if name == 'tanh':
            return nn.Tanh()
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'elu':
            return nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {name}")

    def _init_weights(self):
        """Initialize network weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, task_features: torch.Tensor) -> torch.Tensor:
        """
        Generate control pulses for given gamma task features.

        Args:
            task_features: (batch_size, task_feature_dim) or (task_feature_dim,)

        Returns:
            controls: (batch_size, n_segments, n_controls) or (n_segments, n_controls)
        """
        single_input = task_features.ndim == 1
        if single_input:
            task_features = task_features.unsqueeze(0)

        output = self.network(task_features)


        controls = output.view(-1, self.n_segments, self.n_controls)
        controls = self.output_scale * controls

        if single_input:
            controls = controls.squeeze(0)

        return controls

    def get_lipschitz_constant(self) -> float:
        """
        Estimate Lipschitz constant L_net via spectral norms.

        """
        lipschitz = 1.0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data
                spectral_norm = torch.linalg.matrix_norm(W, ord=2).item()
                lipschitz *= spectral_norm
        return lipschitz

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GammaTaskFeatureEncoder(nn.Module):
    """
    Optional: Learn task representations from raw gamma parameters.

    Can include:
    - Fourier features for better expressiveness
    - Learned embeddings
    - Feature normalization
    """

    def __init__(
        self,
        raw_dim: int = 2,  # (gamma_deph, gamma_relax) before normalization
        feature_dim: int = 16,
        use_fourier: bool = True,
        fourier_scale: float = 1.0
    ):
        super().__init__()

        self.raw_dim = raw_dim
        self.feature_dim = feature_dim
        self.use_fourier = use_fourier

        if use_fourier:
            self.register_buffer(
                'B',
                torch.randn(raw_dim, feature_dim // 2) * fourier_scale
            )
            final_dim = feature_dim
        else:
            self.encoder = nn.Sequential(
                nn.Linear(raw_dim, 32),
                nn.ReLU(),
                nn.Linear(32, feature_dim)
            )
            final_dim = feature_dim

        self.output_dim = final_dim

    def forward(self, raw_features: torch.Tensor) -> torch.Tensor:
        """
        Encode raw gamma parameters.

        Args:
            raw_features: (batch, raw_dim) - (gamma_deph, gamma_relax)

        Returns:
            encoded: (batch, feature_dim)
        """
        if self.use_fourier:
            x_proj = raw_features @ self.B
            encoded = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
        else:
            encoded = self.encoder(raw_features)

        return encoded

def create_gamma_policy(
    config: dict,
    device: torch.device = torch.device('cpu')
) -> GammaPulsePolicy:
    """
    Factory function to create gamma policy from config.

    Args:
        config: Dictionary with policy hyperparameters
        device: torch device

    Returns:
        policy: Initialized GammaPulsePolicy network
    """
    policy = GammaPulsePolicy(
        task_feature_dim=config.get('task_feature_dim', 3),  # 3 for gamma
        hidden_dim=config.get('hidden_dim', 128),
        n_hidden_layers=config.get('n_hidden_layers', 2),
        n_segments=config.get('n_segments', 20),
        n_controls=config.get('n_controls', 2),
        output_scale=config.get('output_scale', 1.0),
        activation=config.get('activation', 'tanh')
    )

    policy = policy.to(device)

    print(f"Created gamma policy with {policy.count_parameters():,} parameters")
    print(f"Estimated Lipschitz constant: {policy.get_lipschitz_constant():.2f}")

    return policy

PulsePolicy = GammaPulsePolicy 
