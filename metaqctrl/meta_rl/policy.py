"""
Policy Networks for Quantum Control. 
Task features --> control pulses 

© 2025 The MITRE Corporation, All Rights Reserved   
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class PulsePolicy(nn.Module):
    """  
    Neural network policy that outputs control pulse sequences.
    """
    
    def __init__(
        self,
        task_feature_dim: int = 3,  
        hidden_dim: int = 128,
        n_hidden_layers: int = 2,
        n_segments: int = 20,
        n_controls: int = 2,  # (X, Y) for single qubit gate 
        output_scale: float = 1.0,
        activation: str = 'tanh'
    ):
        """ Good. 
        Args:
            task_feature_dim: Dimension of task encoding
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
        

        layers = []
        layers.append(nn.Linear(task_feature_dim, hidden_dim))
        layers.append(self._get_activation(activation))
        
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self._get_activation(activation))
        
        layers.append(nn.Linear(hidden_dim, self.output_dim)) 
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        """ 
        Get activation function."""
        if name == 'tanh':
            return nn.Tanh()
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'elu':
            return nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    def _init_weights(self):
        """ 
        Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, task_features: torch.Tensor) -> torch.Tensor:
        """
        Generate control pulses for given task.
        
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


class TaskFeatureEncoder(nn.Module):
    def __init__(
        self,
        raw_dim: int = 3,
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
        Encode raw task parameters.
        
        Args:
            raw_features: (batch, raw_dim) - e.g., (α, A, ωc)
            
        Returns:
            encoded: (batch, feature_dim)
        """
        if self.use_fourier:
            x_proj = raw_features @ self.B
            encoded = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
        else:
            encoded = self.encoder(raw_features)
        
        return encoded

def create_policy(
    config: dict,
    device: torch.device = torch.device('cpu')
) -> PulsePolicy:
    """
    Makes policy . 
    Args:
        config: Dictionary with policy hyperparameters
        device: torch device
        
    Returns:
        policy: Initialized policy network
    """
    policy = PulsePolicy(
        task_feature_dim=config.get('task_feature_dim', 3),
        hidden_dim=config.get('hidden_dim', 128),
        n_hidden_layers=config.get('n_hidden_layers', 2),
        n_segments=config.get('n_segments', 20),
        n_controls=config.get('n_controls', 2),
        output_scale=config.get('output_scale', 1.0),
        activation=config.get('activation', 'tanh')
    )
    
    policy = policy.to(device)
    return policy 
