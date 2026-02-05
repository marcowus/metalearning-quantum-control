"""
Hardware Distortion Models for Quantum Control
© 2025 The MITRE Corporation, All Rights Reserved
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PulseShaper(nn.Module):
    """
    Differentiable model of hardware control channels.
    Simulates bandwidth limitations, amplitude saturation, and slew rate constraints.
    """

    def __init__(
        self,
        n_controls: int = 2,
        dt: float = 1.0,
        cutoff_freq: float = None,  # normalized freq (0, 0.5)
        rise_time: float = None,    # alternative to cutoff_freq
        amp_limit: float = None,
        slew_limit: float = None,
        filter_order: int = 15
    ):
        """
        Args:
            n_controls: Number of control channels.
            dt: Time step duration.
            cutoff_freq: Low-pass filter cutoff frequency (normalized to Nyquist).
            rise_time: Step response rise time (used if cutoff_freq is None).
            amp_limit: Maximum control amplitude (saturation).
            slew_limit: Maximum allowed change per time step.
            filter_order: Length of the FIR filter kernel.
        """
        super().__init__()
        self.n_controls = n_controls
        self.dt = dt
        self.amp_limit = amp_limit
        self.slew_limit = slew_limit

        # Initialize FIR filter
        if cutoff_freq is None and rise_time is not None:
            # Approx relation: rise_time ~ 0.35 / f_c (in Hz)
            # f_c_norm = f_c * dt
            cutoff_freq = 0.35 / (rise_time / dt)

        if cutoff_freq is not None:
            self.use_filter = True
            # Create FIR kernel (Gaussian or Sinc window)
            # Simple Gaussian kernel for smoothness
            sigma = 1.0 / (2 * np.pi * cutoff_freq)
            t = torch.arange(-filter_order, filter_order + 1, dtype=torch.float32)
            kernel = torch.exp(-t**2 / (2 * sigma**2))
            kernel = kernel / kernel.sum()  # Normalize

            # Reshape for Conv1d: (out_channels, in_channels/groups, kernel_size)
            # We use groups=n_controls to filter each channel independently
            self.register_buffer('kernel', kernel.view(1, 1, -1).repeat(n_controls, 1, 1))
            self.padding = filter_order
        else:
            self.use_filter = False

    def forward(self, controls: torch.Tensor) -> torch.Tensor:
        """
        Apply hardware distortions.

        Args:
            controls: (batch, n_segments, n_controls)

        Returns:
            distorted_controls: (batch, n_segments, n_controls)
        """
        x = controls

        # 1. Bandwidth Limitation (FIR Filter)
        if self.use_filter:
            # Rearrange for Conv1d: (batch, channels, length)
            x_perm = x.permute(0, 2, 1)

            # Pad to maintain length
            x_padded = F.pad(x_perm, (self.padding, self.padding), mode='replicate')

            x_filtered = F.conv1d(x_padded, self.kernel, groups=self.n_controls)

            # Revert shape
            x = x_filtered.permute(0, 2, 1)

            # Ensure output length matches input (conv valid might cut)
            if x.shape[1] != controls.shape[1]:
                x = x[:, :controls.shape[1], :]

        # 2. Amplitude Saturation (Smooth tanh)
        if self.amp_limit is not None:
            x = self.amp_limit * torch.tanh(x / self.amp_limit)

        return x

    def compute_slew_penalty(self, controls: torch.Tensor) -> torch.Tensor:
        """
        Compute penalty for violating slew rate limits.

        Args:
            controls: (batch, n_segments, n_controls)

        Returns:
            penalty: Scalar tensor
        """
        if self.slew_limit is None:
            return torch.tensor(0.0, device=controls.device)

        # Diff along time axis
        diffs = torch.abs(controls[:, 1:, :] - controls[:, :-1, :])

        # Relu penalty for exceeding limit
        violation = F.relu(diffs - self.slew_limit)

        return torch.sum(violation**2)
