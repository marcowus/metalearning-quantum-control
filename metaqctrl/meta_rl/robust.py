"""
Robust Loss Functions for Meta-Learning
(CVaR, Entropic Risk/DRO)
© 2025 The MITRE Corporation, All Rights Reserved
"""

import torch

def cvar_loss(losses: torch.Tensor, alpha: float = 0.9) -> torch.Tensor:
    """
    Compute Conditional Value at Risk (CVaR) of a batch of losses.
    Focuses on the worst (1-alpha) fraction of tasks.

    Formula (Rockafellar & Uryasev):
    CVaR_alpha(Z) = min_eta { eta + 1/(1-alpha) * E[ (Z - eta)_+ ] }

    Here we use the "plug-in" estimator where eta is the sample quantile.

    Args:
        losses: 1D tensor of losses.
        alpha: Confidence level (e.g. 0.9 means average of worst 10%).

    Returns:
        cvar: Scalar tensor.
    """
    n = losses.shape[0]
    k = int(n * (1 - alpha))
    if k < 1:
        k = 1 # At least one sample (max loss)

    # Sort losses
    sorted_losses, _ = torch.sort(losses, descending=True)

    # Take top k
    worst_losses = sorted_losses[:k]

    return torch.mean(worst_losses)

def entropic_risk_measure(losses: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Compute Entropic Risk Measure (Log-Sum-Exp).
    Equivalent to dual of KL-DRO.

    Risk = temp * log( E[ exp(loss / temp) ] )

    Args:
        losses: 1D tensor of losses.
        temperature: Controls robustness (lambda).
                     Small temp -> Max loss (Hard robustness).
                     Large temp -> Mean loss.

    Returns:
        risk: Scalar tensor.
    """
    # Use logsumexp for numerical stability
    # log(mean(exp(x))) = log(sum(exp(x))/N) = log(sum(exp(x))) - log(N)
    n = losses.shape[0]
    scaled_losses = losses / temperature

    log_sum_exp = torch.logsumexp(scaled_losses, dim=0)

    risk = temperature * (log_sum_exp - torch.log(torch.tensor(n, dtype=losses.dtype, device=losses.device)))

    return risk
