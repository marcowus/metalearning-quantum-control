# Meta-Reinforcement Learning for Adaptive Quantum Control
Author: Nima Leclerc (nleclerc@mitre.org) -- PI for Adaptive Quantum Sensing and Quantum Research Scientist at MITRE

© 2025 The MITRE Corporation, All Rights Reserved

Approved for Public Release; Distribution Unlimited. Public Release Case Number 25-2936.  

A research implementation of first-order Model-Agnostic Meta-Learning (MAML) for quantum state control under noise. This framework trains a meta-learned policy initialization that rapidly adapts to new quantum noise environments with minimal gradient steps.

## Overview

  Lindblad Master Equation:
  - Full equation with commutator $[H, \rho]$ and dissipator terms
  - CPTP map notation $\mathcal{E}[\rho_0]$

  Control Hamiltonian:
  - Rabi frequency formulation with $\Omega_x(t)/2$
  - Explicit Pauli matrices written out
  - Piecewise-constant pulse discretization formula

  Decoherence Model:
  - Explicit Lindblad operators: $L_1 = \sqrt{\gamma_1}\sigma_-$ and $L_\phi = \sqrt{\gamma_\phi/2}\sigma_z$
  - $T_1$, $T_2$, $T_2^*$ relationships
  - Task parameterization $\tau = (\gamma_{\text{deph}}, \gamma_{\text{relax}})$

  Fidelity:
  - Average gate fidelity formula
  - Process fidelity $F_{\text{pro}} = \frac{1}{d^2}|\text{Tr}(U^\dagger \mathcal{E})|^2$
  - Loss function $\mathcal{L}(\theta; \tau) = 1 - F$

  Two-Qubit:
  - Local control Hamiltonian sum
  - Heisenberg exchange coupling $H_{\text{int}} = J(\sigma_x\sigma_x + \sigma_y\sigma_y + \sigma_z\sigma_z)$ 


This project combines meta-reinforcement learning with quantum control theory, enabling control policies to adapt quickly to different noise profiles by leveraging task-specific structure. The key innovation is a fully differentiable Lindblad master equation simulator that allows end-to-end gradient-based meta-learned optimization of optimal pulse sequences.

The framework supports two noise parameterizations:
- **PSD-based**: Colored noise via power spectral density (1/f, Lorentzian, etc.)
- **Gamma-rate**: Direct Lindblad decoherence rates (γ_deph, γ_relax)

## Project Structure

```
meta-quantum-control/
├── metaqctrl/                    # Main package
│   ├── meta_rl/                  # Meta-learning algorithms
│   │   ├── maml.py               # MAML implementation
│   │   ├── maml_gamma.py         # Gamma-parameterized MAML
│   │   ├── policy.py             # Neural network policies (PSD)
│   │   └── policy_gamma.py       # Gamma-parameterized policies
│   ├── quantum/                  # Quantum simulation
│   │   ├── lindblad.py           # NumPy Lindblad simulator
│   │   ├── lindblad_torch.py     # Differentiable PyTorch simulator
│   │   ├── gates.py              # Fidelity computation
│   │   ├── noise_models_v2.py    # Noise PSD models v2 
│   │   ├── noise_models.py       # Noise PSD models
│   │   ├── noise_models_gamma.py # Gamma-rate noise models
│   │   └── noise_adapter.py      # PSD to Lindblad conversion
│   │   └──  quantum_environment.py    # Unified simulation interface  
│   └── utils/                    # Utilities
│       ├── checkpoint_utils.py   # Model saving/loading 
├── experiments/                  # Reproducible experiment scripts
│   ├── fig_2_lemma_validation/
│   ├── fig_3_adaptation_gap_analysis/
│   ├── fig_4_adaptation_dynamics/
│   ├── fig_5_two_qubit_cz/
│   ├── fig_task_variance_correlation/
│   └── figs_appendix_ablations/
│   └── fig_appendix_meta_training/
│   └── figs_appendix_classical/
│   └── fig_appendix_maml_vs_grape/ 
├── configs/                      # Configuration files
│   ├── experiment_config.yaml         # PSD-based config
│   └── experiment_config_gamma.yaml   # Gamma-rate config
├── checkpoints/                  # Saved model weights
│   └── checkpoints_gamma/        # Gamma-trained checkpoints
```

### Prerequisites

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

## Quick Start

### Training a Gamma-Parameterized Meta-Policy (Recommended)

```bash
cd experiments/fig_5_meta_training
uv run train_meta_gamma.py --config ../../configs/experiment_config_gamma.yaml
```

This will:
1. Train a FOMAML policy for single-qubit and two-qubit control using a Lindblad simulator to capture decoherence effects. 
2. Save checkpoints to `checkpoints_gamma/`
3. Log training metrics to `checkpoints_gamma/training_history.json`
4. Display training progress with pre/post-adaptation validation metrics

### Evaluating a Trained Policy

```python
from metaqctrl.utils.checkpoint_utils import load_policy_from_checkpoint
from metaqctrl.quantum.lindblad_torch import DifferentiableLindbladSimulator
import torch

# Load trained gamma-parameterized policy
policy = load_policy_from_checkpoint("checkpoints/checkpoints_gamma/maml_gamma_pauli_x.pt")

# Create task features: [gamma_deph/0.1, gamma_relax/0.05, sum/0.15]
gamma_deph, gamma_relax = 0.10, 0.05
task_features = torch.tensor([[
    gamma_deph / 0.1,
    gamma_relax / 0.05,
    (gamma_deph + gamma_relax) / 0.15
]])

# Generate control pulses
controls = policy(task_features)
print(f"Control shape: {controls.shape}")  # [1, n_segments, 2]
```

## Reproducing Paper Figures

All experiments are organized by figure number. Each script uses fixed random seeds for reproducibility.

### Main Figures

| Figure | Script | Description |
|--------|--------|-------------|
| Fig. 1 | `experiments/fig_1_overview/fig1.py` | System overview schematic |
| Fig. 2 | `experiments/fig_2_lemma_validation/lemma_validation.py` | Theoretical lemma validation |
| Fig. 3 | `experiments/fig_3_adaptation_gap_analysis/generate_adaptation_gap_figure_gamma_checkpoint.py` | Adaptation gap analysis (exponential saturation + task diversity scaling) |
| Fig. 3 (alt) | `experiments/fig_3_adaptation_gap_analysis/generate_adaptation_gap_figure_actual_variance.py` | Adaptation gap with actual task variance |
| Fig. 4 | `experiments/fig_4_adaptation_dynamics/adaptation_dynamics_figure_gamma_checkpoint.py` | Adaptation dynamics over K steps |
| Fig. 5 | `experiments/fig_5_meta_training/generate_cz_adaptation_gap_figure_fast.py` | Two Qubit gate results |


### Running Figure Generation Script

Highlights general structure for running script. 
```bash
# Figure 3: Adaptation Gap Analysis
python -u experiments/fig_3_adaptation_gap_analysis/generate_adaptation_gap_figure_gamma_checkpoint.py \
    --checkpoint checkpoints/checkpoints_gamma/maml_gamma_pauli_x.pt \
    --n_tasks 60 --max_K 30 --inner_lr 0.0001
```

## Citation

```bibtex
@inproceedings{leclerc2025meta,
  title={Meta-Reinforcement Learning for Quantum Control},
  author={Leclerc, Nima and Miller, Chris and Brawand, Nicholas},
  year={2025}
}
```
