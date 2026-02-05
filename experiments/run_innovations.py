"""
Run Innovations Experiment
Demonstrates Closed-Loop Calibration, Hardware Constraints, and Robust CVaR Training.
© 2025 The MITRE Corporation, All Rights Reserved
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from metaqctrl.meta_rl.closed_loop import (
    ClosedLoopPolicy, ClosedLoopMAML, ClosedLoopMAMLTrainer
)
from metaqctrl.meta_rl.policy import PulsePolicy
from metaqctrl.meta_rl.estimators import NeuralEstimator
from metaqctrl.quantum.hardware import PulseShaper
from metaqctrl.quantum.noise_models_gamma import GammaTaskDistribution
from metaqctrl.quantum.gates import TargetGates

def main():
    # Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N_ITERATIONS = 20
    TASKS_PER_BATCH = 4
    CVAR_ALPHA = 0.8  # Focus on worst 20%

    print(f"Running innovations demo on {DEVICE}")
    print(f"CVaR Alpha: {CVAR_ALPHA}")

    # 1. Setup Task Distribution
    # Range of gamma rates
    task_dist = GammaTaskDistribution(
        dist_type='uniform',
        gamma_deph_range=(0.02, 0.2),  # Tphi ~ 5s to 50s
        gamma_relax_range=(0.01, 0.1)  # T1 ~ 10s to 100s
    )

    # 2. Setup Target (Pauli X Gate)
    target_unitary = TargetGates.pauli_x()
    # Target state |1><1| from |0><0|
    target_state = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64, device=DEVICE)

    # 3. Initialize Components

    # A. Pulse Policy (Controls)
    # Input: 3 features (gamma_deph, gamma_relax, sum)
    # Output: 20 segments, 2 controls (X, Y)
    control_policy = PulsePolicy(
        task_feature_dim=3,
        hidden_dim=64,
        n_segments=20,
        n_controls=2
    ).to(DEVICE)

    # B. Neural Estimator (Calibration)
    # 4 diagnostic pulses -> 4 measurements
    n_diag = 4
    estimator = NeuralEstimator(
        n_diagnostic_pulses=n_diag,
        hidden_dim=32,
        output_dim=2
    ).to(DEVICE)

    # C. Hardware Shaper (Constraints)
    # Bandwidth limit + Slew rate + Saturation
    pulse_shaper = PulseShaper(
        n_controls=2,
        dt=1.0,
        cutoff_freq=0.1, # 10% of Nyquist
        amp_limit=5.0,
        slew_limit=2.0
    ).to(DEVICE)

    # D. Diagnostic Pulses (Fixed for now)
    # Random small pulses
    diag_pulses = torch.randn(n_diag, 20, 2, device=DEVICE) * 0.5

    # E. Closed Loop Policy Container
    cl_policy = ClosedLoopPolicy(
        estimator=estimator,
        control_policy=control_policy,
        pulse_shaper=pulse_shaper,
        diagnostic_pulses=diag_pulses,
        target_state=target_state
    ).to(DEVICE)

    # 4. Initialize MAML Trainer
    maml = ClosedLoopMAML(
        policy=cl_policy,
        meta_lr=0.005,
        inner_lr=0.0,      # Route B: No inner gradient steps
        inner_steps=0,     # Route B
        robust_alpha=CVAR_ALPHA
    )

    trainer = ClosedLoopMAMLTrainer(
        maml=maml,
        task_distribution=task_dist,
        device=DEVICE,
        log_interval=10,
        val_interval=50
    )

    # 5. Run Training
    print("Starting training...")
    trainer.train(n_iterations=N_ITERATIONS, tasks_per_batch=TASKS_PER_BATCH)

    # 6. Evaluation & Plotting
    print("Generating plots...")
    save_dir = "experiments/innovations_demo"
    os.makedirs(save_dir, exist_ok=True)

    # Plot 1: Training History
    plt.figure(figsize=(10, 5))
    plt.plot(trainer.training_history['meta_loss'], label='Meta Loss (CVaR)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Convergence with Robust Objective')
    plt.legend()
    plt.savefig(f"{save_dir}/training_loss.png")
    plt.close()

    # Evaluate on a test batch for detailed stats
    test_tasks = task_dist.sample(50)
    true_params = []
    est_params = []
    fidelities = []

    cl_policy.eval()
    with torch.no_grad():
        for task in test_tasks:
            result = cl_policy(task, device=str(DEVICE))

            true_params.append(result['real_params'].cpu().numpy())
            est_params.append(result['estimated_params'][0].cpu().numpy())
            fidelities.append(result['fidelity'].item())

    true_params = np.array(true_params)
    est_params = np.array(est_params)
    fidelities = np.array(fidelities)

    # Plot 2: Estimation Accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(true_params[:, 0], est_params[:, 0], alpha=0.6)
    plt.plot([0, 0.2], [0, 0.2], 'k--')
    plt.xlabel('True Gamma Deph')
    plt.ylabel('Estimated')
    plt.title('Dephasing Rate Estimation')

    plt.subplot(1, 2, 2)
    plt.scatter(true_params[:, 1], est_params[:, 1], alpha=0.6, color='orange')
    plt.plot([0, 0.1], [0, 0.1], 'k--')
    plt.xlabel('True Gamma Relax')
    plt.ylabel('Estimated')
    plt.title('Relaxation Rate Estimation')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/estimation_accuracy.png")
    plt.close()

    # Plot 3: Fidelity Distribution (Risk Profile)
    plt.figure(figsize=(8, 6))
    plt.hist(fidelities, bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(np.mean(fidelities), color='k', linestyle='--', label=f'Mean: {np.mean(fidelities):.4f}')

    # Calculate CVaR of Fidelity (Average of worst 20%)
    sorted_fids = np.sort(fidelities)
    cutoff = int(len(fidelities) * (1 - CVAR_ALPHA))
    if cutoff > 0:
        cvar_fid = np.mean(sorted_fids[:cutoff])
        plt.axvline(cvar_fid, color='r', linestyle='--', label=f'CVaR-{CVAR_ALPHA}: {cvar_fid:.4f}')

    plt.xlabel('Fidelity')
    plt.ylabel('Count')
    plt.title('Fidelity Distribution (Closed-Loop + Hardware Constrained)')
    plt.legend()
    plt.savefig(f"{save_dir}/fidelity_distribution.png")
    plt.close()

    print(f"Results saved to {save_dir}/")

if __name__ == "__main__":
    main()
