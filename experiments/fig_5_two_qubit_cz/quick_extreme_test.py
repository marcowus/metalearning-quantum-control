"""
Quick test with extreme noise distribution to find where MAML helps.
© 2025 The MITRE Corporation, All Rights Reserved 
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.optim as optim
from copy import deepcopy

from two_qubit_cz_maml_fast import (
    TwoQubitTaskParams, TwoQubitCZPolicy,
    create_two_qubit_simulator, average_gate_fidelity_cz,
    CZ_IDEAL_GATE_TIME
)

torch.manual_seed(42)
np.random.seed(42)
device = 'cpu'

print("=" * 70)
print("QUICK TEST: Extreme noise distribution (500x range)")
print("γ_deph: 0.001 - 0.5, γ_relax: 0.0005 - 0.25")
print("=" * 70)

# Extreme tasks at corners
tasks = [
    TwoQubitTaskParams(0.001, 0.0005, 0.001, 0.0005),  # Low noise
    TwoQubitTaskParams(0.5, 0.25, 0.5, 0.25),          # High noise (EXTREME)
    TwoQubitTaskParams(0.1, 0.05, 0.1, 0.05),          # Medium
    TwoQubitTaskParams(0.25, 0.125, 0.25, 0.125),      # Medium-high
]

print("\n1. TASK-SPECIFIC OPTIMAL (upper bound):")
task_optimal = []
for i, task in enumerate(tasks):
    policy = TwoQubitCZPolicy(
        task_feature_dim=4, hidden_dim=256, n_hidden_layers=4,
        n_segments=30, n_controls=6
    ).to(device)
    policy.train()
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    sim = create_two_qubit_simulator(task, device=device)
    task_features = torch.tensor(task.to_array(normalized=True), dtype=torch.float32, device=device)

    for _ in range(300):
        optimizer.zero_grad()
        controls = policy(task_features)
        fid = average_gate_fidelity_cz(sim, controls, CZ_IDEAL_GATE_TIME, device)
        loss = 1 - fid
        loss.backward()
        optimizer.step()

    policy.eval()
    with torch.no_grad():
        controls = policy(task_features)
        fid = average_gate_fidelity_cz(sim, controls, CZ_IDEAL_GATE_TIME, device).item()
    task_optimal.append(fid)
    print(f"   Task {i+1} (γ_d={task.gamma_deph_1:.3f}): {fid:.4f}")

print(f"\n   Spread: {max(task_optimal):.4f} - {min(task_optimal):.4f} = {max(task_optimal)-min(task_optimal):.4f}")

print("\n2. FIXED AVERAGE (trained on medium noise):")
avg_task = TwoQubitTaskParams(0.1, 0.05, 0.1, 0.05)
fixed_policy = TwoQubitCZPolicy(
    task_feature_dim=4, hidden_dim=256, n_hidden_layers=4,
    n_segments=30, n_controls=6
).to(device)
fixed_policy.train()
optimizer = optim.Adam(fixed_policy.parameters(), lr=0.001)
sim_avg = create_two_qubit_simulator(avg_task, device=device)
task_features_avg = torch.tensor(avg_task.to_array(normalized=True), dtype=torch.float32, device=device)

for i in range(300):
    optimizer.zero_grad()
    controls = fixed_policy(task_features_avg)
    fid = average_gate_fidelity_cz(sim_avg, controls, CZ_IDEAL_GATE_TIME, device)
    loss = 1 - fid
    loss.backward()
    optimizer.step()

fixed_policy.eval()

# Test on all tasks
print("\n   Performance on each task:")
fixed_k0 = []
for i, task in enumerate(tasks):
    sim = create_two_qubit_simulator(task, device=device)
    task_features = torch.tensor(task.to_array(normalized=True), dtype=torch.float32, device=device)
    with torch.no_grad():
        controls = fixed_policy(task_features)
        fid = average_gate_fidelity_cz(sim, controls, CZ_IDEAL_GATE_TIME, device).item()
    fixed_k0.append(fid)
    print(f"   Task {i+1}: {fid:.4f}")

print(f"\n   Mean: {np.mean(fixed_k0):.4f}, Min: {min(fixed_k0):.4f}")

print("\n3. FIXED AVERAGE + ADAPTATION (K=30):")
fixed_k30 = []
inner_lr = 0.001  # Higher LR for faster adaptation
for i, task in enumerate(tasks):
    adapted = deepcopy(fixed_policy)
    adapted.train()
    opt = optim.Adam(adapted.parameters(), lr=inner_lr)
    sim = create_two_qubit_simulator(task, device=device)
    task_features = torch.tensor(task.to_array(normalized=True), dtype=torch.float32, device=device)

    for _ in range(30):
        opt.zero_grad()
        controls = adapted(task_features)
        fid = average_gate_fidelity_cz(sim, controls, CZ_IDEAL_GATE_TIME, device)
        loss = 1 - fid
        loss.backward()
        opt.step()

    adapted.eval()
    with torch.no_grad():
        controls = adapted(task_features)
        fid = average_gate_fidelity_cz(sim, controls, CZ_IDEAL_GATE_TIME, device).item()
    fixed_k30.append(fid)
    improvement = fid - fixed_k0[i]
    print(f"   Task {i+1}: {fixed_k0[i]:.4f} -> {fid:.4f} (Δ = {improvement:+.4f})")

print(f"\n   Mean improvement: {np.mean(fixed_k30) - np.mean(fixed_k0):+.4f}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Task-specific optimal:    {np.mean(task_optimal):.4f} (spread: {max(task_optimal)-min(task_optimal):.4f})")
print(f"Fixed Average K=0:        {np.mean(fixed_k0):.4f} (min: {min(fixed_k0):.4f})")
print(f"Fixed Average K=30:       {np.mean(fixed_k30):.4f} (min: {min(fixed_k30):.4f})")
print(f"Adaptation benefit:       {np.mean(fixed_k30) - np.mean(fixed_k0):+.4f}")
print(f"Generalization gap:       {np.mean(task_optimal) - np.mean(fixed_k0):+.4f}")
print("=" * 70)

if np.mean(task_optimal) - np.mean(fixed_k0) > 0.05:
    print("\n✓ MAML could help here! Generalization gap > 5%")
else:
    print("\n✗ MAML unlikely to help. Tasks too similar.")
