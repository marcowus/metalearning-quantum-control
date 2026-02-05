# Koopman MPC with PDP for Meta-Quantum Control

This document outlines the theoretical framework for applying **Koopman Operator Theory**, **Model Predictive Control (MPC)**, and **Pontryagin Differentiable Programming (PDP)** to the problem of robust quantum control.

## 1. Problem Formulation

The goal is to control a quantum system governed by the Lindblad Master Equation:
$$ \dot{\rho} = -i[H(u(t)), \rho] + \mathcal{L}(\rho) $$
to maximize the fidelity with a target state $\rho_{target}$ at time $T$, despite uncertainties in the noise parameters $\xi$ inside $\mathcal{L}$.

## 2. The Koopman Bridge

### 2.1. Observable Space (Lifting)
The dynamics of the density matrix $\rho$ are linear in the Liouville space, but the control $u(t)$ enters bilinearly (e.g., $u(t) \sigma_x \rho$).
To apply linear MPC, we lift the state into a space of observables $z = \phi(\rho)$.
For a single qubit, the natural observables are the **Bloch vector** components:
$$ z = [r_x, r_y, r_z]^T = [\text{Tr}(\sigma_x \rho), \text{Tr}(\sigma_y \rho), \text{Tr}(\sigma_z \rho)]^T $$

### 2.2. Koopman Approximation
We approximate the discrete-time evolution ($\Delta t$) of these observables using a linear control model (EDMDc):
$$ z_{k+1} \approx A z_k + B u_k $$
where $A$ captures the drift and dissipation, and $B$ captures the control rotation.
*Note: While quantum dynamics are bilinear, a linear approximation is valid for small time steps or small rotations. More advanced Koopman models could include bilinear terms $u \otimes z$.*

## 3. Bilevel Optimization Strategy

We replace the standard black-box Neural Network policy with a structured **MPC Controller** whose internal cost function is learned.

### Inner Loop: Model Predictive Control (MPC)
At each time step $t$, the controller solves a convex optimization problem:
$$
\begin{aligned}
\min_{u_{0:H-1}} \quad & \sum_{k=0}^{H-1} \left( (z_k - z_{ref})^T Q z_k + u_k^T R u_k \right) + (z_H - z_{ref})^T Q_f z_H \\
\text{s.t.} \quad & z_{k+1} = A z_k + B u_k \\
& |u_k| \leq u_{max}
\end{aligned}
$$
Here, the matrices $Q, R, Q_f$ are parameterized by a vector $\theta$.

### Outer Loop: PDP Adjoint Optimization (Meta-Learning)
We want to find the optimal weights $\theta$ (parameterizing $Q, R$) such that the **closed-loop trajectory** driven by the MPC maximizes fidelity on the **true nonlinear system**:

$$ \min_\theta \mathcal{J}_{meta}(\theta) = \mathbb{E}_{\xi \sim p(\xi)} \left[ 1 - \mathcal{F}(\rho_{true}(T), \rho_{target}) \right] $$

**Gradient Computation:**
We compute $\nabla_\theta \mathcal{J}_{meta}$ using the chain rule:
$$ \frac{d \mathcal{J}}{d \theta} = \frac{\partial \mathcal{J}}{\partial \rho} \cdot \frac{\partial \rho}{\partial u} \cdot \frac{\partial u_{MPC}}{\partial \theta} $$

1.  $\frac{\partial \mathcal{J}}{\partial \rho}$: From the Fidelity loss.
2.  $\frac{\partial \rho}{\partial u}$: From the **Differentiable Lindblad Simulator** (Backprop through time or Adjoint method).
3.  $\frac{\partial u_{MPC}}{\partial \theta}$: From the **PDP / Implicit Function Theorem**, differentiating through the KKT conditions of the MPC quadratic program.

## 4. Advantages

1.  **Interpretability:** The learned policy is an MPC controller with physically meaningful weights, not a black-box neural net.
2.  **Constraint Handling:** MPC explicitly handles control amplitude limits ($|u| \leq u_{max}$).
3.  **Efficiency:** The inner loop solves a convex QP, which is fast and stable.
4.  **Sim-to-Real:** The outer loop learns to compensate for the mismatch between the simple Linear Koopman model (used inside MPC) and the complex True Lindblad system.

## 5. Implementation Plan

1.  **System ID:** Generate trajectories from `LindbladSimulator` and fit $(A, B)$ using linear regression (EDMD).
2.  **MPC Setup:** Use CasADi to define the MPC problem with symbolic parameters $\theta$.
3.  **Meta-Training:** Run the bilevel loop:
    *   Rollout true system with MPC.
    *   Compute gradients via adjoints.
    *   Update $\theta$.
