# Mathematical and Control Principles of Meta-Quantum Control

This document details the mathematical framework, control theory, and meta-learning algorithms used in the **Meta-Reinforcement Learning for Adaptive Quantum Control** project.

## 1. Introduction

The objective of this project is to develop a quantum control policy initialization that can rapidly adapt to varying noise environments. This is achieved using Model-Agnostic Meta-Learning (MAML) combined with a differentiable Lindblad master equation solver. The system models open quantum systems subject to relaxation and dephasing noise, where the precise noise parameters (rates) may vary or be unknown initially.

## 2. Quantum System Dynamics

### 2.1. Lindblad Master Equation

The evolution of the quantum state $\rho(t)$ (density matrix) in an open quantum system is governed by the **Lindblad Master Equation**:

$$
\dot{\rho}(t) = -i [H(t), \rho(t)] + \sum_{j} \mathcal{D}[L_j](\rho(t))
$$

where:
*   $\hbar = 1$ (atomic units).
*   $H(t)$ is the system Hamiltonian.
*   $L_j$ are the Lindblad jump operators representing coupling to the environment.
*   $\mathcal{D}[L](\rho) = L \rho L^\dagger - \frac{1}{2} \{ L^\dagger L, \rho \}$ is the dissipator superoperator, with $\{A, B\} = AB + BA$ denoting the anti-commutator.

### 2.2. Hamiltonian Decomposition

For a single qubit, the time-dependent Hamiltonian is composed of a drift term and control terms:

$$
H(t) = H_0 + \sum_{k=1}^{K} u_k(t) H_k
$$

In this project's implementation (see `metaqctrl/quantum/quantum_environment.py`):
*   **Drift Hamiltonian**: $H_0 = \frac{\omega_d}{2} \sigma_z$ (where $\omega_d$ is the drift strength).
*   **Control Hamiltonians**: $H_1 = \sigma_x$, $H_2 = \sigma_y$.
*   **Control Fields**: $u_k(t)$ are the real-valued control amplitudes (Rabi frequencies).

The Pauli matrices are defined as:

$$
\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad
\sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad
\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
$$

### 2.3. Noise Model and Dissipators

The system is subject to two primary types of Markovian noise: **Amplitude Damping (Relaxation)** and **Pure Dephasing**.

#### Relaxation ($T_1$)
Relaxation describes the energy loss from the excited state $|1\rangle$ to the ground state $|0\rangle$.
*   **Rate**: $\gamma_{\text{relax}} = 1/T_1$.
*   **Operator**: $L_{\text{relax}} = \sqrt{\gamma_{\text{relax}}} \sigma_+$, where $\sigma_+ = |0\rangle\langle 1| = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$.
    *   *Note*: In the codebase, this operator is implemented as `[[0, 1], [0, 0]]`. While often denoted as $\sigma_-$ in spin-down ground state conventions, here it represents the transition from the second basis state to the first.

#### Dephasing ($T_\phi$)
Pure dephasing describes the loss of quantum coherence without energy loss.
*   **Rate**: $\gamma_{\text{deph}} = 1/T_\phi$.
*   **Operator**: $L_{\text{deph}} = \sqrt{\frac{\gamma_{\text{deph}}}{2}} \sigma_z$.

The total dephasing rate $1/T_2^*$ is given by:

$$
\frac{1}{T_2^*} = \frac{1}{2T_1} + \frac{1}{T_\phi} = \frac{\gamma_{\text{relax}}}{2} + \gamma_{\text{deph}}
$$

## 3. Control Optimization Problem

### 3.1. Pulse Parameterization
The control fields $u_k(t)$ are parameterized as **Piecewise Constant** functions (PWC). The total time $T$ is divided into $N$ segments of duration $\Delta t = T/N$.

$$
u_k(t) = u_{k,n} \quad \text{for } t \in [(n-1)\Delta t, n\Delta t)
$$

The control parameters are the vector $\theta = \{ u_{k,n} \} \in \mathbb{R}^{K \times N}$.

### 3.2. Objective Function: Fidelity
The goal is to drive the initial state $\rho_0$ (typically $|0\rangle\langle 0|$) to a target state $\rho_{\text{target}}$ (or implement a target unitary $U_{\text{target}}$).

The **State Fidelity** between two density matrices $\rho$ and $\sigma$ is:

$$
F(\rho, \sigma) = \left( \text{Tr} \sqrt{\sqrt{\rho} \sigma \sqrt{\rho}} \right)^2
$$

For a pure target state $\sigma = |\psi\rangle\langle \psi|$, this simplifies to:

$$
F(\rho, |\psi\rangle\langle \psi|) = \langle \psi | \rho | \psi \rangle = \text{Tr}(\rho \sigma)
$$

The **Loss Function** for optimization is the infidelity:

$$
\mathcal{L}(\theta; \tau) = 1 - F(\rho(T; \theta, \tau), \rho_{\text{target}})
$$

where $\tau$ represents the task-specific noise parameters ($\gamma_{\text{relax}}, \gamma_{\text{deph}}$).

## 4. Meta-Reinforcement Learning (MAML)

The project employs **Model-Agnostic Meta-Learning (MAML)** to find a policy initialization $\theta$ that is robust and adaptable.

### 4.1. Problem Formulation
We assume a distribution of tasks $p(\mathcal{T})$, where each task $\mathcal{T}_i$ corresponds to a specific noise environment $\tau_i = (\gamma_{\text{relax}}^{(i)}, \gamma_{\text{deph}}^{(i)})$.

### 4.2. Algorithm
MAML seeks to minimize the expected loss *after* $K$ steps of gradient adaptation.

1.  **Inner Loop (Adaptation)**:
    For a sampled task $\mathcal{T}_i$, we update the policy parameters $\theta$ to $\theta'_i$ using gradient descent on the task loss:

    $$
    \theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta)
    $$

    where $\alpha$ is the inner learning rate.

2.  **Outer Loop (Meta-Update)**:
    We update the initial parameters $\theta$ to minimize the loss of the *adapted* parameters across a batch of tasks:

    $$
    \theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(\theta'_i)
    $$

    where $\beta$ is the meta learning rate.

### 4.3. First-Order MAML (FOMAML)
To reduce computational cost, the project supports First-Order MAML, which ignores second-order derivatives (the Hessian of the inner loop) during the meta-update:

$$
\nabla_\theta \mathcal{L}(\theta'_i) \approx \nabla_{\theta'_i} \mathcal{L}(\theta'_i)
$$

### 4.4. Differentiable Simulation
A key component is the `DifferentiableLindbladSimulator` (in `lindblad_torch.py`), which allows backpropagation of gradients $\nabla_\theta \mathcal{L}$ through the time-evolution of the master equation. This enables end-to-end gradient-based optimization of the pulses.

## 5. Theoretical Foundations

The validity of Gradient Descent and MAML in this landscape relies on certain geometric properties of the loss landscape, which are validated in `experiments/fig_2_lemma_validation`.

### 5.1. Polyak-Lojasiewicz (PL) Condition
The loss landscape satisfies the PL condition locally around the optimum, ensuring exponential convergence rates for gradient descent:

$$
\frac{1}{2} \| \nabla_\theta \mathcal{L}(\theta) \|^2 \ge \mu (\mathcal{L}(\theta) - \mathcal{L}^*)
$$

where $\mu > 0$ is the PL constant.

### 5.2. Lipschitz Continuity
The mapping from task parameters $\xi$ (noise rates) to the system dynamics (Lindbladian) is Lipschitz continuous. Consequently, the loss function is smooth with respect to task variations:

$$
\| \mathcal{L}(\theta; \xi) - \mathcal{L}(\theta; \xi') \| \le C_L \| \xi - \xi' \|
$$

### 5.3. Control Separation
Optimal control parameters vary smoothly with the underlying task parameters. If two tasks $\xi$ and $\xi'$ are close, their optimal controls $\theta^*_\xi$ and $\theta^*_{\xi'}$ are also close:

$$
\| \theta^*_\xi - \theta^*_{\xi'} \| \le K \| \xi - \xi' \|
$$

This property is crucial for MAML, as it implies that a single initialization can be close to the optima of varying tasks within a local region.

## 6. Noise Models

### 6.1. Gamma-Rate Parameterization
The noise environment is parameterized by the vector $\gamma = (\gamma_{\text{relax}}, \gamma_{\text{deph}})$.
The project uses normalized inputs for the neural network policy:
*   $\hat{\gamma}_{\text{deph}} = \gamma_{\text{deph}} / 0.1$
*   $\hat{\gamma}_{\text{relax}} = \gamma_{\text{relax}} / 0.05$

This ensures the neural network inputs are in a standardized range $\approx [0, 1]$ (see `metaqctrl/quantum/noise_models_gamma.py`).

### 6.2. 1/f Noise (PSD)
Alternative to constant rates, the project also supports colored noise defined by a Power Spectral Density (PSD), $S(\omega) = \frac{A}{\omega^\alpha}$. This is converted to effective Lindblad rates via the `NoiseAdapter` or `PSDToLindblad` classes.
