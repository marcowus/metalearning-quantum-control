
"""
Koopman MPC with PDP for Quantum Control
Adapts the Koopman-MPC-PDP framework to the Lindblad Master Equation.
"""

import sys
import numpy as np
import torch
import casadi as ca
from casadi import SX, mtimes, vertcat, nlpsol
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from metaqctrl.quantum.lindblad_torch import DifferentiableLindbladSimulator
from metaqctrl.quantum.gates import TargetGates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reuse the generic Koopman/MPC classes from the previous script
# (In a real refactor, these would be in a shared library)
# For this demo, we redefine them or import if possible.
# To ensure standalone execution, I will include the minimal necessary classes here.

# ==============================================================================
# 1. Quantum System Wrapper
# ==============================================================================
class BlochLindbladSystem:
    """
    Wraps the DifferentiableLindbladSimulator to expose:
    - State: 3D Bloch vector [rx, ry, rz]
    - Control: 2D [ux, uy]
    - Dynamics: discrete step
    """
    def __init__(self, dt=0.02):
        self.dt = dt
        self.device = 'cpu'

        # Define operators (Pauli)
        self.sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        self.sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
        self.sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
        self.sigma_p = torch.tensor([[0, 1], [0, 0]], dtype=torch.complex64) # |0><1| (Relaxation)

        # Standard noise rates
        self.gamma_relax = 0.05
        self.gamma_deph = 0.02

        self.n_state_obs = 3 # rx, ry, rz
        self.n_control = 2   # ux, uy

        self._init_sim()

    def _init_sim(self):
        # Create simulator
        H0 = torch.zeros((2, 2), dtype=torch.complex64)
        H_controls = [self.sigma_x, self.sigma_y]

        L_relax = np.sqrt(self.gamma_relax) * self.sigma_p
        L_deph = np.sqrt(self.gamma_deph / 2.0) * self.sigma_z
        L_operators = [L_relax, L_deph]

        self.sim = DifferentiableLindbladSimulator(
            H0=H0, H_controls=H_controls, L_operators=L_operators,
            dt=self.dt, method='rk4', device=self.device
        )

    def rho_to_bloch(self, rho):
        # rho is (2,2) complex
        # rx = tr(x rho), ry = tr(y rho), rz = tr(z rho)
        rx = torch.real(torch.trace(self.sigma_x @ rho))
        ry = torch.real(torch.trace(self.sigma_y @ rho))
        rz = torch.real(torch.trace(self.sigma_z @ rho))
        return torch.stack([rx, ry, rz])

    def bloch_to_rho(self, bloch):
        # rho = 0.5 * (I + rx X + ry Y + rz Z)
        I = torch.eye(2, dtype=torch.complex64)
        rho = 0.5 * (I + bloch[0]*self.sigma_x + bloch[1]*self.sigma_y + bloch[2]*self.sigma_z)
        return rho

    def dynamics_step(self, x_bloch_np, u_np):
        """
        Discrete dynamics step: x_{k+1} = f(x_k, u_k)
        Inputs are numpy arrays.
        """
        # Convert to torch
        x_torch = torch.tensor(x_bloch_np, dtype=torch.float32)
        rho_0 = self.bloch_to_rho(x_torch)

        u_torch = torch.tensor(u_np, dtype=torch.float32).unsqueeze(0) # (1, 2)

        # Evolve one step
        rho_next, _ = self.sim.evolve(rho_0, u_torch, T=self.dt)

        # Back to bloch
        x_next = self.rho_to_bloch(rho_next)
        return x_next.detach().numpy()

    def linearize_step(self, x_bloch_np, u_np):
        """
        Compute Jacobian Fx, Fu via finite difference (for the 'True' system adjoint).
        In a full implementation, we'd use torch.autograd.
        """
        eps = 1e-4
        n_x = 3
        n_u = 2

        Fx = np.zeros((n_x, n_x))
        Fu = np.zeros((n_x, n_u))

        # Central difference for x
        for i in range(n_x):
            xp = x_bloch_np.copy(); xp[i] += eps
            xm = x_bloch_np.copy(); xm[i] -= eps
            fp = self.dynamics_step(xp, u_np)
            fm = self.dynamics_step(xm, u_np)
            Fx[:, i] = (fp - fm) / (2*eps)

        # Central difference for u
        for i in range(n_u):
            up = u_np.copy(); up[i] += eps
            um = u_np.copy(); um[i] -= eps
            fp = self.dynamics_step(x_bloch_np, up)
            fm = self.dynamics_step(x_bloch_np, um)
            Fu[:, i] = (fp - fm) / (2*eps)

        return Fx, Fu

    def generate_trajectories(self, num_traj, traj_len):
        state_trajs = []
        control_trajs = []

        for _ in range(num_traj):
            # Random initial state on Bloch sphere
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            x0 = np.array([
                np.sin(theta)*np.cos(phi),
                np.sin(theta)*np.sin(phi),
                np.cos(theta)
            ])

            states = [x0]
            controls = []
            xk = x0

            for _ in range(traj_len):
                # Random control pulses
                uk = np.random.uniform(-5.0, 5.0, 2)
                xk_next = self.dynamics_step(xk, uk)
                states.append(xk_next)
                controls.append(uk)
                xk = xk_next

            state_trajs.append(np.array(states))
            control_trajs.append(np.array(controls))

        return state_trajs, control_trajs, state_trajs # obs = full state

# ==============================================================================
# 2. Koopman ID (Linear Fit on Bloch Vector)
# ==============================================================================
class KoopmanLinear:
    def fit(self, X, U):
        # X: list of (T+1, nx), U: list of (T, nu)
        # Form matrices
        Data_X = []
        Data_Y = [] # X next
        Data_U = []

        for x, u in zip(X, U):
            Data_X.append(x[:-1])
            Data_Y.append(x[1:])
            Data_U.append(u)

        X_mat = np.vstack(Data_X).T
        Y_mat = np.vstack(Data_Y).T
        U_mat = np.vstack(Data_U).T

        # Solve Y = [A B] [X; U]
        Gamma = np.vstack([X_mat, U_mat])
        AB = Y_mat @ np.linalg.pinv(Gamma)

        nx = X[0].shape[1]
        self.A = AB[:, :nx]
        self.B = AB[:, nx:]

        return self.A, self.B

# ==============================================================================
# 3. Simple MPC with CasADi
# ==============================================================================
class QuantumMPC:
    def __init__(self, A, B, horizon=10):
        self.A = A
        self.B = B
        self.H = horizon
        self.nx = 3
        self.nu = 2

        self.setup_casadi()

    def setup_casadi(self):
        # Variables
        self.U = ca.SX.sym('U', self.nu, self.H) # Control sequence
        self.x0 = ca.SX.sym('x0', self.nx)       # Initial state
        self.ref = ca.SX.sym('ref', self.nx)     # Target state

        # Cost weights (parameters to learn)
        # We parameterize diagonal Q and R
        self.theta = ca.SX.sym('theta', self.nx + self.nu) # log(Q_diag), log(R_diag)

        Q_diag = ca.exp(self.theta[:self.nx])
        R_diag = ca.exp(self.theta[self.nx:])
        Q = ca.diag(Q_diag)
        R = ca.diag(R_diag)

        cost = 0
        curr_x = self.x0

        # Dynamics constraints
        g = []

        for k in range(self.H):
            uk = self.U[:, k]
            # Cost
            err = curr_x - self.ref
            cost += ca.mtimes([err.T, Q, err]) + ca.mtimes([uk.T, R, uk])

            # Linear Dynamics
            curr_x = ca.mtimes(self.A, curr_x) + ca.mtimes(self.B, uk)

        # Terminal cost (weighted higher)
        err = curr_x - self.ref
        cost += 10.0 * ca.mtimes([err.T, Q, err])

        # NLP
        # We optimize U. Parameters are x0, ref, theta.
        params = ca.vertcat(self.x0, self.ref, self.theta)
        nlp = {'x': ca.reshape(self.U, -1, 1), 'f': cost, 'p': params}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.solver = nlpsol('solver', 'ipopt', nlp, opts)

    def solve(self, x0, target, theta):
        x0 = np.array(x0).flatten()
        target = np.array(target).flatten()
        theta = np.array(theta).flatten()

        params = np.concatenate([x0, target, theta])

        # Warm start with zeros
        u0 = np.zeros(self.nu * self.H)

        sol = self.solver(x0=u0, p=params)
        u_opt = sol['x'].full().reshape(self.nu, self.H).T # (H, nu)

        return u_opt[0], sol # Return first action

# ==============================================================================
# 4. Main Experiment Loop
# ==============================================================================
def main():
    # 1. Setup System
    sys_wrapper = BlochLindbladSystem(dt=0.05)

    # 2. Generate Data for System ID
    logger.info("Generating ID data...")
    X, U, _ = sys_wrapper.generate_trajectories(num_traj=20, traj_len=50)

    # 3. Learn Linear Koopman Model
    logger.info("Fitting Koopman model...")
    kid = KoopmanLinear()
    A, B = kid.fit(X, U)
    print("A matrix:\n", A)
    print("B matrix:\n", B)

    # 4. Setup MPC
    horizon = 10
    mpc = QuantumMPC(A, B, horizon=horizon)

    # 5. Meta-Learning (Simplified PDP)
    # We want to transfer |0> -> |1> (Bloch: [0,0,1] -> [0,0,-1])
    # Ideally south pole is [0,0,-1]

    x0_bloch = np.array([0.0, 0.0, 1.0]) # |0>
    target_bloch = np.array([0.0, 0.0, -1.0]) # |1>

    # Initial weights: Q=1, R=0.1
    # theta = log([1,1,1, 0.1, 0.1])
    theta = np.log(np.array([1.0, 1.0, 1.0, 0.1, 0.1]))

    meta_lr = 0.1
    meta_iters = 10

    costs = []

    logger.info("Starting Meta-Training...")

    for iter in range(meta_iters):
        # Rollout with current MPC weights
        curr_x = x0_bloch.copy()
        traj = [curr_x]
        total_fidelity_loss = 0

        # Simple finite difference gradient for theta (Proof of Concept)
        # In the full file I submitted previously, I used the exact adjoint LQR.
        # Here, for brevity and robustness in this demo, I'll use numerical gradient
        # on the *simulation rollout cost*.

        # Evaluation function
        def evaluate_rollout(theta_val):
            x = x0_bloch.copy()
            loss = 0
            for t in range(horizon):
                u, _ = mpc.solve(x, target_bloch, theta_val)
                x = sys_wrapper.dynamics_step(x, u)
                # Loss = distance to target (proxy for infidelity)
                dist = np.linalg.norm(x - target_bloch)**2
                loss += dist
            return loss

        # 1. Compute current loss
        current_loss = evaluate_rollout(theta)
        costs.append(current_loss)

        logger.info(f"Iter {iter}: Loss = {current_loss:.4f}, Weights = {np.exp(theta)}")

        # 2. Estimate Gradient (Finite Diff)
        grad = np.zeros_like(theta)
        eps = 0.05

        for i in range(len(theta)):
            theta_p = theta.copy(); theta_p[i] += eps
            loss_p = evaluate_rollout(theta_p)
            grad[i] = (loss_p - current_loss) / eps

        # 3. Update
        theta = theta - meta_lr * grad

    # Plot results
    plt.figure()
    plt.plot(costs, marker='o')
    plt.title("Meta-Learning MPC Weights for State Transfer")
    plt.xlabel("Iteration")
    plt.ylabel("Trajectory Error (Euclidean)")
    plt.grid(True)
    plt.savefig("quantum_mpc_learning.png")
    print("Saved quantum_mpc_learning.png")

if __name__ == "__main__":
    main()
