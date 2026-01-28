"""   
© 2025 The MITRE Corporation, All Rights Reserved  
"""  
import numpy as np
from scipy.linalg import solve_continuous_are, solve_lyapunov
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'serif',
})

np.random.seed(42)


def get_system_matrices(mass, damping, stiffness):
    """
    Mass-spring-damper system: m*x'' + c*x' + k*x = u
    State: [position, velocity]
    """
    A = np.array([[0, 1],
                  [-stiffness/mass, -damping/mass]])
    B = np.array([[0], [1/mass]])
    return A, B


def solve_lqr(A, B, Q, R):
    """Solve continuous-time LQR, return gain K and cost matrix P"""
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K, P


def evaluate_cost(A, B, K, Q, R, x0):
    """
    Compute closed-loop LQR cost for gain K on system (A,B) from initial state x0.
    Cost = x0' P_cl x0 where P_cl solves the Lyapunov equation.
    """
    A_cl = A - B @ K

    # Check stability
    eigvals = np.linalg.eigvals(A_cl)
    if np.max(np.real(eigvals)) >= 0:
        return np.inf

    # Cost matrix for closed-loop system
    Q_cl = Q + K.T @ R @ K
    P_cl = solve_lyapunov(A_cl.T, -Q_cl)

    return float(x0.T @ P_cl @ x0)


def policy_gradient_step(K, A, B, Q, R, lr=0.1):
    """
    One step of policy gradient for LQR.
    The gradient of J w.r.t. K involves the solution to Lyapunov equations.
    """
    A_cl = A - B @ K

    # Check stability
    if np.max(np.real(np.linalg.eigvals(A_cl))) >= 0:
        return K, False

    # Solve for P (cost-to-go matrix)
    Q_cl = Q + K.T @ R @ K
    P = solve_lyapunov(A_cl.T, -Q_cl)

    Sigma = solve_lyapunov(A_cl, -np.eye(A.shape[0]))

    grad_K = 2 * (R @ K - B.T @ P) @ Sigma

    # Gradient descent
    K_new = K - lr * grad_K

    return K_new, True


def adapt_controller(K_init, A, B, Q, R, n_steps, lr=0.1):
    """
    Adapt controller via gradient descent for n_steps.
    Returns adapted gain and cost trajectory.
    """
    K = K_init.copy()
    costs = []

    for _ in range(n_steps):
        K, success = policy_gradient_step(K, A, B, Q, R, lr)
        if not success:
            costs.append(np.inf)
            continue

        # Evaluate cost (using a fixed x0 for consistency)
        x0 = np.array([[1.0], [0.0]])
        cost = evaluate_cost(A, B, K, Q, R, x0)
        costs.append(cost)

    return K, costs


def gap_scaling(K, A_inf, beta):
    """Simplified gap scaling: G_K = A_inf * (1 - e^{-beta*K})"""
    return A_inf * (1 - np.exp(-beta * K))


def experiment_gap_vs_K():
    """Test exponential saturation of adaptation gap with K."""

    # Cost matrices
    Q = np.diag([1.0, 0.1])
    R = np.array([[0.1]])

    # Task distribution: vary mass
    mass_mean, mass_std = 1.0, 0.25
    damping, stiffness = 0.5, 2.0
    n_tasks = 100

    masses = np.random.normal(mass_mean, mass_std, n_tasks)
    masses = np.clip(masses, 0.4, 1.8)

    x0 = np.array([[1.0], [0.0]])

    # Robust baseline: optimize for mean system
    A_mean, B_mean = get_system_matrices(mass_mean, damping, stiffness)
    K_robust, _ = solve_lqr(A_mean, B_mean, Q, R)

    print(f"Task distribution: mass ~ N({mass_mean}, {mass_std}^2)")
    print(f"Number of tasks: {n_tasks}")
    print(f"Robust controller gain: K_rob = {K_robust.flatten()}")

    costs_robust = []
    costs_optimal = []
    K_optimal_list = []

    for m in masses:
        A, B = get_system_matrices(m, damping, stiffness)
        K_opt, _ = solve_lqr(A, B, Q, R)
        K_optimal_list.append(K_opt)

        cost_rob = evaluate_cost(A, B, K_robust, Q, R, x0)
        cost_opt = evaluate_cost(A, B, K_opt, Q, R, x0)

        costs_robust.append(cost_rob)
        costs_optimal.append(cost_opt)

    costs_robust = np.array(costs_robust)
    costs_optimal = np.array(costs_optimal)

    gap_infinity = np.mean(costs_robust - costs_optimal)

    # Simulate adaptation for varying K
    K_steps = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30]
    gaps_by_K = {k: [] for k in K_steps}

    K_meta_init = K_robust.copy()
    lr = 0.08

    for i, m in enumerate(masses):
        A, B = get_system_matrices(m, damping, stiffness)

        for n_steps in K_steps:
            if n_steps == 0:
                K_adapted = K_meta_init.copy()
            else:
                K_adapted, _ = adapt_controller(K_meta_init, A, B, Q, R, n_steps, lr)

            cost_adapted = evaluate_cost(A, B, K_adapted, Q, R, x0)
            gap = costs_robust[i] - cost_adapted
            gaps_by_K[n_steps].append(gap)

    # Compute statistics
    K_array = np.array(K_steps)
    mean_gaps = np.array([np.mean(gaps_by_K[k]) for k in K_steps])
    std_gaps = np.array([np.std(gaps_by_K[k]) for k in K_steps])

    print("\nGap vs K:")
    for k, g, s in zip(K_steps, mean_gaps, std_gaps):
        print(f"  K={k:2d}: Gap = {g:.4f} +/- {s:.4f}")

    # Initial guess
    p0 = [mean_gaps[-1], 0.3]

    try:
        popt, pcov = curve_fit(gap_scaling, K_array, mean_gaps, p0=p0,
                               bounds=([0, 0], [10, 5]), maxfev=10000)
        A_inf_fit, beta_fit = popt

        # R^2 calculation
        gap_pred = gap_scaling(K_array, *popt)
        ss_res = np.sum((mean_gaps - gap_pred)**2)
        ss_tot = np.sum((mean_gaps - np.mean(mean_gaps))**2)
        r_squared = 1 - ss_res / ss_tot

        print(f"\nFitted scaling law parameters (simplified):")
        print(f"  A_inf = {A_inf_fit:.4f} (asymptotic gap)")
        print(f"  beta = {beta_fit:.4f} (adaptation rate)")
        print(f"  R^2 = {r_squared:.4f}")

        fit_success = True
    except Exception as e:
        print(f"Curve fitting failed: {e}")
        fit_success = False
        A_inf_fit, beta_fit, r_squared = 0, 0, 0

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel (a): Gap vs K with fit
    ax = axes[0]
    ax.errorbar(K_array, mean_gaps, yerr=std_gaps/np.sqrt(n_tasks),
                fmt='o', capsize=4, capthick=1.5, markersize=8,
                color='#2E86AB', label='Empirical', zorder=3)

    if fit_success:
        K_smooth = np.linspace(0, 35, 200)
        gap_fit = gap_scaling(K_smooth, A_inf_fit, beta_fit)
        ax.plot(K_smooth, gap_fit, '-', color='#E94F37', linewidth=2,
                label=f'Fit: $G_K = {A_inf_fit:.3f}(1-e^{{-{beta_fit:.2f}K}})$')
        ax.axhline(y=A_inf_fit, color='gray', linestyle='--', alpha=0.5,
                   label=f'$A_\\infty = {A_inf_fit:.3f}$')

    ax.set_xlabel('Adaptation Steps $K$')
    ax.set_ylabel('Adaptation Gap $G_K$')
    ax.set_title(f'(a) Exponential Saturation ($R^2 = {r_squared:.3f}$)')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim(-1, 35)

    # Panel (b): Log-scale residuals
    ax = axes[1]
    if fit_success:
        residual = A_inf_fit - mean_gaps
        residual_positive = np.maximum(residual, 1e-6)
        ax.semilogy(K_array, residual_positive, 'o-', color='#2E86AB',
                    markersize=8, linewidth=1.5, label='$A_\\infty - G_K$')

        # Theoretical exponential decay
        K_fit = np.linspace(0, 30, 100)
        decay_theory = A_inf_fit * np.exp(-beta_fit * K_fit)
        ax.semilogy(K_fit, decay_theory, '--', color='#E94F37', linewidth=2,
                    label=f'$\\propto e^{{-{beta_fit:.2f}K}}$')

    ax.set_xlabel('Adaptation Steps $K$')
    ax.set_ylabel('Residual Gap (log scale)')
    ax.set_title('(b) Exponential Convergence')
    ax.legend()
    ax.grid(alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('lqr_gap_vs_K_simplified.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nSaved: lqr_gap_vs_K_simplified.png")

    return {
        'A_inf': A_inf_fit,
        'beta': beta_fit,
        'r_squared': r_squared,
        'task_variance': mass_std**2
    }


def experiment_gap_vs_variance():
    """Test linear scaling of adaptation gap with task variance."""
    print("\n" + "=" * 60)
    print("Experiment 2: Adaptation Gap vs Task Variance sigma^2")
    print("=" * 60)

    Q = np.diag([1.0, 0.1])
    R = np.array([[0.1]])

    mass_mean = 1.0
    damping, stiffness = 0.5, 2.0
    n_tasks = 100
    x0 = np.array([[1.0], [0.0]])

    # Robust baseline
    A_mean, B_mean = get_system_matrices(mass_mean, damping, stiffness)
    K_robust, _ = solve_lqr(A_mean, B_mean, Q, R)

    # Variance levels to test
    std_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    variance_levels = [s**2 for s in std_levels]

    fixed_K = 10  # Fixed adaptation steps
    lr = 0.08

    gaps_at_convergence = []  # Gap when K -> infinity (optimal)
    gaps_at_K = []  # Gap after K steps

    print(f"\nFixed K = {fixed_K} adaptation steps")
    print(f"Testing {len(variance_levels)} variance levels\n")

    for std in std_levels:
        masses = np.random.normal(mass_mean, std, n_tasks)
        masses = np.clip(masses, 0.3, 2.0)

        gaps_inf = []
        gaps_k = []

        for m in masses:
            A, B = get_system_matrices(m, damping, stiffness)

            # Optimal controller for this task
            K_opt, _ = solve_lqr(A, B, Q, R)

            # Costs
            cost_robust = evaluate_cost(A, B, K_robust, Q, R, x0)
            cost_optimal = evaluate_cost(A, B, K_opt, Q, R, x0)

            # Adapted controller after K steps
            K_adapted, _ = adapt_controller(K_robust, A, B, Q, R, fixed_K, lr)
            cost_adapted = evaluate_cost(A, B, K_adapted, Q, R, x0)

            gaps_inf.append(cost_robust - cost_optimal)
            gaps_k.append(cost_robust - cost_adapted)

        gaps_at_convergence.append(np.mean(gaps_inf))
        gaps_at_K.append(np.mean(gaps_k))

        print(f"  sigma^2 = {std**2:.4f}: G_inf = {np.mean(gaps_inf):.4f}, G_K = {np.mean(gaps_k):.4f}")

    variance_array = np.array(variance_levels)
    gaps_inf_array = np.array(gaps_at_convergence)
    gaps_k_array = np.array(gaps_at_K)

    # Linear fit for asymptotic gap
    slope_inf, intercept_inf = np.polyfit(variance_array, gaps_inf_array, 1)
    r2_inf = np.corrcoef(variance_array, gaps_inf_array)[0, 1]**2

    # Linear fit for gap at K
    slope_k, intercept_k = np.polyfit(variance_array, gaps_k_array, 1)
    r2_k = np.corrcoef(variance_array, gaps_k_array)[0, 1]**2

    print(f"\nLinear fit (G_inf vs sigma^2):")
    print(f"  slope = {slope_inf:.4f}, intercept = {intercept_inf:.4f}")
    print(f"  R^2 = {r2_inf:.4f}")

    print(f"\nLinear fit (G_K vs sigma^2):")
    print(f"  slope = {slope_k:.4f}, intercept = {intercept_k:.4f}")
    print(f"  R^2 = {r2_k:.4f}")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel (a): Gap vs variance
    ax = axes[0]
    ax.scatter(variance_array, gaps_inf_array, s=80, color='#2E86AB',
               label=f'$G_\\infty$ (optimal)', zorder=3, marker='o')
    ax.scatter(variance_array, gaps_k_array, s=80, color='#E94F37',
               label=f'$G_{{K={fixed_K}}}$ (adapted)', zorder=3, marker='s')

    var_fit = np.linspace(0, 0.18, 100)
    ax.plot(var_fit, slope_inf * var_fit + intercept_inf, '--',
            color='#2E86AB', linewidth=2, alpha=0.7)
    ax.plot(var_fit, slope_k * var_fit + intercept_k, '--',
            color='#E94F37', linewidth=2, alpha=0.7)

    ax.set_xlabel('Task Variance $\\sigma^2_S$')
    ax.set_ylabel('Adaptation Gap')
    ax.set_title(f'(a) Linear Scaling with Variance')
    ax.legend()
    ax.grid(alpha=0.3)

    # Add R^2 annotations
    ax.text(0.05, 0.95, f'$G_\\infty$: $R^2 = {r2_inf:.3f}$',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            color='#2E86AB')
    ax.text(0.05, 0.87, f'$G_K$: $R^2 = {r2_k:.3f}$',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            color='#E94F37')

    # Panel (b): 2D heatmap of G_K vs (K, sigma^2)
    ax = axes[1]

    # Generate heatmap data
    K_range = [1, 3, 5, 7, 10, 15, 20]
    std_range = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]

    gap_matrix = np.zeros((len(std_range), len(K_range)))

    for i, std in enumerate(std_range):
        masses = np.random.normal(mass_mean, std, 50)
        masses = np.clip(masses, 0.3, 2.0)

        for j, K in enumerate(K_range):
            gaps = []
            for m in masses:
                A, B = get_system_matrices(m, damping, stiffness)
                cost_robust = evaluate_cost(A, B, K_robust, Q, R, x0)
                K_adapted, _ = adapt_controller(K_robust, A, B, Q, R, K, lr)
                cost_adapted = evaluate_cost(A, B, K_adapted, Q, R, x0)
                gaps.append(cost_robust - cost_adapted)
            gap_matrix[i, j] = np.mean(gaps)

    im = ax.imshow(gap_matrix, aspect='auto', origin='lower', cmap='viridis')
    ax.set_xticks(range(len(K_range)))
    ax.set_xticklabels(K_range)
    ax.set_yticks(range(len(std_range)))
    ax.set_yticklabels([f'{s**2:.2f}' for s in std_range])
    ax.set_xlabel('Adaptation Steps $K$')
    ax.set_ylabel('Task Variance $\\sigma^2_S$')
    ax.set_title('(b) Combined Scaling: $G_K(K, \\sigma^2_S)$')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Adaptation Gap $G_K$')

    plt.tight_layout()
    plt.savefig('lqr_gap_vs_variance_simplified.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nSaved: lqr_gap_vs_variance_simplified.png")

    return {
        'slope_inf': slope_inf,
        'slope_k': slope_k,
        'r2_inf': r2_inf,
        'r2_k': r2_k
    }


def experiment_trajectory_comparison():
    """Visualize state trajectories for robust vs adapted controllers."""
    print("\n" + "=" * 60)
    print("Experiment 3: Trajectory Comparison")
    print("=" * 60)

    from scipy.integrate import solve_ivp

    Q = np.diag([1.0, 0.1])
    R = np.array([[0.1]])

    mass_mean = 1.0
    damping, stiffness = 0.5, 2.0

    # Robust controller (for mean mass)
    A_mean, B_mean = get_system_matrices(mass_mean, damping, stiffness)
    K_robust, _ = solve_lqr(A_mean, B_mean, Q, R)

    mass_test = 1.4  # 40% heavier than nominal
    A_test, B_test = get_system_matrices(mass_test, damping, stiffness)

    # Optimal controller for test task
    K_optimal, _ = solve_lqr(A_test, B_test, Q, R)

    # Adapted controller (5 gradient steps)
    K_adapted, _ = adapt_controller(K_robust, A_test, B_test, Q, R, 5, lr=0.1)

    print(f"Test mass: {mass_test} (nominal: {mass_mean})")
    print(f"K_robust: {K_robust.flatten()}")
    print(f"K_optimal: {K_optimal.flatten()}")
    print(f"K_adapted: {K_adapted.flatten()}")

    # Simulate trajectories
    x0 = np.array([1.0, 0.0])
    t_span = (0, 5)
    t_eval = np.linspace(0, 5, 200)

    def dynamics(t, x, A, B, K):
        u = -K @ x.reshape(-1, 1)
        dxdt = A @ x.reshape(-1, 1) + B @ u
        return dxdt.flatten()

    sol_robust = solve_ivp(lambda t, x: dynamics(t, x, A_test, B_test, K_robust),
                           t_span, x0, t_eval=t_eval)
    sol_optimal = solve_ivp(lambda t, x: dynamics(t, x, A_test, B_test, K_optimal),
                            t_span, x0, t_eval=t_eval)
    sol_adapted = solve_ivp(lambda t, x: dynamics(t, x, A_test, B_test, K_adapted),
                            t_span, x0, t_eval=t_eval)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # Panel (a): Position
    ax = axes[0]
    ax.plot(sol_robust.t, sol_robust.y[0], '-', color='#666666',
            linewidth=2, label='Robust')
    ax.plot(sol_adapted.t, sol_adapted.y[0], '-', color='#E94F37',
            linewidth=2, label='Adapted (K=5)')
    ax.plot(sol_optimal.t, sol_optimal.y[0], '--', color='#2E86AB',
            linewidth=2, label='Optimal')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position $x$')
    ax.set_title(f'(a) Position (mass = {mass_test})')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # Panel (b): Velocity
    ax = axes[1]
    ax.plot(sol_robust.t, sol_robust.y[1], '-', color='#666666', linewidth=2)
    ax.plot(sol_adapted.t, sol_adapted.y[1], '-', color='#E94F37', linewidth=2)
    ax.plot(sol_optimal.t, sol_optimal.y[1], '--', color='#2E86AB', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity $\\dot{x}$')
    ax.set_title('(b) Velocity')
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # Panel (c): Control effort
    ax = axes[2]
    u_robust = -K_robust @ sol_robust.y
    u_adapted = -K_adapted @ sol_adapted.y
    u_optimal = -K_optimal @ sol_optimal.y

    ax.plot(sol_robust.t, u_robust.flatten(), '-', color='#666666',
            linewidth=2, label='Robust')
    ax.plot(sol_adapted.t, u_adapted.flatten(), '-', color='#E94F37',
            linewidth=2, label='Adapted')
    ax.plot(sol_optimal.t, u_optimal.flatten(), '--', color='#2E86AB',
            linewidth=2, label='Optimal')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control $u$')
    ax.set_title('(c) Control Effort')
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig('lqr_trajectories_simplified.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Compute costs
    x0_col = np.array([[1.0], [0.0]])
    cost_robust = evaluate_cost(A_test, B_test, K_robust, Q, R, x0_col)
    cost_adapted = evaluate_cost(A_test, B_test, K_adapted, Q, R, x0_col)
    cost_optimal = evaluate_cost(A_test, B_test, K_optimal, Q, R, x0_col)

    print(f"\nCosts on test task (mass = {mass_test}):")
    print(f"  Robust:  {cost_robust:.4f}")
    print(f"  Adapted: {cost_adapted:.4f} (gap recovered: {(cost_robust-cost_adapted)/(cost_robust-cost_optimal)*100:.1f}%)")
    print(f"  Optimal: {cost_optimal:.4f}")

    print("\nSaved: lqr_trajectories_simplified.png")


def create_summary_figure(results_K, results_var):
    """Create a combined 2x2 summary figure for the paper."""
    print("\n" + "=" * 60)
    print("Creating Summary Figure (Simplified Model)")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Re-run experiments with data collection for plotting
    Q = np.diag([1.0, 0.1])
    R = np.array([[0.1]])
    mass_mean = 1.0
    damping, stiffness = 0.5, 2.0
    n_tasks = 100
    x0 = np.array([[1.0], [0.0]])

    A_mean, B_mean = get_system_matrices(mass_mean, damping, stiffness)
    K_robust, _ = solve_lqr(A_mean, B_mean, Q, R)

    # Panel (a): Gap vs K
    ax = axes[0, 0]
    mass_std = 0.25
    masses = np.random.normal(mass_mean, mass_std, n_tasks)
    masses = np.clip(masses, 0.4, 1.8)

    K_steps = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30]
    lr = 0.08

    costs_robust = []
    for m in masses:
        A, B = get_system_matrices(m, damping, stiffness)
        costs_robust.append(evaluate_cost(A, B, K_robust, Q, R, x0))
    costs_robust = np.array(costs_robust)

    mean_gaps = []
    std_gaps = []
    for K in K_steps:
        gaps = []
        for i, m in enumerate(masses):
            A, B = get_system_matrices(m, damping, stiffness)
            if K == 0:
                K_adapted = K_robust.copy()
            else:
                K_adapted, _ = adapt_controller(K_robust, A, B, Q, R, K, lr)
            cost_adapted = evaluate_cost(A, B, K_adapted, Q, R, x0)
            gaps.append(costs_robust[i] - cost_adapted)
        mean_gaps.append(np.mean(gaps))
        std_gaps.append(np.std(gaps))

    K_array = np.array(K_steps)
    mean_gaps = np.array(mean_gaps)
    std_gaps = np.array(std_gaps)

    # Simplified fit
    popt, _ = curve_fit(gap_scaling, K_array, mean_gaps,
                        p0=[mean_gaps[-1], 0.3],
                        bounds=([0, 0], [10, 5]))
    A_inf, beta = popt

    gap_pred = gap_scaling(K_array, *popt)
    r2 = 1 - np.sum((mean_gaps - gap_pred)**2) / np.sum((mean_gaps - np.mean(mean_gaps))**2)

    ax.errorbar(K_array, mean_gaps, yerr=std_gaps/np.sqrt(n_tasks),
                fmt='o', capsize=3, markersize=7, color='#2E86AB', zorder=3)
    K_smooth = np.linspace(0, 35, 200)
    ax.plot(K_smooth, gap_scaling(K_smooth, *popt), '-', color='#E94F37', linewidth=2)
    ax.axhline(y=A_inf, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Adaptation Steps $K$')
    ax.set_ylabel('Adaptation Gap $G_K$')
    ax.set_title(f'(a) Gap vs $K$: $R^2 = {r2:.3f}$')
    ax.text(0.95, 0.15, f'$G_K = {A_inf:.2f}(1-e^{{-{beta:.2f}K}})$',
            transform=ax.transAxes, ha='right', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.grid(alpha=0.3)
    ax.set_xlim(-1, 35)

    # Panel (b): Log residuals
    ax = axes[0, 1]
    residual = np.maximum(A_inf - mean_gaps, 1e-6)
    ax.semilogy(K_array, residual, 'o-', color='#2E86AB', markersize=7)
    K_fit = np.linspace(0, 30, 100)
    ax.semilogy(K_fit, A_inf * np.exp(-beta * K_fit), '--',
                color='#E94F37', linewidth=2)
    ax.set_xlabel('Adaptation Steps $K$')
    ax.set_ylabel('Residual $A_\\infty - G_K$')
    ax.set_title('(b) Exponential Convergence')
    ax.grid(alpha=0.3, which='both')

    # Panel (c): Gap vs variance
    ax = axes[1, 0]
    std_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    gaps_inf = []
    gaps_k = []
    fixed_K = 10

    for std in std_levels:
        masses = np.random.normal(mass_mean, std, n_tasks)
        masses = np.clip(masses, 0.3, 2.0)
        g_inf, g_k = [], []
        for m in masses:
            A, B = get_system_matrices(m, damping, stiffness)
            K_opt, _ = solve_lqr(A, B, Q, R)
            cost_rob = evaluate_cost(A, B, K_robust, Q, R, x0)
            cost_opt = evaluate_cost(A, B, K_opt, Q, R, x0)
            K_ad, _ = adapt_controller(K_robust, A, B, Q, R, fixed_K, lr)
            cost_ad = evaluate_cost(A, B, K_ad, Q, R, x0)
            g_inf.append(cost_rob - cost_opt)
            g_k.append(cost_rob - cost_ad)
        gaps_inf.append(np.mean(g_inf))
        gaps_k.append(np.mean(g_k))

    var_arr = np.array([s**2 for s in std_levels])
    gaps_inf = np.array(gaps_inf)
    gaps_k = np.array(gaps_k)

    slope, intercept = np.polyfit(var_arr, gaps_inf, 1)
    r2_var = np.corrcoef(var_arr, gaps_inf)[0, 1]**2

    ax.scatter(var_arr, gaps_inf, s=70, color='#2E86AB', label='$G_\\infty$', zorder=3)
    ax.scatter(var_arr, gaps_k, s=70, color='#E94F37', marker='s',
               label=f'$G_{{K={fixed_K}}}$', zorder=3)
    var_fit = np.linspace(0, 0.18, 100)
    ax.plot(var_fit, slope * var_fit + intercept, '--', color='#2E86AB', alpha=0.7)
    ax.set_xlabel('Task Variance $\\sigma^2_S$')
    ax.set_ylabel('Adaptation Gap')
    ax.set_title(f'(c) Linear Scaling: $R^2 = {r2_var:.3f}$')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    # Panel (d): Heatmap
    ax = axes[1, 1]
    K_range = [1, 3, 5, 7, 10, 15, 20]
    std_range = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    gap_matrix = np.zeros((len(std_range), len(K_range)))

    for i, std in enumerate(std_range):
        masses = np.random.normal(mass_mean, std, 50)
        masses = np.clip(masses, 0.3, 2.0)
        for j, K in enumerate(K_range):
            gaps = []
            for m in masses:
                A, B = get_system_matrices(m, damping, stiffness)
                cost_rob = evaluate_cost(A, B, K_robust, Q, R, x0)
                K_ad, _ = adapt_controller(K_robust, A, B, Q, R, K, lr)
                cost_ad = evaluate_cost(A, B, K_ad, Q, R, x0)
                gaps.append(cost_rob - cost_ad)
            gap_matrix[i, j] = np.mean(gaps)

    im = ax.imshow(gap_matrix, aspect='auto', origin='lower', cmap='viridis')
    ax.set_xticks(range(len(K_range)))
    ax.set_xticklabels(K_range)
    ax.set_yticks(range(len(std_range)))
    ax.set_yticklabels([f'{s**2:.2f}' for s in std_range])
    ax.set_xlabel('Adaptation Steps $K$')
    ax.set_ylabel('Task Variance $\\sigma^2_S$')
    ax.set_title('(d) Combined Scaling')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('$G_K$')

    plt.tight_layout()
    plt.savefig('lqr_summary_simplified.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved: lqr_summary_simplified.png")


if __name__ == "__main__":
    print("=" * 70)
    print("LQR META-LEARNING VALIDATION (SIMPLIFIED MODEL)")
    print("Testing: Gap(K) = A_inf * (1 - e^{-beta*K})")
    print("=" * 70)

    results_K = experiment_gap_vs_K()
    results_var = experiment_gap_vs_variance()
    experiment_trajectory_comparison()
    create_summary_figure(results_K, results_var)
