""" 
Make Figure 2     
© 2025 The MITRE Corporation, All Rights Reserved 

"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm, norm
import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def build_lindbladian(H, gamma_deph, gamma_relax):
    d = 2
    I = np.eye(d, dtype=np.complex128)
    L = -1j * (np.kron(I, H) - np.kron(H.T, I))
    
    for Lop, g in [(SIGMA_Z, gamma_deph),
                   (np.array([[0, 1], [0, 0]], dtype=np.complex128), gamma_relax)]:
        Ldag_L = Lop.conj().T @ Lop
        L += g * (np.kron(Lop.conj(), Lop) - 0.5*np.kron(I, Ldag_L) - 0.5*np.kron(Ldag_L.T, I))
    return L


def simulate(theta, gamma_deph, gamma_relax):
    n_seg = len(theta) // 2
    rho_vec = np.array([1, 0, 0, 0], dtype=np.complex128)
    dt = 1.0 / n_seg
    for seg in range(n_seg):
        H = 0.5*SIGMA_Z + np.clip(theta[2*seg], -2, 2)*SIGMA_X + np.clip(theta[2*seg+1], -2, 2)*SIGMA_Y
        rho_vec = expm(build_lindbladian(H, gamma_deph, gamma_relax) * dt) @ rho_vec
    return rho_vec.reshape((2, 2), order='F')


def loss_fn(theta, gd, gr):
    rho = simulate(theta, gd, gr)
    return 1.0 - np.real(np.trace(rho @ np.array([[.5, .5], [.5, .5]])))


def grad_fn(theta, gd, gr, eps=1e-6):
    g = np.zeros_like(theta)
    for i in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += eps
        tm[i] -= eps
        g[i] = (loss_fn(tp, gd, gr) - loss_fn(tm, gd, gr)) / (2*eps)
    return g


def find_optimal(gd, gr, n_seg=4):
    best = np.inf
    for _ in range(6):
        res = minimize(lambda t: loss_fn(t, gd, gr), np.random.randn(2*n_seg)*0.5,
                       method='L-BFGS-B', bounds=[(-2, 2)]*(2*n_seg), options={'maxiter': 200})
        if res.fun < best:
            best = res.fun
            best_theta = res.x
    return best_theta, best


# Panel (a): PL Condition


def generate_pl_data():
    """Generate PL condition validation data."""
    tasks = [
        ## gamma_1 , gamma_2 
        (0.01, 0.005, 'b'), 
        (0.05, 0.01, 'r'),  
    ]
    
    all_subopt = []
    all_gradsq = []
    all_colors = []
    
    ##gamma 1, gamma 2, color 
    for gd, gr, color in tasks:
        _, L_star = find_optimal(gd, gr)
        
        for _ in range(5):
            theta = np.random.randn(20) * 0.9    
            for _ in range(60):
                L = loss_fn(theta, gd, gr)
                g = grad_fn(theta, gd, gr)
                sub = L - L_star
                gns = 0.5 * np.sum(g**2)
                if sub > 1e-8:
                    all_subopt.append(sub)
                    all_gradsq.append(gns)
                    all_colors.append(color)
                theta = np.clip(theta - 0.1*g, -2, 2)
    
    return np.array(all_subopt), np.array(all_gradsq), all_colors


# Panel (b): Lipschitz Continuity


def generate_lipschitz_data():
    """Generate Lipschitz validation data."""
    gamma_ref = (0.05, 0.02)
    H = 0.5 * SIGMA_Z
    
    task_dists = []
    lind_dists = []
    
    for _ in range(40):
        gd = 0.01 + 0.12 * np.random.rand()
        gr = 0.005 + 0.05 * np.random.rand()
        
        t_dist = np.abs(gd - gamma_ref[0]) + np.abs(gr - gamma_ref[1])
        
        L1 = build_lindbladian(H, gamma_ref[0], gamma_ref[1])
        L2 = build_lindbladian(H, gd, gr)
        l_dist = norm(L1 - L2, ord='fro')
        
        task_dists.append(t_dist)
        lind_dists.append(l_dist)
    
    return np.array(task_dists), np.array(lind_dists)



# Panel (c): Control Separation 
def generate_control_separation_data():
    """Generate control separation validation data."""
    gamma_ref = (0.05, 0.02)
    
    best_loss = np.inf
    for _ in range(10):
        res = minimize(lambda t: loss_fn(t, gamma_ref[0], gamma_ref[1]), 
                       np.random.randn(8)*0.5, method='L-BFGS-B', 
                       bounds=[(-2, 2)]*8, options={'maxiter': 500})
        if res.fun < best_loss:
            best_loss = res.fun
            theta_star = res.x
    
    delta_gammas = []
    delta_thetas = []
    
    # Systematic perturbations along one direction (cleaner signal)
    epsilons = np.linspace(0.005, 0.04, 15)
    
    for eps in epsilons:
        # Perturb dephasing only (cleaner than random)
        gd_new = gamma_ref[0] + eps
        gr_new = gamma_ref[1]
        
        # Careful optimization for perturbed task
        best_loss_new = np.inf
        for _ in range(8):
            res = minimize(lambda t: loss_fn(t, gd_new, gr_new),
                           theta_star + np.random.randn(8)*0.1,  # warm start
                           method='L-BFGS-B', bounds=[(-2, 2)]*8, 
                           options={'maxiter': 500})
            if res.fun < best_loss_new:
                best_loss_new = res.fun
                theta_new = res.x
        
        delta_gamma = eps
        delta_theta = np.linalg.norm(theta_new - theta_star)
        
        delta_gammas.append(delta_gamma)
        delta_thetas.append(delta_theta)
    
    return np.array(delta_gammas), np.array(delta_thetas)



# Generate All Data

print("Generating PL data...")
np.random.seed(42)
pl_subopt, pl_gradsq, pl_colors = generate_pl_data()

print("Generating Lipschitz data...")
np.random.seed(43)
lip_task, lip_lind = generate_lipschitz_data()

print("Generating control separation data...")
np.random.seed(44)
cs_gamma, cs_theta = generate_control_separation_data()



from matplotlib.patches import Patch
from matplotlib.lines import Line2D

fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
#  Panel (a): PL Condition 
ax = axes[0]
# PL bound 
mu = np.min(pl_gradsq / pl_subopt)  

# Define adaptation region boundary
adaptation_boundary = 0.15

# Create legend elements for both scatter and line
legend_elements = [
    Patch(facecolor='b', label='Task 1'),
    Patch(facecolor='r', label='Task 2'),
    Line2D([0], [0], color='k', linestyle='--', lw=1.5, label=rf'$\mu = {mu:.2f}$')
]

x_max = 0.4 
x_line = np.linspace(0, x_max, 100)

# Add shaded region for adaptation zone
ax.axvspan(0, adaptation_boundary, alpha=0.15, color='green', zorder=0)
ax.axvline(adaptation_boundary, color='green', linestyle=':', lw=1.5, alpha=0.7)

# Add annotation
ax.annotate('Adaptation\nregion', 
            xy=(adaptation_boundary/2, 0.042), 
            ha='center', va='top',
            fontsize=8, color='darkgreen', style='italic')

ax.scatter(pl_subopt, pl_gradsq, c=pl_colors, s=15, alpha=0.5, edgecolors='none')
ax.plot(x_line, mu * x_line, 'k--', lw=1.5)
ax.set_xlabel(r'$L(\theta) - L^*$')
ax.set_ylabel(r'$\frac{1}{2}\|\nabla L \|^2$')
ax.set_title('(a) PL Condition')
ax.set_xlim(0, x_max)
ax.set_ylim(0, None)
ax.grid(True, alpha=0.3)
ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)


# Panel (b): Lipschitz  
ax = axes[1]

ax.scatter(lip_task, lip_lind, s=15, alpha=0.6, c='steelblue', edgecolors='k', linewidths=0.3)

# Bound line
C_L = np.max(lip_lind / lip_task)
x_max = lip_task.max() * 1 
x_line = np.linspace(0, x_max, 100)
ax.plot(x_line, C_L * x_line, 'r--', lw=1.5, label=rf'$C_L = {C_L:.1f}$')

ax.set_xlabel(r'$\|\xi - \xi^\prime \|$')
ax.set_ylabel(r'$\|f_\xi - f_{\xi^\prime}\|$')
ax.set_title('(b) Lipschitz Continuity')
ax.legend(loc='upper left', framealpha=0.9)
ax.set_xlim(0, x_max)
ax.set_ylim(0, None)
ax.grid(True, alpha=0.3)

# Panel (c): Control Separation 
ax = axes[2]

ax.scatter(cs_gamma, cs_theta, s=30, alpha=0.7, c='#e74c3c', edgecolors='k', linewidths=0.3)

# Linear fit
slope = np.sum(cs_gamma * cs_theta) / np.sum(cs_gamma**2)
r2 = 1 - np.sum((cs_theta - slope*cs_gamma)**2) / np.sum((cs_theta - cs_theta.mean())**2)
x_max = cs_gamma.max() * 1.1  
x_line = np.linspace(0, x_max, 100)
ax.plot(x_line, slope * x_line, 'k--', lw=1.5, label=rf'$R^2 = {r2:.2f}$')

ax.set_xlabel(r'$\|\xi - \xi^\prime \|$')
ax.set_ylabel(r'$\|\theta^*_\xi - \theta^*_{\xi^\prime}\|$')
ax.set_title('(c) Control Separation')
ax.legend(loc='upper left', framealpha=0.9)
ax.set_xlim(0, x_max)
ax.set_ylim(0, None)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assumptions_validation_composite.png', dpi=200, bbox_inches='tight')
plt.savefig('assumptions_validation_composite.pdf', bbox_inches='tight')
print("\nSaved: assumptions_validation_composite.png/pdf")
plt.close()
 
