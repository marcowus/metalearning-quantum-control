"""
Make Figure 1  
© 2025 The MITRE Corporation, All Rights Reserved 
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'font.size': 18,
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'legend.fontsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

## Figure b 

ax_b = axes[0]
ax_b.set_title('(a)', fontsize=22, fontweight='bold')

# Create synthetic loss landscape
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = 0.5 * (X**2 + Y**2) + 0.2 * np.sin(2*X) * np.cos(2*Y)

# Plot contours
ax_b.contourf(X, Y, Z, levels=15, cmap='Blues_r', alpha=0.4)
ax_b.contour(X, Y, Z, levels=10, colors='steelblue', alpha=0.4, linewidths=0.6)

# Robust optimum (at center)
robust_pos = (0.0, 0.0)
ax_b.plot(*robust_pos, 'k*', markersize=20, markeredgecolor='black', 
          markerfacecolor='gold', markeredgewidth=1.5, zorder=10)

# Task-specific optima  
task_optima = [(-1.0, 0.8), (1.1, 1.0)]
task_colors = ['#e74c3c', '#27ae60']
task_labels = [r'$\xi_1$', r'$\xi_2$']

for i, (pos, color, label) in enumerate(zip(task_optima, task_colors, task_labels)):
    ax_b.plot(*pos, 'o', markersize=14, markerfacecolor=color, 
              markeredgecolor='white', markeredgewidth=2, zorder=10)
    ax_b.annotate('', xy=pos, xytext=robust_pos,
                  arrowprops=dict(arrowstyle='-|>', color=color, lw=2.5, 
                                mutation_scale=20,
                                connectionstyle='arc3,rad=0.15'))

ax_b.set_xlabel(r'$\theta_1$')
ax_b.set_ylabel(r'$\theta_2$')
ax_b.set_xlim(-1.8, 1.8)
ax_b.set_ylim(-1.8, 1.8)
ax_b.set_xticks([-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5])
ax_b.set_yticks([-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5])
ax_b.set_aspect('equal')

# Legend - positioned in clear area
legend_elements = [
    Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
           markeredgecolor='black', markersize=14, label=r'Robust $\theta^*_{\mathrm{rob}}$'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
           markeredgecolor='white', markersize=10, label=r'Task $\xi_1$'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#27ae60',
           markeredgecolor='white', markersize=10, label=r'Task $\xi_2$'),
]
ax_b.legend(handles=legend_elements, loc='upper left', fontsize=16,
            framealpha=0.95, edgecolor='gray')

# Add small inset for pulse sequences - positioned to avoid arrow overlap
inset_ax = ax_b.inset_axes([0.58, 0.08, 0.40, 0.32])
t = np.linspace(0, 1, 100)

# Different pulses for 2 tasks
pulses = [
    0.8 * np.sin(2 * np.pi * t) * np.exp(-t),
    0.6 * np.sin(3 * np.pi * t + 0.5) * np.exp(-0.5*t),
]

for pulse, color in zip(pulses, task_colors):
    inset_ax.plot(t, pulse, color=color, lw=2, alpha=0.9)

inset_ax.set_xlabel(r'$t$', fontsize=15, labelpad=-1)
inset_ax.set_ylabel(r'$u(t)$', fontsize=15)
# Title placed inside to avoid arrow overlap
inset_ax.text(0.5, 0.95, 'Adapted Pulses', fontsize=14, fontweight='bold',
              ha='center', va='top', transform=inset_ax.transAxes)
inset_ax.set_xlim(0, 1)
inset_ax.set_ylim(-0.6, 0.9)
inset_ax.set_xticks([])
inset_ax.set_yticks([])
inset_ax.set_facecolor('white')
for spine in inset_ax.spines.values():
    spine.set_edgecolor('gray')
    spine.set_linewidth(1)


ax_c = axes[1]
ax_c.set_title('(b)', fontsize=22, fontweight='bold')

# Analytical curve
K = np.linspace(0, 20, 200)
A_inf = 1.0
beta = 0.25
G_K = A_inf * (1 - np.exp(-beta * K))

# Main curve
ax_c.plot(K, G_K, 'b-', lw=3)

# Asymptote
ax_c.axhline(A_inf, color='dimgray', linestyle='--', lw=2)
ax_c.text(1, A_inf + 0.06, r'$A_\infty \propto \sigma_\tau^2$', fontsize=18,
          color='dimgray', va='bottom')

# Shade diminishing returns region
ax_c.fill_between(K, G_K, A_inf, where=(K > 10), alpha=0.15, color='gray')
ax_c.text(17, 0.92, 'Diminishing\nreturns', fontsize=16, ha='center',
          style='italic', color='dimgray', va='top')

# Shade early adaptation region
ax_c.fill_between(K, 0, G_K, where=(K < 5), alpha=0.12, color='green')
ax_c.text(2.5, 0.08, 'Rapid\ngains', fontsize=16, ha='center',
          style='italic', color='darkgreen', va='bottom')

# Rate annotation - positioned clearly
ax_c.annotate(r'Rate $\beta = \eta \mu$',
              xy=(3, G_K[30]), xytext=(7, 0.25),
              fontsize=18, color='blue',
              arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

# K* annotation
K_star = 10
G_star = A_inf * (1 - np.exp(-beta * K_star))
ax_c.plot([K_star, K_star], [0, G_star], 'g:', lw=2)
ax_c.plot(K_star, G_star, 'go', markersize=10, markeredgecolor='white', markeredgewidth=2)
ax_c.annotate(r'$K^*$', xy=(K_star, 0.08), fontsize=19, color='darkgreen',
              ha='center', va='bottom',
              bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))

ax_c.set_xlabel('Adaptation Steps $K$', fontsize=20)
ax_c.set_ylabel('Adaptation Gap $G_K$', fontsize=20)
ax_c.set_xlim(0, 20)
ax_c.set_ylim(-0.05, 1.2)
# Equation label positioned below the curve
ax_c.text(12, 0.42, r'$G_K = A_\infty(1 - e^{-\beta K})$', fontsize=18,
          color='blue', ha='left', va='bottom')
ax_c.grid(True, alpha=0.3)

# Clean up
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)


plt.tight_layout()
plt.savefig('fig1.png', dpi=150, bbox_inches='tight')
plt.savefig('fig1.pdf', bbox_inches='tight')  
plt.show()
