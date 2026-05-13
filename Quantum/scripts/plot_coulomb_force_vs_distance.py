#!/usr/bin/env python3
"""
Generate plot showing Coulomb's law: Force vs Distance (constant charge product).
Publication-quality figure for Chapter 3.1.1
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

# Set publication-quality style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times', 'DejaVu Serif'],
    'text.usetex': False,
    'axes.linewidth': 1.2,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Constants
k = 8.99e9  # Coulomb's constant in N⋅m²/C²
q1q2 = 1e-18  # Constant charge product in C²

# Distance range
r = np.linspace(0.1, 5.0, 1000)  # meters

# Calculate force: F = k * (q1*q2) / r²
F = k * q1q2 / (r**2)

# ========== Left plot: Linear scale ==========
ax1.plot(r, F * 1e9, 'b-', linewidth=2, label='$F(r)$')
ax1.set_xlabel('Distance $r$ (m)', fontsize=12)
ax1.set_ylabel('Force $F$ ($\\times 10^{-9}$ N)', fontsize=12)
ax1.set_title('Coulomb\'s Law: Force vs Distance\n(Linear Scale)', 
             fontsize=13, pad=10)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax1.legend(loc='best', fontsize=10)

# Add 1/r² reference line
r_ref = np.linspace(0.1, 5.0, 100)
F_ref = 1.0 / (r_ref**2)
F_ref_normalized = F_ref / F_ref[0] * F[0] * 1e9
ax1.plot(r_ref, F_ref_normalized, 'r--', linewidth=1.5, alpha=0.7, 
        label='$\\propto 1/r^2$')
ax1.legend(loc='best', fontsize=10)

# Add equation
ax1.text(0.05, 0.95, r'$F = k \frac{q_1 q_2}{r^2}$', 
        transform=ax1.transAxes, fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# ========== Right plot: Log-log scale ==========
ax2.loglog(r, F * 1e9, 'b-', linewidth=2, label='$F(r)$')
ax2.set_xlabel('Distance $r$ (m)', fontsize=12)
ax2.set_ylabel('Force $F$ ($\\times 10^{-9}$ N)', fontsize=12)
ax2.set_title('Coulomb\'s Law: Force vs Distance\n(Log-Log Scale)', 
             fontsize=13, pad=10)
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')

# Add reference line with slope -2
r_log = np.logspace(np.log10(0.1), np.log10(5.0), 100)
F_log = 1.0 / (r_log**2)
F_log_normalized = F_log / F_log[0] * F[0] * 1e9
ax2.loglog(r_log, F_log_normalized, 'r--', linewidth=1.5, alpha=0.7, 
          label='Slope = -2')
ax2.legend(loc='best', fontsize=10)

# Add annotation about inverse square law
ax2.text(0.05, 0.95, 'Inverse Square Law:\n$F \\propto 1/r^2$', 
        transform=ax2.transAxes, fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()

# Save figure
output_path = '../images/chapter03/coulomb_force_vs_distance.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")
plt.close()

