#!/usr/bin/env python3
"""
Generate plot showing Coulomb's law: Force vs Charge product (constant radius).
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
fig, ax = plt.subplots(figsize=(8, 6))

# Constants
k = 8.99e9  # Coulomb's constant in N⋅m²/C²
r = 1.0  # Constant distance in meters

# Charge product range (q1 * q2)
q_product = np.linspace(-2e-18, 2e-18, 1000)  # C²

# Calculate force: F = k * (q1*q2) / r²
F = k * q_product / (r**2)

# Plot
ax.plot(q_product * 1e18, F * 1e9, 'b-', linewidth=2, label='Attractive (opposite charges)')
ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axvline(x=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)

# Fill areas for attractive and repulsive
ax.fill_between(q_product[q_product < 0] * 1e18, 0, F[q_product < 0] * 1e9, 
                alpha=0.3, color='green', label='Attractive force')
ax.fill_between(q_product[q_product > 0] * 1e18, 0, F[q_product > 0] * 1e9, 
                alpha=0.3, color='red', label='Repulsive force')

# Labels and title
ax.set_xlabel('Charge Product $q_1 q_2$ ($\\times 10^{-18}$ C²)', fontsize=12)
ax.set_ylabel('Force $F$ ($\\times 10^{-9}$ N)', fontsize=12)
ax.set_title('Coulomb\'s Law: Force vs Charge Product\n(Constant Distance $r = 1$ m)', 
             fontsize=14, pad=15)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.legend(loc='best', fontsize=10)

# Add equation annotation
ax.text(0.05, 0.95, r'$F = k \frac{q_1 q_2}{r^2}$', 
        transform=ax.transAxes, fontsize=14,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Save figure
output_path = '../images/chapter03/coulomb_force_vs_charge.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")
plt.close()

