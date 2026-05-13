#!/usr/bin/env python3
"""
Generate visualization of Coulomb's law - electric field around a point charge.
Publication-quality figure for Chapter 3.1.1
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

# Set publication-quality style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times', 'DejaVu Serif'],
    'text.usetex': False,  # Set to True if LaTeX is available
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
fig, ax = plt.subplots(figsize=(8, 8))

# Grid for field calculation
x = np.linspace(-3, 3, 30)
y = np.linspace(-3, 3, 30)
X, Y = np.meshgrid(x, y)

# Charge position (at origin)
q_x, q_y = 0, 0
q = 1.0  # Positive charge

# Calculate electric field
# E = k * q / r^2 * r_hat
# E_x = k * q * (x - q_x) / r^3
# E_y = k * q * (y - q_y) / r^3

# Distance from charge
R = np.sqrt((X - q_x)**2 + (Y - q_y)**2)
# Avoid division by zero
R[R < 0.1] = 0.1

# Electric field components
k = 1.0  # Normalized constant
E_x = k * q * (X - q_x) / R**3
E_y = k * q * (Y - q_y) / R**3

# Normalize field vectors for visualization
E_mag = np.sqrt(E_x**2 + E_y**2)
E_x_norm = E_x / E_mag
E_y_norm = E_y / E_mag

# Plot field vectors
skip = 2  # Skip some vectors for clarity
ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
          E_x_norm[::skip, ::skip], E_y_norm[::skip, ::skip],
          E_mag[::skip, ::skip], cmap='viridis', scale=15, width=0.003,
          alpha=0.7)

# Draw charge
charge_circle = Circle((q_x, q_y), 0.15, color='red', zorder=5)
ax.add_patch(charge_circle)
ax.text(q_x, q_y, '+', ha='center', va='center', fontsize=20, 
        color='white', weight='bold', zorder=6)

# Add field lines (equipotential approach)
theta = np.linspace(0, 2*np.pi, 100)
for r in [0.5, 1.0, 1.5, 2.0, 2.5]:
    x_line = r * np.cos(theta)
    y_line = r * np.sin(theta)
    ax.plot(x_line, y_line, 'k--', alpha=0.3, linewidth=0.8)

# Labels and title
ax.set_xlabel('$x$ (arbitrary units)', fontsize=12)
ax.set_ylabel('$y$ (arbitrary units)', fontsize=12)
ax.set_title("Electric Field Around a Positive Point Charge\n(Coulomb's Law)", 
             fontsize=14, pad=15)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

# Add text annotation
ax.text(2.5, 2.5, r'$E = \frac{1}{4\pi\epsilon_0}\frac{q}{r^2}\hat{r}$',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        fontsize=11, ha='right')

# Save figure
output_path = '../images/chapter03/coulomb_law_field.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")
plt.close()

