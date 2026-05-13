#!/usr/bin/env python3
"""
Generate visualization of Ampère's law - magnetic field around a current-carrying wire.
Publication-quality figure for Chapter 3.1.1
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.patches as mpatches

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
fig, ax = plt.subplots(figsize=(8, 8))

# Grid for field calculation
x = np.linspace(-3, 3, 30)
y = np.linspace(-3, 3, 30)
X, Y = np.meshgrid(x, y)

# Wire position (at origin, coming out of page)
wire_x, wire_y = 0, 0
I = 1.0  # Current (positive = out of page)

# Calculate magnetic field (circular around wire)
# B = μ₀ * I / (2πr) in azimuthal direction
# B_x = -μ₀ * I * (y - wire_y) / (2πr²)
# B_y = μ₀ * I * (x - wire_x) / (2πr²)

# Distance from wire
R = np.sqrt((X - wire_x)**2 + (Y - wire_y)**2)
R[R < 0.1] = 0.1  # Avoid division by zero

# Magnetic field components (circular)
mu0 = 1.0  # Normalized constant
B_x = -mu0 * I * (Y - wire_y) / (2 * np.pi * R**2)
B_y = mu0 * I * (X - wire_x) / (2 * np.pi * R**2)

# Normalize field vectors
B_mag = np.sqrt(B_x**2 + B_y**2)
B_x_norm = B_x / B_mag
B_y_norm = B_y / B_mag

# Plot field vectors
skip = 2
ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
          B_x_norm[::skip, ::skip], B_y_norm[::skip, ::skip],
          B_mag[::skip, ::skip], cmap='plasma', scale=15, width=0.003,
          alpha=0.7)

# Draw wire (cross-section)
wire_circle = Circle((wire_x, wire_y), 0.1, color='darkblue', zorder=5)
ax.add_patch(wire_circle)
# Draw current direction indicator (out of page = dot)
ax.plot(wire_x, wire_y, 'o', color='white', markersize=8, zorder=6)
ax.plot(wire_x, wire_y, '.', color='darkblue', markersize=12, zorder=7)

# Add circular field lines
theta = np.linspace(0, 2*np.pi, 100)
for r in [0.5, 1.0, 1.5, 2.0, 2.5]:
    x_line = r * np.cos(theta)
    y_line = r * np.sin(theta)
    ax.plot(x_line, y_line, 'b--', alpha=0.4, linewidth=0.8)

# Labels and title
ax.set_xlabel('$x$ (arbitrary units)', fontsize=12)
ax.set_ylabel('$y$ (arbitrary units)', fontsize=12)
ax.set_title("Magnetic Field Around a Current-Carrying Wire\n(Ampère's Law)", 
             fontsize=14, pad=15)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

# Add text annotation
ax.text(2.5, 2.5, r'$B = \frac{\mu_0 I}{2\pi r}\hat{\phi}$',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
        fontsize=11, ha='right')

# Add current direction label
ax.text(0.3, 0.3, '$I$ (out)', fontsize=10, color='darkblue', weight='bold')

# Save figure
output_path = '../images/chapter03/ampere_law_field.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")
plt.close()

