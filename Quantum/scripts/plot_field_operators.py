#!/usr/bin/env python3
"""
Generate visualization of vector field operators: divergence and curl.
Publication-quality figure for Chapter 3.1.2
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

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 14))

# ========== Top Left: Divergence - Source Field ==========
ax = axes[0, 0]
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)

# Source field (positive divergence)
# F = (x, y) - radial outward
F_x = X
F_y = Y

# Normalize for visualization
F_mag = np.sqrt(F_x**2 + F_y**2)
F_x_norm = F_x / (F_mag + 1e-10)
F_y_norm = F_y / (F_mag + 1e-10)

ax.quiver(X, Y, F_x_norm, F_y_norm, F_mag, cmap='Reds', 
          scale=25, width=0.003, alpha=0.7)

# Add source point
source = Circle((0, 0), 0.15, color='red', zorder=5)
ax.add_patch(source)
ax.text(0, 0, '+', ha='center', va='center', fontsize=16, 
        color='white', weight='bold', zorder=6)

ax.set_title('Divergence > 0 (Source)\n$\\nabla \\cdot F > 0$', 
             fontsize=12, pad=10)
ax.set_xlabel('$x$', fontsize=10)
ax.set_ylabel('$y$', fontsize=10)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)

# ========== Top Right: Divergence - Sink Field ==========
ax = axes[0, 1]
# Sink field (negative divergence)
# F = (-x, -y) - radial inward
F_x = -X
F_y = -Y

F_mag = np.sqrt(F_x**2 + F_y**2)
F_x_norm = F_x / (F_mag + 1e-10)
F_y_norm = F_y / (F_mag + 1e-10)

ax.quiver(X, Y, F_x_norm, F_y_norm, F_mag, cmap='Blues', 
          scale=25, width=0.003, alpha=0.7)

# Add sink point
sink = Circle((0, 0), 0.15, color='blue', zorder=5)
ax.add_patch(sink)
ax.text(0, 0, '-', ha='center', va='center', fontsize=16, 
        color='white', weight='bold', zorder=6)

ax.set_title('Divergence < 0 (Sink)\n$\\nabla \\cdot F < 0$', 
             fontsize=12, pad=10)
ax.set_xlabel('$x$', fontsize=10)
ax.set_ylabel('$y$', fontsize=10)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)

# ========== Bottom Left: Curl - Clockwise Rotation ==========
ax = axes[1, 0]
# Rotational field (curl)
# F = (-y, x) - clockwise rotation
F_x = -Y
F_y = X

F_mag = np.sqrt(F_x**2 + F_y**2)
F_x_norm = F_x / (F_mag + 1e-10)
F_y_norm = F_y / (F_mag + 1e-10)

ax.quiver(X, Y, F_x_norm, F_y_norm, F_mag, cmap='viridis', 
          scale=25, width=0.003, alpha=0.7)

# Add rotation indicator
ax.text(0, 0, '↻', ha='center', va='center', fontsize=24, 
        color='darkgreen', weight='bold', zorder=6)

ax.set_title('Curl (Clockwise Rotation)\n$\\nabla \\times F \\neq 0$', 
             fontsize=12, pad=10)
ax.set_xlabel('$x$', fontsize=10)
ax.set_ylabel('$y$', fontsize=10)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)

# ========== Bottom Right: Zero Divergence and Curl ==========
ax = axes[1, 1]
# Uniform field (zero divergence, zero curl)
# F = (1, 0) - uniform horizontal
F_x = np.ones_like(X)
F_y = np.zeros_like(Y)

F_mag = np.sqrt(F_x**2 + F_y**2)
F_x_norm = F_x / (F_mag + 1e-10)
F_y_norm = F_y / (F_mag + 1e-10)

ax.quiver(X, Y, F_x_norm, F_y_norm, F_mag, cmap='gray', 
          scale=25, width=0.003, alpha=0.7)

ax.set_title('Uniform Field\n$\\nabla \\cdot F = 0$, $\\nabla \\times F = 0$', 
             fontsize=12, pad=10)
ax.set_xlabel('$x$', fontsize=10)
ax.set_ylabel('$y$', fontsize=10)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)

# Add overall title
fig.suptitle("Vector Field Operators: Divergence and Curl", 
             fontsize=16, y=0.98, weight='bold')

plt.tight_layout()

# Save figure
output_path = '../images/chapter03/field_operators.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")
plt.close()

