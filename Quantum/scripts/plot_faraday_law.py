#!/usr/bin/env python3
"""
Generate visualization of Faraday's law - induced electric field from changing magnetic field.
Publication-quality figure for Chapter 3.1.1
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# ========== Left plot: Changing B field ==========
# Grid
x = np.linspace(-2, 2, 25)
y = np.linspace(-2, 2, 25)
X, Y = np.meshgrid(x, y)

# Uniform magnetic field pointing into page (increasing)
B_mag = 1.0
dB_dt = 1.0  # Rate of change (positive = increasing)

# Magnetic field (uniform, into page)
# Represented as circles with X (into page)
for i in range(0, len(x), 3):
    for j in range(0, len(y), 3):
        circle = Circle((X[i,j], Y[i,j]), 0.15, fill=False, 
                        edgecolor='blue', linewidth=1.5)
        ax1.add_patch(circle)
        # Draw X to indicate into page
        ax1.plot(X[i,j], Y[i,j], 'x', color='blue', markersize=8, 
                markeredgewidth=2)

# Induced electric field (circular, opposing the change)
# E = -r/2 * dB/dt (for uniform changing B)
R = np.sqrt(X**2 + Y**2)
R[R < 0.1] = 0.1

# E-field is azimuthal (circular)
E_x = -0.5 * dB_dt * (-Y) / R
E_y = -0.5 * dB_dt * X / R

# Normalize
E_mag = np.sqrt(E_x**2 + E_y**2)
E_x_norm = E_x / (E_mag + 1e-10)
E_y_norm = E_y / (E_mag + 1e-10)

# Plot E field vectors
skip = 2
ax1.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
           E_x_norm[::skip, ::skip], E_y_norm[::skip, ::skip],
           E_mag[::skip, ::skip], cmap='Reds', scale=20, width=0.003,
           alpha=0.8)

ax1.set_xlabel('$x$ (arbitrary units)', fontsize=12)
ax1.set_ylabel('$y$ (arbitrary units)', fontsize=12)
ax1.set_title('Induced Electric Field from\nIncreasing Magnetic Field', 
              fontsize=13, pad=10)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax1.set_xlim(-2.5, 2.5)
ax1.set_ylim(-2.5, 2.5)

# Add annotation
ax1.text(1.8, 1.8, r'$\frac{\partial B}{\partial t} > 0$', 
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
         fontsize=12, ha='right')
ax1.text(1.8, -1.8, r'$\nabla \times E = -\frac{\partial B}{\partial t}$',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         fontsize=10, ha='right')

# ========== Right plot: Coil with changing flux ==========
ax2.set_xlim(-3, 3)
ax2.set_ylim(-2, 2)
ax2.set_aspect('equal')

# Draw coil
coil_radius = 1.0
coil_center = (0, 0)
theta_coil = np.linspace(0, 2*np.pi, 100)
coil_x = coil_center[0] + coil_radius * np.cos(theta_coil)
coil_y = coil_center[1] + coil_radius * np.sin(theta_coil)
ax2.plot(coil_x, coil_y, 'k-', linewidth=3, label='Coil')

# Draw magnetic field lines (into page, increasing)
for angle in np.linspace(0, 2*np.pi, 12):
    x_start = 0.3 * np.cos(angle)
    y_start = 0.3 * np.sin(angle)
    ax2.plot(x_start, y_start, 'x', color='blue', markersize=10, 
            markeredgewidth=2)
    # Add arrow showing increasing
    ax2.annotate('', xy=(x_start*1.5, y_start*1.5), 
                xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5, alpha=0.6))

# Draw induced current (counter-clockwise to oppose increasing B)
theta_arrow = np.linspace(0, 2*np.pi, 8)
for i, angle in enumerate(theta_arrow):
    x_pos = 1.3 * np.cos(angle)
    y_pos = 1.3 * np.sin(angle)
    # Arrow direction (tangential, counter-clockwise)
    dx = -np.sin(angle) * 0.2
    dy = np.cos(angle) * 0.2
    ax2.arrow(x_pos, y_pos, dx, dy, head_width=0.15, head_length=0.1,
              fc='red', ec='red', linewidth=2, alpha=0.8)

ax2.set_xlabel('$x$ (arbitrary units)', fontsize=12)
ax2.set_ylabel('$y$ (arbitrary units)', fontsize=12)
ax2.set_title('Faraday\'s Law: Changing Flux\nInduces Current in Coil', 
              fontsize=13, pad=10)
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Add legend
blue_patch = mpatches.Patch(color='blue', label='Magnetic field (increasing)')
red_patch = mpatches.Patch(color='red', label='Induced current')
ax2.legend(handles=[blue_patch, red_patch], loc='upper right', fontsize=10)

# Add equation
ax2.text(2.5, -1.5, r'$\oint E \cdot dl = -\frac{d\Phi_B}{dt}$',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         fontsize=11, ha='right')

plt.tight_layout()

# Save figure
output_path = '../images/chapter03/faraday_law_induction.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")
plt.close()

