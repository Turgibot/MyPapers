#!/usr/bin/env python3
"""
Generate visualization of all four Maxwell's equations showing their relationships.
Publication-quality figure for Chapter 3.1.1
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
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
fig = plt.figure(figsize=(14, 14))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# ========== Equation 1: Gauss's Law for Electricity ==========
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax1.set_aspect('equal')

# Positive charge
charge1 = Circle((0, 0), 0.2, color='red', zorder=5)
ax1.add_patch(charge1)
ax1.text(0, 0, '+', ha='center', va='center', fontsize=16, 
        color='white', weight='bold', zorder=6)

# Electric field lines (radial, outward)
theta = np.linspace(0, 2*np.pi, 8)
for angle in theta:
    r = np.linspace(0.3, 1.8, 20)
    x_line = r * np.cos(angle)
    y_line = r * np.sin(angle)
    ax1.plot(x_line, y_line, 'r-', linewidth=1.5, alpha=0.6)
    # Arrow at end
    ax1.arrow(x_line[-2], y_line[-2], 
             (x_line[-1] - x_line[-2]) * 0.3, 
             (y_line[-1] - y_line[-2]) * 0.3,
             head_width=0.1, head_length=0.08, fc='red', ec='red', alpha=0.6)

ax1.set_title('(1) Gauss\'s Law for Electricity\n$\\nabla \\cdot E = 0$', 
              fontsize=12, pad=10)
ax1.set_xlabel('$x$', fontsize=10)
ax1.set_ylabel('$y$', fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# ========== Equation 2: Gauss's Law for Magnetism ==========
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlim(-2, 2)
ax2.set_ylim(-2, 2)
ax2.set_aspect('equal')

# Magnetic dipole (bar magnet)
# North pole
north = Circle((-0.3, 0), 0.15, color='red', zorder=5)
ax2.add_patch(north)
ax2.text(-0.3, 0, 'N', ha='center', va='center', fontsize=10, 
        color='white', weight='bold', zorder=6)
# South pole
south = Circle((0.3, 0), 0.15, color='blue', zorder=5)
ax2.add_patch(south)
ax2.text(0.3, 0, 'S', ha='center', va='center', fontsize=10, 
        color='white', weight='bold', zorder=6)

# Magnetic field lines (closed loops)
for r in [0.8, 1.2, 1.6]:
    theta = np.linspace(0, 2*np.pi, 100)
    x_line = r * np.cos(theta)
    y_line = r * np.sin(theta)
    ax2.plot(x_line, y_line, 'b-', linewidth=1.5, alpha=0.6)
    # Add arrows
    for i in range(0, len(theta), 15):
        idx = i
        dx = -np.sin(theta[idx]) * 0.2
        dy = np.cos(theta[idx]) * 0.2
        ax2.arrow(x_line[idx], y_line[idx], dx, dy,
                 head_width=0.08, head_length=0.06, fc='blue', ec='blue', alpha=0.6)

ax2.set_title('(2) Gauss\'s Law for Magnetism\n$\\nabla \\cdot B = 0$', 
              fontsize=12, pad=10)
ax2.set_xlabel('$x$', fontsize=10)
ax2.set_ylabel('$y$', fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# ========== Equation 3: Faraday's Law ==========
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_xlim(-2, 2)
ax3.set_ylim(-2, 2)
ax3.set_aspect('equal')

# Changing B field (into page, increasing)
for i in range(-1, 2):
    for j in range(-1, 2):
        circle = Circle((i*0.6, j*0.6), 0.12, fill=False, 
                        edgecolor='blue', linewidth=1.2)
        ax3.add_patch(circle)
        ax3.plot(i*0.6, j*0.6, 'x', color='blue', markersize=6, 
                markeredgewidth=1.5)

# Induced E field (circular)
theta = np.linspace(0, 2*np.pi, 8)
for angle in theta:
    r = np.linspace(0.4, 1.6, 15)
    x_line = r * np.cos(angle)
    y_line = r * np.sin(angle)
    # Rotate 90 degrees for circular E field
    x_rot = -y_line
    y_rot = x_line
    ax3.plot(x_rot, y_rot, 'r-', linewidth=1.5, alpha=0.6)
    # Arrow
    if len(x_rot) > 2:
        ax3.arrow(x_rot[-3], y_rot[-3], 
                 (x_rot[-1] - x_rot[-3]) * 0.3, 
                 (y_rot[-1] - y_rot[-3]) * 0.3,
                 head_width=0.1, head_length=0.08, fc='red', ec='red', alpha=0.6)

ax3.set_title('(3) Faraday\'s Law\n$\\nabla \\times E = -\\frac{\\partial B}{\\partial t}$', 
              fontsize=12, pad=10)
ax3.set_xlabel('$x$', fontsize=10)
ax3.set_ylabel('$y$', fontsize=10)
ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# ========== Equation 4: Ampère-Maxwell Law ==========
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_xlim(-2, 2)
ax4.set_ylim(-2, 2)
ax4.set_aspect('equal')

# Changing E field (radial, increasing)
for i in range(-1, 2):
    for j in range(-1, 2):
        if i == 0 and j == 0:
            # Charge at center
            charge = Circle((0, 0), 0.15, color='red', zorder=5)
            ax4.add_patch(charge)
            ax4.text(0, 0, '+', ha='center', va='center', fontsize=14, 
                    color='white', weight='bold', zorder=6)
        else:
            # E field lines
            x_start, y_start = i*0.6, j*0.6
            r = np.sqrt(x_start**2 + y_start**2)
            if r > 0.3:
                x_end = x_start * 1.3
                y_end = y_start * 1.3
                ax4.arrow(x_start, y_start, x_end-x_start, y_end-y_start,
                         head_width=0.1, head_length=0.08, fc='red', ec='red', 
                         linewidth=1.5, alpha=0.6)

# Induced B field (circular around changing E)
theta = np.linspace(0, 2*np.pi, 8)
for angle in theta:
    r = np.linspace(0.5, 1.5, 15)
    x_line = r * np.cos(angle)
    y_line = r * np.sin(angle)
    # Circular B field
    x_rot = -y_line
    y_rot = x_line
    ax4.plot(x_rot, y_rot, 'b-', linewidth=1.5, alpha=0.6)
    # Arrow
    if len(x_rot) > 2:
        ax4.arrow(x_rot[-3], y_rot[-3], 
                 (x_rot[-1] - x_rot[-3]) * 0.3, 
                 (y_rot[-1] - y_rot[-3]) * 0.3,
                 head_width=0.1, head_length=0.08, fc='blue', ec='blue', alpha=0.6)

ax4.set_title('(4) Ampère-Maxwell Law\n$\\nabla \\times B = \\mu_0\\epsilon_0\\frac{\\partial E}{\\partial t}$', 
              fontsize=12, pad=10)
ax4.set_xlabel('$x$', fontsize=10)
ax4.set_ylabel('$y$', fontsize=10)
ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Add overall title
fig.suptitle("Maxwell's Four Equations: The Complete Picture", 
             fontsize=16, y=0.98, weight='bold')

# Save figure
output_path = '../images/chapter03/maxwell_equations_complete.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")
plt.close()

