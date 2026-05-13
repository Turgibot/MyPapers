#!/usr/bin/env python3
"""
Generate schematic diagram of Faraday's electromagnetic induction experiment.
Publication-quality figure for Chapter 3.1.1
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, Polygon
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
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(-4, 4)
ax.set_ylim(-1, 7)
ax.set_aspect('equal')
ax.axis('off')

# Draw coil of wire
coil_center_x, coil_center_y = 0, 3
coil_radius = 1.2
num_turns = 8

# Draw coil turns
for i in range(num_turns):
    angle = np.linspace(0, 2*np.pi, 100)
    radius = coil_radius + (i - num_turns/2) * 0.05
    x_coil = coil_center_x + radius * np.cos(angle)
    y_coil = coil_center_y + radius * np.sin(angle)
    # Only show visible parts
    mask = (y_coil > coil_center_y - coil_radius) & (y_coil < coil_center_y + coil_radius)
    ax.plot(x_coil[mask], y_coil[mask], 'brown', linewidth=2, alpha=0.8)

# Draw coil leads
ax.plot([-coil_radius, -2.5], [coil_center_y, 1], 'brown', linewidth=3)
ax.plot([coil_radius, 2.5], [coil_center_y, 1], 'brown', linewidth=3)

# Draw galvanometer (ammeter)
galvanometer = Circle((0, 1), 0.6, facecolor='white', edgecolor='black', linewidth=2)
ax.add_patch(galvanometer)
# Draw needle
ax.plot([0, 0.4], [1, 1], 'r-', linewidth=2)
ax.plot([0, 0], [0.7, 1.3], 'k-', linewidth=1)
ax.text(0, 0.3, 'Galvanometer', ha='center', fontsize=9, weight='bold')
ax.text(0.6, 1, 'Current $I$', ha='left', fontsize=9, style='italic', color='red')

# Draw magnet
magnet_x, magnet_y = 0, 5.5
# North pole
north = Rectangle((magnet_x-0.3, magnet_y), 0.6, 0.4, facecolor='red', edgecolor='black', linewidth=2)
ax.add_patch(north)
ax.text(magnet_x, magnet_y+0.2, 'N', ha='center', fontsize=12, weight='bold', color='white')
# South pole
south = Rectangle((magnet_x-0.3, magnet_y-0.6), 0.6, 0.4, facecolor='blue', edgecolor='black', linewidth=2)
ax.add_patch(south)
ax.text(magnet_x, magnet_y-0.4, 'S', ha='center', fontsize=12, weight='bold', color='white')

# Draw magnetic field lines
for i in range(5):
    y_field = magnet_y - 0.3 - i * 0.3
    if y_field > coil_center_y + coil_radius:
        # Field lines above coil
        x_field = np.linspace(-1.5, 1.5, 50)
        y_field_line = np.full_like(x_field, y_field)
        ax.plot(x_field, y_field_line, 'b--', linewidth=1, alpha=0.5)
    elif y_field < coil_center_y - coil_radius:
        # Field lines below coil
        x_field = np.linspace(-1.5, 1.5, 50)
        y_field_line = np.full_like(x_field, y_field)
        ax.plot(x_field, y_field_line, 'b--', linewidth=1, alpha=0.5)

# Draw movement arrows
arrow_down = FancyArrowPatch((magnet_x+0.8, magnet_y-0.1), (magnet_x+0.8, magnet_y-1.2),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='green')
ax.add_patch(arrow_down)
ax.text(magnet_x+1.2, magnet_y-0.65, 'Moving\nmagnet', ha='left', fontsize=9, style='italic', color='green')

arrow_up = FancyArrowPatch((magnet_x-0.8, magnet_y-1.2), (magnet_x-0.8, magnet_y-0.1),
                           arrowstyle='->', mutation_scale=20, linewidth=2, color='green', linestyle='--', alpha=0.5)
ax.add_patch(arrow_up)
ax.text(magnet_x-1.2, magnet_y-0.65, 'Or moving\nup', ha='right', fontsize=9, style='italic', color='green', alpha=0.7)

# Draw induced current direction in coil
# Current flows when magnet moves
theta_current = np.linspace(0, 2*np.pi, 20)
for i, angle in enumerate(theta_current[::2]):
    x_arrow = (coil_radius - 0.2) * np.cos(angle)
    y_arrow = (coil_radius - 0.2) * np.sin(angle) + coil_center_y
    dx = -np.sin(angle) * 0.15
    dy = np.cos(angle) * 0.15
    arrow = FancyArrowPatch((x_arrow, y_arrow), (x_arrow+dx, y_arrow+dy),
                            arrowstyle='->', mutation_scale=10, linewidth=1.5, color='red', alpha=0.7)
    ax.add_patch(arrow)

# Add labels
ax.text(0, 6.8, 'Faraday\'s Induction Experiment (1831)', ha='center', fontsize=14, weight='bold')
ax.text(-2.5, 4, 'Moving the magnet\nchanges the magnetic\nflux through the coil,\ninducing a current', 
        ha='center', fontsize=10, style='italic',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Add equation annotation
ax.text(2.5, 1, r'$\mathcal{E} = -\frac{d\Phi_B}{dt}$', ha='left', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Save figure
output_path = '../images/chapter03/faraday_experiment.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")
plt.close()

