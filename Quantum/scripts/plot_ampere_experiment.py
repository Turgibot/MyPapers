#!/usr/bin/env python3
"""
Generate schematic diagram of Ørsted's experiment showing current and magnetic field.
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

# Draw battery/voltage source
battery = Rectangle((-3.5, 1), 0.8, 1.5, facecolor='lightblue', edgecolor='black', linewidth=2)
ax.add_patch(battery)
ax.text(-3.1, 1.75, '+', ha='center', fontsize=16, weight='bold')
ax.text(-3.1, 1.25, '-', ha='center', fontsize=16, weight='bold')
ax.text(-3.1, 0.5, 'Battery', ha='center', fontsize=9)

# Draw wire (vertical)
ax.plot([0, 0], [0.5, 5.5], 'k-', linewidth=4, label='Wire')

# Draw current direction arrows
for y in [1.5, 2.5, 3.5, 4.5]:
    arrow = FancyArrowPatch((0.3, y), (0.3, y+0.6),
                            arrowstyle='->', mutation_scale=15, linewidth=2, color='red')
    ax.add_patch(arrow)
ax.text(0.7, 3, 'Current $I$', ha='left', fontsize=11, weight='bold', color='red')

# Draw compass needles around wire
# Top compass
compass_top = Circle((1.5, 5), 0.4, facecolor='white', edgecolor='black', linewidth=2)
ax.add_patch(compass_top)
# Needle pointing tangent to field
ax.plot([1.5-0.3, 1.5+0.3], [5, 5], 'r-', linewidth=3)
ax.plot([1.5, 1.5], [5-0.2, 5+0.2], 'k-', linewidth=1)
ax.text(1.5, 4.3, 'N', ha='center', fontsize=8)
ax.text(1.5, 5.7, 'S', ha='center', fontsize=8)

# Right compass
compass_right = Circle((1.5, 3), 0.4, facecolor='white', edgecolor='black', linewidth=2)
ax.add_patch(compass_right)
ax.plot([1.5-0.3, 1.5+0.3], [3, 3], 'r-', linewidth=3)
ax.plot([1.5, 1.5], [3-0.2, 3+0.2], 'k-', linewidth=1)
ax.text(1.5, 2.3, 'N', ha='center', fontsize=8)
ax.text(1.5, 3.7, 'S', ha='center', fontsize=8)

# Bottom compass
compass_bottom = Circle((1.5, 1), 0.4, facecolor='white', edgecolor='black', linewidth=2)
ax.add_patch(compass_bottom)
ax.plot([1.5-0.3, 1.5+0.3], [1, 1], 'r-', linewidth=3)
ax.plot([1.5, 1.5], [1-0.2, 1+0.2], 'k-', linewidth=1)
ax.text(1.5, 0.3, 'N', ha='center', fontsize=8)
ax.text(1.5, 1.7, 'S', ha='center', fontsize=8)

# Draw magnetic field lines (circular)
theta = np.linspace(0, 2*np.pi, 100)
for r in [0.8, 1.3, 1.8]:
    x_field = r * np.cos(theta)
    y_field = r * np.sin(theta) + 3
    # Only show part of the circle
    mask = (y_field > 0.5) & (y_field < 5.5)
    ax.plot(x_field[mask], y_field[mask], 'b--', linewidth=1.5, alpha=0.6)

# Add field direction arrows
for angle in [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]:
    r = 1.0
    x_start = r * np.cos(angle)
    y_start = r * np.sin(angle) + 3
    dx = -np.sin(angle) * 0.3
    dy = np.cos(angle) * 0.3
    arrow = FancyArrowPatch((x_start, y_start), (x_start+dx, y_start+dy),
                            arrowstyle='->', mutation_scale=12, linewidth=1.5, color='blue', alpha=0.7)
    ax.add_patch(arrow)

# Draw connecting wires
ax.plot([-3.5, 0], [1.75, 0.5], 'k-', linewidth=3)
ax.plot([-3.5, 0], [1.25, 5.5], 'k-', linewidth=3)

# Add switch
switch = Rectangle((-1.5, 2.5), 0.3, 0.1, facecolor='gray', edgecolor='black', linewidth=1)
ax.add_patch(switch)
ax.plot([-1.5, -1.2], [2.55, 2.55], 'k-', linewidth=2)
ax.text(-1.35, 2.2, 'Switch', ha='center', fontsize=8)

# Add labels
ax.text(0, 6.5, 'Ørsted\'s Experiment (1820)', ha='center', fontsize=14, weight='bold')
ax.text(-2.5, 3, 'When current flows,\nthe compass needles\ndeflect perpendicular\nto the wire', 
        ha='center', fontsize=10, style='italic',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Add equation annotation
ax.text(2.5, 1, r'$B \propto \frac{I}{r}$', ha='left', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Save figure
output_path = '../images/chapter03/ampere_experiment.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")
plt.close()

