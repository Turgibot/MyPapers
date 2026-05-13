#!/usr/bin/env python3
"""
Generate schematic diagram of Coulomb's torsion balance experiment.
Publication-quality figure for Chapter 3.1.1
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, FancyArrowPatch
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
ax.set_xlim(-5, 5)
ax.set_ylim(-1, 7)
ax.set_aspect('equal')
ax.axis('off')

# Draw support structure
support = Rectangle((-0.3, 6), 0.6, 0.5, facecolor='gray', edgecolor='black', linewidth=2)
ax.add_patch(support)

# Draw suspension wire
ax.plot([0, 0], [6, 4], 'k-', linewidth=2)

# Draw horizontal rod (torsion balance)
rod_length = 2.5
ax.plot([-rod_length, rod_length], [4, 4], 'k-', linewidth=3)

# Draw charged spheres
# Left sphere (fixed)
left_sphere = Circle((-rod_length-0.3, 4), 0.3, facecolor='red', edgecolor='black', linewidth=2)
ax.add_patch(left_sphere)
ax.text(-rod_length-0.3, 3.3, '$q_1$', ha='center', fontsize=12, weight='bold')
ax.text(-rod_length-0.3, 2.8, 'Fixed', ha='center', fontsize=9, style='italic')

# Right sphere (movable, on torsion balance)
right_sphere = Circle((rod_length+0.3, 4), 0.3, facecolor='blue', edgecolor='black', linewidth=2)
ax.add_patch(right_sphere)
ax.text(rod_length+0.3, 3.3, '$q_2$', ha='center', fontsize=12, weight='bold')
ax.text(rod_length+0.3, 2.8, 'Movable', ha='center', fontsize=9, style='italic')

# Draw distance indicator
distance = rod_length * 2 + 0.6
ax.plot([-rod_length-0.3, rod_length+0.3], [1.5, 1.5], 'k--', linewidth=1, alpha=0.5)
ax.plot([-rod_length-0.3, -rod_length-0.3], [1.3, 1.7], 'k-', linewidth=1)
ax.plot([rod_length+0.3, rod_length+0.3], [1.3, 1.7], 'k-', linewidth=1)
ax.text(0, 1.2, f'$r$ (distance)', ha='center', fontsize=11, style='italic')

# Draw force arrows
# Repulsive force on right sphere
arrow1 = FancyArrowPatch((rod_length+0.6, 4), (rod_length+1.2, 4),
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='red')
ax.add_patch(arrow1)
ax.text(rod_length+1.0, 4.4, '$F$', ha='center', fontsize=11, weight='bold', color='red')

# Repulsive force on left sphere
arrow2 = FancyArrowPatch((-rod_length-0.6, 4), (-rod_length-1.2, 4),
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='red')
ax.add_patch(arrow2)
ax.text(-rod_length-1.0, 4.4, '$F$', ha='center', fontsize=11, weight='bold', color='red')

# Draw scale/mirror for measuring deflection
scale = Rectangle((3.5, 2), 1, 3, facecolor='lightgray', edgecolor='black', linewidth=1)
ax.add_patch(scale)
ax.text(4, 3.5, 'Scale', ha='center', fontsize=9, rotation=90)

# Draw light beam indicator (for optical lever)
ax.plot([rod_length+0.3, 4], [4, 3.5], 'b--', linewidth=1, alpha=0.5)
ax.plot([4, 4.5], [3.5, 3.5], 'b-', linewidth=1.5)
ax.text(4.7, 3.5, 'Light beam', ha='left', fontsize=9, style='italic')

# Add labels
ax.text(0, 6.8, 'Coulomb\'s Torsion Balance (1785)', ha='center', fontsize=14, weight='bold')
ax.text(0, 0.5, 'The balance measures the electrostatic force between two charged spheres\nby detecting the angular deflection of the suspended rod.', 
        ha='center', fontsize=10, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Save figure
output_path = '../images/chapter03/coulomb_experiment.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")
plt.close()

