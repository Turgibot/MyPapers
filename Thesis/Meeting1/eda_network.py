#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for Porto SUMO Network
Analyzes network structure, edges, junctions, and generates statistics.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Configuration
NETWORK_PATH = "/home/guy/Projects/Traffic/Develop/Projects/PortoForSumo/config/porto.net.xml"
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

def parse_sumo_network(network_path):
    """Parse SUMO network XML file."""
    print(f"Loading SUMO network from {network_path}...")
    
    tree = ET.parse(network_path)
    root = tree.getroot()
    
    # Parse junctions
    junctions = []
    for junction in root.findall('junction'):
        junc_data = {
            'id': junction.get('id'),
            'type': junction.get('type'),
            'x': float(junction.get('x', 0)),
            'y': float(junction.get('y', 0)),
        }
        junctions.append(junc_data)
    
    # Parse edges
    edges = []
    for edge in root.findall('edge'):
        edge_id = edge.get('id')
        if edge_id.startswith(':'):  # Internal edges
            continue
            
        edge_data = {
            'id': edge_id,
            'from': edge.get('from'),
            'to': edge.get('to'),
            'priority': int(edge.get('priority', 0)),
            'length': 0.0,
            'num_lanes': 0,
            'speed_limit': 0.0,
        }
        
        # Get lane information
        lanes = edge.findall('lane')
        if lanes:
            first_lane = lanes[0]
            edge_data['length'] = float(first_lane.get('length', 0))
            edge_data['num_lanes'] = len(lanes)
            edge_data['speed_limit'] = float(first_lane.get('speed', 0))
        
        edges.append(edge_data)
    
    return junctions, edges

def analyze_network():
    """Main network analysis function."""
    junctions, edges = parse_sumo_network(NETWORK_PATH)
    
    print(f"\nNetwork Statistics:")
    print(f"  Junctions: {len(junctions)}")
    print(f"  Edges: {len(edges)}")
    
    # Calculate statistics
    edge_lengths = [e['length'] for e in edges if e['length'] > 0]
    edge_speeds = [e['speed_limit'] for e in edges if e['speed_limit'] > 0]
    num_lanes_list = [e['num_lanes'] for e in edges if e['num_lanes'] > 0]
    
    # Calculate bounding box
    junc_x = [j['x'] for j in junctions]
    junc_y = [j['y'] for j in junctions]
    
    stats = {
        'junctions': len(junctions),
        'edges': len(edges),
        'total_length_km': sum(edge_lengths) / 1000,
        'avg_edge_length_m': np.mean(edge_lengths) if edge_lengths else 0,
        'median_edge_length_m': np.median(edge_lengths) if edge_lengths else 0,
        'bbox': {
            'x_min': min(junc_x) if junc_x else 0,
            'x_max': max(junc_x) if junc_x else 0,
            'y_min': min(junc_y) if junc_y else 0,
            'y_max': max(junc_y) if junc_y else 0,
        }
    }
    
    # Junction types
    junc_types = defaultdict(int)
    for j in junctions:
        junc_types[j['type']] += 1
    
    stats['junction_types'] = dict(junc_types)
    
    # Save statistics
    save_network_statistics(stats, edge_lengths, edge_speeds, num_lanes_list)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_network_structure(junctions, edges)
    plot_edge_length_distribution(edge_lengths)
    plot_speed_limit_distribution(edge_speeds)
    plot_lane_distribution(num_lanes_list)
    
    print("\nNetwork Analysis Complete!")
    print_network_statistics(stats)
    
    return stats, junctions, edges

def save_network_statistics(stats, edge_lengths, edge_speeds, num_lanes_list):
    """Save network statistics to text files for LaTeX inclusion."""
    
    (OUTPUT_DIR / "network_junctions.tex").write_text(f"{stats['junctions']:,}")
    (OUTPUT_DIR / "network_edges.tex").write_text(f"{stats['edges']:,}")
    (OUTPUT_DIR / "network_length.tex").write_text(f"{stats['total_length_km']:.2f}")
    (OUTPUT_DIR / "network_avg_edge_length.tex").write_text(f"{stats['avg_edge_length_m']:.1f}")
    
    bbox = stats['bbox']
    bbox_str = f"({bbox['x_min']:.2f}, {bbox['y_min']:.2f}) to ({bbox['x_max']:.2f}, {bbox['y_max']:.2f})"
    (OUTPUT_DIR / "network_bbox.tex").write_text(bbox_str)

def plot_network_structure(junctions, edges):
    """Plot network structure visualization."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot edges (sample for performance)
    sample_edges = edges[:min(5000, len(edges))]
    for edge in sample_edges:
        # Find junction coordinates
        from_junc = next((j for j in junctions if j['id'] == edge['from']), None)
        to_junc = next((j for j in junctions if j['id'] == edge['to']), None)
        
        if from_junc and to_junc:
            ax.plot([from_junc['x'], to_junc['x']], 
                   [from_junc['y'], to_junc['y']],
                   'b-', alpha=0.1, linewidth=0.5)
    
    # Plot junctions
    junc_x = [j['x'] for j in junctions]
    junc_y = [j['y'] for j in junctions]
    ax.scatter(junc_x, junc_y, c='red', s=10, alpha=0.5, label='Junctions')
    
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    ax.set_title('Porto SUMO Network Structure')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "network_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_edge_length_distribution(edge_lengths):
    """Plot distribution of edge lengths."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(edge_lengths, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Edge Length (meters)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Edge Lengths in Porto Network')
    ax.axvline(np.median(edge_lengths), color='r', linestyle='--',
               label=f'Median: {np.median(edge_lengths):.1f} m')
    ax.legend()
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "network_edge_lengths.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_speed_limit_distribution(edge_speeds):
    """Plot distribution of speed limits."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert m/s to km/h
    speeds_kmh = [s * 3.6 for s in edge_speeds]
    
    ax.hist(speeds_kmh, bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Speed Limit (km/h)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Speed Limits in Porto Network')
    ax.axvline(np.median(speeds_kmh), color='r', linestyle='--',
               label=f'Median: {np.median(speeds_kmh):.1f} km/h')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "network_speed_limits.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_lane_distribution(num_lanes_list):
    """Plot distribution of number of lanes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lane_counts = defaultdict(int)
    for n in num_lanes_list:
        lane_counts[n] += 1
    
    lanes = sorted(lane_counts.keys())
    counts = [lane_counts[l] for l in lanes]
    
    ax.bar(lanes, counts, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Lanes')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Number of Lanes per Edge')
    
    for lane, count in zip(lanes, counts):
        ax.text(lane, count, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "network_lanes.png", dpi=300, bbox_inches='tight')
    plt.close()

def print_network_statistics(stats):
    """Print network statistics."""
    print("\n" + "="*60)
    print("NETWORK STATISTICS")
    print("="*60)
    print(f"Junctions: {stats['junctions']:,}")
    print(f"Edges: {stats['edges']:,}")
    print(f"Total Length: {stats['total_length_km']:.2f} km")
    print(f"Average Edge Length: {stats['avg_edge_length_m']:.1f} m")
    print(f"Median Edge Length: {stats['median_edge_length_m']:.1f} m")
    print(f"\nJunction Types:")
    for jtype, count in stats['junction_types'].items():
        print(f"  {jtype}: {count}")

if __name__ == "__main__":
    analyze_network()

