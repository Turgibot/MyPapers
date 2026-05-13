#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for Porto Taxi Dataset
Analyzes trajectory data, GPS quality, and generates statistics and visualizations.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import pandas as pd
import numpy as np
import ast
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Configuration
# Use train.csv for full dataset analysis (test.csv is only a small sample)
DATASET_PATH = "/home/guy/Projects/Traffic/Multi-Variant-Simulated-Traffic-Dataset-Creator-and-Model-Tester/Porto/dataset/train.csv"
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Porto bounding box (approximate)
PORTO_BBOX = {
    'lon_min': -8.7,
    'lon_max': -8.5,
    'lat_min': 41.1,
    'lat_max': 41.2
}

def parse_polyline(polyline_str):
    """Parse polyline string to list of [lon, lat] coordinates."""
    try:
        if pd.isna(polyline_str) or polyline_str == '[]':
            return []
        # Handle both string and already-parsed formats
        if isinstance(polyline_str, str):
            return ast.literal_eval(polyline_str)
        return polyline_str
    except:
        return []

def haversine_distance(lon1, lat1, lon2, lat2):
    """Calculate great circle distance between two points in meters."""
    from math import radians, cos, sin, asin, sqrt
    
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371000  # Earth radius in meters
    return c * r

def is_valid_coordinate(lon, lat):
    """Check if coordinate is within Porto bounding box."""
    return (PORTO_BBOX['lon_min'] <= lon <= PORTO_BBOX['lon_max'] and
            PORTO_BBOX['lat_min'] <= lat <= PORTO_BBOX['lat_max'])

def analyze_dataset():
    """Main EDA analysis function."""
    print("Loading Porto dataset...")
    print(f"Reading from: {DATASET_PATH}")
    
    # For very large datasets, we'll compute statistics efficiently
    # First, get basic counts without loading everything into memory
    print("Computing basic statistics...")
    
    # Count total lines (subtract 1 for header)
    with open(DATASET_PATH, 'r') as f:
        total_lines = sum(1 for _ in f) - 1
    
    print(f"Total trips in dataset: {total_lines:,}")
    
    # Read in chunks for efficient processing
    chunk_size = 100000
    chunks_processed = 0
    total_trips = 0
    valid_trips = 0
    total_gps_points = 0
    unique_taxis = set()
    num_points_list = []
    timestamps = []
    
    print("Processing dataset in chunks...")
    for chunk in pd.read_csv(DATASET_PATH, chunksize=chunk_size):
        chunks_processed += 1
        total_trips += len(chunk)
        
        # Parse trajectories
        chunk['trajectory'] = chunk['POLYLINE'].apply(parse_polyline)
        chunk['num_points'] = chunk['trajectory'].apply(len)
        
        # Filter valid trajectories
        chunk_valid = chunk[chunk['num_points'] > 0]
        valid_trips += len(chunk_valid)
        total_gps_points += chunk_valid['num_points'].sum()
        num_points_list.extend(chunk_valid['num_points'].tolist())
        
        # Collect unique taxis
        unique_taxis.update(chunk['TAXI_ID'].unique())
        
        # Collect timestamps
        timestamps.extend(chunk_valid['TIMESTAMP'].tolist())
        
        if chunks_processed % 5 == 0:
            print(f"  Processed {chunks_processed} chunks ({total_trips:,} trips)...")
    
    print(f"Dataset processing complete!")
    print(f"  Total trips: {total_trips:,}")
    print(f"  Valid trips: {valid_trips:,}")
    print(f"  Unique taxis: {len(unique_taxis):,}")
    
    # Basic statistics
    stats = {
        'total_trips': total_trips,
        'valid_trips': valid_trips,
        'unique_taxis': len(unique_taxis),
        'total_gps_points': total_gps_points,
        'avg_points_per_trip': np.mean(num_points_list) if num_points_list else 0,
        'median_points_per_trip': np.median(num_points_list) if num_points_list else 0,
    }
    
    # Parse timestamps for date range
    print("Computing date range...")
    if timestamps:
        timestamps_dt = pd.to_datetime(timestamps, unit='s')
        stats['date_range'] = f"{timestamps_dt.min().date()} to {timestamps_dt.max().date()}"
    else:
        stats['date_range'] = "N/A"
    
    # Analyze GPS quality - sample for detailed analysis
    print("\nAnalyzing GPS quality (sampling for detailed metrics)...")
    invalid_coords = 0
    large_jumps = 0
    total_segments = 0
    jump_distances = []
    trajectory_lengths = []
    travel_times = []
    speeds = []
    short_trajectories = 0
    
    # Sample up to 100k trajectories for detailed GPS quality analysis
    sample_size = min(100000, valid_trips)
    print(f"Sampling {sample_size:,} trajectories for detailed GPS quality analysis...")
    
    # Re-read and sample
    sample_count = 0
    for chunk in pd.read_csv(DATASET_PATH, chunksize=chunk_size):
        chunk['trajectory'] = chunk['POLYLINE'].apply(parse_polyline)
        chunk['num_points'] = chunk['trajectory'].apply(len)
        chunk_valid = chunk[chunk['num_points'] > 0]
        
        # Count short trajectories
        short_trajectories += len(chunk_valid[chunk_valid['num_points'] < 5])
        
        # Sample for detailed analysis
        if sample_count < sample_size:
            remaining = sample_size - sample_count
            sample_chunk = chunk_valid.head(remaining)
            
            for idx, row in sample_chunk.iterrows():
                traj = row['trajectory']
                if len(traj) < 2:
                    continue
                    
                traj_length = 0
                for i in range(len(traj)):
                    lon, lat = traj[i]
                    
                    # Check if coordinate is valid
                    if not is_valid_coordinate(lon, lat):
                        invalid_coords += 1
                    
                    # Calculate jumps
                    if i > 0:
                        prev_lon, prev_lat = traj[i-1]
                        if is_valid_coordinate(prev_lon, prev_lat) and is_valid_coordinate(lon, lat):
                            dist = haversine_distance(prev_lon, prev_lat, lon, lat)
                            jump_distances.append(dist)
                            total_segments += 1
                            traj_length += dist
                            
                            if dist > 1000:  # Large jump threshold
                                large_jumps += 1
                
                trajectory_lengths.append(traj_length)
                
                # Estimate travel time (assuming 15 seconds between points)
                if len(traj) > 1:
                    estimated_time = (len(traj) - 1) * 15  # seconds
                    travel_times.append(estimated_time)
                    
                    # Estimate average speed
                    if traj_length > 0:
                        avg_speed = (traj_length / 1000) / (estimated_time / 3600)  # km/h
                        speeds.append(avg_speed)
            
            sample_count += len(sample_chunk)
            if sample_count >= sample_size:
                break
    
    # Scale statistics based on sample
    sample_ratio = sample_count / valid_trips if valid_trips > 0 else 1
    total_gps_points_sampled = sum(len(parse_polyline(row['POLYLINE'])) for _, row in 
                                   pd.read_csv(DATASET_PATH, nrows=sample_count).iterrows() 
                                   if len(parse_polyline(row['POLYLINE'])) > 0)
    
    print("Calculating statistics...")
    stats['invalid_coords_pct'] = (invalid_coords / total_gps_points_sampled * 100) if total_gps_points_sampled > 0 else 0
    stats['large_jumps_pct'] = (large_jumps / total_segments * 100) if total_segments > 0 else 0
    stats['short_trajectories_pct'] = (short_trajectories / valid_trips * 100) if valid_trips > 0 else 0
    
    # Save statistics to files for LaTeX
    save_statistics(stats, trajectory_lengths, travel_times, speeds, jump_distances)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    # Create sample dataframe for visualization
    sample_df = pd.DataFrame({
        'num_points': num_points_list[:min(100000, len(num_points_list))]
    })
    plot_trajectory_length_distribution(sample_df, trajectory_lengths)
    plot_travel_time_distribution(travel_times)
    plot_speed_distribution(speeds)
    plot_gps_jumps(jump_distances)
    
    # For spatial coverage, sample trajectories
    print("  Generating spatial coverage plot (sampling trajectories)...")
    plot_spatial_coverage_sampled(DATASET_PATH, sample_size=1000)
    
    plot_data_quality_summary(None, stats)
    
    print("\nEDA Analysis Complete!")
    print_statistics(stats)
    
    return stats

def save_statistics(stats, trajectory_lengths, travel_times, speeds, jump_distances):
    """Save statistics to text files for LaTeX inclusion."""
    
    # Note: Network statistics are saved by eda_network.py
    # Do not overwrite them here
    
    # Dataset statistics
    (OUTPUT_DIR / "dataset_total_trips.tex").write_text(f"{stats['total_trips']:,}")
    (OUTPUT_DIR / "dataset_unique_taxis.tex").write_text(f"{stats['unique_taxis']:,}")
    (OUTPUT_DIR / "dataset_total_points.tex").write_text(f"{stats['total_gps_points']:,}")
    (OUTPUT_DIR / "dataset_avg_points.tex").write_text(f"{stats['avg_points_per_trip']:.1f}")
    (OUTPUT_DIR / "dataset_date_range.tex").write_text(stats['date_range'])
    
    # Data quality
    (OUTPUT_DIR / "missing_points_pct.tex").write_text(f"{0:.2f}")  # Placeholder
    (OUTPUT_DIR / "invalid_coords_pct.tex").write_text(f"{stats['invalid_coords_pct']:.2f}")
    (OUTPUT_DIR / "large_jumps_pct.tex").write_text(f"{stats['large_jumps_pct']:.2f}")
    (OUTPUT_DIR / "short_trajectories_pct.tex").write_text(f"{stats['short_trajectories_pct']:.2f}")
    
    # Statistical measures
    if trajectory_lengths:
        (OUTPUT_DIR / "median_trajectory_length.tex").write_text(f"{np.median(trajectory_lengths):.0f}")

def plot_trajectory_length_distribution(df_valid, trajectory_lengths):
    """Plot distribution of trajectory lengths."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Points distribution
    axes[0].hist(df_valid['num_points'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Number of GPS Points per Trajectory')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of GPS Points per Trajectory')
    axes[0].axvline(df_valid['num_points'].median(), color='r', linestyle='--', 
                     label=f'Median: {df_valid["num_points"].median():.0f}')
    axes[0].legend()
    axes[0].set_yscale('log')
    
    # Distance distribution
    if trajectory_lengths:
        axes[1].hist(trajectory_lengths, bins=50, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Trajectory Length (meters)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Trajectory Lengths')
        axes[1].axvline(np.median(trajectory_lengths), color='r', linestyle='--',
                        label=f'Median: {np.median(trajectory_lengths):.0f} m')
        axes[1].legend()
        axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "trajectory_length.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_travel_time_distribution(travel_times):
    """Plot distribution of travel times."""
    if not travel_times:
        return
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to minutes
    travel_times_min = np.array(travel_times) / 60
    
    ax.hist(travel_times_min, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Travel Time (minutes)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Estimated Travel Times')
    ax.axvline(np.median(travel_times_min), color='r', linestyle='--',
               label=f'Median: {np.median(travel_times_min):.1f} min')
    ax.legend()
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "travel_time.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_speed_distribution(speeds):
    """Plot distribution of speeds."""
    if not speeds:
        return
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    speeds = [s for s in speeds if 0 < s < 200]  # Filter outliers
    
    ax.hist(speeds, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Average Speed (km/h)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Average Speeds')
    ax.axvline(np.median(speeds), color='r', linestyle='--',
               label=f'Median: {np.median(speeds):.1f} km/h')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "speed_dist.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_gps_jumps(jump_distances):
    """Plot distribution of GPS jumps."""
    if not jump_distances:
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Full distribution
    axes[0].hist(jump_distances, bins=100, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('GPS Jump Distance (meters)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of GPS Coordinate Jumps')
    axes[0].axvline(1000, color='r', linestyle='--', label='Large Jump Threshold (1000m)')
    axes[0].legend()
    axes[0].set_yscale('log')
    
    # Zoomed in view (0-500m)
    filtered_jumps = [j for j in jump_distances if j < 500]
    axes[1].hist(filtered_jumps, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('GPS Jump Distance (meters)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of GPS Jumps (0-500m)')
    axes[1].axvline(np.median(filtered_jumps), color='r', linestyle='--',
                    label=f'Median: {np.median(filtered_jumps):.1f} m')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gps_jumps.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_spatial_coverage_sampled(dataset_path, sample_size=1000):
    """Plot spatial distribution of GPS points from sampled trajectories."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    print(f"    Sampling {sample_size} trajectories for spatial plot...")
    chunk_size = 100000
    sampled = 0
    
    for chunk in pd.read_csv(dataset_path, chunksize=chunk_size):
        if sampled >= sample_size:
            break
        
        chunk['trajectory'] = chunk['POLYLINE'].apply(parse_polyline)
        chunk['num_points'] = chunk['trajectory'].apply(len)
        chunk_valid = chunk[chunk['num_points'] > 0]
        
        remaining = sample_size - sampled
        sample_chunk = chunk_valid.head(remaining)
        
        for idx, row in sample_chunk.iterrows():
            traj = row['trajectory']
            if len(traj) > 0:
                lons = [p[0] for p in traj if is_valid_coordinate(p[0], p[1])]
                lats = [p[1] for p in traj if is_valid_coordinate(p[0], p[1])]
                if len(lons) > 0:
                    ax.plot(lons, lats, alpha=0.1, linewidth=0.5, color='blue')
        
        sampled += len(sample_chunk)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Spatial Coverage of GPS Trajectories (Sample of {sampled} trips)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "spatial_coverage.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_data_quality_summary(df_valid, stats):
    """Plot summary of data quality issues."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Invalid\nCoordinates', 'Large Jumps\n(>1000m)', 'Short Trajectories\n(<5 points)']
    percentages = [
        stats['invalid_coords_pct'],
        stats['large_jumps_pct'],
        stats['short_trajectories_pct']
    ]
    
    bars = ax.bar(categories, percentages, color=['#ff6b6b', '#ffa500', '#ffd93d'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Data Quality Issues in Porto Dataset')
    max_pct = max(percentages) if percentages else 1
    ax.set_ylim(0, max_pct * 1.2)
    
    # Add value labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.2f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "data_quality.png", dpi=300, bbox_inches='tight')
    plt.close()

def print_statistics(stats):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    for key, value in stats.items():
        print(f"{key:30s}: {value}")

if __name__ == "__main__":
    analyze_dataset()

