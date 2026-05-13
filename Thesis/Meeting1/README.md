# Thesis Progress Report: Porto Dataset Conversion Tool

This directory contains the progress report document and supporting analysis scripts for the Porto dataset conversion tool development.

## Files

- `main.tex` - LaTeX source for the progress report
- `main.pdf` - Compiled PDF document
- `eda_porto.py` - Python script for Porto dataset EDA analysis
- `eda_network.py` - Python script for SUMO network analysis
- `figures/` - Directory containing generated plots and statistics files

## Requirements

### Python Dependencies
```bash
pip install pandas numpy matplotlib seaborn
```

### LaTeX Dependencies
- pdflatex
- Standard LaTeX packages (amsmath, graphicx, booktabs, etc.)

## Usage

### Running EDA Analysis

1. **Analyze SUMO Network:**
   ```bash
   python3 eda_network.py
   ```
   This generates:
   - Network statistics (junctions, edges, lengths, etc.)
   - Network visualization plots
   - Statistics files for LaTeX inclusion

2. **Analyze Porto Dataset:**
   ```bash
   python3 eda_porto.py
   ```
   This generates:
   - Dataset statistics (trips, GPS points, etc.)
   - Trajectory analysis plots
   - Data quality metrics
   - Statistics files for LaTeX inclusion

### Compiling the PDF

After running the EDA scripts, compile the LaTeX document:

```bash
pdflatex main.tex
```

Run twice to resolve all references and cross-references.

## Dataset Paths

The scripts use the following default paths (modify in the scripts if needed):

- **Porto Dataset:** `/home/guy/Projects/Traffic/Multi-Variant-Simulated-Traffic-Dataset-Creator-and-Model-Tester/Porto/dataset/test.csv`
- **SUMO Network:** `/home/guy/Projects/Traffic/Develop/Projects/PortoForSumo/config/porto.net.xml`

## Generated Outputs

### Figures
- `network_visualization.png` - Network structure overview
- `network_edge_lengths.png` - Distribution of edge lengths
- `network_speed_limits.png` - Distribution of speed limits
- `network_lanes.png` - Distribution of number of lanes
- `trajectory_length.png` - Distribution of trajectory lengths
- `travel_time.png` - Distribution of travel times
- `speed_dist.png` - Distribution of speeds
- `gps_jumps.png` - Distribution of GPS coordinate jumps
- `spatial_coverage.png` - Spatial distribution of GPS points
- `data_quality.png` - Summary of data quality issues

### Statistics Files (for LaTeX)
All `.tex` files in the `figures/` directory contain statistics that are automatically included in the document.

## Notes

- The Porto dataset analysis currently uses a sample/test file. For full analysis, update the path in `eda_porto.py`.
- The network visualization may take time to generate due to the large number of edges (80,980 edges).
- GPS jump detection uses a 1000-meter threshold as specified in the requirements.

