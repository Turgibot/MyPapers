# Multi-Variant Dataset Generation Project Summary

## Overview

This document summarizes the changes made to the Advanced Project paper and outlines the requirements and goals for developing a multi-variant dataset generation framework that enables fair comparison across different model architectures for both flow forecasting and ETA prediction.

## Changes Made to Advanced Project Paper

### 1. Abstract
- **Changed**: Rewritten to emphasize multi-variant dataset generation framework
- **Key Points**:
  - Addresses critical gap in traffic prediction research
  - Enables fair comparison across different model architectures
  - Supports both flow forecasting and ETA prediction
  - Provides standardized benchmark for research community

### 2. Introduction
- **Problem Statement**: Updated to highlight lack of unified datasets preventing fair model comparison
- **Motivation**: Emphasizes multi-variant dataset generator as research contribution
- **Objectives**: Added goals for multi-variant dataset generation
- **Deliverables**: Includes multi-variant dataset as research contribution

### 3. Literature Review (Section 2)
- **Added**: Model-Specific Data Requirements subsection
- **Added**: Multi-Variant Dataset Requirements subsection
- **Updated**: Comparative summary table with route information column
- **Key Content**: Details on what each variant must provide

### 4. System Design (Section 3)
- **Updated**: SUMO Service to include multi-variant extraction capabilities
- **Added**: Functionality for extracting all four variants

### 5. Implementation (Section 4)
- **Rewritten**: Dataset generation section (TrafficLab section unchanged)
- **Added**: Detailed description of all four variants
- **Added**: Synchronization and consistency requirements
- **Added**: Multi-variant extraction tools

### 6. Results & Achievements (Section 5)
- **Updated**: Emphasizes multi-variant dataset as primary achievement
- **Added**: Research contribution section highlighting standardized benchmark
- **Updated**: Impact section to focus on research community benefits

### 7. Conclusion (Section 6)
- **Updated**: Emphasizes multi-variant dataset contribution
- **Added**: Standardized benchmark for research community
- **Updated**: Future work includes dataset expansion

## Project Requirements

### Core Requirements

#### 1. Multi-Variant Dataset Generation
The system must generate **four synchronized variants** from a single SUMO simulation:

##### Variant 1: Dynamic Graph
- **Purpose**: For DSTRA-GNN and similar dynamic GNN models
- **Format**: PyTorch Geometric .pt files with temporal sequences
- **Structure**:
  - Static junction nodes with spatial and topological features
  - Dynamic vehicle nodes with position, speed, acceleration, route information
  - Time-varying edges creating junction-vehicle-vehicle-junction relations
  - Explicit route encoding with complete remaining route per vehicle
- **Node Features**: 28 dimensions
  - Junction Nodes: Zone, position, type, traffic state
  - Vehicle Nodes: Speed, acceleration, position, route progress, destination, route_left information
- **Edge Features**: 7 dimensions
  - Static Edges: Length, lanes, speed limits
  - Dynamic Edges: Current speed, demand, occupancy
- **Labels**: 
  - Flow: Speed, volume, occupancy at junctions
  - ETA: Per-vehicle travel times
- **Temporal**: Sequences of graph snapshots at regular intervals (default: 30 seconds)

##### Variant 2: Trajectory
- **Purpose**: For DeepTTE, STAD, and similar trajectory-based models
- **Format**: CSV/JSON files with trajectory sequences
- **Structure**:
  - GPS trajectory sequences: Ordered lists of (latitude, longitude, timestamp) tuples per trip
  - Trip-level metadata: Origin, destination, start time, duration
- **Labels**: ETA (per-trip travel times)
- **Note**: No explicit route or graph structure information

##### Variant 3: Static Graph
- **Purpose**: For DCRNN, ST-GCN, and similar static sensor network models
- **Format**: PyTorch Geometric .pt files with static graph and time-series features
- **Structure**:
  - Aggregated sensor readings at fixed locations (junctions or road segments)
  - Static road network topology with fixed node and edge structure
  - Time-series of aggregated measurements at each sensor
- **Features**: 
  - Speed (average vehicle speed at sensor)
  - Flow/Volume (vehicles per hour at sensor)
  - Occupancy (percentage of time sensor is occupied)
- **Labels**: Flow prediction (speed, volume, occupancy at fixed locations)
- **Note**: No individual vehicle trajectories or dynamic relationships

##### Variant 4: Route Segment
- **Purpose**: For DuETA and similar route-aware models
- **Format**: CSV/JSON files with route sequences and segment features
- **Structure**:
  - Trip-level records with explicit route sequences
  - Route represented as ordered list of road segments or waypoints
  - Segment-level aggregated features: Length, speed limit, historical travel time
- **Labels**: ETA (per-trip travel times) with duration-aware categorization
- **Trip Metadata**: Duration-aware categorization (short, medium, long trips)

#### 2. Synchronization Requirements
All variants must share:
- **Same underlying traffic simulation**: Ensures identical traffic patterns
- **Consistent train/validation/test splits**: Chronological partitioning (e.g., 2 weeks train, 1 week validation, 1 week test)
- **Identical trip identifiers**: Enables cross-variant analysis
- **Synchronized timestamps**: For temporal alignment across variants
- **Same traffic patterns and scenarios**: Across all variants

#### 3. Label Generation
The system must generate labels for both prediction tasks:

##### Flow Prediction Labels
- **Speed**: Average vehicle speed at each sensor location (junctions/road segments)
- **Volume/Flow**: Number of vehicles per hour at each sensor
- **Occupancy**: Percentage of time sensor location is occupied
- **Temporal**: Predictions for future time steps (e.g., 5, 10, 15, 30, 60 minutes ahead)

##### ETA Prediction Labels
- **Per-vehicle travel times**: Time remaining for each vehicle to reach destination
- **Per-trip travel times**: Total travel time for each trip
- **Ground truth**: Actual arrival times from simulation

#### 4. Data Processing Pipeline
The system must implement a six-stage pipeline:

1. **Simulation Stage**: Generate traffic snapshots using SUMO with configurable scenarios
2. **Multi-Variant Extraction Stage**: Extract four synchronized variants from same simulation
3. **Labeling Stage**: Generate per-snapshot labels for both flow and ETA prediction
4. **EDA Stage**: Generate feature statistics and analysis for each variant
5. **Conversion Stage**: Convert each variant to its native format
6. **Validation Stage**: Verify dataset integrity, temporal synchronization, and consistent splits

## Project Goals

### Primary Goals

1. **Enable Fair Model Comparison**
   - Provide synchronized variants from single simulation
   - Ensure all models evaluated on identical traffic patterns
   - Eliminate dataset bias in performance comparisons

2. **Support Multiple Model Architectures**
   - Trajectory-based models (DeepTTE, STAD)
   - Static graph models (DCRNN, ST-GCN)
   - Route-aware models (DuETA)
   - Dynamic graph models (DSTRA-GNN)

3. **Support Both Prediction Tasks**
   - Flow forecasting: Aggregate speed, volume, occupancy at fixed locations
   - ETA prediction: Individual vehicle travel times

4. **Provide Standardized Benchmark**
   - Consistent train/validation/test splits
   - Reproducible evaluation framework
   - Publicly accessible dataset for research community

### Technical Goals

1. **Efficient Extraction**
   - Extract all variants simultaneously from single simulation
   - Minimize computational overhead
   - Maintain temporal synchronization

2. **Data Quality**
   - Validate dataset integrity
   - Ensure consistency across variants
   - Verify label accuracy

3. **Scalability**
   - Support long simulation periods (e.g., 4 weeks)
   - Handle large numbers of vehicles and sensors
   - Efficient storage and processing

### Research Contribution Goals

1. **Address Research Gap**
   - Fill gap in unified dataset availability
   - Enable fair comparison across model architectures
   - Support both flow and ETA prediction from same source

2. **Reproducibility**
   - Provide consistent evaluation framework
   - Enable reproducible research
   - Standardize comparison methodology

3. **Community Resource**
   - Publicly accessible dataset
   - Open-source generation tools
   - Documentation and usage examples

## Implementation Notes

### Key Components to Implement

1. **Simulation Manager** (`graph/entities.py`)
   - Multi-variant extraction coordination
   - Temporal synchronization management
   - State tracking across variants

2. **Variant Extractors**
   - `trajectory_extractor.py`: GPS trajectory sequences
   - `static_graph_aggregator.py`: Sensor readings aggregation
   - `route_segment_extractor.py`: Route segment extraction
   - `dataset_creator.py`: Dynamic graph conversion (existing)

3. **Label Generators**
   - `create_labels_json.py`: Flow and ETA labels
   - Flow labels: Speed, volume, occupancy at sensors
   - ETA labels: Per-vehicle and per-trip travel times

4. **Validation Tools**
   - `synchronization_validator.py`: Temporal synchronization verification
   - Split consistency validation
   - Cross-variant consistency checks

5. **EDA Tools**
   - `EDA.py`: Feature statistics for each variant
   - Variant-specific analysis
   - Cross-variant comparison

### SUMO Integration Requirements

- **TraCI Integration**: Real-time simulation control
- **Vehicle Tracking**: Extract positions, speeds, routes for all variants
- **Sensor Aggregation**: Aggregate data at fixed locations for static graph variant
- **Route Extraction**: Extract route sequences for route segment variant
- **Temporal Synchronization**: Ensure all variants extracted at same time steps

### Output Formats

- **Dynamic Graph**: PyTorch Geometric .pt files
- **Trajectory**: CSV/JSON files
- **Static Graph**: PyTorch Geometric .pt files
- **Route Segment**: CSV/JSON files

## Success Criteria

1. **Functional Requirements**
   - All four variants generated successfully
   - Labels generated for both flow and ETA prediction
   - Temporal synchronization maintained
   - Consistent train/validation/test splits

2. **Quality Requirements**
   - Dataset integrity validated
   - Cross-variant consistency verified
   - Label accuracy confirmed

3. **Research Impact**
   - Enables fair comparison across model architectures
   - Supports both flow and ETA prediction tasks
   - Provides standardized benchmark for research community

## References

- **Paper Location**: `/home/guy/Projects/Traffic/MyPapers/22997/Advanced_Project/`
- **Dataset Generator**: `22997/Traffic-DSTG-Gen/`
- **Related Work**: Thesis proposal sections on multi-variant dataset requirements

## Next Steps

1. Implement multi-variant extraction logic in SUMO integration
2. Develop variant-specific extractors
3. Implement flow label generation (speed, volume, occupancy at sensors)
4. Implement synchronization validation
5. Test with sample simulation
6. Generate full dataset
7. Validate all variants and labels
8. Prepare dataset for public release




