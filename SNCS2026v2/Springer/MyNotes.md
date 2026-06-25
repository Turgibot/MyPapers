1. Introduction
Contains
Why ETA matters
Applications (navigation, logistics, ride-sharing, smart cities)
Challenges of ETA prediction
Limitations of existing methods
Main contributions
End with contributions

For example:

Route-Aware Dynamic Traffic Graph (RAD-TG) representation.
Unified pipeline for simulation and real trajectory conversion.
Dynamic spatio-temporal GNN framework for ETA prediction.
Extensive evaluation demonstrating improved accuracy.
2. Related Work
2.1 ETA Prediction
Historical/statistical methods
DeepTTE
ST-NN
MetaTTE
DuETA
2.2 Graph-Based Traffic Learning
DCRNN
STGCN
Traffic forecasting GNNs
2.3 Traffic Datasets
Porto
T-Drive
Chengdu
Sensor datasets
Gap Summary

Explain that:

Existing ETA methods often use trajectories only.
Existing GNN traffic datasets focus on forecasting.
Few datasets explicitly encode route intent and dynamic interactions.
3. Route-Aware Dynamic Traffic Graph Framework

This is the heart of the paper.

3.1 Graph Representation

Describe:

Nodes:

Junction nodes
Vehicle nodes

Edges:

Road edges
Traversal edges
Interaction edges
Intent edges

Include a figure.

3.2 Temporal Graph Construction

Describe:

Snapshot interval (30 seconds)
Dynamic graph evolution
Time encoding

Include temporal graph illustration.

3.3 Feature Engineering

Node features:

Speed
Acceleration
Vehicle type
Route progress
Destination
Time encoding
etc.

Edge features:

Length
Lanes
Demand
Occupancy
Average speed

Could be a table.

4. Dataset Generation Pipeline

This section explains where the graphs come from.

4.1 Simulation-Based Generation

SUMO
OSM
Traffic demand generation

4.2 Real-Trajectory Conversion

Porto trajectory conversion pipeline

Map matching
Route reconstruction
Graph creation

4.3 Dataset Statistics

Tables:

journeys
snapshots
nodes
edges
duration

Distributions:

Trip duration
Route length
ETA labels
5. ETA Prediction Model

Now explain the actual GNN.

5.1 Problem Formulation

Input:
G1...Gt

Output:
Remaining travel time

5.2 Architecture

Explain:

Route Encoder

↓

Spatial GNN

↓

Temporal Module (GRU/Transformer)

↓

ETA Head

Include architecture figure.

5.3 Training Objective

Loss function

MAE

MSE

Optimization

6. Experimental Setup
6.1 Dataset Splits
Train
Validation
Test

Include percentages.

6.2 Baselines

DeepTTE

ST-NN

MetaTTE

DuETA

Simple MLP

GRU

etc.

6.3 Evaluation Metrics

MAE

RMSE

MAPE

R² (optional)

6.4 Training Configuration

Hardware

Epochs

Batch size

Learning rate

7. Results and Discussion
7.1 Overall Performance

Main results table.

7.2 Ablation Study

Remove:

Route intent edges
Vehicle interaction edges
Temporal encoding
Dynamic snapshots

Show contribution.

This section is extremely important.

7.3 Error Analysis

Performance by:

Trip duration
Congestion level
Route length
7.4 Generalization Analysis

Simulation → real

Different traffic densities

Different city regions

8. Limitations

Be honest:

Single-city simulation
Dependence on map matching
Computational cost
Need for additional real-world validation

Reviewers appreciate this.

9. Conclusion and Future Work

Summarize:

RAD-TG representation
GNN framework
Accuracy improvements

Future:

Multi-city datasets
Online learning
Dynamic rerouting
Shortest-path integration
Robotics and autonomous navigation applications

For your specific thesis work, Sections 3–5 are the novelty. Sections 6–7 are where reviewers decide whether the novelty actually works. If page budget becomes tight, I'd spend most of it on:

Graph representation
GNN architecture
Ablation studies
Error analysis

and keep Related Work relatively compact. Those are the sections that will most clearly distinguish your work from DeepTTE, ST-NN, and MetaTTE.