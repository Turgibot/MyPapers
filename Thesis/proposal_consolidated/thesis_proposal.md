# Thesis Proposal

## Cover Page

**The Open University of Israel**  
Department of Mathematics and Computer Science

**Thesis Proposal**

### Route-Aware ETA Prediction with Dynamic Graph Neural Networks

Thesis proposal is submitted as a partial fulfillment of the requirements towards an M.Sc. degree in Computer Science  
The Open University of Israel  
Department of Mathematics and Computer Science

**By**  
Guy Tordjman

**Prepared under the supervision of**  
Professor Ehud Gudes

**November 2025**

---

## Personal Information

**Student Personal Information**

Full name: Guy Tordjman  
ID: 036238772  
Address: 86 Wietsman St. Kfar Yona, Israel  
Email: turgibot@gmail.com

---

## 1. Work Objective

This thesis aims to improve ETA prediction accuracy by developing a dynamic Graph Neural Network (GNN) architecture that models traffic at the vehicle level with explicit route information. Unlike existing approaches that use static sensor networks or infer routes implicitly, this work proposes a hybrid graph structure combining static infrastructure with dynamic vehicle nodes and time-varying edges. The primary research objective is to investigate whether explicit route information significantly enhances ETA prediction accuracy, comparing the proposed approach against baseline models including trajectory-based (DeepTTE, STAD), static graph (DCRNN, ST-GCN), and route-aware (DuETA) methods.

---

## 2. Background with Reference to Sources

### 2.1 The ETA Prediction Challenge

Estimated Time of Arrival (ETA) prediction is a fundamental problem in intelligent transportation systems, where the goal is to accurately predict the time remaining for a vehicle to reach its destination given its current position, planned route, and the dynamic state of the traffic network. Unlike static shortest-path algorithms that assume constant travel times, ETA prediction must account for evolving traffic conditions, congestion buildup, and vehicle interactions that occur during the journey.

Figure 1 illustrates a key challenge in ETA prediction: traffic conditions are highly dynamic and change continuously as vehicles move through the network. The figure shows two snapshots of the same road network at different times (07:30 AM and 07:40 AM), tracking two vehicles (Car 1 and Car 2) traveling from a city to their homes. At 07:30 AM, Car 1's ETA is predicted to be 07:55 AM based on current traffic conditions. However, the presence of multiple vehicles at point A that are also heading toward point B indicates potential future congestion. Ten minutes later, at 07:40 AM, these vehicles have reached point B, creating significant congestion that increases the travel time from City to B from 15 minutes to 35 minutes. As a result, Car 1's ETA is updated to 08:15 AM—a 20-minute increase from the original prediction.

This example demonstrates the core research challenge addressed in this thesis: **how to accurately predict ETAs by anticipating future traffic conditions and congestion buildup, rather than relying solely on current traffic state**. Traditional navigation systems that base ETA calculations on real-time traffic data fail to account for how traffic will evolve during the journey, leading to inaccurate predictions that must be continuously updated. The research objective is to develop models that can predict future congestion patterns and incorporate this information into ETA calculations, enabling more accurate and stable predictions that account for the dynamic nature of traffic.

*Figure 1: Illustration of the ETA prediction challenge: traffic conditions evolve dynamically, causing ETA predictions to change as congestion builds up. Figure adapted from SmartSimulativeRoute2025.*

### 2.2 Previous Work in Route-Aware Traffic Prediction

This research builds upon previous work in traffic prediction and route-aware modeling. Voloch and Voloch-Bloch demonstrated real-time future traffic estimations for navigation route optimization, establishing the foundation for dynamic traffic prediction. Building on this, Voloch et al. introduced smart simulative route predictions that anticipate future congestion through heuristic simulation, improving ETA accuracy. These works laid the methodological groundwork for graph-based analysis of traffic dynamics and route-aware modeling.

### 2.3 State-of-the-Art ETA Prediction Methods

Machine learning models have been applied to ETA prediction with varying approaches. Tree-based methods such as XGBoost leverage handcrafted features and perform well on aggregated trip records, but cannot capture fine-grained spatio-temporal interactions. Trajectory-based neural models address this gap: DeepTTE learns ETA from raw GPS traces, while STAD adjusts routing-engine outputs with spatio-temporal corrections. Production systems such as STANN evaluate on large ride-hailing ETA datasets.

Graph-based spatio-temporal learning has advanced traffic prediction by representing road networks as graphs. DCRNN combined diffusion graph convolutions with recurrent units and became a common backbone for traffic speed/flow prediction on sensor benchmarks (METR-LA, PEMS-BAY). ST-GCN introduced spatio-temporal graph convolutional networks, while Graph WaveNet advanced deep spatial-temporal graph modeling. Most recently, DuETA introduced duration-aware ETA modeling at Baidu Maps, categorizing trips into short (0–3 km), medium (3–10 km), and long (>10 km) segments, achieving MAEs of 27s, 46s, and 98s respectively. However, DuETA focuses on aggregated road segments rather than individual vehicle trajectories with explicit route intent.

### 2.4 Limitations of Existing Approaches

Existing benchmarks such as NYC, Porto, Chengdu/DiDi, and Geolife lack explicit representation of pre-planned routes. Models must infer likely paths between origin and destination, introducing ambiguity and reducing accuracy. Most ETA prediction methods either use static sensor networks (e.g., DCRNN, ST-GCN) that aggregate traffic at fixed locations, or rely on trajectory-based learning (e.g., DeepTTE, STAD) that infers routes implicitly from origin-destination pairs.

Beyond methodological limitations, existing datasets present significant constraints. Public datasets such as NYC Taxi and Porto Taxi provide trip-level records but lack explicit route information and graph structures. Proprietary datasets used by production systems, such as the Baidu Maps dataset employed by DuETA, are not publicly accessible, preventing independent validation and fair comparison. The absence of a unified dataset that supports all model types creates a fundamental challenge: trajectory-based models (DeepTTE, STAD) require GPS trajectory sequences, static graph models (DCRNN, ST-GCN) require aggregated sensor network data, and route-aware models (DuETA) require route segment information. No existing public dataset provides the combination of explicit routes, dynamic graph structure, and vehicle-level data required for evaluating all model types simultaneously. This limitation necessitates the creation of a simulated dataset with multiple variants, where each variant is tailored to a specific model's requirements while maintaining the same underlying traffic simulation, enabling fair comparison across all models.

These limitations highlight the need for approaches that: (1) model traffic at the vehicle level rather than aggregated sensor networks, (2) explicitly encode predefined routes rather than inferring them implicitly, (3) combine dynamic graph structures with temporal learning to capture evolving traffic conditions, and (4) provide a unified evaluation framework through a multi-variant dataset generated from the same underlying traffic simulation.

---

## 3. Work Description

### 3.1 What Has Already Been Done

The DSTRA-GNN architecture has been implemented and evaluated on a four-week simulated SUMO dataset, demonstrating promising results. The model achieved a mean absolute error (MAE) of 46.2 seconds, representing an 82.3% improvement over the average baseline (260.6 seconds MAE). These initial results provide evidence for the effectiveness of the route-aware dynamic graph neural network approach for ETA prediction.

The current implementation includes:
- Complete DSTRA-GNN architecture with dynamic graph representation, route encoding, and temporal learning mechanisms
- Evaluation framework tested on simulated SUMO traffic data
- Initial comparison showing significant improvement over average baseline models
- Four-week simulated dataset with dynamic graph snapshots at 30-second intervals

While these results demonstrate the potential of the approach, they are based on simulated data. The next phase of this research will focus on validating the model's effectiveness on real-world data and establishing fair comparison with state-of-the-art baseline methods.

### 3.2 Planned Research Work

The planned research work is organized into four main phases, building upon the foundation established in the initial implementation:

#### Phase 1: Multi-Variant Dataset Generation (Precondition)

Before conducting comprehensive model evaluation, a multi-variant dataset generation framework must be developed. This is a prerequisite for enabling fair comparison across different model architectures. The framework will generate four synchronized variants from a single underlying traffic simulation: (1) dynamic graph variant for DSTRA-GNN, (2) trajectory variant for trajectory-based models, (3) static graph variant for static sensor network models, and (4) route segment variant for route-aware models. This precondition work is estimated to require approximately two months and will ensure that all models are evaluated on identical traffic patterns while receiving data in their native format.

#### Phase 2: Real Data Enhancement and Ground Truth Establishment

To validate the DSTRA-GNN model on real-world data, existing benchmark datasets (such as NYC Taxi, Porto Taxi, or similar publicly available traffic datasets) will be enhanced with the additional features required by the DSTRA-GNN architecture. These datasets typically provide trip-level records with origin, destination, timestamps, and travel times, but lack explicit route information, dynamic graph structures, and vehicle-level interactions that DSTRA-GNN requires.

The enhancement process will add missing features in a statistically valid manner by introducing controlled randomness and normal distributions that preserve the statistical properties of the original real data. This approach is critical for several reasons: (1) it maintains the ground truth travel times from real-world observations, ensuring that model evaluation reflects actual traffic conditions rather than purely simulated scenarios, (2) it preserves the statistical distribution and temporal patterns of the original dataset, maintaining realism while adding necessary structural information, (3) it enables fair comparison with baseline models that can use the original dataset features, while DSTRA-GNN benefits from the enhanced features, and (4) it provides a common evaluation framework where all models are tested on the same underlying real-world traffic patterns, with ground truth established from actual observed travel times.

The ground truth for evaluation will be the actual travel times recorded in the original real-world dataset. This ensures that performance metrics reflect the model's ability to predict real traffic conditions, rather than simulated approximations.

#### Phase 3: Architecture Enhancements

Building upon the initial DSTRA-GNN implementation, this phase will explore architectural improvements to further enhance ETA prediction accuracy. The enhancements will focus on refining the model's ability to capture spatio-temporal traffic dynamics and route-aware patterns. Specific improvements will be determined through systematic analysis of the model's performance on the enhanced real-world dataset, identifying areas where the architecture can be strengthened.

#### Phase 4: Comprehensive Baseline Comparison and Validation

Once the enhanced real-world dataset is available, comprehensive evaluation will be conducted comparing DSTRA-GNN against state-of-the-art baseline methods. This phase will:

- Implement and evaluate baseline models (DeepTTE, STAD, DCRNN, ST-GCN, DuETA) on the enhanced dataset
- Establish performance benchmarks using standard metrics (MAE, RMSE, MAPE)
- Compare DSTRA-GNN performance against all baseline methods on the same real-world enhanced dataset
- Conduct ablation studies to quantify the contribution of different architectural components
- Analyze performance across different trip characteristics and traffic conditions

This comprehensive evaluation will demonstrate whether the improvements observed on simulated data translate to real-world scenarios, providing stronger evidence for the effectiveness of the route-aware dynamic graph neural network approach.

### 3.3 Expected Deliverables

The work will produce: (1) an enhanced DSTRA-GNN architecture validated on real-world data, (2) a multi-variant dataset generation framework enabling fair comparison across model architectures, (3) a real-world dataset enhancement tool that adds necessary features while preserving statistical validity, (4) comprehensive experimental results comparing DSTRA-GNN against state-of-the-art baselines on real-world data, (5) detailed ablation studies quantifying architectural contributions, and (6) analysis demonstrating the effectiveness of route-aware dynamic graph modeling for ETA prediction in real-world scenarios.

---

## 4. Work Importance

This thesis addresses fundamental limitations in current ETA prediction research by developing a comprehensive evaluation framework and validating route-aware dynamic graph neural networks on real-world data.

**Research Community Impact:** Current ETA prediction research suffers from fragmentation, where different model architectures are evaluated on incompatible datasets, making fair comparison impossible. This thesis addresses this gap by developing a multi-variant dataset generation framework that enables rigorous comparison across architectural paradigms (trajectory-based, static graph, dynamic graph, route-aware) on identical traffic patterns, providing the research community with a standardized evaluation framework.

**Theoretical and Methodological Contribution:** Initial results from the DSTRA-GNN model (detailed in Appendix A) on simulated data demonstrate that explicit route information combined with vehicle-level dynamic graph modeling significantly improves ETA prediction accuracy (46.2 s MAE, 82.3% improvement over average baseline). This thesis will provide the first comprehensive validation of route-aware dynamic graph neural networks on enhanced real-world datasets, establishing whether theoretical advantages translate to practical improvements. The real-world dataset enhancement approach enables evaluation of advanced models on actual traffic conditions while preserving ground truth travel times, establishing a new paradigm that combines real data realism with structural richness required by modern architectures.

**Practical Value:** Accurate ETA prediction directly impacts navigation systems, ride-sharing platforms, fleet management, and logistics operations. By validating route-aware dynamic graph models on real-world data, this research provides evidence-based guidance for deploying advanced ETA prediction systems in production environments, with potential benefits for millions of daily users of transportation services.

---

## 5. Methods for Work Execution

The work will involve literature review and analysis, software development and programming (implementing baseline models and extending DSTRA-GNN), experimental design and execution (baseline comparisons, ablation studies, architectural enhancements), data analysis and interpretation, and documentation and writing.

**Data Collection and Preprocessing:** For validation on real-world data, existing benchmark datasets (NYC Taxi, Porto Taxi, or similar) will be enhanced with additional features required by DSTRA-GNN. The enhancement process will add missing features (explicit routes, dynamic graph structures, vehicle-level interactions) in a statistically valid manner using controlled randomness and normal distributions that preserve the statistical properties of the original real data. Data preprocessing will include chronological partitioning, feature normalization, temporal window construction (H=30 snapshots), route sequence encoding, and target normalization.

**Analysis Techniques:** Performance evaluation using standard regression metrics (MAE, RMSE, MAPE) computed per-vehicle and aggregated across trip categories and traffic conditions. Ablation studies will systematically compare route-aware vs. route-agnostic variants, temporal vs. non-temporal variants, dynamic vs. static graph structures, and different route encoding mechanisms. Statistical analysis will include confidence intervals, significance testing, and effect size calculations. Error analysis will identify failure modes across different scenarios.

**Validation Methods:** All experiments will follow rigorous validation protocols including chronological data partitioning, hyperparameter tuning with early stopping, multiple runs with different random seeds (42, 43, 44) for reproducibility, and model selection based on lowest validation MAE.

**Ground Truth and Evaluation:** The ground truth for evaluation will be the actual travel times recorded in the original real-world dataset. This ensures that performance metrics reflect the model's ability to predict real traffic conditions, rather than simulated approximations. The enhanced dataset maintains the original real-world travel times as ground truth while adding necessary structural features for DSTRA-GNN. This approach enables fair comparison with baseline models that can use the original dataset features, while DSTRA-GNN benefits from the enhanced features, all evaluated against the same real-world ground truth.

**Baseline Comparison:** Comprehensive baseline comparison will be conducted on the enhanced real-world dataset, ensuring all models (DSTRA-GNN, DeepTTE, STAD, DCRNN, ST-GCN, DuETA) are evaluated on identical traffic patterns with the same ground truth. All code and configurations will be documented for reproducibility.

---

## 6. Work Execution Timeline

The work will be executed over ten months, including a two-month precondition phase for multi-variant dataset generation, followed by four main research phases aligned with the work description.

**Months 1-2: Precondition - Multi-Variant Dataset Generation** - Development of the multi-variant dataset generation framework. Implementation of tools to extract four synchronized variants (dynamic graph, trajectory, static graph, route segment) from a single SUMO simulation. Framework testing and validation. This precondition work is essential for enabling fair comparison across different model architectures.

**Month 3: Phase 1 - Real Data Enhancement Tool Development** - Development of the real-world dataset enhancement tool. Design and implementation of statistical methods to add missing features (explicit routes, dynamic graph structures, vehicle-level interactions) to existing benchmark datasets while preserving statistical validity. Tool testing on sample datasets (NYC Taxi, Porto Taxi).

**Month 4: Phase 1 Completion and Phase 2 Initiation** - Completion of real data enhancement tool. Generation of enhanced real-world dataset with ground truth travel times preserved. Begin Phase 2: Architecture enhancement analysis and design.

**Month 5: Phase 2 - Architecture Enhancements** - Systematic analysis of DSTRA-GNN performance on enhanced real-world dataset. Identification of areas for architectural improvement. Design and implementation of enhancements focusing on spatio-temporal traffic dynamics and route-aware patterns.

**Month 6: Phase 2 Completion and Phase 3 Initiation** - Completion of architectural enhancements. Training and initial evaluation of enhanced DSTRA-GNN model. Begin Phase 3: Baseline model implementation and evaluation framework setup.

**Month 7: Phase 3 - Baseline Model Implementation and Evaluation** - Implementation of baseline models (DeepTTE, STAD, DCRNN, ST-GCN, DuETA). Evaluation of all baseline models on the enhanced real-world dataset. Establishment of performance benchmarks using standard metrics (MAE, RMSE, MAPE).

**Month 8: Phase 3 Completion and Phase 4 Initiation** - Completion of baseline evaluations. Begin Phase 4: Comprehensive comparison of DSTRA-GNN against all baseline methods on the enhanced dataset. Ablation study design and initial experiments.

**Month 9: Phase 4 - Comprehensive Evaluation and Analysis** - Comprehensive evaluation comparing DSTRA-GNN against all baseline methods. Ablation studies quantifying architectural contributions. Performance analysis across different trip characteristics and traffic conditions. Error analysis and visualization of learned representations. Statistical analysis and synthesis of experimental results.

**Month 10: Thesis Writing and Finalization** - Writing thesis chapters. Preparation of figures, tables, and visualizations. Final review, editing, proofreading, and formatting. Final submission preparation.

---

## 7. Initial Source List

The proposal body references sources throughout the text using standard citation format. All sources are managed using BibTeX and are automatically generated from the references.bib file. The complete bibliography is provided in the following section.

---

## Appendix A: Previous Work

### Dynamic Route-Aware Graph Neural Networks for Accurate ETA Prediction

This appendix presents the paper "Dynamic Route-Aware Graph Neural Networks for Accurate ETA Prediction" by Tordjman and Voloch, which serves as the foundation for this thesis research. The paper was submitted to IEEE Transactions on Intelligent Transportation Systems. The full paper is included in the DSTRA-GNN project directory.

---

*End of Thesis Proposal*

