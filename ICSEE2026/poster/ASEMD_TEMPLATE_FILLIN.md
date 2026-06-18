# ICSEE 2026 poster — fill-in guide for `Poster-Template-ASEMD2023-2.ppt`

Use the ASEMD template **as-is** (fonts, colors, layout). Replace placeholder text only.  
Poster size: keep the template slide size; export to PDF at **90 cm × 125 cm** if your printer requires it.

## Typography (do not change)


| Element                           | Font size          | Style                                   |
| --------------------------------- | ------------------ | --------------------------------------- |
| Paper title                       | 30 pt              | Bold                                    |
| Author names                      | 24 pt              | Bold, **one line**                      |
| Affiliation(s)                    | 20 pt              | **One line**                            |
| Author email                      | (template default) |                                         |
| **Abstract** heading              | 22 pt              | Bold                                    |
| Abstract body                     | 17 pt              | Single-spaced                           |
| **Keywords** heading              | 22 pt              | Bold                                    |
| Section headings (II, III, V, VI) | 22 pt              | Bold                                    |
| Body text in sections             | 17 pt              | Justify columns; last column left-align |
| Fig. captions                     | 16 pt              | e.g. `Fig. 1. ...`                      |
| Table heading                     | 16 pt              | e.g. `Table I. ...`                     |
| Acknowledgements                  | 14 pt              | Optional                                |


**Colors:** light-yellow background, **dark green** body text (per template). Use transparent figure backgrounds where possible.

**Font:** Arial (per template).

---

## Header

**Title (30 pt, bold)**  
Integrated Dataset Generation and Testing Platform for Dynamic Graph Neural Network ETA Prediction

**Conference line (replace ASEMD sample text; 20 pt if separate from title, or match template box)**  
2026 International Conference on the Science of Electrical Engineering (ICSEE), Jerusalem, Israel, 10--11 June 2026

*(Official site: [icsee2026.org](https://www.icsee2026.org/) — IEEE Israel Section flagship meeting; technical sessions June 10--11, 2026; welcome reception June 9; wrap-up June 12.)*

**Authors (24 pt, one line)**  
Guy Tordjman, Nadav Voloch

**Affiliation (20 pt, one line)**  
The Open University, Ra'anana, Israel; Ruppin Academic Center, Emek Hefer, Israel

**Email**  
[turgibot@gmail.com](mailto:turgibot@gmail.com); [nadavv@ruppin.ac.il](mailto:nadavv@ruppin.ac.il)

**IEEE logo**  
Replace `IEEE LOG` placeholder with IEEE and/or institutional logos as allowed by ICSEE.

---

## Abstract (22 pt bold heading + 17 pt body)

Accurate Estimated Time of Arrival (ETA) prediction is essential for navigation, traffic management, and smart transportation. Dynamic Graph Neural Networks (GNNs) model evolving road networks, but research is hindered by limited datasets and the lack of a unified environment for data generation and model testing. We present an integrated open-source platform combining **Traffic-DSTG-Gen** (route-aware dynamic spatio-temporal graphs from SUMO simulation or GPS trajectories) and **TrafficLab** (interactive web-based ETA evaluation). The system supports 80,000+ simulation snapshots with 28-dimensional node and 7-dimensional edge features, and logs thousands of real-time test journeys with MAE/RMSE analysis. Both tools are publicly available for reproducible ETA research.

---

## Keywords (22 pt bold heading)

Graph Neural Networks; ETA Prediction; SUMO Simulation; Dataset Generation; Route-Aware Graphs; Testing Platform

---

## II. Principle (22 pt bold)

ETA prediction requires rich spatio-temporal data that links road topology, vehicle motion, and trip labels. Static-sensor and plain trajectory datasets often lack explicit routes and fine-grained features needed by dynamic GNNs.

**Our principle:** one shared **route-conditioned dynamic graph** representation, built along two paths:

1. **Simulation path** — controlled SUMO microsimulation (TraCI), zone/landmark configuration, reproducible snapshots.
2. **Trajectory path** — real GPS traces (e.g. Porto taxi), map matching and conversion to the same graph schema.

Both paths export identical bundles for training and for the same web evaluation client—no silent schema mismatch between dataset creation and testing.

---

## III. Theoretical Modelling (22 pt bold)

**Hybrid graph structure**

- **Static nodes:** junctions (road network substrate).  
- **Dynamic nodes:** vehicles with route and motion state.  
- **Static edges:** roads between junctions.  
- **Dynamic edges:** junction–vehicle–vehicle–junction relations, updated each snapshot.  
- **Features:** 28-D node vectors (speed, position, route progress, destination, …); 7-D edge features; configurable time windows.

**Platform components**


| Component        | Role                                                                                    |
| ---------------- | --------------------------------------------------------------------------------------- |
| Traffic-DSTG-Gen | Desktop tool (PySide6 + SUMO/TraCI): simulation & trajectory conversion, export         |
| TrafficLab       | Web client (Vue.js + FastAPI + PostgreSQL): route selection, inference, journey logging |


**Paste figures**

- **Fig. 1** → `../nodes.png` — Static and dynamic graph layers.  
- **Fig. 2** → `../../Academia/images/dataset_projects.png` — Desktop: 3-zone city + Porto projects.  
- **Fig. 3** → `../../Academia/images/web_sim.png` — Web evaluation dashboard.

**Fig. 1 caption (16 pt)**  
Fig. 1. Static and dynamic graph structure: junction nodes (black), vehicle nodes (green), static roads (grey), dynamic relations (blue dashed).

**Fig. 2 caption**  
Fig. 2. Traffic-DSTG-Gen desktop: simulation-based and trajectory-based project examples.

**Fig. 3 caption**  
Fig. 3. TrafficLab web UI: interactive ETA testing on a SUMO network.

---

## V. Results and Analyses (22 pt bold)

**Dataset generation:** 80,000+ snapshots; multi-zone urban scenarios; route-aware graphs for ETA-oriented learning.

**Model testing (TrafficLab):** logged journeys independent of training data; aggregate MAE, RMSE, MedAE, P90/P95.

**Paste figures**

- **Fig. 4** → `../mae_vs_time_plot_inverted.png` (paper) and/or `../../Academia/images/3onesmae.png` (per-journey errors).

**Table I (16 pt heading)**  
Table I. Web-tool ETA error vs. held-out training evaluation (seconds, except MAPE in %).


| Data source        | *n*   | MAE  | RMSE  | MedAE |
| ------------------ | ----- | ---- | ----- | ----- |
| Simulation         | 2,707 | 46.2 | 107.2 | 31.2  |
| Trajectory (Porto) | 3,055 | 52.4 | 121.5 | 35.5  |


Web-tool MAE is within ~2 s of training evaluation on both paths—confirming end-to-end integration.

**Fig. 4 caption**  
Fig. 4. MAE vs. trip duration and per-journey error distribution from logged web evaluation.

**Optional callout:** Journey #2709 — 13.5 km, predicted 17:11, error 19 s (98.1% accuracy). Screenshot: `../../Academia/images/web3.png`.

---

## VI. Conclusion (22 pt bold)

We introduced an integrated platform for dynamic GNN ETA research: **Traffic-DSTG-Gen** for route-aware dataset generation from simulation or trajectories, and **TrafficLab** for real-time model benchmarking. The workflow from SUMO to logged interactive tests supports reproducible smart-transportation research.

**Open source:**  
[https://github.com/Ruppin-SmartTransportation/Traffic-DSTG-Gen](https://github.com/Ruppin-SmartTransportation/Traffic-DSTG-Gen)  
[https://github.com/Ruppin-SmartTransportation/TrafficLab](https://github.com/Ruppin-SmartTransportation/TrafficLab)  
**Demo:** [https://demo.smart-transport.cloud/](https://demo.smart-transport.cloud/)

**Future work:** federated learning, additional real-world data sources, larger-network cloud deployment.

---

## Acknowledgements (14 pt, optional)

Supported by Ruppin Academic Center and the Israeli Ministry of Innovation, Science and Technology (Proposal no. 0007846). This study was funded by grant number 34836.

---

## Figure file quick reference


| Fig.       | File                                                       |
| ---------- | ---------------------------------------------------------- |
| 1          | `ICSEE2026/nodes.png`                                      |
| 2          | `Academia/images/dataset_projects.png`                     |
| 3          | `Academia/images/web_sim.png`                              |
| 4          | `ICSEE2026/mae_vs_time_plot_inverted.png`                  |
| (optional) | `Academia/images/web3.png`, `Academia/images/3onesmae.png` |


---

## Checklist before printing

- Conference line: ICSEE 2026, Jerusalem, 10--11 June 2026 (replace ASEMD2023 sample text)
- Author line fits **one line** at 24 pt
- Affiliation fits **one line** at 20 pt
- No font size/color changes vs. template
- Figures with transparent background on yellow
- Export PDF at required print size (90 × 125 cm)

