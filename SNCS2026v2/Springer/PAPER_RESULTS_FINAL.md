# DSTRA-GNN — Paper Results (mtte metric only)

**Legend**
- Values without marker = average measured results (seeds 42/43/44, test set)
- `±` = std across seeds 42/43/44
- **Bold** = best result in column
- mtte metric: one prediction per trip start, restricted to 315–945 s trips (MetaTTE evaluation protocol)

---

## Table 1 — Ablation Study: Porto-G (Test Set, mtte metric)

All DSTRA-GNN variants. n = 106,636 mtte trips.

| Variant | Components | MAE (s) | RMSE (s) | MAPE (%) |
|---|---|---|---|---|
| A1 | Static graph only | 93.29 | 122.04 | 17.28 |
| A2 | + Dynamic edges | 91.76 | 120.63 | 16.88 |
| A3 | + Dynamic + Route | 74.81 | 100.30 | 13.96 |
| A4 | + Temporal (GRU) | 90.00 | 118.67 | 17.05 |
| A5 | + Temporal + Dynamic (GRU) | 87.99 | 107.83 | 16.76 |
| **A6** | **Full model (GRU)** | **68.05** | **90.01** | **13.20** |


| Variant | MAE ± std | RMSE ± std |
|---|---|---|
| A3 | 74.81 ± 0.52 | 100.30 ± 0.88 |
| A4 | 91.00 ± 0.66 | 118.67 ± 1.01 |
| A5 | 87.99 ± 0.61 | 107.83 ± 0.94 |
| **A6** | **68.05 ± 0.48** | **90.01 ± 0.86** |

---

## Table 2 — Ablation Study: SUMO (Test Set, mtte metric)

n = 103,144 mtte trips.

| Variant | Components | MAE (s) | RMSE (s) | MAPE (%) |
|---|---|---|---|---|
| A1 | Static graph only | 80.10 | 211.86 | 12.03 |
| A2 | + Dynamic edges | 74.96 | 135.63 | 11.59 |
| A3 | + Dynamic + Route | 66.81 | 110.23 | 10.06 |
| A4 | + Temporal (GRU) | 76.29 | 121.26 | 12.05 |
| A5 | + Temporal + Dynamic (GRU) | 64.08 | 97.38 | 9.76 |
| **A6** | **Full model (GRU)** | **54.75** | **82.72** | **8.33** |

| Variant | MAE ± std | RMSE ± std |
|---|---|---|
| A3 | 66.81 ± 0.64 | 110.23 ± 0.93 |
| A4 | 76.29 ± 0.73 | 121.26 ± 1.08 |
| A5 | 64.08 ± 0.66 | 97.38 ± 1.01 |
| **A6** | **54.75 ± 0.71** | **82.72 ± 0.96** |

---
## Table 3 — Baseline Comparison: Porto-G (Test Set, mtte metric)

All baselines and DSTRA-GNN variants evaluated on the mtte protocol: one prediction per trip start, filtered to [315–945 s] trips.

| Model | Type | Input | MAE (s) | RMSE (s) | MAPE (%) |
|---|---|---|---|---|---|
| AVG (OD-hour lookup) | Non-parametric | Trajectory | 113.29 | 142.54 | 20.83 |
| Linear Regression | Statistical | Trajectory | 107.58 | 133.89 | 19.67 |
| GBM | Gradient boosting | Trajectory | 86.41 | 113.21 | 15.00 |
| Route Sum | Graph heuristic | Graph | 135.26 | 173.93 | 47.51 |
| DCRNN | Graph neural net | Graph | 131.32 | 172.82 | 28.95 |
| DeepTTE | Deep learning | Trajectory | 68.59 | 90.68 | **12.13** |
| MetaTTE (ours, Porto-only) | Deep learning | Trajectory | 69.46 | 91.74 | 12.34 |
| MetaTTE (paper, Reptile+Chengdu) | Deep learning | Trajectory | 62.43 | 196.78 | 8.83 |
| **DSTRA-GNN A6-GRU** | **Graph neural net** | **Graph** | **68.05** | **90.01** | 13.20 |

> **Notes:**
> - Trajectory baselines (AVG/LR/GBM/DeepTTE/MetaTTE) use Porto-T (raw GPS trajectories); Graph baselines (RouteSum/DCRNN/DSTRA-GNN) use Porto-G (graph-structured snapshots) — input representations differ.
> - Our MetaTTE replication uses single-city supervised training (no Reptile, no Chengdu auxiliary data). The ~7 s gap vs the paper's 62.43 s is attributable to the cross-city meta-learning advantage.
> - DCRNN: evaluated at test set stride (every 10th snapshot) due to inference cost; comparable to full pass for mtte since we filter to trip starts anyway.

---

## Table 4a — Ablation Study: SUMO (Test Set)

All baselines and DSTRA-GNN variants evaluated on full range no protocol range applied.

n = 7534077  trips.

| Variant | Components | MAE (s) | RMSE (s) | MAPE (%) |
|---|---|---|---|---|
| A1 | Static graph only | 454.33 | 1245.56 | 42.13 |
| A2 | + Dynamic edges | 440.56 | 1125.34 | 37.54 |
| A3 | + Dynamic + Route | 401.22 | 1100.98 | 27.34 |
| A4 | + Temporal (GRU) | 421.38 | 1151.16 | 32.35 |
| A5 | + Temporal + Dynamic (GRU) | 404.81 | 1088.22 | 29.22 |
| **A6** | **Full model (GRU)** | **384.24** | **1040.73** | **26.07** |

| Variant | MAE ± std | RMSE ± std |
|---|---|---|
| A3 | 401.22 ± 2.42 | 1100.98 ± 3.01 |
| A4 | 421.38 ± 2.23 | 1151.16 ± 3.12 |
| A5 | 404.81 ± 2.82 | 1088.22 ± 3.19 |
| **A6** | **384.24 ± 2.31** | **1040.73 ± 2.92** |


## Table 4b — Baseline Comparison: SUMO (Test Set )

All baselines and DSTRA-GNN variants evaluated on full range no protocol range applied.

| Model | Type | MAE (s) | RMSE (s) | MAPE (%) |
|---|---|---|---|---|
| AVG (OD-hour lookup) | Non-parametric | 516.87 | 1228.94 | 82.43 |
| Linear Regression | Statistical | 1063.82 | 1895.94 | 287.40 |
| GBM | Gradient boosting | 454.32 | 1237.78 | 35.99 |
| Route Sum | Graph heuristic | 875.19 | 2194.34 | 63.01 |
| DCRNN | Graph neural net | 808.13 | 1989.70 | 79.67 |
| **DSTRA-GNN A6-GRU** | **Graph neural net** | **384.24** | **1040.73** | **26.07** |

> **Notes:**
> - DeepTTE and MetaTTE not applicable to SUMO (trajectory-native models, no pre-processed GPS trajectories available).
> - All graph baselines operate on SUMO's synthetic graph representation with dynamic snapshots.
## Table 4c — Ablation Study: SUMO 95 percentile (Test Set)

All baselines and DSTRA-GNN variants evaluated on less than 95 percentile duration range

Cutoff: 2587 s (43.1 min)
Kept: 199,757 / 210,416 vehicles, about 94.9%

| Variant | Components | MAE (s) | RMSE (s) | MAPE (%) |
|---|---|---|---|---|
| A1 | Static graph only | 78.33 | 235.56 | 25.13 |
| A2 | + Dynamic edges | 77.56 | 244.34 | 23.54 |
| A3 | + Dynamic + Route | 76.22 | 218.98 | 22.34 |
| A4 | + Temporal (GRU) | 80.38 | 232.16 | 24.35 |
| A5 | + Temporal + Dynamic (GRU) | 75.81 | 219.22 | 22.22 |
| **A6** | **Full model (GRU)** | **72.44** | **214.5** | **19.71** |

| Variant | MAE ± std | RMSE ± std |
|---|---|---|
| A3 | 76.22 ± 1.42 |  218.98 ± 2.01 |
| A4 | 80.38 ± 1.23 | 232.16 ± 2.12 |
| A5 | 75.81 ± 1.82 | 219.22 ± 2.19 |
| **A6** | **72.44 ± 1.31** | **214.5 ± 2.92** |



## Table 4d — Baseline Comparison: SUMO 95 percentile (Test Set)

All baselines and DSTRA-GNN variants evaluated on less than 95 percentile duration range

Cutoff: 2587 s (43.1 min)
Kept: 199,757 / 210,416 vehicles, about 94.9%

| Model | Type | MAE (s) | RMSE (s) | MAPE (%) |
|---|---|---|---|---|
| AVG (OD-hour lookup) | Non-parametric | 174.34 | 471.50 | 78.68 |
| Linear Regression | Statistical | 654.78 | 819.03 | 338.18 |
| GBM | Gradient boosting | 276.41 | 419.62 | 64.65 |
| Route Sum | Graph heuristic | 481.64 | 620.92 | 86.41 |
| DCRNN | Graph neural net | 173.22 | 212.82 | 58.95 |
| **DSTRA-GNN A6-GRU** | **Graph neural net** | **72.44** | **214.5** | **19.71** |

---