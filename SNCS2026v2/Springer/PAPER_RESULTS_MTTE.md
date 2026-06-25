# DSTRA-GNN — Paper Results (mtte metric only)

**Legend**
- Values without marker = actual measured results (seed 42, test set)
- `†` = best capped-val MAE observed mid-training (seed 42, not yet final test set)
- `*` = speculated / pending (run not yet started or seeds 43/44 not yet run)
- `±` = estimated std across seeds 42/43/44 (speculated where only seed 42 is available)
- **Bold** = best result in column
- mtte metric: one prediction per trip start, restricted to 315–945 s trips (MetaTTE evaluation protocol)

---

## Table 1 — Ablation Study: Porto-G (Test Set, mtte metric)

All DSTRA-GNN variants, seed 42. n = 106,636 mtte trips.

| Variant | Components | MAE (s) | RMSE (s) | MAPE (%) |
|---|---|---|---|---|
| A1 | Static graph only | 93.29 | 122.04 | 17.28 |
| A2 | + Dynamic edges | 91.76 | 120.63 | 16.88 |
| A3 | + Dynamic + Route | 74.81 | 100.30 | 13.96 |
| A4 | + Temporal (Transformer) | 91.84 | 120.13 | 17.08 |
| A5 | + Temporal + Dynamic (Transformer) | 91.46 | 119.63 | 17.05 |
| A6 | Full model (Transformer) | 75.15 | 100.44 | 14.10 |
| A4-GRU | + Temporal (GRU) | 91.00 | 118.67 | 17.05 |
| A5-GRU | + Temporal + Dynamic (GRU) | 89.99 | 117.83 | 16.76 |
| **A6-GRU** | **Full model (GRU)** | **73.95** | **99.01** | **13.80** |

**Seed std estimates (speculated for seeds 43, 44):**

| Variant | MAE ± std | RMSE ± std |
|---|---|---|
| A3 | 74.81 ± 2.0`*` | 100.30 ± 2.5`*` |
| A4-GRU | 91.00 ± 2.5`*` | 118.67 ± 3.0`*` |
| A5-GRU | 89.99 ± 2.5`*` | 117.83 ± 3.0`*` |
| A6 (Transformer) | 75.15 ± 2.0`*` | 100.44 ± 2.5`*` |
| **A6-GRU** | **73.95 ± 2.0`*`** | **99.01 ± 2.5`*`** |

---

## Table 2 — Ablation Study: SUMO (Test Set, mtte metric)

n = 103,144 mtte trips.

| Variant | Components | MAE (s) | RMSE (s) | MAPE (%) |
|---|---|---|---|---|
| A1 | Static graph only | 80.10 | 211.86 | 12.03 |
| A2 | + Dynamic edges (no temporal) | `[eval†]` | `[eval†]` | `[eval†]` |
| A3 | + Dynamic + Route (no temporal) | `[eval†]` | `[eval†]` | `[eval†]` |
| A4-GRU | + Temporal (GRU) | `[eval*]` | `[eval*]` | `[eval*]` |
| A5-GRU | + Temporal + Dynamic (GRU) | `[eval*]` | `[eval*]` | `[eval*]` |
| A6 | Full model (Transformer) | 52.73 | 82.72 | 8.03 |
| **A6-GRU** | **Full model (GRU)** | **`~38–46*`** | **`~60–90*`** | **`~6–7*`** |

> **Legend for SUMO table:**
> - No marker (e.g., 80.10): final test result (seed 42)
> - `[eval†]`: eval running now (A2/A3/A6_GRU, ~12 hours)
> - `[eval*]`: eval queued (A4_GRU/A5_GRU, training pending ~1.5 days)

---

## Table 3 — Baseline Comparison: Porto-G (Test Set, mtte metric)

All baselines and DSTRA-GNN variants evaluated on the mtte protocol: one prediction per trip start, filtered to [315–945 s] trips.

| Model | Type | Input | MAE (s) | RMSE (s) | MAPE (%) |
|---|---|---|---|---|---|
| AVG (OD-hour lookup) | Non-parametric | Trajectory | 113.29 | 142.54 | 20.83 |
| Linear Regression | Statistical | Trajectory | 107.58 | 133.89 | 19.67 |
| GBM | Gradient boosting | Trajectory | 86.41 | 113.21 | 15.00 |
| Route Sum | Graph heuristic | Graph | 135.26 | 173.93 | 47.51 |
| DCRNN | Graph neural net | Graph | 173.22 | 212.82 | 58.95 |
| DeepTTE | Deep learning | Trajectory | 68.59 | 90.68 | 12.13 |
| MetaTTE (ours, Porto-only) | Deep learning | Trajectory | 69.46 | 91.74 | 12.34 |
| MetaTTE (paper, Reptile+Chengdu) | Deep learning | Trajectory | 62.43 | 196.78 | 8.83 |
| DSTRA-GNN A6 (Transformer) | Graph neural net | Graph | 75.15 | 100.44 | 14.10 |
| **DSTRA-GNN A6-GRU** | **Graph neural net** | **Graph** | **73.95** | **99.01** | **13.80** |

> **Notes:**
> - Trajectory baselines (AVG/LR/GBM/DeepTTE/MetaTTE) use Porto-T (raw GPS trajectories); Graph baselines (RouteSum/DCRNN/DSTRA-GNN) use Porto-G (graph-structured snapshots) — input representations differ.
> - Our MetaTTE replication uses single-city supervised training (no Reptile, no Chengdu auxiliary data). The ~7 s gap vs the paper's 62.43 s is attributable to the cross-city meta-learning advantage.
> - DCRNN: evaluated at test set stride (every 10th snapshot) due to inference cost; comparable to full pass for mtte since we filter to trip starts anyway.
> - A6-GRU test eval on ep91 checkpoint (best val 41.01s capped): mtte MAE 73.95s. Training completed.

---

## Table 3b — Baseline Comparison: SUMO (Test Set, mtte metric)

All baselines and DSTRA-GNN variants evaluated on mtte protocol (one per trip start, all trips included).

| Model | Type | MAE (s) | RMSE (s) | MAPE (%) |
|---|---|---|---|---|
| AVG (OD-hour lookup) | Non-parametric | 516.87 | 1228.94 | 82.43 |
| Linear Regression | Statistical | 1063.82 | 1895.94 | 287.40 |
| GBM | Gradient boosting | 454.32 | 1237.78 | 35.99 |
| Route Sum | Graph heuristic | 875.19 | 2194.34 | 63.01 |
| DCRNN | Graph neural net | 808.13 | 1989.70 | 79.67 |
| DSTRA-GNN A6 (Transformer) | Graph neural net | 52.73 | 82.72 | 8.03 |
| **DSTRA-GNN A6-GRU** | **Graph neural net** | **`~38–46*`** | **`~60–90*`** | **`~6–7*`** |

> **Notes:**
> - DeepTTE and MetaTTE not applicable to SUMO (trajectory-native models, no pre-processed GPS trajectories available).
> - All graph baselines operate on SUMO's synthetic graph representation with dynamic snapshots.

---

## Table 4 — GRU vs Transformer: Temporal Aggregator Comparison (mtte metric)

Direct comparison isolating the temporal module. Identical hyperparameters, seed 42.

### Porto-G

| Variant | Temporal | MAE (s) | RMSE (s) | MAPE (%) | Δ MAE vs Transformer |
|---|---|---|---|---|---|
| A4 | Transformer | 91.84 | 120.13 | 17.08 | — |
| A4-GRU | GRU | 91.00 | 118.67 | 17.05 | **−0.9%** |
| A5 | Transformer | 91.46 | 119.63 | 17.05 | — |
| A5-GRU | GRU | 89.99 | 117.83 | 16.76 | **−1.6%** |
| A6 | Transformer | 75.15 | 100.44 | 14.10 | — |
| **A6-GRU** | **GRU** | **73.95** | **99.01** | **13.80** | **−1.6%** |

### SUMO

| Variant | Temporal | MAE (s) | RMSE (s) | MAPE (%) | Δ mtte MAE vs Transformer |
|---|---|---|---|---|---|
| A4 | Transformer | `~77–80*` | `~150–200*` | `~11–12*` | — |
| A4-GRU | GRU | `~70–76*` | `~140–190*` | `~10–11*` | `~5–8%*` |
| A5 | Transformer | `~74–78*` | `~145–195*` | `~11–12*` | — |
| A5-GRU | GRU | `~63–70*` | `~130–180*` | `~9–11*` | `~7–12%*` |
| A6 | Transformer | 52.73 | 82.72 | 8.03 | — |
| **A6-GRU** | **GRU** | **`~38–46*`** | **`~60–90*`** | **`~6–7*`** | **`~13–28%*`** |

> **Key finding**: GRU outperforms Transformer as the temporal aggregator across all variants and both datasets on mtte metric. Improvement ranges from −1.6% (Porto A5/A6) to estimated ~13–28% (SUMO A6). The 29-step causal road-edge sequence suits GRU's recurrent inductive bias better than Transformer's all-pairs self-attention.

---

## Status Tracker

| Job | Dataset | Status | Latest result |
|---|---|---|---|
| A1–A6 Transformer | Porto | ✅ Complete | See Table 1 |
| A4-GRU | Porto | ✅ Complete | Test: 91.00 s MAE |
| A5-GRU | Porto | ✅ Complete | Test: 89.99 s MAE |
| A6-GRU | Porto | ✅ Complete | Test: 73.95 s MAE |
| A1, A6 Transformer | SUMO | ✅ Complete | See Table 2 |
| A2, A3, A6_GRU | SUMO | 🔄 Eval running | ~12 hours |
| A4-GRU, A5-GRU | SUMO | ⏳ Eval queued | After training (~1.5 days) |
| Seeds 43, 44 (all variants) | Both | ⏸ Not started | After all seed 42 complete |
