Dear Editor and Reviewers,

We thank the editor and reviewers for their careful reading of our manuscript and for the constructive comments. We have substantially revised the manuscript to address the concerns about real-world validation, baseline strength, evaluation protocol clarity, leakage risks, reproducibility, and the balance of claims. Below we provide a point-by-point response.

# Response to Reviewer #1

## Comment 1: Real-world validation beyond SUMO

The reviewer noted that the original evaluation relied entirely on a SUMO-generated simulation dataset and requested stronger justification of simulation realism or additional experiments on public real-world data.

**Response:** We agree. The revised manuscript now includes a real-world validation track based on the Porto taxi trajectory dataset from the ECML-PKDD 2015 challenge. We construct two aligned Porto views: Porto-T, a filtered trajectory-native dataset used for trajectory baselines, and Porto-G, a duration-matched graph-converted representation used by graph-native models. The revised Methodology describes the map-matching, route reconstruction, graph conversion, snapshot aggregation, and duration-matching process. The Experimental Evaluation reports dataset statistics and distribution plots for SUMO, Porto-T, and Porto-G. The Discussion and Limitations now explicitly state that Porto-G covers one taxi fleet in one city, is not a trip-identical copy of Porto-T, and uses reconstructed routes rather than real navigation logs.

## Comment 2: Limited comparison with state-of-the-art methods

The reviewer noted that the original experimental comparison relied mainly on internal ablations and a simple average baseline, without direct implementation-based comparisons to established methods.

**Response:** We have expanded the evaluation with a stronger baseline ladder. For Porto, the revised manuscript compares against AVG, Linear Regression, GBM, TEMP, DeepTTE, MetaTTE, WDR, STNN, Route-sum, and DCRNN, with input modality clearly labeled as Porto-T or Porto-G. For SUMO, we added a compact p95 external-baseline comparison against AVG, LR, GBM, Route-sum, and DCRNN. The text now distinguishes between graph-native baselines on SUMO/Porto-G and trajectory-native baselines on Porto-T, avoiding claims that all baselines operate on identical inputs.

## Comment 3: Methodology clarity and motivation

The reviewer requested clearer explanation and intuitive justification for temporal aggregation, route encoding, and the Mixture-of-Experts configuration.

**Response:** We revised the Methodology to improve the intuitive explanation of the architecture. The route feature is now defined as `vehicle_route_left`, the ordered sequence of remaining edge IDs, and the route encoder is described as edge-ID embedding followed by mean pooling. The temporal module is now described as a GRU over persistent static-road edge sequences, mean-pooled to a graph-level context because road edges persist across the window whereas vehicle/interaction edges are transient. The MoE justification has also been refined: the six-expert configuration is now described as a shared default selected because it performs best on SUMO while remaining very close to the best Porto-G configuration, avoiding per-dataset expert-count tuning.

## Comment 4: Overstated cross-study improvement claims

The reviewer noted that the original paper claimed large improvements relative to existing approaches despite differing datasets, task definitions, and evaluation contexts.

**Response:** We agree and have softened the positioning. The revised manuscript no longer presents cross-paper percentage improvements as direct evidence of superiority. Instead, it reports direct same-paper experiments under the revised evaluation protocol. The abstract now says that DSTRA-GNN performs competitively with strong trajectory-based baselines, rather than claiming to match or exceed them decisively. The Discussion explicitly notes that headline numbers from related systems are not directly comparable across cities, label definitions, and horizons.

## Comment 5: Reproducibility details

The reviewer requested additional details about hyperparameters, computational requirements, training stability, and implementation transparency.

**Response:** The revised manuscript now includes a clearer training protocol, target normalization, optimizer settings, seeds, batch size, training schedule, compute hardware, training time, inference latency, and code/data availability information. The Declarations section includes dataset availability and an exact code commit for the DSTRA-GNN model, Porto-G conversion pipeline, training scripts, and retrained baselines. We also added a MoE expert-count sensitivity table and clarified that validation is used only for model selection while final metrics are reported on the held-out test split.

## Comment 6: Language refinement and balance

The reviewer requested language refinement, reduced repetition, and a more balanced discussion of limitations.

**Response:** We revised the abstract, introduction, results, discussion, and conclusion to reduce overstatement and clarify limitations. The revised Discussion now explicitly addresses Porto-G vs Porto-T differences, route provenance differences across SUMO and Porto-G, the lack of observed route re-planning events in Porto-G, sensing/observability assumptions, fairness considerations, and the limits of drawing broad density-dependent conclusions from one real-world city.

# Response to Reviewer #2

## Major comment 1: Baselines are not competitive enough

The reviewer requested stronger classical, tabular, sequence, and graph baselines, including route-sum and graph-native comparisons.

**Response:** We have substantially expanded the baseline evaluation. The revised Porto baseline table includes classical/statistical, historical segment-speed, trajectory-native neural, graph heuristic, graph neural, and DSTRA-GNN rows. The revised SUMO p95 table adds AVG, LR, GBM, Route-sum, and DCRNN baselines against the full DSTRA-GNN model. Route-sum uses per-edge historical speeds from the training split, and DCRNN is adapted to the road-edge line graph. We also clarified the baseline input modality for each method so that Porto-T trajectory baselines and Porto-G graph-native baselines are not presented as trip-identical evaluations.

## Major comment 2: Validation vs test reporting

The reviewer noted inconsistent validation/test wording and requested dedicated test-set reporting, split boundaries, and seed variability.

**Response:** We corrected this throughout the manuscript. The revised Evaluation Protocol states that validation is used only for early stopping and checkpoint selection, while all headline numbers are computed on the held-out test split. For graph windows, the first `H-1` snapshots of each split are discarded so that windows do not cross split boundaries. A3-A6 variants report mean ± standard deviation across seeds 42, 43, and 44 for MAE and RMSE; A1-A2 are explicitly marked as seed-42 only. We also clarified that protocol-range metrics use one trip-start prediction, whereas SUMO p95 metrics are retained-row metrics over vehicles whose trip-start duration falls below the p95 cutoff.

## Major comment 3: Feature availability and leakage risks

The reviewer asked how route construction and demand/count features avoid leakage.

**Response:** We added a dedicated Leakage Considerations subsection. For SUMO, routes are computed using TraCI Dijkstra with static edge costs based on length divided by speed limit, not historical or future traffic speed. For Porto-G, the route is the map-matched path of the vehicle's observed GPS trajectory. The former `edge_route_count` concern corresponds to the current `edge_demand` feature, which is computed only from vehicles already active in the current snapshot and their already-fixed remaining routes. It is therefore a current observed-demand feature, not a corpus-wide future route count.

## Major comment 4: Missing design details and MAPE estimation

The reviewer asked for a definition of route-left inputs and for MAPE to be recomputed directly rather than estimated.

**Response:** The revised manuscript defines `vehicle_route_left` as the ordered sequence of remaining edge IDs along the vehicle's path from the current edge to the destination. We removed the ambiguous “route left splits” wording. The Evaluation Metrics subsection now states that MAPE is computed directly from stored predictions and ground-truth labels, not estimated. We also clarify that MAPE is reported as a point estimate because it is secondary and less stable for small ETA denominators, while seed variability is reported for MAE and RMSE.

## Major comment 5: Dynamic-edge contribution is not fully isolated

The reviewer noted that dynamic edges alone were not convincingly established and requested additional ablations and diagnostics.

**Response:** We revised the discussion to frame this point more carefully. The updated ablations show that dynamic edges alone provide smaller gains than route features, and the manuscript no longer overstates dynamic edges as the sole driver. We explicitly acknowledge that the current ablation grid isolates route gains when route features are added to dynamic-graph variants, but it does not include a route-only static-graph variant. We therefore interpret the route gains as evidence for explicit route intent within the tested dynamic-graph family rather than as a complete factorial decomposition of route and dynamic-edge effects. We also leave density-stratified dynamic-edge diagnostics and relation-specific modeling as future work.

## Major comment 6: References

The reviewer noted incorrect DOI/URL entries, DiDi citation ambiguity, and a missing DOI for the foundation paper.

**Response:** We revised the bibliography and recompiled the manuscript. We added the missing DOI for the foundation paper (`10.1007/978-3-032-06164-5_1`), corrected a malformed BibTeX entry, and reviewed the cited DOI/URL fields. The DiDi reference is cited only as an example of trajectory-based ETA data rather than as a road-graph/route-aware benchmark.

# Minor comments and technical questions

## Duration bins and route-length terminology

**Response:** We clarified the duration-bin figure caption. The caption now lists the exact bins used by the plotting script: `<315`, `315-525`, `525-735`, `735-945`, `946-1200`, `1201-1800`, and `1801-2587` seconds, with the three interior bins splitting the shared `[315,945]` second protocol band.

## Standard deviations across seeds

**Response:** The revised ablation tables report mean ± standard deviation for MAE and RMSE across seeds 42, 43, and 44 for A3-A6, and clearly mark A1-A2 as seed-42 only.

## Global temporal pooling

**Response:** We expanded the temporal module explanation. The revised text explains that static road edges are persistent across the temporal window, whereas vehicle and interaction edges are transient, and that the pooled road-edge context is a network-level signal broadcast to vehicles at prediction time.

## Number of windows, batch interpretation, compute, and runtime

**Response:** The revised manuscript describes graph windows as `H=30` snapshots with stride 10 and batch size 2. It also reports hardware, approximate step time, training duration, and inference throughput, including approximately 58 windows/s on the full Porto-G test split.

## Route encoder design

**Response:** We acknowledge that the current route encoder uses edge-ID embedding with mean pooling. We did not add a sequence route encoder in this revision because the main revision focused on real-world validation, stronger baselines, leakage clarity, and test-set reporting. We now frame this design as part of the tested architecture rather than as a claim that order-invariant pooling is optimal.

## Relation-specific GNN parameters

**Response:** We clarified the relation inventory and revised the Discussion to avoid overclaiming relation-specific dynamic-edge effects. Relation-specific or heterogeneous GNN parameterizations are identified as a future direction.

## MoE contribution and expert count

**Response:** We added a MoE expert-count sensitivity table and revised the explanation of the six-expert setting. Six experts are used as a shared default because this setting gives the best SUMO result by a larger margin while remaining within 0.19 seconds MAE of the best Porto-G setting, avoiding per-dataset tuning.

## Runtime and scalability

**Response:** The revised Complexity and Inference paragraph reports per-window inference latency and throughput on the hardware used in the experiments. Broader scaling curves are left for future work.

## Dataset access and code availability

**Response:** The Declarations section now identifies public Kaggle datasets for SUMO and Porto-G, the dataset generation tool, and the exact code commit containing the DSTRA-GNN model, Porto-G conversion pipeline, training scripts, and retrained baselines.

## Ethics, observability, and fairness

**Response:** We expanded the Discussion to address observability, privacy/surveillance assumptions, and fairness. The revised text notes that vehicle positions, interaction edges, and vehicle-count features require realistic sensing assumptions outside SUMO. It also states that better ETA can support mobility reliability, but the present evaluation measures prediction accuracy rather than downstream emissions, accessibility, or equity outcomes.

We thank the reviewers again for their constructive feedback. We believe the revised manuscript is substantially stronger, with real-world validation, stronger baselines, clearer test-set reporting, improved leakage discussion, and more balanced claims.
