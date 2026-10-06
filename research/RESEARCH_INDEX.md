# Research Artifact Index

**Branch:** `research/task-relative-reliability`  
**Isolation:** none of these artifacts are part of the AISTATS submission unless explicitly promoted later.

| Theme | Core note | Reproduction | Key result |
|---|---|---|---|
| Representation/task interaction | README R1-R6 | run_extended_reliability_studies.py | scalar difficulty useful but directional geometry matters |
| K=3 orientation | README R3 | run_extended_reliability_studies.py | same eps/lambda_min can yield ~6x different error |
| State granularity | STATE_REFINEMENT_PARADOX.md | task_aware_state_coarsening.py | BIC-favored fine states can be causally inefficient |
| Data-driven abstraction | STATE_REFINEMENT_PARADOX.md | data_driven_task_coarsening.py; crossfit_task_coarsening.py | held-out task coarsening near oracle |
| Local weak-overlap rates | STATE_REFINEMENT_PARADOX.md | weak_overlap_refinement_paradox.py | fine target can diverge while pooled target is root-N |
| Bias-information frontier | TASK_SUFFICIENT_ABSTRACTION_SYNTHESIS.md | abstraction_tradeoff.py | coarsening helps only below heterogeneity boundary |
| Adaptive partition selection | README R14 | risk_selected_abstraction.py | train-only risk selector useful away from crossover |
| Dual resolution | DUAL_RESOLUTION_PRINCIPLE.md | dual_resolution_experiment.py | fine adjustment + coarse target avoids bias and weak-state rate failure |
| Causal resolution limit | CAUSAL_RESOLUTION_LIMIT.md | causal_resolution_power.py | alpha+beta=1/2 detection boundary |
| Resolution spectrum | CAUSAL_RESOLUTION_SPECTRUM.md | real_causal_resolution_spectrum.py | direction-specific resolvable effect scale 1/sqrt(N v'Jv) |
| Resolution thresholding | README R20 | resolution_adaptive_target.py | unresolved does not mean zero; estimand change must be explicit |
| Resolution divergence | RESOLUTION_DIVERGENCE_THEOREM.md | resolution_divergence_simulation.py | state evidence can rise while causal information vanishes |
| Honest weak-resolution inference | HONEST_INFERENCE_BELOW_RESOLUTION.md | honest_inference_resolution.py | honest intervals can widen while naive root-N intervals become falsely precise |
| Granularity weak ID | GRANULARITY_INDUCED_WEAK_IDENTIFICATION.md | granularity_weak_identification.py | growing state count consumes causal information even with healthy within-state overlap |
| Sequential resolution budget | SEQUENTIAL_CAUSAL_RESOLUTION_BUDGET.md | sequential_resolution_budget.py | score dependence adds an effective-sample-size tax; state persistence alone is insufficient |
| Resolution-adaptive targets | CAUSAL_RESOLUTION_SPECTRUM.md | resolution_adaptive_target.py | truncation helps only when target projection is explicit; unresolved is not zero |
| Multitask incompatibility | MULTITASK_ABSTRACTION_GAP.md | multitask_abstraction.py | crossing partitions; exponential universal-compression gap construction |
| Real K sensitivity | README R8/R19 | latent_state_count_sensitivity.py | BIC improves while downstream lambda_min collapses |
| Telemetry sensitivity | README R7 | frozen_sensor_sensitivity.py | Mumbai decomposition materially sensitive to constant runs |
| Literature/novelty audit | LITERATURE_POSITIONING.md | n/a | records closest known neighboring work and prohibited overclaims |

## Highest-priority candidate contributions

1. **Causal resolution limit:** a perfectly observed state can have an asymptotically undetectable effect contrast under local weak overlap.
2. **Resolution divergence:** observational evidence for a state split can diverge while causal information about its effect contrast vanishes.
3. **Dual-resolution principle:** retain fine latent detail for identification but adapt target-effect resolution to supported causal information.
4. **Causal resolution spectrum/effective rank:** replace one worst-case eigenvalue with direction-specific detectable effect scales.
5. **State-refinement rate paradox:** full fine-state target parameterization can lose root-N behavior or diverge even with perfect state recovery.
6. **Unified causal-resolution budget:** growing target granularity, weak overlap, subtle effect scale, and score dependence combine through an information exponent; no single diagnostic should be interpreted in isolation.

## Strong negative results retained

- eps/lambda_min is not sufficient in K>=3 because perturbation orientation matters.
- scalar difficulty poorly predicts the oracle regularization parameter.
- naive spectral thresholding can have large approximation error when unresolved directions carry real signal.
- coarsening is harmful when within-group causal heterogeneity exceeds the information benefit.
- real-data transition locations are inferred, not ground truth.
- Mumbai regime estimates are telemetry-sensitive.
- broad task-aware abstraction/coarsening ideas have substantial prior art.

## Reproducibility discipline

Every claimed numeric result should point to a committed CSV under `research/results/` or be explicitly labeled analytic/exploratory. Default scripts expose replication-count environment variables where appropriate. Publication-grade use requires rerunning with larger Monte Carlo counts and recording software/environment hashes.
