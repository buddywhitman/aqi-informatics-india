# Master Artifact Index

This index groups the research branch by scientific purpose. For chronological claims and exact numbers, use research/README.md. For final claim discipline, use FINAL_AUDIT_CHECKLIST.md.

## Core synthesis and audits
- FINAL_CANDIDATE_SYNTHESIS.md
- NOVELTY_COLLISION_AUDIT.md
- FINAL_REAL_DATA_COLLISION.md
- REPRESENTATION_ZOO_REANALYSIS.md
- FINAL_AUDIT_CHECKLIST.md
- SATURATION_STATUS.md
- REMAINING_RESEARCH_PLAN.md
- UNIFIED_THEOREM_SKETCH.md

## Causal resolution and abstraction
- CAUSAL_RESOLUTION_SPECTRUM.md
- DUAL_RESOLUTION_PRINCIPLE.md
- FINE_ADJUSTMENT_COARSE_TARGET.md
- TASK_CONDITIONED_PARETO_FRONTIER.md
- TASK_FAMILY_DECISION_THEORY.md
- REPRESENTATION_METRIC_IMPOSSIBILITY.md
- MULTITASK_CALIBRATION_REVERSAL.md
- SUBSPACE_RELIABILITY_PRINCIPLE.md
- TARGET_SUBSPACE_RESOLUTION_PARADOX.md
- TARGET_AWARE_INFERENCE.md
- ROBUST_TARGET_FAMILIES.md

## Phantom resolution / generated-state geometry
- PHANTOM_CAUSAL_RESOLUTION.md
- POSTERIOR_SECOND_MOMENT_COMPLETION.md
- COMPLETION_ROBUSTNESS_LIMITS.md
- ROBUST_CAUSAL_GEOMETRY.md
- OPERATIONAL_PERTURBATION_BOUNDS.md
- TARGET_AWARE_VALIDATION_DESIGN.md

## Directional and task-relative calibration
- DIRECTIONAL_RELIABILITY_OPERATOR.md
- DIRECTIONAL_TASK_CALIBRATION.md
- TASK_WEIGHTED_CALIBRATION_PRINCIPLE.md
- LEVERAGE_CALIBRATION_IMPOSSIBILITY.md
- PARTIAL_REPRESENTATION_ORTHOGONALITY.md
- REPRESENTATION_ORTHOGONALITY.md
- REPRESENTATION_ACCURACY_RATE.md
- RELIABILITY_BUDGET.md

## Nonlinearity, validity and inference
- NONLINEAR_RELIABILITY_BOUNDARY.md
- OPERATOR_SCOPE.md
- PROXY_ORACLE_TARGET_DECOMPOSITION.md
- TARGET_SELECTION_VALIDITY.md
- TARGET_FAMILY_INFERENCE.md
- SEQUENTIAL_SCORE_INFORMATION.md

## Real-data robustness
- frozen_sensor_sensitivity.py
- real_causal_resolution_spectrum.py
- real_data_collision_panel.py
- results/frozen_sensor_sensitivity.csv
- results/real_transition_diagnostics.csv

## Representation zoo
- zoo_task_relative_reanalysis.py
- ../reports/representation_zoo_world_evaluations.csv
- ../reports/representation_zoo_reliability_auc.csv
- ../reports/representation_zoo_reliability_correlations.csv
- ../reports/representation_zoo_calibration_intervention.csv
- ../reports/representation_zoo_lowo_evaluation.csv
- ../reports/representation_zoo_hierarchical_regression.csv

## Reproducible experiment scripts

All Python files under research/ are exploratory scripts. Major families include:
- causal-resolution / state-count / abstraction experiments;
- phantom-resolution and concentration experiments;
- posterior moment completion and misspecification;
- task-weighted/directional calibration;
- target-aware selection/regularization;
- representation-rate/cross-fitting/orthogonality;
- operator tightness/nonlinear breakdown;
- validation-bound/allocation studies;
- sequential score dependence;
- adaptive target and simultaneous-family inference.

Generated summaries are stored under research/results/; larger workflow outputs are also available as GitHub Actions artifacts.

## Submission boundary

The exploratory research branch contains findings that contradict or narrow some earlier manuscript claims. Nothing should be promoted into a submission merely because it is favorable. Promotion requires:
1. reproducible committed evidence;
2. consistency with negative/falsification results;
3. novelty audit;
4. no conflict with the main manuscript/supplement;
5. compliance with AISTATS anonymity, page-limit and AI-disclosure requirements.
