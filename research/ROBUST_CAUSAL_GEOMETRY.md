# Robust Causal Geometry and Task-Weighted Calibration

Exploratory; not submission material.

## Core problem
A single proxy or completed information matrix can be confidently wrong. Robust diagnosis should distinguish sampling uncertainty from latent-model uncertainty.

## Geometry disagreement
Construct at least three operational geometries: posterior-outer plug-in; posterior diagonal completion under representation-only information; and joint state-treatment posterior completion. Large spectral disagreement is a falsification signal, not a nuisance to average away. The script geometry_disagreement_diagnostic.py tests whether log spread among these constructions predicts large error in nominal completed lambda_min.

## Task-weighted calibration
Ordinary posterior calibration weights observations roughly equally. Causal geometry weights observations by score leverage such as residual treatment squared. Define weighted error E_w = E[w ||gamma-H||^2] / E[w]. For treatment-information geometry, w=T_tilde^2 is natural. A representation may have good global Brier/ECE yet poor E_w if its mistakes concentrate on high-leverage observations.

The script task_weighted_calibration.py compares ordinary and treatment-information-weighted Brier error against oracle/proxy spectral distortion.

## Robust reporting proposal
Do not report a naked latent causal lambda_min. Report the nominal completed spectrum, model-sensitivity envelope, disagreement spread across admissible geometry constructions, task-weighted posterior calibration, and scientific-scale causal effective rank. If the lower sensitivity bound approaches zero or geometry disagreement is large, label the corresponding contrast unresolved.

## Open methodological target
Construct a lower confidence/sensitivity bound on oracle v'Jv that combines finite-sample uncertainty, posterior calibration error, state-treatment model uncertainty, and generated-state uncertainty. The aim is one-sided honesty: false warnings are tolerable; false declarations of healthy causal resolution are not.
