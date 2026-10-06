# Literature Positioning Notes for Task-Sufficient Latent Abstraction

**Exploratory literature scan, 2026-10-06. Not a novelty claim.**

The state-refinement paradox/task-aware coarsening direction overlaps with several established literatures. Any future paper must position itself carefully.

## Closest conceptual neighbors

1. **Task-specific state abstraction / bisimulation in reinforcement learning.** Wang et al. (AAAI 2024), *Building Minimal and Reusable Causal State Abstractions for Reinforcement Learning*, explicitly derives minimal task-specific state abstractions from causal dynamics and reward relationships. This means the generic statement "task-specific abstractions are new" would be false.

2. **Causal abstraction.** Massidda et al. (CLeaR 2023; UAI 2024), D'Acunto et al. (ICML 2025), and Xia & Bareinboim (ICML 2025) study mappings between low- and high-level causal models, including lossy/projected abstractions. Therefore "causal coarsening" by itself is not novel.

3. **Bisimulation metrics/state aggregation.** A large RL literature studies aggregating states that preserve rewards/value/dynamics and the resulting approximation/sample-efficiency tradeoffs.

## What appears more specific in our exploratory finding

The present phenomenon concerns **statistical estimation of a causal functional under weak overlap/weak residual treatment information**, not policy-value preservation in an MDP and not merely existence/identification of a high-level SCM.

The distinctive candidate statement is:

> Refining a perfectly observed latent confounder into causally redundant microstates can worsen the statistical rate of downstream causal-effect estimation; under a local weak-overlap sequence, the refined estimator can be O_p(N^{alpha-1/2}) or diverge while a task-sufficient coarsening remains root-N.

The associated model-selection tension is also specific:
- generative likelihood/BIC increasingly favors the fine microstate representation;
- the fine representation can simultaneously lose downstream causal information;
- a sample-split task-aware merge can recover near-oracle causal risk.

This needs a much deeper literature review before any novelty claim. Search specifically for:
- causal effect estimation after state aggregation;
- covariate coarsening under positivity/overlap;
- propensity-score subclassification/coarsening;
- weak identification after conditioning/refinement;
- causal representation sufficient statistics;
- semiparametric efficiency under coarsened covariates;
- post-selection inference for data-driven causal partitions.

## Relevant papers found in initial scan

- Wang et al., 2024, AAAI: Building Minimal and Reusable Causal State Abstractions for Reinforcement Learning.
- Massidda et al., 2023, CLeaR: Causal Abstraction with Soft Interventions.
- Massidda et al., 2024, UAI: Learning Causal Abstractions of Linear Structural Causal Models.
- D'Acunto et al., 2025, ICML: Causal Abstraction Learning based on the Semantic Embedding Principle.
- Xia & Bareinboim, 2025, ICML: Causal Abstraction Inference under Lossy Representations.
- Kemertas & Aumentado-Armstrong, 2021, NeurIPS: Towards Robust Bisimulation Metric Learning.

## Novelty discipline

Do not describe "task-aware abstraction", "minimal task-specific representation", "causal abstraction", or "state coarsening" as new in isolation.

A defensible future contribution would need to center the **weak-overlap statistical-rate paradox**, an estimable causal-functional-specific abstraction criterion, and valid post-selection inference.
