# Representation Metric Impossibility Under Unrestricted Downstream Tasks

Status: elementary decision-theoretic construction; not submission material.

Suppose a task-agnostic scalar metric ranks representation A above B based on global error. If A and B have crossing pointwise error profiles, a downstream task can concentrate leverage on a region where B is better. As the task leverage ratio on that region increases, downstream regret from choosing A can grow arbitrarily toward the local error ratio, despite the global metric continuing to prefer A.

This is not a claim that every pair of representations can be reversed: if A pointwise dominates B for all task-relevant sufficient statistics, then all nonnegative leverage tasks preserve the ordering. The exact condition for universal dominance over an unrestricted nonnegative leverage family is therefore pointwise/almost-sure dominance of the relevant error functional.

## Consequence

A universal scalar representation leaderboard is defensible only if:
1. one representation dominates others pointwise in the task-relevant error functional; or
2. the downstream task/leverage family is restricted and declared.

Otherwise task-conditioned risk vectors, Pareto frontiers, or regret over a declared task family are the appropriate objects.

## Connection to latent causal inference

Causal contrasts induce different score leverage functions and information directions. Therefore global state F1, likelihood, ECE, or unweighted posterior error cannot universally order latent representations for causal use unless representation errors are uniformly dominated in every causal-relevant region/direction.

The substantive research question is to characterize realistic causal task families tightly enough that robust representation comparison becomes possible without requiring impossible pointwise dominance.

## Reproduction

research/representation_metric_impossibility.py
