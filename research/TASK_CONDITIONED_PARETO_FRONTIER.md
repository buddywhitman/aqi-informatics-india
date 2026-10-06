# Task-Conditioned Representation Pareto Frontier

Status: automated construction; not submission material.

Five representations are evaluated on four downstream leverage tasks. Every representation is Pareto non-dominated even though their global calibration errors differ.

The representation with the best global error (global_best, 0.00835) is excellent for left/right tasks (about 0.00326) but much worse for the center task (0.02895). The center specialist, despite worse global error (0.01162), achieves center-task error 0.00247, about 11.7x lower. Left/right specialists similarly dominate their corresponding tasks.

Thus a global representation leaderboard collapses a genuine multi-objective frontier. There need not be a universally best representation even within a fixed finite task family.

## Decision-theoretic implication

Choosing a representation requires a specified task, a distribution/weights over tasks, a minimax criterion over a declared task family, or reporting the Pareto frontier. Without one of these, saying representation A is better than B is under-specified.

This is not claimed as a new general fact in multi-objective learning. Its relevance here is that latent causal reliability inherits the same structure through task-induced leverage and causal-resolution geometry.

## Reproduction

research/task_conditioned_representation_dominance.py
research/results/task_conditioned_representation_pareto.csv
