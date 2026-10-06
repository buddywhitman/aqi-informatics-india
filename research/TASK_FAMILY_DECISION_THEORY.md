# Task-Family Decision Theory for Representation Selection

Status: exploratory decision-theoretic synthesis; not submission material.

The task-conditioned risk matrix shows that representation selection is not well-defined until a downstream task family and decision rule are declared.

For the family containing only the uniform task, the globally best representation is also average-risk and minimax optimal.

For the family {uniform,left,right}, global_best remains optimal.

When the center task is added, the decision changes sharply:
- global-calibration selection keeps global_best, with average risk 0.01095 and worst-task risk 0.02895;
- average-task and minimax selection switch to the uniform representation, with average/worst risk 0.01000;
- the per-task oracle average is 0.00393.

Thus adding one legitimate downstream task changes the robustly preferred representation even though nothing about the representations themselves changed.

The globally selected representation's worst-case risk is 2.895x the minimax risk on the expanded family.

## Principle

Representation quality is a decision problem over a declared task family. Without specifying the task distribution or robustness criterion, a universal leaderboard implicitly assumes a utility function.

Natural reporting objects:
1. task-conditioned risk vector;
2. Pareto frontier;
3. average risk under an explicit task distribution;
4. minimax/worst-task risk over a declared family;
5. regret relative to a task-specific oracle.

## Connection to causal resolution

In latent causal inference, tasks correspond to causal contrasts/functionals and induce different leverage/information geometries. Therefore model selection by state F1, likelihood, ECE, or any single global reliability score silently chooses a task utility that may be unrelated to the causal query.

This is decision-theoretic framing, not claimed as a novel general theorem. The research contribution would be to instantiate it for latent causal-resolution geometry and show when standard representation metrics incur large causal regret.
