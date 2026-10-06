# Multitask Calibration Ranking Reversal

**Status:** exact construction + automated Monte Carlo. Not submission material.

The same observational representation can have opposite reliability rankings for two downstream tasks.

Two representations have essentially identical global posterior squared error (~0.016). Representation L concentrates errors on the left 20% of feature space; representation R concentrates errors on the right 20%. Task A assigns leverage 100 to the left region and 1 elsewhere; Task B does the reverse.

At N=50,000:
- Representation L: Task A error 0.07693, Task B error 0.000770.
- Representation R: Task A error 0.000768, Task B error 0.07692.

Thus each representation is about 100x worse for one task and 100x better for the other, while their global calibration errors are indistinguishable.

The ranking reversal is stable from N=2,000 to 50,000, so it is structural.

## Consequence

There is no task-independent total ordering of representations by downstream reliability when tasks induce different leverage measures. A representation can be simultaneously excellent and poor depending on the functional consuming it.

This strengthens the earlier multitask abstraction result:
- task dependence governs which latent distinctions should be retained;
- task dependence also governs where posterior accuracy matters.

A universal representation metric would need restrictions on the family of downstream leverage functions. Without such restrictions, task-conditioned evaluation is mathematically unavoidable.

## Reproduction
- research/multitask_calibration_gap.py
- research/results/multitask_calibration_gap_summary.csv
