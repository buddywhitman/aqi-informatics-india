# Validity Under Data-Adaptive Target Selection

Status: exploratory caution; not submission material.

Target-aware inference is cleanest when the scientific target or target family is declared independently of the outcome noise used for final estimation.

If the target is chosen after inspecting noisy estimated effects, the target map A becomes data-dependent. Standard fixed-target inference can then suffer winner's curse / selective-inference bias.

A simple null experiment with K independent root-N effect estimates compares:
1. predeclared coordinate;
2. post-hoc coordinate with largest absolute estimated effect;
3. split-sample selection followed by independent estimation.

As K grows, post-hoc selection error increases roughly with the Gaussian maximum scale sqrt(log K)/sqrt(N), whereas predeclared and split-sample estimation remain at root-N constant scale.

## Consequence

Task/target awareness does not license outcome-adaptive target shopping.

Safe architectures:
- confirmatory: predeclare A or target family F;
- adaptive discovery: use one split to choose A, another for estimation/inference;
- selective inference: explicitly condition/adjust for target selection;
- reusable representation: retain rich nuisance representation even if target estimation is specialized.

This creates another important separation:
    target-aware != target-adaptive-without-penalty.

The same sample splitting already used for task-aware state abstraction can potentially be reused for valid target selection.

Reproduction:
research/pretest_target_selection_bias.py
