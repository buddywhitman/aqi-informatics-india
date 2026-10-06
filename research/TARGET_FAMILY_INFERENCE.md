# Simultaneous Inference for Target Families

If target-aware analysis protects a declared family rather than one fixed contrast, inference must account for multiplicity/selection.

A simple Bonferroni benchmark for M standardized targets uses z_{1-alpha/(2M)} instead of 1.96. The width penalty grows slowly (roughly sqrt(log M)) but is real.

This creates a continuum:
- fixed predeclared target: narrowest valid inference;
- finite declared family: simultaneous coverage with multiplicity cost;
- data-adaptive target: split/selective inference;
- unrestricted target space: global/operator-norm protection.

Target-family robustness therefore has an inferential price, not just a regularization price.

Reproduction: research/simultaneous_target_family_inference.py
