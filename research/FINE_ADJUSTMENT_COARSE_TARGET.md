# Fine Adjustment, Coarse Target

A rich latent state may be necessary for confounding adjustment without implying that every state deserves its own reported causal effect.

With K balanced states and healthy within-state treatment information, separate state-specific effects have SE scale sqrt(K/N), while a pooled task-equivalent target retains 1/sqrt(N). The ratio is sqrt(K).

Thus increasing representation granularity can be valuable for nuisance adjustment yet unnecessarily destroy target precision if causal parameter granularity is forced to match representation granularity.

Principle:
    representation resolution >= identification/adjustment resolution,
while
    target resolution should be only as fine as the scientific estimand and causal information support.

This is the operational form of dual resolution.

Reproduction: research/granularity_adjustment_target.py
