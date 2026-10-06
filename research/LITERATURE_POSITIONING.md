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


## Additional closest weak-overlap/coarsening work found

- **Clivio et al., AISTATS 2026, _Deconfounding Scores and Representation Learning for Causal Effect Estimation with Weak Overlap_.** This is highly relevant and substantially narrows any novelty claim. They explicitly optimize feature representations for improved overlap subject to a deconfounding-score constraint that preserves identification/target. A future task-abstraction paper must distinguish latent-state *refinement/coarsening* and the local-rate paradox from their overlap-optimal observed-feature representation problem.
- **Ou & Nabi, UAI 2026, _Coarsening Bias from Variable Discretization in Causal Functionals_.** They formalize population-level approximation bias induced by discretization/coarsening. This reinforces the need for the bias term in our R12 frontier; coarsening is not free.
- Classical epidemiologic work on unnecessary adjustment/overadjustment already notes that adding adjustment variables can reduce precision without changing bias. The candidate novelty cannot be the generic observation that "more adjustment can hurt precision."

## Sharpened candidate gap after this scan

The potentially distinctive phenomenon is therefore narrower:

1. the variable being refined is an **estimated/persistent latent confounder state**, not merely an observed covariate representation;
2. fine microstates may be perfectly recovered and increasingly preferred by generative likelihood;
3. the downstream target is a vector of regime-specific causal effects / task functional;
4. local weak residual treatment information creates a **rate separation** where fine-state estimation is $O_p(N^{\alpha-1/2})$ while a task-sufficient merge remains $O_p(N^{-1/2})$;
5. the abstraction itself can be selected on a training split using downstream effect similarity and validated on held-out outcomes.

Whether this exact combination is novel remains unproven; a systematic scholarly review is still required.


## Dual-resolution correction and prior art

The distinction between **confounder adjustment** and **effect-modifier/subgroup parameterization** is classical. Reviews of heterogeneous treatment-effect estimation explicitly separate confounders required for ignorability from effect modifiers defining subgroups. Therefore "fine for adjustment, coarse for target" is not novel as a generic causal principle.

What the branch adds as a candidate specialized result is the latent-state/local-weak-overlap rate separation:
- fine latent microstate detail may remain necessary in nuisance functions to remove confounding;
- estimating a separate target effect for every fine microstate can have O_p(N^{alpha-1/2}) error under local weak residual treatment information;
- pooling only the target parameters across task-equivalent microstates can restore root-N while retaining fine-state adjustment;
- coarsening the nuisance representation itself can induce persistent omitted-microstate bias.

This should be described as a **dual-resolution latent-state estimation problem**, not as generic invention of the confounder/effect-modifier distinction.

Also relevant: Kalavasis, Mehrotra & Zampetakis (COLT 2024) introduce data-dependent coarsened IPW to improve robustness/confidence intervals under inaccurate propensity scores and extreme propensities. This is close in spirit and must be distinguished from the latent-state target-resolution problem. Their result makes a broad novelty claim about "coarsening improves weak-overlap causal estimation" untenable.


## Multitask sufficiency prior art

The broad observation that a representation sufficient/minimal for one task can discard information needed by other downstream tasks is established in representation learning; e.g. Wang et al. (CVPR 2022), _Rethinking Minimal Sufficient Representation in Contrastive Learning_, proves that a minimal representation for the pretext/shared-view objective may lose downstream-task information. Therefore the generic statement "no single minimal representation is optimal for all tasks" is not novel.

Our exact crossing-partition/exponential construction should be treated as an explanatory lemma. Its research value is in motivating a **rich shared latent substrate + task-specific causal target heads**, particularly when universal fine target parameterization pays weak-overlap costs. A future contribution must quantify that statistical cost rather than rely on the elementary partition argument alone.


## Targeted search for the causal-resolution boundary (2026-10-06)

A focused search for weak-overlap heterogeneous/subgroup effects found substantial neighboring work on:
- doubly robust estimation and inference under weak overlap (Ma, Sant'Anna, Sasaki & Ura);
- subgroup causal effects with overlap weighting;
- trimming/thresholding and positivity violations;
- representation learning that improves overlap while preserving causal identification (Clivio et al.);
- coarsening/discretization bias (Ou & Nabi).

This search did **not** surface, in the queried literature, the exact local-sequence result currently derived on this branch: a perfectly observed latent subgroup whose residual treatment variance is N^{-2 alpha}, with causal-effect separation N^{-beta}, yielding KL order N^{1-2 alpha-2 beta} and an information-theoretic heterogeneity-detection boundary alpha+beta=1/2. Absence from this search is not evidence of novelty. A formal literature review should search semiparametric weak identification, local alternatives, subgroup testing under positivity violations, and nonregular inference before any novelty claim.

The dual-resolution principle is also supported by recent work on coarsened exact matching showing that coarse confounder adjustment can leave residual confounding that persists with sample size; this is conceptually consistent with our analytic coarse-adjustment bias construction.


## Weak-identification correction after causal-resolution work

The broad inferential phenomena behind the causal-resolution limit are classical weak-identification phenomena, not new:
- Kaji (Econometrica 2021), _Theory of Weak Identification in Semiparametric Models_, develops local weak-identification embeddings and efficiency theory.
- Andrews & Cheng (Econometrica 2012), _Estimation and Inference With Weak, Semi-Strong, and Strong Identification_, studies inference across identification-strength regimes.
- Stock & Wright's GMM weak-identification theory and the wider weak-IV literature establish nonstandard rates and failure of conventional inference.
- The zero-information-limit literature explicitly studies sequences where Fisher information vanishes and standard inference becomes spurious.
- Weak-identification-robust confidence sets can be unbounded/wide rather than falsely precise.

Therefore the following are **not** novelty claims by themselves:
- confidence intervals widening as information vanishes;
- local alternatives with KL/noncentrality boundaries;
- weak eigenvalues causing non-root-N rates;
- spectral regularization of inverse problems.

The candidate contribution must instead be the specific structural connection:
**latent representation granularity / state refinement -> downstream causal information spectrum -> task-specific target resolution**, including the possibility that generative evidence for finer latent states increases while the corresponding causal contrast enters a weak-identification regime.

## Growing-number-of-subgroups correction

There is established work on simultaneous estimation/testing of many subgroup treatment effects and the familiar fact that finer subgrouping reduces per-group sample size and raises multiplicity/variance. Thus "more subgroups means noisier subgroup estimates" is not novel.

The new granularity calculation on this branch should be treated as a unifying rate law, not an isolated novelty claim:
    K_N=N^kappa,
    treatment SD=N^-alpha,
    effect separation=N^-beta
implies per-state causal KL order
    N^(1-kappa-2alpha-2beta).
The potentially useful contribution is tying the subgroup-growth exponent kappa directly to **learned latent-state resolution** and combining it with overlap and local effect scale in one causal-resolution budget.


## Phantom-resolution / latent-class uncertainty literature check (2026-10-06)

A targeted search found substantial prior work showing that classify-analyze latent subgroup causal effects can be biased or unstable when class membership is uncertain. In particular:
- Diemer et al. (2022/2023), _Evaluating sensitivity to classification uncertainty in latent subgroup effect analyses_, explicitly studies sensitivity of subgroup causal effects to posterior/classification uncertainty and recommends perturbation/bootstrap sensitivity analysis.
- Lyu, Kim & Suk (2023), _Estimating Heterogeneous Treatment Effects Within Latent Class Multilevel Models_, jointly models latent classes and heterogeneous treatment effects so that sequential misclassification does not obstruct inference.
- Recent latent-variable causal work on misclassified treatment similarly models the measurement-error channel jointly rather than treating labels as known.

Therefore **classification uncertainty affecting latent subgroup causal effects is not novel**.

The more specific phenomenon found on this branch is different: misclassification can *inflate the smallest eigenvalue of the downstream treatment-information/Gram matrix*, making the proxy-weighted causal geometry look healthier than the oracle true-state geometry. In the symmetric two-state calculation, chance-level classification drives inferred state-specific information toward equality and apparent condition number toward 1 even when true state information is arbitrarily imbalanced.

The targeted search did not surface a paper framing this as optimistic distortion of a downstream causal-information spectrum or studying serial inversion of a latent-state confusion operator followed by a weak causal moment operator. This absence is not evidence of novelty. Search should be expanded to errors-in-variables inverse problems, mixture deconvolution, latent-class distal-outcome corrections, and information geometry under misclassification before making any claim.
