# Focused Novelty Collision Audit

Checkpoint date: 2026-10-06. This is a conservative audit, not a claim of exhaustive literature coverage.

## 1. Generated covariates are established
Mammen, Rothe & Schienle, "Semiparametric Estimation with Generated Covariates" (Econometric Theory, 2016) develops root-n theory, asymptotic variance, and bootstrap validity for semiparametric estimators with generated covariates. Therefore:
- "estimated/generated representations affect downstream semiparametric inference" is not novel;
- generic first-stage generated-covariate rate effects are not novel.

Potential distinction here: generated *latent sequential state distributions* interacting with target-specific weak causal information and proxy-induced spectral distortion.

## 2. General semiparametric weak identification is established
Kaji (Econometrica, 2021) gives a general theory of weak identification in semiparametric models, including weakly regular parameters and minimal sufficient underlying regular parameters. Therefore:
- weak identification, drifting information, impossibility, and local nonregularity are not novel;
- "some directions matter and irrelevant nuisance directions should be excluded" has deep antecedents.

Bennett et al. (COLT 2023), "Inference on Strongly Identified Functionals of Weakly Identified Functions," is especially close to our target-subspace theme: a functional can be strongly identified even when an underlying nuisance function is weakly identified. Therefore the broad claim "global weak identification can coexist with a well-identified scientific functional" is not novel.

Potential distinction here: learned latent-state representation resolution, proxy spectral distortion, and task-specific generated-representation perturbation as the mechanism.

## 3. Latent-class classification correction/sensitivity is established
Latent-class/BCH/three-step literature corrects downstream analyses for classification error. Diemer et al. (2022/2023) explicitly studies sensitivity of latent subgroup causal effects to classification uncertainty. Recent causal latent-class work combines propensity weighting and BCH/ML corrections.

Therefore:
- classification uncertainty biasing subgroup causal effects is not novel;
- posterior perturbation/bootstrap sensitivity to latent memberships is not novel;
- correcting downstream latent-class analysis for misclassification is not novel.

Potential distinction: misclassification can *optimistically inflate the downstream causal-information spectrum*, making lambda_min/condition number look healthier, and can self-cancel an entropy-over-conditioning reliability score.

## 4. Task-aware calibration exists
Tomov et al. (2026), "Task-Aware Calibration: Provably Optimal Decoding in LLMs," explicitly develops task calibration in a task-induced latent space and a task calibration error tied to downstream loss. Therefore:
- "calibration should be task-aware" is not novel in general;
- application-aware calibration metrics are not novel.

Potential distinction: calibration under the causal score/leverage measure, directional calibration paired with weak causal information, and the impossibility/ranking-reversal construction for latent causal representations.

## 5. Surviving candidate novelty claims

The following are the strongest claims that did not collide directly in this focused audit, but each still requires a broader formal search before publication:

### A. Phantom causal resolution
Imperfect latent-state recovery can inflate weak eigenvalues of a downstream causal-information/Gram matrix, making generated-state conditioning appear *better* than oracle conditioning. Chance-level mixing can drive apparent information toward isotropy even when oracle state information is arbitrarily imbalanced.

### B. Self-canceling reliability diagnostics
When the same uncertain representation drives both entropy and proxy conditioning, worse representation can raise entropy while simultaneously inflating proxy lambda_min; H/lambda_proxy can therefore erase its own warning. Controlled experiment: AUC 0.470 versus entropy 0.640, while oracle-conditioned score reaches 0.994.

### C. Double ill-posedness
Correcting representation confusion and then solving a weak causal moment can create serial inverse problems: representation-channel deconvolution followed by causal inversion. This needs careful positioning against generic inverse-problem composition.

### D. Posterior causal-geometry completion
For one-hot latent states, downstream information moments require posterior expected latent second moments, not posterior-mean outer products. The identity itself is elementary conditional expectation and not novel; the candidate contribution is its use to diagnose/correct phantom causal information, plus the finding that the posterior information set must include variables entering the weighted causal moment.

### E. Directional generated-representation reliability
The exact/local perturbation object A J^{-1} b_gamma combines scientific target, causal information geometry, and representation-induced score error. The algebra is standard influence/GMM sensitivity; novelty cannot be claimed for the formula itself. The candidate contribution is using it to unify latent-representation reliability phenomena and derive operational diagnostics/validation allocation in generated-state causal inference.

### F. Representation-resolution budget
Growing latent/target granularity K_N, weak treatment information, effect-resolution scale, and score dependence jointly consume causal information. Individual ingredients are classical; the combined learned-state resolution interpretation may be novel but should be presented as a synthesis unless a theorem materially exceeds existing many-parameter/weak-ID theory.

### G. Representation orthogonality
Neyman orthogonality is established. The candidate new method would require an operational score orthogonal to *latent representation/posterior perturbations* under identifiable tangent information. The current branch only has proof-of-concept toys, so this is not yet a contribution.

## 6. Claims to avoid

Do not claim novelty for:
- weak identification;
- target-specific versus global conditioning in general;
- generated-covariate asymptotics;
- latent-class misclassification correction;
- task-aware calibration in general;
- sample splitting after adaptive target selection;
- Pareto/multiobjective representation selection;
- Neyman orthogonality itself.

## 7. Best current novelty package

The most defensible potential package is:
1. identify and formalize **phantom causal resolution** in generated latent-state causal inference;
2. show it causes **self-canceling proxy reliability diagnostics**;
3. derive a task/direction-aware reliability analysis that separates representation perturbation from oracle causal information and scientific target;
4. provide posterior-moment/sensitivity/validation mechanisms for conservative diagnosis;
5. demonstrate the phenomena in sequential latent-state simulations and, where possible, repository real/semi-synthetic data.

This is narrower than the branch's full conceptual exploration, but much more defensible.
