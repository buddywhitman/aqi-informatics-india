# Remaining Research Plan and Saturation Criteria

Status: active research plan. Submission branches/materials remain untouched.

## Objective

Drive the isolated research branch to scientific saturation: every materially promising direction exposed by the repository and findings should be either validated, falsified, connected to prior art, or explicitly left as an identified open problem. Literal exhaustion of all possible mathematical directions is impossible; the stopping criterion is that additional experiments become variants rather than changing the scientific conclusion.

## Work packages

### WP1 — Representation orthogonality
- Validate partial/tangent-space correction.
- Stress-test correction-moment misspecification.
- Add estimated rather than oracle correction moments.
- Combine with weak information.
- Extend to Markov/generated posterior settings.
- Determine whether first-order cancellation is operationally identifiable.

### WP2 — Integrated stress-test DGP
Build one controlled experiment containing:
- latent-state uncertainty;
- state persistence;
- weak overlap/information;
- directional representation error;
- target-specific heterogeneity;
- phantom resolution;
- generated-state nuisance estimation.

Evaluate whether the directional operator framework predicts downstream bias/error across the full factorial design.

### WP3 — Unified theorem
Formalize
    A(theta_hat-theta) = A J^{-1} b_gamma + A J^{-1} xi/sqrt(N) + remainder.
Derive:
- target-specific rate requirements;
- target-family version;
- representation-orthogonal second-order version;
- weak-information/local sequences;
- causal-resolution lower bounds;
- state-granularity corollaries.

### WP4 — Operational estimation of representation perturbation
Investigate:
- posterior second-moment completion;
- validation samples;
- repeated proxies;
- perturbation tangent models;
- sensitivity/partial-identification sets;
- conservative bounds on target-projected b_gamma.

### WP5 — Real-data exploitation
Re-analyze Indian-city data for:
- target/subspace conditioning;
- causal effective rank;
- K-dependent generative versus causal resolution;
- task-weighted calibration;
- phantom-overlap signatures;
- sensor-quality dependence;
- season/time-of-day dependence;
- city-transfer reliability.

No oracle-state claim may be made from observational data.

### WP6 — Representation-zoo reanalysis
Re-evaluate HMM/GRU/Transformer/SSM/etc. using:
- directional/task-weighted calibration;
- target-subspace information;
- completed/proxy geometry sensitivity;
- downstream error orientation.
Test whether previous apparent contradictions become predictable.

### WP7 — Sequential theory
Replace stylized effective-sample-size arguments with score-process dependence:
- Markov/mixing sequences;
- filtering versus smoothing;
- regime duration/persistence;
- transition uncertainty;
- temporal cross-fitting;
- local-to-persistent regimes.

### WP8 — Granularity and target-resolution selection
Combine:
- K_N growth;
- causal effective rank;
- likelihood/BIC gains;
- weak identification;
- fine adjustment/coarse target principle.
Seek a criterion for retaining a latent state for nuisance adjustment without automatically allocating a separate causal parameter.

### WP9 — Honest inference
Study:
- target-specific weak-ID intervals/tests;
- generated-state uncertainty propagation;
- adaptive target/subspace selection;
- split-sample/selective inference;
- target-family simultaneous coverage.

### WP10 — Adversarial falsification
Search deliberately for cases where:
- AJ^{-1}b_gamma poorly predicts actual error;
- moment completion is misleading;
- task-weighted calibration fails;
- target-aware regularization loses badly;
- representation orthogonality introduces new bias/non-identification.

### WP11 — Novelty audit
Search and position against:
- weak identification;
- semiparametric generated regressors;
- latent-class causal inference;
- measurement error;
- mixture deconvolution;
- inverse problems;
- sufficient/task-aware representations;
- calibration under covariate shift/weighted measures;
- selective inference;
- proximal causal inference;
- information geometry.

Every final novelty claim must survive this audit.

### WP12 — Final synthesis
Compress the branch into 3–5 defensible contributions. Explicitly classify every major finding as:
- candidate novel theorem/method;
- known idea with a new connection/application;
- empirical finding;
- negative result;
- failed hypothesis;
- unresolved/open.

## Current candidate synthesis

The strongest unifying local object is

    A J^{-1} b_gamma,

where:
- A defines the scientific target/subspace;
- J is downstream causal information geometry;
- b_gamma is the target-relevant score perturbation induced by generated representation error.

Candidate overarching principle:

> Latent representations should be evaluated and learned through the information geometry of the scientific functional that consumes them, rather than through state-recovery/reconstruction quality alone.

Supporting phenomena:
1. observational resolution != causal resolution;
2. phantom causal resolution;
3. directional/target-specific reliability;
4. dual resolution: fine adjustment, task-appropriate target;
5. causal-resolution budget/rate laws;
6. self-canceling proxy diagnostics;
7. task-weighted calibration impossibility;
8. posterior latent second-moment completion;
9. representation orthogonality;
10. target-aware inference and regularization.

## Saturation criteria

Research is considered saturated only when:
1. WP1–WP10 have either reproducible results or documented reasons they cannot be resolved with current data/model assumptions;
2. every favorable claim has at least one targeted falsification attempt;
3. all key negative results remain documented;
4. real-data claims are separated from oracle/synthetic claims;
5. WP11 identifies closest prior art for each surviving contribution;
6. WP12 reduces the work to a small coherent contribution set;
7. further experiments are predominantly parameter variants rather than conceptual changes.

Expected remaining effort at this checkpoint: approximately 8–12 substantive research cycles, with likely diminishing returns after ~8.
