# Research Note: The State-Refinement Paradox and Task-Sufficient Latent Abstraction

**Status:** exploratory research note; not part of the AISTATS submission.  
**Branch:** `research/task-relative-reliability`.

## 1. Empirical phenomenon

A latent representation can be observationally more accurate and generatively better fitting while being statistically worse for a downstream causal functional.

The clean construction has four perfectly distinguishable microstates. States 0/1 share causal effect theta_A and states 2/3 share theta_B. One microstate in each pair has weak residual treatment information. Gaussian-mixture BIC overwhelmingly prefers K=4, and a K=4 GMM recovers the microstates with ARI about 0.99. Nevertheless, pooling causally equivalent states reduces downstream causal error by roughly 4-5x at ordinary fixed overlap and can change the asymptotic rate completely under local weak overlap.

## 2. Rate calculation

Let microstate k have n_k proportional to N observations and residual treatment variance sigma_{k,N}^2. For a simple residual slope,
```
theta_hat_k - theta_k = sum_i V_i U_i / sum_i V_i^2,
```
so conditional variance is approximately
```
Var(theta_hat_k | S=k) = sigma_U^2 / (n_k sigma_{k,N}^2).
```
If sigma_{k,N}=N^{-alpha}, then
```
sd(theta_hat_k) = O(N^{alpha-1/2}).
```
Suppose microstates 0 and 1 share the same causal effect but state 1 is weak while state 0 has variance bounded away from zero. Estimating both microstate slopes separately and then occupancy-averaging retains a nonvanishing weight on the weak estimate. Therefore the refined macro estimate inherits O_p(N^{alpha-1/2}) error.

If instead states 0 and 1 are pooled before estimating the shared effect, total treatment information includes the strong state and is O(N). The pooled slope is O_p(N^{-1/2}).

Hence:
- alpha < 1/2: refined representation is consistent but slower than root-N;
- alpha = 1/2: refined representation has O_p(1) error;
- alpha > 1/2: refined representation error diverges;
- task-sufficient coarsening remains root-N throughout, provided a positive fraction of the pooled state has nondegenerate treatment variation.

This is not a latent-state recovery failure: the microstate labels may be perfectly observed.

## 3. Simulation

`research/weak_overlap_refinement_paradox.py` confirms the rate phenomenon over N=400..6400.

At alpha=0.75:
- refined mean error: 5.58 -> 10.64 as N grows 400 -> 6400;
- task-coarsened error: 0.126 -> 0.0327.

At alpha=1:
- refined mean error: 24.9 -> 95.2;
- task-coarsened error: 0.126 -> 0.0327.

## 4. Data-driven coarsening proof of concept

`research/data_driven_task_coarsening.py` does not use the true macro grouping. It:
1. fits a K=4 Gaussian mixture to emissions;
2. estimates microstate residual slopes and standard errors;
3. evaluates the three possible pairings of four states;
4. selects the pairing minimizing within-pair Wald discrepancy;
5. pools treatment information within the selected macro-states.

At N=800 over 500 replications:
- microstate recovery ARI = 0.991;
- true causal pairing recovered = 94.8%;
- fully refined causal error = 0.459;
- task-coarsened error = 0.085;
- oracle macro-state error = 0.085;
- task coarsening beats refinement = 97.6%.

This proof-of-concept is selection-biased because grouping and effect estimation use the same sample. A rigorous method should cross-fit state merging or use sample splitting.

## 5. Connection to the directional reliability finding

The K=3 orientation experiment shows that equal scalar proxy error and nearly equal lambda_min(J) can produce 6x different causal errors. Thus there are two distinct failures of representation-only thinking:

1. **Granularity failure:** distinguishing more latent states can destroy downstream information.
2. **Orientation failure:** equal representation error magnitudes can have different downstream effects depending on score direction.

Both suggest replacing task-agnostic representation quality with a task-induced object such as
```
R_task(gamma) = || (J + lambda I)^(-1) b_gamma ||,
```
where b_gamma is the downstream score perturbation induced by the representation.

The existing scalar epsilon_gamma/lambda_min(J) is naturally interpreted as a worst-case upper bound/order diagnostic, not a complete task-relative metric.

## 6. Potential general statement

> A latent representation should not be judged solely by its ability to reconstruct the hidden data-generating state. The statistically appropriate representation for a downstream functional is the coarsest representation that preserves task-relevant heterogeneity while maintaining sufficient downstream information.

This resembles a task-specific sufficient abstraction rather than generative state recovery.

## 7. Required work before claiming a theorem/method

- formalize the class of permissible state partitions and task functionals;
- prove the rate statement under estimated rather than observed states;
- cross-fit the merge-selection step;
- derive a penalty balancing effect heterogeneity against information loss;
- compare against BIC/AIC, held-out emission likelihood, predictive state F1, and fixed-K baselines;
- test wrong K, semi-Markov durations, non-Gaussian emissions, gradual transitions, and heterogeneous nuisance functions;
- establish post-selection inference after data-driven merging;
- determine whether the task-aware abstraction transfers across downstream functionals or must be relearned per task.

No claim of general optimality is warranted yet.
