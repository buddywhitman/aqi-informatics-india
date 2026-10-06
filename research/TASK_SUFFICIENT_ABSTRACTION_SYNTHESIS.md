# Task-Sufficient Latent Abstraction: Exploratory Research Synthesis

**Not part of the AISTATS submission.**

## Core empirical facts established on this branch

1. **State-refinement paradox.** Under local weak residual treatment information sigma_N=N^{-alpha}, estimating perfectly observed microstate effects separately has error O_p(N^{alpha-1/2}); a causally sufficient merge can remain O_p(N^{-1/2}). For alpha>1/2 the refined estimator diverges while the coarsened estimator converges.

2. **Generative/downstream conflict.** Four-state BIC beats two-state BIC in 100% of simulations and by a growing margin with N, while the task-aware two-state estimator has 4-5x lower causal error under fixed weak overlap.

3. **Held-out learnability.** A sample-split Wald merger recovers the correct task partition 93-97% of the time and reduces held-out causal error about 5x despite microstate ARI about 0.99.

4. **Bias-information frontier.** Coarsening helps when within-group task heterogeneity is small and hurts when it is large. In the current sweep, average error gain moves from +65% at delta=0 to +10% at delta=.5, then -56% at delta=1 and -175% at delta=2.

5. **Task dependence.** The same latent representation requires incompatible abstractions for two different causal functionals; sample-split selection recovers both task-specific partitions.

6. **Directional representation risk.** In K=3, equal average posterior error and similar lambda_min can yield 6x different causal errors. Scalar eps/lambda_min is a worst-case/order diagnostic, not a sufficient general representation metric.

7. **Adaptive selection is possible but imperfect.** A train-only heterogeneity+variance risk criterion nearly matches oracle partition choice in low/moderate heterogeneity, but struggles near the crossover where refinement becomes preferable.

## Candidate formal framework

Let S be a fine latent state and phi:S->A a task abstraction/partition. Let tau(S) be the fine-state causal functional and tau_phi(A) the estimand after pooling. Define

    R(phi) = B_task(phi)^2 + V_task(phi) + U_repr(phi),

where:
- B_task measures heterogeneity erased by pooling;
- V_task is estimation variance/information loss under the downstream score geometry;
- U_repr is error from estimating/assigning the latent abstraction.

The best generative representation minimizes a likelihood/reconstruction objective. The best downstream abstraction minimizes R(phi). These objectives need not agree.

A sharper local representation-risk object is

    R_dir(gamma) = || (J + lambda I)^(-1) b_gamma ||,

with b_gamma the downstream score perturbation caused by representation error. The familiar eps_gamma/lambda_min(J) is a worst-case bound when ||b_gamma|| is controlled by eps_gamma.

## Candidate theorem sequence for future work

**Theorem A (Refinement rate penalty).** If a refined microstate has probability bounded away from zero but residual treatment variance N^{-2alpha}, its separately estimated effect has standard error O(N^{alpha-1/2}). Any aggregate retaining nonvanishing weight on that estimate inherits this rate.

**Corollary (Refinement inconsistency).** For alpha>=1/2, the refined aggregate is not root-N; for alpha>1/2 it is inconsistent/divergent under the local sequence.

**Theorem B (Task-sufficient pooling rescue).** If a weak microstate and a strong-information microstate share the target effect and are pooled, with strong-state treatment information bounded away from zero, the pooled effect remains root-N.

**Theorem C (Approximate pooling).** If within-group effect heterogeneity is delta_N, pooling risk is approximately delta_N^2 times an occupancy/information factor plus O(1/N) variance. This gives a threshold comparing delta_N against the information loss of refinement.

**Theorem D (Directional proxy perturbation).** First-order target displacement under a generated latent representation is J^{-1}b_gamma; eps_gamma/lambda_min(J) follows as a worst-case bound under ||b_gamma||<=C eps_gamma.

## Critical caveats

- Similar ideas exist in causal abstraction, RL state abstraction/bisimulation, weak-overlap representation learning, unnecessary adjustment, and causal coarsening. Novelty must rest on the precise latent-confounder/weak-overlap rate phenomenon and an estimable downstream abstraction procedure, if literature review confirms the gap.
- Current coarsening algorithms use very small K and enumerate partitions. Scaling requires graph/agglomerative/continuous relaxations.
- Valid confidence intervals after learned partition selection remain unsolved.
- If heterogeneity is large, coarsening is harmful; no universal preference for coarse states is justified.
- Real-data K sensitivity is suggestive only because the true state/effects are unknown.
