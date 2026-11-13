# Background and Literature Survey

## 2.1 Causal inference from observational data

Under *unconfoundedness*, the effect of a treatment on an outcome is identified by comparing units that are alike on all confounders. With a continuous treatment and many confounders this is done by regression, by propensity-style weighting, or by combinations of both. The partially linear model $Y = \theta T + g(X) + U$ with $T = m(X) + V$ is the workhorse: the effect $\theta$ is the coefficient on the *residual* of $T$ after removing what the controls explain (Robinson-style residualization).

## 2.2 Double / Debiased Machine Learning

Chernozhukov et al. (2018) showed that if the nuisance functions $m(X)=\mathbb{E}[T\mid X]$ and $\mu(X)=\mathbb{E}[Y\mid X]$ are estimated by flexible machine learning, then the residual-on-residual estimator of $\theta$ is still $\sqrt N$-consistent and asymptotically normal, provided two ingredients are present.

* **Neyman orthogonality.** The estimating equation is insensitive, to first order, to errors in the nuisances. First-order nuisance mistakes then contribute only second-order terms, which are negligible when nuisance rates are $o(N^{-1/4})$.
* **Cross-fitting.** Nuisances are estimated on data not used to evaluate the equation, removing own-observation overfitting bias.

Extensions include automatic debiasing (Chernozhukov, Newey and Singh, 2022), orthogonal statistical learning (Foster and Syrgkanis, 2023) and quasi-oracle heterogeneous effects (Nie and Wager, 2021). These results are stated for independent data and *observed* confounders.

## 2.3 Dependent data: time-series DML

Time series break the independence assumption. Random $K$-fold cross-fitting leaks information across neighbouring observations, because training and evaluation folds are correlated. Lewis and Syrgkanis (2020) treat dynamic treatment effects with g-estimation; Ciganovic et al. (2026) develop DML for time series. In financial machine learning, *purging* and *embargo* of training observations adjacent to the evaluation block are standard remedies (López de Prado, 2018). Our implementation uses contiguous blocks with an embargo of at least $C\log N$ steps, matching the mixing conditions needed for asymptotic theory. These works assume the confounders are observed.

## 2.4 Latent confounders and regime models

Several lines handle unobserved confounding.

* **Representation approaches.** Sequential deconfounders and the time-series deconfounder learn a latent confounder with neural encoders (Bica et al., 2020; Hatt and Feuerriegel, 2021). Posterior-based adjustment is known to fail without identification (Wang and Blei, 2019; D'Amour, 2019). Our results sharpen this caution: *representation quality and downstream reliability are different objects*.
* **Proximal causal inference.** Proxies of an unmeasured confounder identify effects under completeness conditions (Miao, Geng and Tchetgen Tchetgen, 2018; Tchetgen Tchetgen et al., 2020; Mastouri et al., 2021; Cui et al., 2024). The target there is identification; we study what a *plug-in* posterior leaves behind.
* **Regime-switching models.** Hamilton (1989) introduced Markov regime switching. The identifiability and asymptotics of hidden Markov models are classical (Leroux, 1992; Bickel, Ritov and Rydén, 1998; Douc, Moulines and Rydén, 2004; Allman, Matias and Rhodes, 2009). Causal inference for latent Markov models has been studied with the parametric g-formula (Clouth et al., 2025).
* **Stepwise latent-model analysis.** Estimating a latent-class model first and using its posteriors in a second stage is common, and classification error already motivates bias-adjusted corrections (Di Mari, Oberski and Vermunt, 2016). We test the natural two-stage heuristic, not joint likelihood methods.

## 2.5 Omitted-variable bias, amplification and overlap

Our bias laws are omitted-variable identities (Greenland, 1980; Cinelli and Hazlett, 2020; Chernozhukov et al., 2022), and their non-monotonicity is the algebra of *bias amplification*: conditioning on variables that explain the treatment but not the confounder can magnify the residual confounding (Pearl, 2012; Ding, VanderWeele and Robins, 2017). We claim neither identity as new. What is new is their calibrated, dependent-data and regime-coupled form.

Weak overlap is diagnosed in the literature by Gram-matrix or propensity diagnostics (Crump et al., 2009; D'Amour et al., 2021). A central finding here is that such conditioning diagnostics can move in the wrong direction when the state is generated.

## 2.6 Sensitivity analysis

When identification cannot be certified, the standard response is a sensitivity analysis that reports how strong an unmeasured shift would have to be to explain the estimate away (Cinelli and Hazlett, 2020). We adapt this to the HMM-regime channel as a one-parameter *robustness value* $b^\star$.

## 2.7 Research gap

@tab gap | Positioning against the closest lines of work
| Line of work | Observed confounders | Dependent data | Latent regime | Exact law for plug-in posterior | Reliability diagnostics |
|---|---|---|---|---|---|
| DML (Chernozhukov et al.) | yes | no | no | n/a | n/a |
| Time-series / dynamic DML | yes | yes | no | n/a | n/a |
| Deconfounders, sequential encoders | learned | yes | learned | no | accuracy of encoder |
| Proximal causal inference | proxies | partly | latent | identification only | completeness |
| Hamilton / HMM regime models | n/a | yes | yes | n/a | likelihood |
| **This work (OR-DML)** | yes | **purged block** | **HMM** | **yes** | **residual-state law, $b^\star$, placebos** |

The gap is therefore not "another estimator", but a quantitative account of what a generated state does to the downstream estimating equation, together with the estimator and reporting rules that account implies.
