Thank you. We accept that the presentation was below publication standard. As revisions cannot be uploaded now, we show the rewritten abstract here.

**Revised abstract.** *Observational time series are often driven by an unobserved, persistent state, or regime (for example the mixing regime of the atmosphere), that affects both a treatment and an outcome. A common remedy infers the regime from noisy auxiliary measurements (proxies) with a hidden Markov model and then applies Double Machine Learning (DML). Practitioners judge such pipelines by two signs: the inferred regime looks confident, and the matrix in the estimating equation looks well conditioned. We show that both signs can mislead: when the inferred regime is wrong, the conditioning can improve while the estimate gets worse ("phantom causal resolution"). We derive the exact bias, which depends on the uncertainty about the regime that remains after inference, give conditions under which the regime-specific effects are identified, and derive an estimator that removes this bias when those conditions hold. On simulated data, a benchmark calibrated to four Indian cities, and real sensor data, we show when each design works and when none does.*

*Causal error* is the distance between estimated and true regime-specific effects. *Regime-aware* means a separate effect for each latent regime.

**Other changes.**
1. Remove all draft-history language (e.g. "earlier drafts", p. 2) and all numbers from the introduction; use the space for a worked example (NO2 and PM2.5 under stagnant vs. ventilated air).
2. Replace the phrases you cite: "state that nobody measured" becomes "unobserved state"; "geometry of the equation" becomes "conditioning of the estimating equation"; "the closed form says how to build the estimator" becomes "the formula suggests the estimator"; "locates the damage in the nuisance step" becomes "shows the bias comes from the nuisance regressions".
3. Fix unclear references: "both quantities are routinely reported" will cite where; "its circularity" becomes "the circularity of using synthetic proxies whose error we set ourselves"; the "The mechanism..." paragraph gets a lead-in.
4. Add intuition before Theorems 1-2, split Corollary 1 into result and interpretation, and expand the description of the experimental setup and the semi-synthetic construction.

**W2.** The "implicit rule" is the practice of reading a confident posterior and a well-conditioned score matrix as evidence of reliability; we will cite where these diagnostics are reported. A "regime" is the value of the latent discrete state, not a confounder in general.

**W3.** We studied the two-stage heuristic because it is the widely used pipeline. Following the reviews, we now also compare joint-likelihood estimators and a new deconvolution estimator derived from an identification result (see our reply to Reviewer Qg1H).

**Q1.** Pearl (2010) shows that adding observed controls can amplify omitted-variable bias. Phantom resolution is a different object, a diagnostic failure for an *inferred* state: the conditioning number improves (70x in our factorial design) while error is non-monotone in proxy error. The algebra is shared; the generated-state setting is new.

**Q2.** We checked the hourly grid against the raw timestamps. Half-hour stamps had been rounded with ties-to-even, which collapsed them onto a two-hour grid. After rebuilding the grid, Delhi's single-index estimate moved from 0.58 to 0.37.

**Q3.**
- *Bartlett bands* are confidence bands for residual autocorrelation, used to choose how many time steps to leave out between folds.
- The *asymmetric factorial design* crosses proxy error with regime-specific treatment-noise scales, so the closed form can be checked cell by cell.
- *State confusion* means the posterior puts mass on the wrong regime.
- *Placebos* are leads of the treatment, aligned with the outcome's information set; their effect should be zero if the temporal interpretation holds.

**Title.** We propose to drop "Reliable", and to correct "Casual" in the OpenReview title, subject to the chairs' rules on title changes.
