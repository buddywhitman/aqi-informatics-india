# Conclusions, Limitations and Future Work

## 11.1 Conclusions

The intuition that motivated the project was that a confident latent state together with a well-conditioned score Jacobian certifies a causal estimate. It does not. A generated state enters the downstream estimating equation twice, in its bias and in its apparent information, and the two move together. The scalar $\varepsilon_\gamma/\lambda_{\min}$ is an honest worst-case bound and an excellent ranking device when the state error is synthetic and the regime shifts are fixed. It degrades sharply once posteriors are learned (Spearman 0.995 to 0.517) because its denominator is partly made of the error that it is meant to measure.

Two rules survive.

* **Design.** Condition the nuisances on the posterior and use the posterior only to weight the target. This makes the coupled score exactly orthogonal and gave the most accurate regime estimates in every benchmark of the project.
* **Reporting.** State the residual state variance, the regime shifts it multiplies and a robustness value; compare against hard fixed-effects and posterior-feature baselines; and test temporal validity with aligned placebos.

Against the objectives of Chapter 1: the dataset was assembled and audited (O1); the propagation of state error was characterized exactly (O2); an estimator with proven orthogonality was implemented and verified (O3); it was validated from fully synthetic to semi-synthetic to real data (O4); a peer-reviewed research submission was produced (O5); and every number is regenerable and machine-checked (O6).

## 11.2 Limitations

* The laws are probability limits under linear regime shifts, exogenous treatment innovations and calibrated posteriors.
* The robustness value bounds rather than corrects, and it depends on calibration under misspecified emissions.
* Under weak observability every proxy-based estimator, OR-DML included, has large bias and no coverage. For the pooled ATE, posterior-feature DML is sometimes more accurate.
* The four-city analysis is observational: proxies overlap controls, lagged pollutants may be mediators, NO₂ is chemically related to PM₂.₅, and Delhi's lead placebo rejects.
* Inference theory covers posterior-adjusted coupled moments and excludes full-sample smoothing; implemented intervals condition on the fitted HMM.
* The raw data span about sixteen months, and Kolkata's about four, which limits seasonal generalization.

## 11.3 Threats to validity

* **Construct validity.** NO₂ stands in for traffic and combustion emissions, and PM₂.₅ for particulate burden. Neither is a policy intervention, and NO₂ takes part in the chemistry that forms secondary PM₂.₅. The estimated quantity is therefore an exposure association unless a policy-shock design supplies exogenous variation.
* **Internal validity.** The simulation designs use linear regime shifts and Gaussian proxies, so the exact laws are exact for those designs only. The real-data-calibrated benchmark relaxes this by sampling states from an HMM fitted to each city, but the injected causal effects are still known constants. Most experiments use two regimes, which could favour a two-state method; the continuous-mixture experiment (Section 7.3) and the scalability check to $K=5$ (Section 7.7) address this only in part.
* **Statistical conclusion validity.** Replication counts are 30 to 100 datasets per cell. Differences between estimators that are small relative to Monte Carlo error (for example at high proxy separation) are reported as ties, not wins. Observational intervals condition on the fitted HMM and ignore its estimation uncertainty.
* **External validity.** The data cover about sixteen months and four cities (about four months for Kolkata), with sensors of uneven quality. Seasonal and multi-year generalization is untested, and conclusions about specific cities should not be transferred to other airsheds.
* **Reproducibility risk.** Every number is regenerated from committed files, but the raw sensor files are third-party data whose upstream records may be revised.

## 11.4 Future work

**The dual-factor view of reliability.** Preliminary experiments (documented in `docs/FUTURE_WORK_AND_FRONTIERS.md`) suggest that failure is predicted by the *product* of representation ambiguity and task conditioning, rather than by either factor alone, and that selective abstention on the hardest windows can reduce risk sharply. These results predate the corrections of Chapter 10 and are therefore proposed as hypotheses to be re-tested under the learned-posterior protocol, not as findings.

**Software.** Packaging the estimator as an open-source library (`py-ordml`) with a scikit-learn-style interface, modular regime models, EconML compatibility and interactive documentation.

**Theory.** Extension beyond first-order Markov chains to semi-Markov dwell times; end-to-end training of the state model with a causal loss while preserving orthogonality; distribution-free (conformal) guarantees for abstention.

**Domains.** The mechanism is not specific to HMMs. Any generated state, such as a neural encoder's regime embedding, a clustering label or a changepoint segmentation, enters a downstream estimating equation both as an adjustment and as a source of apparent information. Applications that share the structure include intensive-care treatment under latent physiological phases, execution-cost estimation under latent market regimes, and offline reinforcement learning in partially observable environments.

**Air quality.** Chemistry-aware treatment definitions and instrument-based designs (for example, policy shocks such as lockdowns or festival-related restrictions) are the natural route to emission-policy claims that the present observational analysis cannot make.

## 11.5 Closing remark

The project began by trying to estimate how much traffic-related pollution matters. It ended by establishing, exactly and checkably, when such an estimate from latent-state time series should and should not be believed, and by building the tools to report that distinction. In a field where confident numbers are cheap, this is the contribution we are most willing to defend.
