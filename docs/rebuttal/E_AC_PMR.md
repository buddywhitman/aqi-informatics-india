We thank the AC. Briefly, by concern:

(1) **Inference.** We no longer claim nominal structural coverage for OR-DML. Theorem 5 covers the proxy target; the implementation (full-sample smoothing, fixed λ rule) is outside it, and we will say so.

(2) **Identification.** We now prove it. If the proxies affect (T,Y) only through the regime and E[γγᵀ|X] is nonsingular (informative proxies; Theorem 2 is the singular case), every regime-conditional moment, and hence θ*_k, is identified. Assumption 3 is not needed; it becomes a remark.

(3) **Estimator and baselines.** The identification result gives a deconvolution estimator (DC-DML). At Δ_Z=1 it has mean absolute ATE bias 0.51 vs. 2.13 for OR-DML, and 95% vs. 0% coverage. On the city-calibrated benchmark its coverage is 90-100% vs. 0%. We ran the Table 10 baselines on that benchmark too: a regime-specific joint MLE is best in 3 of 4 cities. These move to the main text, and OR-DML is presented as the best two-stage design, not the best estimator.

(4) **Clarity.** Rewritten abstract and introduction, definitions, unified notation, and the R-learner relation (reply to BBdA).
