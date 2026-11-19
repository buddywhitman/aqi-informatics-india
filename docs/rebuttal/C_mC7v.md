Thank you. Both weaknesses are correct, and we have addressed them as follows.

**W1 (calibration).** Agreed: $\hat v$ measures residual state uncertainty only if the posterior is calibrated given the adjustment variables. A misspecified HMM can be overconfident; under correlated $t_5$ emissions the plug-in understates the realized residual variance by a factor of two (App. G). We will state this caveat in the abstract and in Sec. 4, and recommend held-out likelihood and multistart checks. The sensitivity value $b^\star$ is reported as conditional on calibration.

**W2 (inference vs. procedure).** Agreed. Theorem 5 targets the proxy estimand and assumes a separate calibration window, while the implementation uses full-sample smoothing. The implemented penalty, $\lambda=0.05\,\mathrm{tr}(\hat J)/(K\sqrt N)$, also has $\sqrt N\lambda$ constant, so it does not meet Theorem 5's $\lambda=o(N^{-1/2})$; its shrinkage bias is about 0.004 on our frontier. We will state that nominal structural coverage is **not** claimed for OR-DML.

Prompted by this review we added two things.

(i) **An identification result.** Under an exclusion restriction, $(T,Y)\perp Z\mid(X,S)$, and informative proxies ($E[\gamma\gamma^\top\mid X]$ nonsingular), every regime-conditional moment is identified as $M(X)^{-1}E[\gamma h\mid X]$, and so is $\theta_k^*$.

(ii) **An estimator built on it (DC-DML), with a 95% interval for the structural effect.** On our frontier at $\Delta_Z=1$, coverage is 95% vs. 0% for OR-DML, with mean absolute bias 0.51 vs. 2.13. On the real-data-calibrated benchmark coverage is 90-100% vs. 0%. The intervals use a block bootstrap with the posterior held fixed, so they ignore HMM-estimation error (as the paper's intervals do). Joint coverage of both regime effects is 81% at $\Delta_Z=1$. Details are in our reply to Reviewer Qg1H.
