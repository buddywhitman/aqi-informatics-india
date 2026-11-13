# Theory: How State Uncertainty Propagates

Let $H_t = e_{S_t}$ be the one-hot true state, let $\varepsilon_\gamma = \mathbb{E}\|\gamma_t - H_t\|_1$ be the posterior error, and let $J, S$ be the population versions of the coupled system. Define $\theta_\gamma = J^{-1}S$ and $b_\gamma = S - J\theta^*$, so that $\theta_\gamma - \theta^* = J^{-1} b_\gamma$ exactly.

## 5.1 Exact orthogonality

@thm Lemma 1 (Exact orthogonality)
Let $\psi_k(\theta; m, \mu) = \gamma_k (T - m)\{Y - \mu - \sum_j \gamma_j \theta_j (T-m)\}$. At $m = \mathbb{E}[T\mid F]$ and $\mu = \mathbb{E}[Y \mid F]$, the Gateaux derivative of $\mathbb{E}\psi_k$ in every direction $(\delta_m, \delta_\mu)$ of square-integrable functions of $F$ vanishes, for every $\theta$ and every posterior $\gamma$; so does that of $\mathbb{E}[\gamma_j\gamma_k (T-m)^2]$.
@end

*Proof.* Each derivative is an expectation of a product of an $F$-measurable factor (the weights $\gamma$ and the perturbation $\delta$) and either $T-m$ or $Y - \mu - \sum_j\gamma_j\theta_j(T-m)$, whose conditional mean given $F$ is zero. $\square$

The natural alternative, used in earlier drafts, fits regime-$k$ nuisances on $X$ with weights $\gamma_{\cdot k}$. Its Gram part is **not** orthogonal for soft weights: under a nuisance perturbation its score derivative is 85% of the Jacobian, against 0.4% for the posterior-adjusted design of Lemma 1. This is the formal reason OR-DML conditions on the posterior rather than weighting by it.

## 5.2 Graceful degradation

@thm Theorem 3 (Graceful degradation)
Suppose the nuisance targets are within $C_\eta\varepsilon_\gamma$ of the oracle regime nuisances in $L^1$, and $\lambda_{\min}(J) > 0$. Then $\|\theta_\gamma - \theta^*\|_2 \le C\varepsilon_\gamma/\lambda_{\min}(J)$ and
$$\|\hat\theta_\gamma - \theta^*\|_2 \le \frac{C\varepsilon_\gamma}{\lambda_{\min}(J)} + \mathcal{O}_P\!\Big(\frac{N^{-1/2} + \rho_N}{\lambda_{\min}(J)}\Big),$$
with $\rho_N = r_N^2$ for posterior-adjusted nuisances and $\rho_N = r_N$ for posterior-weighted ones, where $r_N$ is the nuisance rate.
@end

The bound separates the two errors that the problem involves: **state recovery** ($\varepsilon_\gamma$) and **nuisance estimation** ($\rho_N$). Posterior adjustment improves the nuisance term from first to second order. In a controlled design the ratio $\varepsilon_\gamma/\lambda_{\min}(\hat J)$ ranks causal error almost perfectly (Spearman 0.995 over 4,900 runs, against 0.626 for $\varepsilon_\gamma$ alone and 0.521 for $1/\lambda_{\min}$). The next results explain why that success is fragile: $\lambda_{\min}(J)$ is itself a function of the state error.

## 5.3 The coupled score under state confusion

Consider two regimes with stationary weights $\pi_s$, treatment means $m_s$, outcome means $\mu_s = \theta_s m_s + g_s$ and innovation scales $\sigma_s$. Take regime-wise nuisances and a proxy that places weight $1-\alpha$ on the true regime and $\alpha = \varepsilon_\gamma/2$ on the other.

@thm Proposition 1 (Exact law for regime-weighted scores)
With $\gamma_j(s) = 1-\alpha$ if $j=s$ and $\alpha$ otherwise,
$$J_{jk} = \sum_s \pi_s \gamma_j(s)\gamma_k(s)\big[(m_s - m_j)(m_s - m_k) + \sigma_s^2\big],$$
$$S_j = \sum_s \pi_s \gamma_j(s)\big[(m_s - m_j)(\mu_s - \mu_j) + \theta_s \sigma_s^2\big].$$
In the symmetric case ($\pi_s = \tfrac12$, $\sigma_s = \sigma$, $\Delta = m_1 - m_0$, $G = \mu_1 - \mu_0$), $J$ has eigenvectors $(1, \pm 1)$ with eigenvalues $\lambda_+ = \tfrac12\sigma^2 + \tfrac12\alpha^2\Delta^2$ and $\lambda_- = \tfrac12\sigma^2(1-2\alpha)^2 + \tfrac12\alpha^2\Delta^2$.
@end

@thm Corollary 1 (Phantom resolution and the error hump)
In the symmetric case: **(a)** $\lambda_{\min}(J) = \lambda_-$ *exceeds* its oracle value $\sigma^2/2$ whenever $\varepsilon_\gamma > 8\sigma^2/(\Delta^2 + 4\sigma^2)$, so a confused state can look *better* conditioned than the true one; **(b)** when $|\Delta G| \gg \sigma^2|\theta_0 - \theta_1|$ the error is $\approx \sqrt2\,\tfrac{\varepsilon_\gamma}{2}|\Delta G|/(\sigma^2 + \varepsilon_\gamma^2\Delta^2/4)$, maximized at $\varepsilon^\star = 2\sigma/|\Delta|$ and *decreasing* beyond it; **(c)** the dominant displacement lies on the $\lambda_+$ eigenvector, so the worst-direction bound of Theorem 3 is loose by $\lambda_+/\lambda_-$.
@end

Proposition 1 is exact and involves no sampling. Applied to the asymmetric factorial design it reproduces every cell's median error to within 0.98–1.02 times the observed value and locates the error peaks at $\varepsilon_\gamma \in \{0.05, 0.10, 0.20\}$ for $\sigma_1 \in \{0.08, 0.15, 0.30\}$, exactly where they are observed (Figure {{fig:phantom}}). Mixing regimes leaks between-regime treatment contrasts into the residualized treatment, which inflates the Gram matrix. The factorial's near-perfect rank correlation is therefore an *algebraic identity of the design*, not a statistical regularity: replacing $\lambda_{\min}(\hat J)$ by the design value $\sigma_1^2$ lowers the cell-level correlation from 0.997 to 0.958.

@fig phantom | fig1_phantom_resolution_hi.png | **Phantom causal resolution.** (a) Coupled regime-score error against proxy error $\varepsilon_\gamma$ for four treatment-innovation scales $\sigma_1$ (dots: median of 100 runs of the $7\times7$ factorial; lines: the exact law of Proposition 1, no fitted parameters). Error is non-monotone, peaking near $\varepsilon^\star \approx 2\sigma/|\Delta_m|$. (b) The smallest eigenvalue of the generated-state Gram matrix: unless $\sigma_1$ is large it *grows* with proxy error (70-fold at $\sigma_1 = 0.08$). (c) With posteriors learned by EM (720 runs) $\lambda_{\min}(\hat J)$ exceeds the oracle-state value in 82% of runs. | w

## 5.4 Posterior-adjusted designs and the residual-state bias law

Posterior-adjusted nuisances remove the contrasts that weighted nuisances leak, up to what the posterior cannot explain. Let $v_t = \gamma_t(1-\gamma_t)$ be the posterior variance of the regime indicator, and let $T_t = m(X_t) + aS_t + V_t$ and $Y_t = \theta_{S_t}T_t + g(X_t) + bS_t + U_t$, with treatment regime shift $a$ and outcome regime shift $b$.

@thm Theorem 4 (Residual-state bias law)
Let $K=2$ and suppose the posterior is calibrated given $F$, $V_t$ is independent of $(S,Z,X)$ with variance $\sigma^2$, nuisances are cross-fitted at rates with $r_N^2 = o(N^{-1/2})$, and the posterior forgets distant proxies. With score weights $w_t = (1-\gamma_t, \gamma_t)$ (OR-DML) or $w_t \equiv 1$ (posterior-feature DML),
$$\hat\theta \to_p \bar\theta + J^{-1} b_\gamma, \qquad J = \mathbb{E}\big[w w^\top (a^2 v + \sigma^2)\big], \quad b_\gamma = a\,\mathbb{E}[\,w\,c\,v\,],$$
with $c_t = b + (\theta_1 - \theta_0)\{m(X_t) + a(1-\gamma_t)\}$ and $\bar\theta = \theta^*$ for OR-DML. Hence: **(a)** with no treatment regime shift ($a=0$) both are consistent for any calibrated posterior, whatever $b$ and the effect heterogeneity; **(b)** for homogeneous effects the pooled bias is $abv/(a^2v + \sigma^2)$, hump-shaped in $a$ with peak at $|a| = \sigma/\sqrt v$ and $\sup|B| = |b|\sqrt v/(2\sigma)$, increasing in $v$, and a soft posterior feature is never worse than its hard label; **(c)** $J$ grows with $a^2v$, so phantom resolution is not specific to the weighted design.
@end

The law is exact up to sampling error. Over 9 configurations with $N = 10^6$ the estimator and the law agree to within 0.008, and with $a = 0$ the bias is 0.005. With learned HMM posteriors, the plug-in form of the law matches measured bias across 72 cells with correlation 0.996 and median relative error 5.2% (Figure {{fig:law}}).

Both laws have the same anatomy: a numerator proportional to *state error times the product of the treatment and outcome regime shifts*, divided by a *residual treatment variance* that the same treatment shift inflates. This is why $\mathrm{Var}(\tilde T)$, the scalar analogue of $\lambda_{\min}(\hat J)$, cannot be read as a monotone certificate.

@fig law | fig2_bias_law_hi.png | **The residual-state bias law with learned HMM posteriors.** (a) Measured bias is hump-shaped in the treatment regime shift and matches the law evaluated with the observable plug-in $\hat v$ (lines). (b) The same bias against $\mathrm{Var}(\tilde T)$: conditioning improves monotonically while bias first rises and then falls. (c) Soft posterior features beat hard labels; DML on $X$ alone is far worse; the oracle state is unbiased. | w

## 5.5 Observability, sensitivity and inference

For a calibrated posterior, $\hat v = N^{-1}\sum_t \hat\gamma_t(1-\hat\gamma_t)$ satisfies $|\mathbb{E}\hat v - v| \le \sqrt{CE} + CE$ in the $L^2$ calibration error $CE$, and $\mathrm{Var}(\tilde T)$ is estimable. If the calibrated regime-mean model identifies the treatment shift $a$, only the outcome shift $b$ remains unidentified, so the **robustness value**
$$b^\star = \frac{|\hat\theta|\,\widehat{\mathrm{Var}}(\tilde T)}{\|\hat\Sigma\hat a\|}$$
(the smallest outcome shift that explains the estimate away) gives a one-parameter sensitivity analysis. Under misspecification $\hat v$ may understate or overstate the residual state variance, so sensitivity values require calibration assumptions.

@thm Theorem 5 (Inference around the proxy target)
For posterior-adjusted nuisances, under Theorem 3 and boundedness, augmented-process $\beta$-mixing, nuisance-rate and differentiability conditions, with $\tau^* \ge C\log N$ and a separated HMM calibration window, $\sqrt N(\hat\theta_\lambda - \theta_\lambda) \to \mathcal{N}(0, \Sigma^{\rm tot}_\lambda)$, including representation uncertainty. Here $\theta_\lambda - \theta^* = (J + \lambda I)^{-1}(b_\gamma - \lambda\theta^*)$ and $\|b_\gamma\| \le C\varepsilon_\gamma$. Structural coverage requires displacement $o(N^{-1/2})$.
@end

When proxy displacement is not negligible, structural Wald tests need not have nominal size. The report therefore pairs estimates with sensitivity analysis rather than an oracle bias correction. Implemented HAC intervals condition on the fitted HMM; **full-sample smoothing lies outside this theorem**, a limit recorded explicitly in the manuscript.

## 5.6 Summary of what the theory says

@tab thsum | Summary of the theoretical results
| Result | Statement | Practical consequence |
|---|---|---|
| Theorem 1 | Standard DML is biased by omitted regimes; more controls can amplify | do not rely on standard or block DML |
| Theorem 2 | No state information gives non-identified regime effects | proxies must carry information |
| Lemma 1 | Posterior-adjusted coupled score is exactly Neyman-orthogonal | condition nuisances on the posterior |
| Theorem 3 | Error $\le C\varepsilon_\gamma/\lambda_{\min}$ plus nuisance terms | state and nuisance errors separate |
| Proposition 1, Cor. 1 | Exact error hump and phantom resolution | $\lambda_{\min}$ is not a monotone certificate |
| Theorem 4 | Residual-state bias law | report $\hat v$, regime shifts, $b^\star$ |
| Theorem 5 | Asymptotic normality around the proxy target | intervals conditional on fitted HMM only |
