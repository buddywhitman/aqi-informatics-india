# Overlap-Aware Regime DML: Method and Implementation

## 4.1 Design principle

The estimator is built from one observation: the posterior enters the downstream estimating equation **twice**. It acts as an *adjustment feature* inside the nuisance regressions, where it supplies fine-grained information about the state, and as a *weight* in the score that defines the regime effects, where it selects a coarse target. The failure analysis of Chapter 5 shows that these two roles must be kept apart: using the posterior as a weight on the nuisance fit leaks regime contrasts into the score, while using it as a feature removes them up to what the posterior cannot explain.

## 4.2 The estimator

**Step 1: latent-state inference.** A Gaussian HMM is fitted to the proxies $Z$ by EM with multiple restarts. Posteriors are either smoothed, $\gamma_{tk} = P(S_t{=}k \mid Z_{1:N})$, or causally filtered, $P(S_t{=}k \mid Z_{1:t})$.

**Step 2: purged block cross-fitting.** The series is split into $B$ contiguous blocks. For each block, nuisances are fitted on the other blocks after dropping an embargo of $\tau^*$ steps on each side of the evaluation block (Figure {{fig:cf}}). The embargo is $\tau^* = \max\{h : \max_j |\hat\rho_j(h)| \ge 1.96/\sqrt N\} + 1$, and never less than $C\log N$, a floor required by the mixing argument.

@svg cf | crossfit | Purged block cross-fitting. Training observations within the embargo $\tau^*$ of the evaluation block are dropped so that temporal dependence cannot leak between folds.

**Step 3: posterior-adjusted nuisances.** One pair of nuisances is cross-fitted on the posterior-augmented features $F_t = (X_t, \gamma_t, \gamma_t \otimes X_t)$: $\hat m(F) \approx \mathbb{E}[T \mid F]$ and $\hat\mu(F) \approx \mathbb{E}[Y \mid F]$. With $\tilde T_t = T_t - \hat m(F_t)$ and $\tilde Y_t = Y_t - \hat\mu(F_t)$ the coupled system is

$$\hat J_{jk} = \frac1N \sum_t \gamma_{tj}\gamma_{tk}\,\tilde T_t^2, \qquad \hat S_k = \frac1N\sum_t \gamma_{tk}\,\tilde T_t \tilde Y_t, \tag{4.1}$$
$$\hat{\boldsymbol\theta}_\lambda = (\hat J + \lambda I)^{-1}\hat S. \tag{4.2}$$

**Step 4: spectral regularization.** The penalty $\lambda \ge 0$ stabilizes the inversion when $\lambda_{\min}(\hat J)$ is small. On a documented path it reduces variance sharply but introduces bias, so $\lambda$ is a tuning knob, not free lunch (Section 7.6).

**Step 5: inference.** A HAC sandwich variance uses the observation-level moments actually solved, including off-diagonal terms. A residual bootstrap uses the same moment matrices as the point solve.

@thm Algorithm 1: OR-DML with a reliability report
**Input:** $\{Y_t, T_t, X_t, Z_t\}_{t=1}^N$, number of folds $B$, penalty $\lambda$.
1. Fit a Gaussian HMM to $Z$ (multi-restart EM); compute posteriors $\gamma_t$ (smoothing or causal filter).
2. Choose embargo $\tau^* = \max\{h : \max_j|\hat\rho_j(h)| \ge 1.96/\sqrt N\} + 1$, at least $C\log N$.
3. Cross-fit $\hat m, \hat\mu$ on $F_t = (X_t, \gamma_t, \gamma_t\otimes X_t)$ over purged contiguous blocks.
4. Form $\hat J, \hat S$ by (4.1); return $\hat\theta_\lambda = (\hat J + \lambda I)^{-1}\hat S$ with a HAC sandwich.
5. **Report:** hard-FE and posterior-feature baselines; $\hat v$, $\hat a$, $\widehat{\mathrm{Var}}(\tilde T)$ and the robustness value $b^\star$; raw and standardized $\lambda_{\min}$; lead placebos aligned to the outcome's information set.
@end

Complexity is $\mathcal{O}(B\,C_{\rm nuis} + NK^2 + K^3)$, where $C_{\rm nuis}$ is the cost of one nuisance fit. A scalability experiment checks $K \le 5$ regimes (Section 7.7).

## 4.3 Baselines

Two simpler estimators on the same inputs serve as the main baselines, alongside standard DML, block DML and DML with $Z$ as extra controls.

* **Hard regime fixed effects:** one-hot $\arg\max$ weights, the practitioner's default after HMM inference.
* **Posterior-feature DML:** a single-index DML whose nuisances are on $F_t$, with no regime-coupled score.
* **OR-DML with weighted nuisances:** the design of earlier project drafts, in which regime-$k$ nuisances are fitted on $X$ with weights $\gamma_{\cdot k}$. It is kept as an *ablation*; its Gram part is not orthogonal for soft weights (Section 5.1).

## 4.4 Software architecture

The estimator is implemented in `src/or_dml.py` (648 lines) in scikit-learn style. The package splits into three decoupled parts, shown with the classes that implement them.

@tab arch | Software components of the estimator
| Component | Class / function | Role |
|---|---|---|
| Latent-state inference | `LatentRegimeHMM` | multi-restart Gaussian-HMM EM; smoothed and filtered posteriors; BIC |
| Temporal validity | `estimate_optimal_embargo`, `PurgedBlockKFold` | embargo choice with a $C\log N$ floor; purged contiguous folds |
| Coupled estimator | `OverlapAwareRegimeDML` | nuisance cross-fitting, $\hat J$, $\hat S$, spectral solve, HAC and bootstrap uncertainty, dynamic IRF |
| Regularization | `AdaptiveSpectralOptimizer` | data-driven $\lambda$ with an oracle risk bound |

The *posterior* nuisance mode, which implements the paper's design, builds the augmented feature matrix and fits one pair of nuisances per fold (full listings in Annexure C):

```python
G = weights[:, 1:]                      # posterior probabilities, reference regime dropped
F = np.column_stack([X, G] + [G[:, [k]] * X for k in range(G.shape[1])])
for train_idx, test_idx in splitter.split(N):      # purged blocks
    m_y = clone(self.nuisance_model).fit(F[train_idx], Y[train_idx])
    m_t = clone(self.nuisance_model).fit(F[train_idx], T[train_idx])
    ry = Y[test_idx] - m_y.predict(F[test_idx])    # out-of-fold residuals
    rt = T[test_idx] - m_t.predict(F[test_idx])
```

and the coupled solve forms the Gram matrix and applies the spectral penalty:

```python
for j in range(K):
    S_vec[j] = np.mean(w[:, j] * tilde_T[j] * tilde_Y[j])
    for k in range(K):
        J_mat[j, k] = np.mean(w[:, j] * w[:, k] * tilde_T[j] * tilde_T[k])
theta_vec = np.linalg.inv(J_mat + eff_lambda * np.eye(K)) @ S_vec
```

## 4.5 Verification of the implementation

Regression tests in `src/test_inference_regression.py` cover the four score/bread mode combinations, score-mean invariance of the HAC, the occupation cross-covariance, bootstrap moment consistency and the uncapped logarithmic embargo floor. Each defect that these tests guard was first reproduced against the earlier implementation (Chapter 10). `verify_science.py` and `verify_submission.py` provide higher-level checks (Chapter 9).
