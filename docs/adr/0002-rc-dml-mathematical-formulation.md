# ADR 0002: Mathematical Specification of Regime-Conditional Double Machine Learning (RC-DML)

## Status
Accepted

## Context
Standard Double Machine Learning (DML; Chernozhukov et al., 2018) is designed for cross-sectional independent and identically distributed (i.i.d.) observations under the conditional independence assumption $Y(t) \perp T \mid X$.

When applied to environmental, atmospheric, or financial time series, standard DML suffers two fatal failure modes:
1. **Omitted Regime Bias**: The true data generating process is governed by a latent Markov-switching regime $S_t \in \{1,\dots,K\}$ (e.g., atmospheric stagnation vs. advective clearance). Both treatment emissions $T_t$ and outcome pollutant retention $Y_t$ depend critically on $S_t$. Omission of $S_t$ induces severe confounding:
   $$\text{Bias}(\hat{\theta}_{\text{naive}}) = \frac{\mathbb{E}\left[(T_t - \mathbb{E}[T_t \mid X_t]) \cdot (\mathbb{E}[Y_t \mid X_t, S_t] - \mathbb{E}[Y_t \mid X_t])\right]}{\mathbb{E}\left[(T_t - \mathbb{E}[T_t \mid X_t])^2\right]} \neq 0$$
   This bias can exceed the magnitude of the causal effect and flip its sign (as observed in our audit of naive DML in coastal airsheds).
2. **Temporal Leakage in Cross-Fitting**: Standard random $K$-fold cross-fitting breaks down under temporal autocorrelation ($\alpha$-mixing), causing nuisance prediction errors to correlate across train/test splits and destroying $\sqrt{N}$-consistency.

## Decision

We formulate and implement **Regime-Conditional Double Machine Learning (RC-DML)** with the following mathematical architecture:

### 1. Structural Causal Model
$$S_t \sim \text{Markov}(\Pi)$$
$$T_t = m(X_t, S_t) + V_t, \quad \mathbb{E}[V_t \mid X_t, S_t] = 0$$
$$Y_t = \theta(S_t) T_t + g(X_t, S_t) + U_t, \quad \mathbb{E}[U_t \mid X_t, S_t, T_t] = 0$$
where $\theta(S_t) = \sum_{k=1}^K \theta_k^* \mathbb{I}(S_t = k)$ is the regime-dependent causal elasticity.

### 2. Latent Regime Inference
Latent states are inferred using an Expectation-Maximization (EM) algorithm for Hidden Markov Models over atmospheric state dynamics $Z_t \in \mathbb{R}^p$ (wind velocity, thermal gradient, boundary layer proxy), yielding posterior smoothing probabilities:
$$\gamma_{tk} = P(S_t = k \mid Z_{1:N})$$

### 3. Regime-Conditioned Neyman-Orthogonal Scores
For each regime $k \in \{1,\dots,K\}$, we define the doubly robust score:
$$\psi_k(W_t; \theta_k, \eta_k) = \gamma_{tk} \cdot \left[ (Y_t - \ell_k(X_t)) - \theta_k (T_t - m_k(X_t)) \right] (T_t - m_k(X_t))$$
where the nuisance vector is $\eta_k = (\ell_k, m_k)$ with $\ell_k(X) = \mathbb{E}[Y \mid X, S=k]$ and $m_k(X) = \mathbb{E}[T \mid X, S=k]$.

The estimator is given in closed form:
$$\hat{\theta}_k = \frac{\sum_{t=1}^N \gamma_{tk} (Y_t - \hat{\ell}_k(X_t)) (T_t - \hat{m}_k(X_t))}{\sum_{t=1}^N \gamma_{tk} (T_t - \hat{m}_k(X_t))^2}$$

### 4. Purged Block-Temporal Cross-Fitting
To guarantee that nuisance estimation satisfies the rate conditions $\|\hat{\ell}_k - \ell_k\|_2 \|\hat{m}_k - m_k\|_2 = o_P(N^{-1/2})$ under temporal dependence:
1. Divide time series $\{1,\dots,N\}$ into $B$ contiguous chronological blocks $\mathcal{B}_1, \dots, \mathcal{B}_B$.
2. For each block $b$, construct training set $\mathcal{T}_b = \{t : \min_{t' \in \mathcal{B}_b} |t - t'| > \tau\}$, where $\tau$ is an embargo buffer exceeding the empirical autocorrelation horizon of the system.
3. Fit nuisance estimators $\hat{\ell}_k^{(-b)}$ and $\hat{m}_k^{(-b)}$ on $\mathcal{T}_b$, and evaluate scores on $\mathcal{B}_b$.

### 5. Asymptotic Inference
Under $\alpha$-mixing conditions with mixing coefficients $\alpha(m) \le C \rho^m$ ($\rho < 1$) and mild overlap, $\hat{\theta}_k$ is asymptotically normal:
$$\sqrt{N}(\hat{\theta}_k - \theta_k^*) \xrightarrow{d} \mathcal{N}(0, \sigma_k^2)$$
with asymptotic variance estimated via the heteroskedasticity-and-autocorrelation-consistent (HAC) sandwich estimator.

## Consequences
- Solves the sign inversion and omitted variable bias of naive DML on non-stationary time series.
- Provides a rigorous, mathematically complete foundation for an AISTATS 2027 methodological contribution.
- Implemented in `src/rc_dml.py` with reproducible benchmarks in `src/synthetic_dgp_benchmark.py`.
