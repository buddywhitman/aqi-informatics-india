"""
or_dml.py
=========
Overlap-Aware Regime Double Machine Learning (OR-DML)
=====================================================
A mathematically principled causal effect estimator under latent, persistent Markov
regimes with imperfect state separation and temporal dependence.

Methodological Highlights:
1. Impossibility & Proxy Formulation: Explicitly models latent state proxy error
   \\varepsilon_\\gamma = E||\\gamma_t - e_{S_t}||_1.
2. Spectral Regularization: Solves (J_hat + \\lambda I)^{-1} S_hat to trade off
   regularization bias against collinearity noise amplification under weak overlap.
3. Diagnostic Frontier: Tracks lambda_min(J), condition number kappa(J), and
   posterior entropy H(gamma) as physical observability diagnostics.
4. Temporal Safety: Purged Block-Temporal Cross-Fitting with Bartlett embargo tau.
5. Dynamic Impulse Responses: Multi-horizon local projections h = 0, ..., H.
6. Filtering vs Smoothing: Evaluates retrospective smoothing gamma_{t|1:N} vs
   online causal forward filtering gamma_{t|1:t}.
"""

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.base import clone
from typing import Dict, List, Tuple, Optional, Union


class LatentRegimeHMM:
    """
    Gaussian Hidden Markov Model for temporal regime discovery from exogenous
    meteorological dynamics Z_t. Supports both forward-filtering and retrospective smoothing.
    """
    def __init__(self, n_regimes: int = 3, n_iter: int = 25, tol: float = 1e-3, random_state: int = 42):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.pi = None       # Initial state distribution (K,)
        self.A = None        # Transition matrix (K, K)
        self.means = None    # Emission means (K, d)
        self.covs = None     # Emission diagonal variances (K, d)
        self._hmm_backend = None

    def fit(self, Z: np.ndarray) -> 'LatentRegimeHMM':
        """Fit Gaussian HMM parameters on exogenous state dynamics Z (N, d) via EM."""
        N, d = Z.shape
        K = self.n_regimes
        
        try:
            from hmmlearn import hmm
            m = hmm.GaussianHMM(
                n_components=K,
                covariance_type='diag',
                n_iter=self.n_iter,
                tol=self.tol,
                random_state=self.random_state
            )
            m.fit(Z)
            order = np.argsort(m.means_[:, 0])
            self.means = m.means_[order]
            self.covs = np.array([np.diag(c) if c.ndim == 2 else c for c in m.covars_[order]])
            self.pi = m.startprob_[order]
            self.A = m.transmat_[order][:, order]
            self._hmm_model = m
            self._order = order
            self._hmm_backend = 'hmmlearn'
            return self
        except Exception:
            pass

        # Standalone NumPy implementation fallback
        self.pi = np.full(K, 1.0 / K)
        self.A = np.full((K, K), 0.1 / (K - 1))
        np.fill_diagonal(self.A, 0.9)
        self.A /= self.A.sum(axis=1, keepdims=True)
        
        z_mean_sort = np.argsort(Z[:, 0])
        chunks = np.array_split(z_mean_sort, K)
        self.means = np.array([Z[c].mean(axis=0) for c in chunks])
        self.covs = np.array([np.var(Z[c], axis=0) + 1e-2 for c in chunks])
        
        log_likelihood_old = -np.inf
        
        for iteration in range(self.n_iter):
            log_B = np.zeros((N, K))
            for k in range(K):
                diff = Z - self.means[k]
                var = np.maximum(self.covs[k], 1e-4)
                log_B[:, k] = -0.5 * np.sum(np.log(2 * np.pi * var) + (diff ** 2) / var, axis=1)
            
            log_A = np.log(np.maximum(self.A, 1e-12))
            log_alpha = np.zeros((N, K))
            log_alpha[0] = np.log(np.maximum(self.pi, 1e-12)) + log_B[0]
            for t in range(1, N):
                log_alpha[t] = logsumexp(log_alpha[t-1][:, None] + log_A, axis=0) + log_B[t]
            
            log_beta = np.zeros((N, K))
            log_beta[-1] = 0.0
            for t in range(N - 2, -1, -1):
                log_beta[t] = logsumexp(log_A + (log_B[t+1] + log_beta[t+1])[None, :], axis=1)
            
            log_gamma = log_alpha + log_beta
            log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)
            
            log_xi = log_alpha[:-1][:, :, None] + log_A[None, :, :] + (log_B[1:] + log_beta[1:])[:, None, :]
            log_xi -= logsumexp(log_xi, axis=(1, 2), keepdims=True)
            xi_sum = np.sum(np.exp(log_xi), axis=0)
            
            self.pi = gamma[0] / np.sum(gamma[0])
            self.A = xi_sum / np.maximum(np.sum(gamma[:-1], axis=0)[:, None], 1e-12)
            self.A /= self.A.sum(axis=1, keepdims=True)
            
            for k in range(K):
                gamma_k = gamma[:, k][:, None]
                sum_gk = np.maximum(np.sum(gamma_k), 1e-12)
                self.means[k] = np.sum(gamma_k * Z, axis=0) / sum_gk
                diff = Z - self.means[k]
                self.covs[k] = np.sum(gamma_k * (diff ** 2), axis=0) / sum_gk + 1e-4
            
            current_ll = logsumexp(log_alpha[-1])
            if abs(current_ll - log_likelihood_old) < self.tol:
                break
            log_likelihood_old = current_ll
            
        return self

    def predict_posteriors(self, Z: np.ndarray, mode: str = 'smooth') -> np.ndarray:
        """
        Compute posterior regime distribution.
        Args:
            Z: Exogenous observations (N, d)
            mode: 'smooth' for retrospective P(S_t | Z_{1:N}),
                  'filter' for causal online P(S_t | Z_{1:t}).
        """
        N, d = Z.shape
        K = self.n_regimes
        log_B = np.zeros((N, K))
        for k in range(K):
            diff = Z - self.means[k]
            var = np.maximum(self.covs[k], 1e-4)
            log_B[:, k] = -0.5 * np.sum(np.log(2 * np.pi * var) + (diff ** 2) / var, axis=1)
            
        log_A = np.log(np.maximum(self.A, 1e-12))
        log_alpha = np.zeros((N, K))
        log_alpha[0] = np.log(np.maximum(self.pi, 1e-12)) + log_B[0]
        for t in range(1, N):
            log_alpha[t] = logsumexp(log_alpha[t-1][:, None] + log_A, axis=0) + log_B[t]
            
        if mode == 'filter':
            # Online filtering distribution P(S_t | Z_1:t)
            log_filter = log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)
            return np.exp(log_filter)
        elif mode == 'smooth':
            # Retrospective smoothing distribution P(S_t | Z_1:N)
            log_beta = np.zeros((N, K))
            log_beta[-1] = 0.0
            for t in range(N - 2, -1, -1):
                log_beta[t] = logsumexp(log_A + (log_B[t+1] + log_beta[t+1])[None, :], axis=1)
            log_gamma = log_alpha + log_beta
            log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
            return np.exp(log_gamma)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'smooth' or 'filter'.")

    def compute_bic(self, Z: np.ndarray) -> float:
        N, d = Z.shape
        K = self.n_regimes
        log_B = np.zeros((N, K))
        for k in range(K):
            diff = Z - self.means[k]
            var = np.maximum(self.covs[k], 1e-4)
            log_B[:, k] = -0.5 * np.sum(np.log(2 * np.pi * var) + (diff ** 2) / var, axis=1)
        log_A = np.log(np.maximum(self.A, 1e-12))
        log_alpha = np.zeros((N, K))
        log_alpha[0] = np.log(np.maximum(self.pi, 1e-12)) + log_B[0]
        for t in range(1, N):
            log_alpha[t] = logsumexp(log_alpha[t-1][:, None] + log_A, axis=0) + log_B[t]
        log_lik = float(logsumexp(log_alpha[-1]))
        n_params = K * (K - 1) + (K - 1) + 2 * K * d
        return float(-2.0 * log_lik + n_params * np.log(N))


def estimate_optimal_embargo(X: np.ndarray, Z: np.ndarray, max_tau: int = 72) -> int:
    """Estimate optimal embargo buffer tau* using Bartlett 95% autocorrelation bound."""
    N = len(X)
    crit = 1.96 / np.sqrt(N)
    V = np.column_stack([X, Z])
    V_centered = V - np.mean(V, axis=0)
    var_V = np.var(V, axis=0) + 1e-12
    max_tau = min(max_tau, max(12, N // 10))
    
    significant_lags = [0]
    for h in range(1, max_tau + 1):
        cov_h = np.mean(V_centered[:-h] * V_centered[h:], axis=0)
        corr_h = np.abs(cov_h / var_V)
        if np.any(corr_h >= crit):
            significant_lags.append(h)
    return int(max(max(significant_lags) + 1, 1))


class PurgedBlockKFold:
    """Purged Block-Temporal Cross-Validation with temporal embargo buffers."""
    def __init__(self, n_splits: int = 5, embargo_tau: int = 24):
        self.n_splits = n_splits
        self.embargo_tau = embargo_tau

    def split(self, N: int):
        indices = np.arange(N)
        fold_sizes = np.full(self.n_splits, N // self.n_splits, dtype=int)
        fold_sizes[:N % self.n_splits] += 1
        
        current = 0
        for fold_size in fold_sizes:
            test_start = current
            test_end = current + fold_size
            test_idx = indices[test_start:test_end]
            
            train_mask = np.ones(N, dtype=bool)
            purge_start = max(0, test_start - self.embargo_tau)
            purge_end = min(N, test_end + self.embargo_tau)
            train_mask[purge_start:purge_end] = False
            
            train_idx = indices[train_mask]
            yield train_idx, test_idx
            current = test_end


class OverlapAwareRegimeDML:
    """
    Overlap-Aware Regime Double Machine Learning (OR-DML).
    
    Provides:
    - Spectral regularized causal estimation: theta_hat_lambda = (J_hat + lambda I)^{-1} S_hat.
    - Observability and overlap diagnostics: lambda_min(J), condition number kappa(J), posterior entropy H_bar.
    - Asymptotically valid HAC sandwich standard errors and un-truncated p-values.
    - Both retrospective smoothing (offline) and forward filtering (online) regimes.
    """
    def __init__(self,
                 n_regimes: int = 3,
                 n_splits: int = 5,
                 embargo_tau: int = 24,
                 auto_embargo: bool = False,
                 reg_lambda: Optional[float] = None,
                 reg_alpha: float = 0.05,
                 posterior_mode: str = 'smooth',
                 nuisance_model = None,
                 hac_lag: int = 12,
                 random_state: int = 42):
        self.n_regimes = n_regimes
        self.n_splits = n_splits
        self.embargo_tau = embargo_tau
        self.auto_embargo = auto_embargo
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.posterior_mode = posterior_mode
        self.nuisance_model = nuisance_model if nuisance_model is not None else HistGradientBoostingRegressor(max_iter=100, min_samples_leaf=20)
        self.hac_lag = hac_lag
        self.random_state = random_state
        
        # Fitted attributes
        self.theta_regimes_ = {}
        self.se_regimes_ = {}
        self.p_regimes_ = {}
        self.ate_ = None
        self.ate_se_ = None
        self.ate_p_ = None
        self.gamma_ = None
        self.regime_weights_ = None
        self.J_mat_ = None
        self.S_vec_ = None
        self.effective_lambda_ = 0.0
        self.lambda_min_ = None
        self.lambda_max_ = None
        self.kappa_ = None
        self.mean_entropy_ = None
        self.effective_embargo_tau_ = embargo_tau

    def fit(self, Y: np.ndarray, T: np.ndarray, X: np.ndarray, Z: np.ndarray) -> 'OverlapAwareRegimeDML':
        N = len(Y)
        assert len(T) == N and len(X) == N and len(Z) == N, "Input dimension mismatch"
        K = self.n_regimes
        
        # 1. Determine embargo buffer
        if self.auto_embargo:
            self.effective_embargo_tau_ = estimate_optimal_embargo(X, Z)
        else:
            self.effective_embargo_tau_ = self.embargo_tau
            
        # 2. Fit HMM on exogenous meteorological variables Z
        self.hmm_ = LatentRegimeHMM(n_regimes=K, random_state=self.random_state)
        self.hmm_.fit(Z)
        gamma = self.hmm_.predict_posteriors(Z, mode=self.posterior_mode)
        self.gamma_ = gamma
        self.regime_weights_ = np.mean(gamma, axis=0)
        
        # Posterior entropy diagnostic: H_bar(gamma) = -1/N sum_t sum_k gamma_tk log(gamma_tk)
        eps = 1e-12
        entropy_t = -np.sum(gamma * np.log(np.maximum(gamma, eps)), axis=1)
        self.mean_entropy_ = float(np.mean(entropy_t))
        
        # 3. Purged Block Temporal Cross-Fitting for Nuisances
        splitter = PurgedBlockKFold(n_splits=self.n_splits, embargo_tau=self.effective_embargo_tau_)
        
        tilde_Y = np.zeros((K, N))
        tilde_T = np.zeros((K, N))
        
        for k in range(K):
            w_k = gamma[:, k]
            for train_idx, test_idx in splitter.split(N):
                weights_train = np.maximum(w_k[train_idx], 1e-4)
                
                # Nuisance outcome: ell_k(X) = E[Y | X, S=k]
                m_y = clone(self.nuisance_model)
                m_y.fit(X[train_idx], Y[train_idx], sample_weight=weights_train)
                tilde_Y[k, test_idx] = Y[test_idx] - m_y.predict(X[test_idx])
                
                # Nuisance treatment: m_k(X) = E[T | X, S=k]
                m_t = clone(self.nuisance_model)
                m_t.fit(X[train_idx], T[train_idx], sample_weight=weights_train)
                tilde_T[k, test_idx] = T[test_idx] - m_t.predict(X[test_idx])
                
        # 4. Construct Empirical Coupled Gram Matrix J and Score Vector S
        J_mat = np.zeros((K, K))
        S_vec = np.zeros(K)
        for j in range(K):
            S_vec[j] = np.mean(gamma[:, j] * tilde_T[j] * tilde_Y[j])
            for k in range(K):
                J_mat[j, k] = np.mean(gamma[:, j] * gamma[:, k] * tilde_T[j] * tilde_T[k])
                
        self.J_mat_ = J_mat
        self.S_vec_ = S_vec
        
        # Spectral properties of J
        eigvals = np.linalg.eigvalsh(J_mat)
        self.lambda_min_ = float(max(eigvals[0], 0.0))
        self.lambda_max_ = float(eigvals[-1])
        self.kappa_ = float(self.lambda_max_ / max(self.lambda_min_, 1e-12))
        
        # 5. Spectral Regularization
        # If reg_lambda is explicitly provided, use it; otherwise scale by Tr(J)/K * N^{-1/2}
        if self.reg_lambda is not None:
            eff_lambda = float(self.reg_lambda)
        else:
            trace_scale = np.trace(J_mat) / K
            eff_lambda = float(self.reg_alpha * trace_scale * (1.0 / np.sqrt(N)))
        self.effective_lambda_ = eff_lambda
        
        # Regularized solve: theta_hat = (J + lambda * I)^{-1} S
        inv_J_reg = np.linalg.inv(J_mat + eff_lambda * np.eye(K))
        theta_vec = inv_J_reg @ S_vec
        
        # 6. Joint HAC Sandwich Covariance Matrix
        # Influence score for each observation t: psi_t = gamma_t * tilde_T_t * (tilde_Y_t - theta * tilde_T_t)
        scores = np.zeros((N, K))
        for k in range(K):
            scores[:, k] = gamma[:, k] * tilde_T[k] * (tilde_Y[k] - theta_vec[k] * tilde_T[k])
            
        Omega = (scores.T @ scores) / N
        for lag in range(1, self.hac_lag + 1):
            weight = 1.0 - (lag / (self.hac_lag + 1.0))
            Gamma_lag = (scores[lag:].T @ scores[:-lag]) / N
            Omega += weight * (Gamma_lag + Gamma_lag.T)
            
        # Sandwich variance formula: Sigma = (J_lambda)^{-1} Omega (J_lambda)^{-1} / N
        Sigma = (inv_J_reg @ Omega @ inv_J_reg) / N
        
        for k in range(K):
            th = float(theta_vec[k])
            se = float(np.sqrt(max(Sigma[k, k], 1e-12)))
            z_score = abs(th / se) if se > 0 else 0.0
            p_val = float(2.0 * norm.sf(z_score))
            
            self.theta_regimes_[k] = th
            self.se_regimes_[k] = se
            self.p_regimes_[k] = p_val
            
        # 7. Aggregate ATE Inference (with Markov state occupation covariance)
        pi = self.regime_weights_
        self.ate_ = float(pi @ theta_vec)
        var_sate = float(pi @ Sigma @ pi)
        
        # Proposition 4 Markov regime occupation variance
        try:
            Pi = self.hmm_.A
            ones = np.ones((K, 1))
            Z_ergodic = np.linalg.inv(np.eye(K) - Pi + ones @ pi.reshape(1, -1))
            D_pi = np.diag(pi)
            Cov_occ = (D_pi @ Z_ergodic + Z_ergodic.T @ D_pi - D_pi - np.outer(pi, pi)) / N
            var_occ = max(float(theta_vec.T @ Cov_occ @ theta_vec), 0.0)
        except Exception:
            var_occ = 0.0
            
        self.ate_se_ = float(np.sqrt(max(var_sate + var_occ, 1e-12)))
        ate_z = abs(self.ate_ / self.ate_se_) if self.ate_se_ > 0 else 0.0
        self.ate_p_ = float(2.0 * norm.sf(ate_z))
        
        return self

    def summary(self) -> pd.DataFrame:
        """Return formatted statistical summary with genuine asymptotic p-values."""
        records = []
        for k in range(self.n_regimes):
            th = self.theta_regimes_[k]
            se = self.se_regimes_[k]
            p = self.p_regimes_[k]
            records.append({
                "Regime": f"Regime {k+1}",
                "Effect (Theta)": round(th, 4),
                "Std Error": round(se, 4),
                "95% CI Lower": round(th - 1.96 * se, 4),
                "95% CI Upper": round(th + 1.96 * se, 4),
                "z-stat": round(th / se, 3) if se > 0 else 0.0,
                "p-value": "< 0.0001" if p < 0.0001 else round(p, 4)
            })
        records.append({
            "Regime": "Overall ATE",
            "Effect (Theta)": round(self.ate_, 4),
            "Std Error": round(self.ate_se_, 4),
            "95% CI Lower": round(self.ate_ - 1.96 * self.ate_se_, 4),
            "95% CI Upper": round(self.ate_ + 1.96 * self.ate_se_, 4),
            "z-stat": round(self.ate_ / self.ate_se_, 3) if self.ate_se_ > 0 else 0.0,
            "p-value": "< 0.0001" if self.ate_p_ < 0.0001 else round(self.ate_p_, 4)
        })
        return pd.DataFrame(records)

    def fit_dynamic_irf(self,
                        Y: np.ndarray,
                        T: np.ndarray,
                        X: np.ndarray,
                        Z: np.ndarray,
                        horizons: List[int] = [0, 1, 2, 3, 6, 12, 24]) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Estimate dynamic causal impulse-response functions:
        Y_{t+h} = theta_h(S_t) * T_t + g_h(X_t, S_t) + U_{t+h}.
        """
        irf_results = {k: {'horizons': np.array(horizons), 'effects': [], 'ses': []} for k in range(self.n_regimes)}
        irf_results['ate'] = {'horizons': np.array(horizons), 'effects': [], 'ses': []}
        
        N = len(Y)
        max_h = max(horizons)
        
        for h in horizons:
            if h == 0:
                Y_h = Y
                T_h = T
                X_h = X
                Z_h = Z
            else:
                Y_h = Y[h:]
                T_h = T[:-h]
                X_h = X[:-h]
                Z_h = Z[:-h]
                
            model_h = OverlapAwareRegimeDML(
                n_regimes=self.n_regimes,
                n_splits=self.n_splits,
                embargo_tau=self.effective_embargo_tau_,
                reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha,
                posterior_mode=self.posterior_mode,
                nuisance_model=self.nuisance_model,
                hac_lag=self.hac_lag,
                random_state=self.random_state
            )
            model_h.fit(Y_h, T_h, X_h, Z_h)
            
            for k in range(self.n_regimes):
                irf_results[k]['effects'].append(model_h.theta_regimes_[k])
                irf_results[k]['ses'].append(model_h.se_regimes_[k])
                
            irf_results['ate']['effects'].append(model_h.ate_)
            irf_results['ate']['ses'].append(model_h.ate_se_)
            
        for key in irf_results:
            irf_results[key]['effects'] = np.array(irf_results[key]['effects'])
            irf_results[key]['ses'] = np.array(irf_results[key]['ses'])
            
        return irf_results
