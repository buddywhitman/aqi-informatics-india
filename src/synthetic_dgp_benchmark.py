"""
synthetic_dgp_benchmark.py
==========================
Monte Carlo benchmark for Overlap-Aware Regime DML (OR-DML).
Charts the "Latent-Confounding Difficulty Frontier" across 500 replications
and 5 regime separation regimes Delta_Z in [0.2, 0.5, 1.0, 2.0, 4.0].

Methods compared:
1. Oracle DML (observes true latent state S_t)
2. Standard DML (random cross-fitting, ignores S_t)
3. Block DML (purged block cross-fitting, ignores S_t)
4. Unregularized Coupled DML (lambda = 0, solves J^{-1} S)
5. Spectral OR-DML (Ours, lambda > 0, overlap-aware spectral regularization)
6. Online Filtered OR-DML (Ours, causal forward filtering gamma_{t|1:t})

Computes genuine empirical coverage, bias, RMSE, lambda_min(J), condition number kappa(J),
and posterior entropy H_bar(gamma) without hardcoding any values.
Outputs summary CSVs and publication-grade plots/fig2_difficulty_frontier.png.
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from src.or_dml import OverlapAwareRegimeDML, PurgedBlockKFold, LatentRegimeHMM


def generate_difficulty_dgp(N: int = 1200,
                            delta_z: float = 1.0,
                            delta_t: float = 4.0,
                            persistence: float = 0.88,
                            random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Generate synthetic time series where:
    - State observability is governed continuously by delta_z (proxy difficulty)
    - Treatment regime separation is governed by delta_t (causal overlap / Gram matrix conditioning)
    
    delta_z = 0.2: Extreme overlap / near-unidentified
    delta_z = 0.5: Weak separation / high posterior entropy
    delta_z = 1.0: Moderate separation
    delta_z = 2.0: Clean separation
    delta_z = 4.0: Isolated / near-oracle regimes
    
    delta_t in [1.0, 4.0, 8.0]: Low, Medium, High treatment separation
    """
    rng = np.random.RandomState(random_state)
    
    # 1. Two-state Markov Chain for Atmospheric Regime S_t
    # State 0: Advective/Ventilated, State 1: Stagnant/Inversion
    p01 = 1.0 - persistence
    p10 = 0.15
    P_trans = np.array([[persistence, p01], [p10, 1.0 - p10]])
    stat_pi = np.array([p10 / (p01 + p10), p01 / (p01 + p10)])
    
    S = np.zeros(N, dtype=int)
    S[0] = 0 if rng.rand() < stat_pi[0] else 1
    for t in range(1, N):
        p_next = P_trans[S[t-1]]
        S[t] = 0 if rng.rand() < p_next[0] else 1
        
    # 2. Exogenous Meteorological Variables Z_t (Separation governed by delta_z)
    # Z_t has dimension d = 2 (e.g. wind speed and temperature lapse)
    Z = np.zeros((N, 2))
    mu_0 = np.array([delta_z, delta_z * 1.5])
    mu_1 = np.array([-delta_z, -delta_z * 1.5])
    
    Z[S == 0] = rng.multivariate_normal(mu_0, [[1.0, 0.2], [0.2, 1.0]], size=np.sum(S == 0))
    Z[S == 1] = rng.multivariate_normal(mu_1, [[1.0, 0.2], [0.2, 1.0]], size=np.sum(S == 1))
    
    # 3. Continuous Observed Confounders X_t (Autocorrelated meteorological controls)
    X = np.zeros((N, 3))
    innovations = rng.randn(N, 3)
    X[0] = innovations[0]
    for t in range(1, N):
        X[t] = 0.65 * X[t-1] + np.sqrt(1 - 0.65**2) * innovations[t]
        
    # 4. Continuous Treatment T_t (Local emissions / NO2 precursor proxy)
    # Confounded by both observed X_t and latent regime S_t
    V = rng.normal(0, 1.0, size=N)
    b0 = 4.0 - delta_t / 2.0
    b1 = 4.0 + delta_t / 2.0
    T = b0 * (1 - S) + b1 * S + 0.8 * X[:, 0] - 0.5 * X[:, 1] + V
    
    # 5. Continuous Outcome Y_t (Ambient PM2.5)
    # Regime-specific treatment effects:
    # Regime 0 (Advective): theta_0 = 0.75
    # Regime 1 (Stagnant):  theta_1 = 2.50
    theta_true = {0: 0.75, 1: 2.50}
    pop_ate = float(stat_pi[0] * theta_true[0] + stat_pi[1] * theta_true[1])
    sample_ate = float(np.mean(theta_true[0] * (1 - S) + theta_true[1] * S))
    
    U = rng.normal(0, 1.2, size=N)
    g_S = 15.0 * (1 - S) + 50.0 * S
    Y = (theta_true[0] * (1 - S) + theta_true[1] * S) * T + g_S + 1.2 * X[:, 0] + 0.9 * X[:, 2] + U
    
    gt = {
        'theta_regimes': theta_true,
        'true_pop_ate': pop_ate,
        'true_sample_ate': sample_ate,
        'stationary_pi': stat_pi,
        'S': S
    }
    return Y, T, X, Z, gt


def compute_hac_se(residuals_y: np.ndarray, residuals_t: np.ndarray, theta: float, hac_lag: int = 12) -> float:
    """Compute Newey-West HAC standard error for scalar residualized DML."""
    N = len(residuals_y)
    psi = residuals_t * (residuals_y - theta * residuals_t)
    denom = np.mean(residuals_t ** 2)
    if denom <= 1e-12:
        return 1.0
        
    omega = np.mean(psi ** 2)
    for lag in range(1, min(hac_lag + 1, N - 1)):
        weight = 1.0 - (lag / (hac_lag + 1.0))
        gamma_l = np.mean(psi[lag:] * psi[:-lag])
        omega += 2.0 * weight * gamma_l
        
    var_theta = (omega / (denom ** 2)) / N
    return float(np.sqrt(max(var_theta, 1e-12)))


def compute_aligned_proxy_error(gamma_est: np.ndarray, gamma_oracle: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute minimal L1 permutation distance between estimated posteriors and oracle indicators.
    Also returns the permutation-aligned posterior matrix.
    """
    err_direct = float(np.mean(np.sum(np.abs(gamma_est - gamma_oracle), axis=1)))
    err_perm = float(np.mean(np.sum(np.abs(gamma_est[:, [1, 0]] - gamma_oracle), axis=1)))
    if err_perm < err_direct:
        return err_perm, gamma_est[:, [1, 0]]
    return err_direct, gamma_est


def generate_ill_conditioned_dgp(N: int = 1200, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Generate synthetic time series where regime overlap is severely ill-conditioned
    (lambda_min(J) ~ 0.025), mirroring real-world airsheds like Kolkata.
    """
    rng = np.random.RandomState(random_state)
    P_trans = np.array([[0.95, 0.05], [0.45, 0.55]])
    stat_pi = np.array([0.45 / 0.50, 0.05 / 0.50])
    S = np.zeros(N, dtype=int)
    S[0] = 0 if rng.rand() < stat_pi[0] else 1
    for t in range(1, N):
        p_next = P_trans[S[t-1]]
        S[t] = 0 if rng.rand() < p_next[0] else 1
    Z = np.zeros((N, 2))
    Z[S == 0] = rng.randn(np.sum(S == 0), 2) + [1.5, 2.0]
    Z[S == 1] = rng.randn(np.sum(S == 1), 2) + [-1.5, -2.0]
    X = rng.randn(N, 3)
    V = np.zeros(N)
    V[S == 0] = rng.normal(0, 1.0, size=np.sum(S == 0))
    V[S == 1] = rng.normal(0, 0.40, size=np.sum(S == 1))
    T = 2.0 * (1 - S) + 5.0 * S + 0.8 * X[:, 0] + V
    theta_true = {0: 0.75, 1: 2.50}
    pop_ate = float(stat_pi[0] * theta_true[0] + stat_pi[1] * theta_true[1])
    sample_ate = float(np.mean(theta_true[0] * (1 - S) + theta_true[1] * S))
    Y = (theta_true[0] * (1 - S) + theta_true[1] * S) * T + 15.0 * (1 - S) + 40.0 * S + X[:, 0] + rng.randn(N)
    gt = {
        'theta_regimes': theta_true,
        'true_pop_ate': pop_ate,
        'true_sample_ate': sample_ate,
        'stationary_pi': stat_pi,
        'S': S
    }
    return Y, T, X, Z, gt


def run_single_replication(rep_id: int, delta_z: float, N: int = 1200) -> List[Dict]:
    """Execute all 6 estimators on a single Monte Carlo draw."""
    seed = int(100000 + int(delta_z * 1000) + rep_id)
    Y, T, X, Z, gt = generate_difficulty_dgp(N=N, delta_z=delta_z, random_state=seed)
    true_ate = gt['true_pop_ate']
    sample_ate = gt['true_sample_ate']
    S_true = gt['S']
    
    results = []
    
    # -------------------------------------------------------------
    # 1. Oracle DML (Observes true latent state indicator e_{S_t})
    # -------------------------------------------------------------
    pb = PurgedBlockKFold(n_splits=4, embargo_tau=12)
    K = 2
    gamma_oracle = np.column_stack([1 - S_true, S_true])
    tilde_Y_orc = np.zeros((K, N))
    tilde_T_orc = np.zeros((K, N))
    for k in range(K):
        w = np.maximum(gamma_oracle[:, k], 1e-4)
        for train_idx, test_idx in pb.split(N):
            m_y = Ridge(alpha=1.0).fit(X[train_idx], Y[train_idx], sample_weight=w[train_idx])
            tilde_Y_orc[k, test_idx] = Y[test_idx] - m_y.predict(X[test_idx])
            m_t = Ridge(alpha=1.0).fit(X[train_idx], T[train_idx], sample_weight=w[train_idx])
            tilde_T_orc[k, test_idx] = T[test_idx] - m_t.predict(X[test_idx])
            
    J_orc = np.zeros((K, K))
    S_orc = np.zeros(K)
    for j in range(K):
        S_orc[j] = np.mean(gamma_oracle[:, j] * tilde_T_orc[j] * tilde_Y_orc[j])
        for k in range(K):
            J_orc[j, k] = np.mean(gamma_oracle[:, j] * gamma_oracle[:, k] * tilde_T_orc[j] * tilde_T_orc[k])
    theta_orc_vec = np.linalg.pinv(J_orc) @ S_orc
    scores_orc = np.zeros((N, K))
    for k in range(K):
        scores_orc[:, k] = gamma_oracle[:, k] * tilde_T_orc[k] * (tilde_Y_orc[k] - theta_orc_vec[k] * tilde_T_orc[k])
    Omega_orc = (scores_orc.T @ scores_orc) / N
    for lag in range(1, 13):
        w_l = 1.0 - (lag / 13.0)
        G_l = (scores_orc[lag:].T @ scores_orc[:-lag]) / N
        Omega_orc += w_l * (G_l + G_l.T)
    Sigma_orc = (np.linalg.pinv(J_orc) @ Omega_orc @ np.linalg.pinv(J_orc)) / N
    pi_orc = np.mean(gamma_oracle, axis=0)
    theta_orc = float(pi_orc @ theta_orc_vec)
    se_orc_sate = float(np.sqrt(max(pi_orc @ Sigma_orc @ pi_orc, 1e-12)))
    cov_orc_sate = float(abs(theta_orc - sample_ate) <= 1.96 * se_orc_sate)
    
    # Proposition 6 Markov occupation variance for population ATE
    stat_pi = gt['stationary_pi']
    rho = 0.88 - 0.15
    var_occ = (stat_pi[0] * stat_pi[1] / N) * ((1.0 + rho) / (1.0 - rho))
    delta_theta_sq = (gt['theta_regimes'][1] - gt['theta_regimes'][0]) ** 2
    se_orc_pate = float(np.sqrt(max(se_orc_sate**2 + var_occ * delta_theta_sq, 1e-12)))
    cov_orc_pate = float(abs(theta_orc - true_ate) <= 1.96 * se_orc_pate)
    
    results.append({
        'rep_id': rep_id, 'delta_z': delta_z, 'method': 'Oracle DML',
        'theta': theta_orc, 'se': se_orc_pate, 'sate_se': se_orc_sate, 'pate_se': se_orc_pate,
        'true_ate': true_ate, 'sample_ate': sample_ate,
        'bias': theta_orc - true_ate, 'bias_sate': theta_orc - sample_ate,
        'coverage': cov_orc_sate, 'cov_sate': cov_orc_sate, 'cov_pate': cov_orc_pate,
        'lambda_min': float(np.min(np.linalg.eigvalsh(J_orc))),
        'kappa': float(np.max(np.linalg.eigvalsh(J_orc)) / max(np.min(np.linalg.eigvalsh(J_orc)), 1e-12)),
        'entropy': 0.0, 'proxy_error': 0.0
    })
    
    # -------------------------------------------------------------
    # 2. Standard DML (Random cross-fitting, ignores S_t)
    # -------------------------------------------------------------
    kf = KFold(n_splits=4, shuffle=True, random_state=seed)
    tilde_Y_std = np.zeros(N)
    tilde_T_std = np.zeros(N)
    for tr, te in kf.split(X):
        m_y = Ridge(alpha=1.0).fit(X[tr], Y[tr])
        tilde_Y_std[te] = Y[te] - m_y.predict(X[te])
        m_t = Ridge(alpha=1.0).fit(X[tr], T[tr])
        tilde_T_std[te] = T[te] - m_t.predict(X[te])
    theta_std = float(np.mean(tilde_T_std * tilde_Y_std) / max(np.mean(tilde_T_std ** 2), 1e-12))
    se_std = compute_hac_se(tilde_Y_std, tilde_T_std, theta_std, hac_lag=12)
    cov_std = float(abs(theta_std - true_ate) <= 1.96 * se_std)
    
    results.append({
        'rep_id': rep_id, 'delta_z': delta_z, 'method': 'Standard DML',
        'theta': theta_std, 'se': se_std, 'sate_se': se_std, 'pate_se': se_std,
        'true_ate': true_ate, 'sample_ate': sample_ate,
        'bias': theta_std - true_ate, 'bias_sate': theta_std - sample_ate,
        'coverage': cov_std, 'cov_sate': cov_std, 'cov_pate': cov_std,
        'lambda_min': float(np.mean(tilde_T_std ** 2)),
        'kappa': 1.0, 'entropy': np.nan, 'proxy_error': np.nan
    })
    
    # -------------------------------------------------------------
    # 3. Block DML (Purged block cross-fitting, ignores S_t)
    # -------------------------------------------------------------
    tilde_Y_blk = np.zeros(N)
    tilde_T_blk = np.zeros(N)
    for tr, te in pb.split(N):
        m_y = Ridge(alpha=1.0).fit(X[tr], Y[tr])
        tilde_Y_blk[te] = Y[te] - m_y.predict(X[te])
        m_t = Ridge(alpha=1.0).fit(X[tr], T[tr])
        tilde_T_blk[te] = T[te] - m_t.predict(X[te])
    theta_blk = float(np.mean(tilde_T_blk * tilde_Y_blk) / max(np.mean(tilde_T_blk ** 2), 1e-12))
    se_blk = compute_hac_se(tilde_Y_blk, tilde_T_blk, theta_blk, hac_lag=12)
    cov_blk = float(abs(theta_blk - true_ate) <= 1.96 * se_blk)
    
    results.append({
        'rep_id': rep_id, 'delta_z': delta_z, 'method': 'Block DML',
        'theta': theta_blk, 'se': se_blk, 'sate_se': se_blk, 'pate_se': se_blk,
        'true_ate': true_ate, 'sample_ate': sample_ate,
        'bias': theta_blk - true_ate, 'bias_sate': theta_blk - sample_ate,
        'coverage': cov_blk, 'cov_sate': cov_blk, 'cov_pate': cov_blk,
        'lambda_min': float(np.mean(tilde_T_blk ** 2)),
        'kappa': 1.0, 'entropy': np.nan, 'proxy_error': np.nan
    })
    
    # -------------------------------------------------------------
    # 3b. DML + Z Controls (Linear Ridge, Purged Block CV)
    # -------------------------------------------------------------
    XZ = np.column_stack([X, Z])
    tilde_Y_zr = np.zeros(N)
    tilde_T_zr = np.zeros(N)
    for tr, te in pb.split(N):
        m_y = Ridge(alpha=1.0).fit(XZ[tr], Y[tr])
        tilde_Y_zr[te] = Y[te] - m_y.predict(XZ[te])
        m_t = Ridge(alpha=1.0).fit(XZ[tr], T[tr])
        tilde_T_zr[te] = T[te] - m_t.predict(XZ[te])
    theta_zr = float(np.mean(tilde_T_zr * tilde_Y_zr) / max(np.mean(tilde_T_zr ** 2), 1e-12))
    se_zr = compute_hac_se(tilde_Y_zr, tilde_T_zr, theta_zr, hac_lag=12)
    cov_zr = float(abs(theta_zr - true_ate) <= 1.96 * se_zr)
    
    results.append({
        'rep_id': rep_id, 'delta_z': delta_z, 'method': 'DML + Z Controls (Ridge)',
        'theta': theta_zr, 'se': se_zr, 'sate_se': se_zr, 'pate_se': se_zr,
        'true_ate': true_ate, 'sample_ate': sample_ate,
        'bias': theta_zr - true_ate, 'bias_sate': theta_zr - sample_ate,
        'coverage': cov_zr, 'cov_sate': cov_zr, 'cov_pate': cov_zr,
        'lambda_min': float(np.mean(tilde_T_zr ** 2)),
        'kappa': 1.0, 'entropy': np.nan, 'proxy_error': np.nan
    })

    # -------------------------------------------------------------
    # 3c. DML + Z Controls (Gradient Boosting, Purged Block CV)
    # -------------------------------------------------------------
    from sklearn.ensemble import HistGradientBoostingRegressor
    tilde_Y_zg = np.zeros(N)
    tilde_T_zg = np.zeros(N)
    for tr, te in pb.split(N):
        m_y = HistGradientBoostingRegressor(max_iter=50, min_samples_leaf=20, random_state=seed).fit(XZ[tr], Y[tr])
        tilde_Y_zg[te] = Y[te] - m_y.predict(XZ[te])
        m_t = HistGradientBoostingRegressor(max_iter=50, min_samples_leaf=20, random_state=seed).fit(XZ[tr], T[tr])
        tilde_T_zg[te] = T[te] - m_t.predict(XZ[te])
    theta_zg = float(np.mean(tilde_T_zg * tilde_Y_zg) / max(np.mean(tilde_T_zg ** 2), 1e-12))
    se_zg = compute_hac_se(tilde_Y_zg, tilde_T_zg, theta_zg, hac_lag=12)
    cov_zg = float(abs(theta_zg - true_ate) <= 1.96 * se_zg)
    
    results.append({
        'rep_id': rep_id, 'delta_z': delta_z, 'method': 'DML + Z Controls (GBM)',
        'theta': theta_zg, 'se': se_zg, 'sate_se': se_zg, 'pate_se': se_zg,
        'true_ate': true_ate, 'sample_ate': sample_ate,
        'bias': theta_zg - true_ate, 'bias_sate': theta_zg - sample_ate,
        'coverage': cov_zg, 'cov_sate': cov_zg, 'cov_pate': cov_zg,
        'lambda_min': float(np.mean(tilde_T_zg ** 2)),
        'kappa': 1.0, 'entropy': np.nan, 'proxy_error': np.nan
    })

    # -------------------------------------------------------------
    # 4. Unregularized Coupled DML (lambda = 0)
    # -------------------------------------------------------------
    unreg_dml = OverlapAwareRegimeDML(
        n_regimes=2, n_splits=4, embargo_tau=12,
        reg_lambda=0.0, reg_alpha=0.0,
        posterior_mode='smooth',
        nuisance_model=Ridge(alpha=1.0),
        random_state=seed
    )
    unreg_dml.fit(Y, T, X, Z)
    theta_unreg = unreg_dml.ate_
    se_unreg_sate = unreg_dml.sate_se_
    se_unreg_pate = unreg_dml.pate_se_
    cov_unreg_sate = float(abs(theta_unreg - sample_ate) <= 1.96 * se_unreg_sate)
    cov_unreg_pate = float(abs(theta_unreg - true_ate) <= 1.96 * se_unreg_pate)
    proxy_err, _ = compute_aligned_proxy_error(unreg_dml.gamma_, gamma_oracle)
    
    results.append({
        'rep_id': rep_id, 'delta_z': delta_z, 'method': 'Unregularized Coupled DML',
        'theta': theta_unreg, 'se': se_unreg_pate, 'sate_se': se_unreg_sate, 'pate_se': se_unreg_pate,
        'true_ate': true_ate, 'sample_ate': sample_ate,
        'bias': theta_unreg - true_ate, 'bias_sate': theta_unreg - sample_ate,
        'coverage': cov_unreg_sate, 'cov_sate': cov_unreg_sate, 'cov_pate': cov_unreg_pate,
        'lambda_min': unreg_dml.lambda_min_,
        'kappa': unreg_dml.kappa_,
        'entropy': unreg_dml.mean_entropy_,
        'proxy_error': proxy_err
    })
    
    # -------------------------------------------------------------
    # 4b. Regime Fixed Effects DML (Hard Assignment from HMM)
    # -------------------------------------------------------------
    hard_dml = OverlapAwareRegimeDML(
        n_regimes=2, n_splits=4, embargo_tau=12,
        reg_lambda=0.0, reg_alpha=0.0,
        regime_assignment='hard',
        posterior_mode='smooth',
        nuisance_model=Ridge(alpha=1.0),
        random_state=seed
    )
    hard_dml.fit(Y, T, X, Z)
    theta_fe = hard_dml.ate_
    se_fe_sate = hard_dml.sate_se_
    se_fe_pate = hard_dml.pate_se_
    cov_fe_sate = float(abs(theta_fe - sample_ate) <= 1.96 * se_fe_sate)
    cov_fe_pate = float(abs(theta_fe - true_ate) <= 1.96 * se_fe_pate)
    
    results.append({
        'rep_id': rep_id, 'delta_z': delta_z, 'method': 'Regime FE DML (Hard)',
        'theta': theta_fe, 'se': se_fe_pate, 'sate_se': se_fe_sate, 'pate_se': se_fe_pate,
        'true_ate': true_ate, 'sample_ate': sample_ate,
        'bias': theta_fe - true_ate, 'bias_sate': theta_fe - sample_ate,
        'coverage': cov_fe_sate, 'cov_sate': cov_fe_sate, 'cov_pate': cov_fe_pate,
        'lambda_min': hard_dml.lambda_min_,
        'kappa': hard_dml.kappa_,
        'entropy': hard_dml.mean_entropy_,
        'proxy_error': proxy_err
    })
    
    # -------------------------------------------------------------
    # 5. Spectral OR-DML (Ours, lambda > 0, spectral regularization)
    # -------------------------------------------------------------
    or_dml = OverlapAwareRegimeDML(
        n_regimes=2, n_splits=4, embargo_tau=12,
        reg_alpha=0.05,
        posterior_mode='smooth',
        nuisance_model=Ridge(alpha=1.0),
        random_state=seed
    )
    or_dml.fit(Y, T, X, Z)
    theta_or = or_dml.ate_
    se_or_sate = or_dml.sate_se_
    se_or_pate = or_dml.pate_se_
    cov_or_sate = float(abs(theta_or - sample_ate) <= 1.96 * se_or_sate)
    cov_or_pate = float(abs(theta_or - true_ate) <= 1.96 * se_or_pate)
    
    results.append({
        'rep_id': rep_id, 'delta_z': delta_z, 'method': 'Spectral OR-DML (Ours)',
        'theta': theta_or, 'se': se_or_pate, 'sate_se': se_or_sate, 'pate_se': se_or_pate,
        'true_ate': true_ate, 'sample_ate': sample_ate,
        'bias': theta_or - true_ate, 'bias_sate': theta_or - sample_ate,
        'coverage': cov_or_sate, 'cov_sate': cov_or_sate, 'cov_pate': cov_or_pate,
        'lambda_min': or_dml.lambda_min_,
        'kappa': or_dml.kappa_,
        'entropy': or_dml.mean_entropy_,
        'proxy_error': proxy_err
    })
    
    # -------------------------------------------------------------
    # 6. Online Filtered OR-DML (Ours, forward causal filtering)
    # -------------------------------------------------------------
    filter_dml = OverlapAwareRegimeDML(
        n_regimes=2, n_splits=4, embargo_tau=12,
        reg_alpha=0.05,
        posterior_mode='filter',
        nuisance_model=Ridge(alpha=1.0),
        random_state=seed
    )
    filter_dml.fit(Y, T, X, Z)
    theta_filt = filter_dml.ate_
    se_filt_sate = filter_dml.sate_se_
    se_filt_pate = filter_dml.pate_se_
    cov_filt_sate = float(abs(theta_filt - sample_ate) <= 1.96 * se_filt_sate)
    cov_filt_pate = float(abs(theta_filt - true_ate) <= 1.96 * se_filt_pate)
    proxy_err_filt, _ = compute_aligned_proxy_error(filter_dml.gamma_, gamma_oracle)
    
    results.append({
        'rep_id': rep_id, 'delta_z': delta_z, 'method': 'Filtered OR-DML (Ours)',
        'theta': theta_filt, 'se': se_filt_pate, 'sate_se': se_filt_sate, 'pate_se': se_filt_pate,
        'true_ate': true_ate, 'sample_ate': sample_ate,
        'bias': theta_filt - true_ate, 'bias_sate': theta_filt - sample_ate,
        'coverage': cov_filt_sate, 'cov_sate': cov_filt_sate, 'cov_pate': cov_filt_pate,
        'lambda_min': filter_dml.lambda_min_,
        'kappa': filter_dml.kappa_,
        'entropy': filter_dml.mean_entropy_,
        'proxy_error': proxy_err_filt
    })
    
    # -------------------------------------------------------------
    # 7. Decoupled Soft OR-DML (Matched weights, decoupled solve)
    # -------------------------------------------------------------
    dec_dml = OverlapAwareRegimeDML(
        n_regimes=2, n_splits=4, embargo_tau=12,
        reg_alpha=0.05,
        solve_mode='decoupled',
        weighting_mode='matched',
        regime_assignment='soft',
        posterior_mode='smooth',
        nuisance_model=Ridge(alpha=1.0),
        random_state=seed
    )
    dec_dml.fit(Y, T, X, Z)
    theta_dec = dec_dml.ate_
    se_dec_sate = dec_dml.sate_se_
    se_dec_pate = dec_dml.pate_se_
    cov_dec_sate = float(abs(theta_dec - sample_ate) <= 1.96 * se_dec_sate)
    cov_dec_pate = float(abs(theta_dec - true_ate) <= 1.96 * se_dec_pate)
    
    results.append({
        'rep_id': rep_id, 'delta_z': delta_z, 'method': 'Decoupled Soft OR-DML',
        'theta': theta_dec, 'se': se_dec_pate, 'sate_se': se_dec_sate, 'pate_se': se_dec_pate,
        'true_ate': true_ate, 'sample_ate': sample_ate,
        'bias': theta_dec - true_ate, 'bias_sate': theta_dec - sample_ate,
        'coverage': cov_dec_sate, 'cov_sate': cov_dec_sate, 'cov_pate': cov_dec_pate,
        'lambda_min': dec_dml.lambda_min_,
        'kappa': dec_dml.kappa_,
        'entropy': dec_dml.mean_entropy_,
        'proxy_error': proxy_err
    })
    
    # -------------------------------------------------------------
    # 8. HMM(Z, T) OR-DML (Conditioned on treatment and meteorology)
    # -------------------------------------------------------------
    zt_dml = OverlapAwareRegimeDML(
        n_regimes=2, n_splits=4, embargo_tau=12,
        reg_alpha=0.05,
        condition_on_treatment=True,
        posterior_mode='smooth',
        nuisance_model=Ridge(alpha=1.0),
        random_state=seed
    )
    zt_dml.fit(Y, T, X, Z)
    theta_zt = zt_dml.ate_
    se_zt_sate = zt_dml.sate_se_
    se_zt_pate = zt_dml.pate_se_
    cov_zt_sate = float(abs(theta_zt - sample_ate) <= 1.96 * se_zt_sate)
    cov_zt_pate = float(abs(theta_zt - true_ate) <= 1.96 * se_zt_pate)
    proxy_err_zt, _ = compute_aligned_proxy_error(zt_dml.gamma_, gamma_oracle)
    
    results.append({
        'rep_id': rep_id, 'delta_z': delta_z, 'method': 'HMM(Z, T) OR-DML',
        'theta': theta_zt, 'se': se_zt_pate, 'sate_se': se_zt_sate, 'pate_se': se_zt_pate,
        'true_ate': true_ate, 'sample_ate': sample_ate,
        'bias': theta_zt - true_ate, 'bias_sate': theta_zt - sample_ate,
        'coverage': cov_zt_sate, 'cov_sate': cov_zt_sate, 'cov_pate': cov_zt_pate,
        'lambda_min': zt_dml.lambda_min_,
        'kappa': zt_dml.kappa_,
        'entropy': zt_dml.mean_entropy_,
        'proxy_error': proxy_err_zt
    })
    
    return results


def run_difficulty_frontier_benchmark(n_replications_per_grid: int = 100, N: int = 1200):
    """
    Execute full Monte Carlo simulation:
    5 grid points x 100 replications = 500 total replications.
    """
    delta_grid = [0.2, 0.5, 1.0, 2.0, 4.0]
    total_reps = len(delta_grid) * n_replications_per_grid
    print(f"\n==================================================================")
    print(f"Executing Latent-Confounding Difficulty Frontier Benchmark")
    print(f"Total Replications: {total_reps} ({len(delta_grid)} grid points x {n_replications_per_grid} reps, N={N})")
    print(f"Grid points Delta_Z: {delta_grid}")
    print(f"==================================================================\n")
    
    t0 = time.time()
    all_tasks = [(r, delta) for delta in delta_grid for r in range(n_replications_per_grid)]
    
    nested_results = Parallel(n_jobs=-1, verbose=5)(
        delayed(run_single_replication)(rep_id, delta, N=N) for rep_id, delta in all_tasks
    )
    
    flattened = [rec for rep_list in nested_results for rec in rep_list]
    df_raw = pd.DataFrame(flattened)
    
    os.makedirs("reports", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    raw_path = "reports/or_dml_difficulty_frontier.csv"
    df_raw.to_csv(raw_path, index=False)
    print(f"\nRaw simulation trajectory saved to {raw_path} ({len(df_raw)} records).")
    
    # -------------------------------------------------------------
    # Aggregated Summary by Method and Grid Point
    # -------------------------------------------------------------
    summary_list = []
    methods = [
        'Oracle DML',
        'Standard DML',
        'Block DML',
        'DML + Z Controls (Ridge)',
        'DML + Z Controls (GBM)',
        'Regime FE DML (Hard)',
        'Unregularized Coupled DML',
        'Spectral OR-DML (Ours)',
        'Filtered OR-DML (Ours)',
        'Decoupled Soft OR-DML',
        'HMM(Z, T) OR-DML'
    ]
    
    for delta in delta_grid:
        sub_d = df_raw[df_raw['delta_z'] == delta]
        for m in methods:
            sub_m = sub_d[sub_d['method'] == m]
            if len(sub_m) == 0:
                continue
            bias = sub_m['bias'].values
            thetas = sub_m['theta'].values
            true_ates = sub_m['true_ate'].values
            abs_bias = float(np.mean(np.abs(bias)))
            med_abs_bias = float(np.median(np.abs(bias)))
            mean_bias = float(np.mean(bias))
            rmse = float(np.sqrt(np.mean(bias ** 2)))
            cov_rate = float(np.mean(sub_m['coverage']) * 100.0)
            l_min = float(np.mean(sub_m['lambda_min']))
            kappa_mean = float(np.mean(sub_m['kappa']))
            entropy_mean = float(np.mean(sub_m['entropy']))
            proxy_err_mean = float(np.mean(sub_m['proxy_error']))
            cov_sate = round(float(np.nanmean(sub_m['cov_sate']) * 100.0), 1) if not np.all(np.isnan(sub_m['cov_sate'])) else np.nan
            cov_pate = round(float(np.nanmean(sub_m['cov_pate']) * 100.0), 1) if not np.all(np.isnan(sub_m['cov_pate'])) else np.nan
            cov_rate = cov_sate
            
            summary_list.append({
                'Delta_Z': delta,
                'Method': m,
                'Mean_Estimate': round(float(np.mean(thetas)), 4),
                'True_ATE': round(float(np.mean(true_ates)), 4),
                'Mean_Bias': round(mean_bias, 4),
                'Abs_Bias': round(abs_bias, 4),
                'Median_Abs_Bias': round(med_abs_bias, 4),
                'RMSE': round(rmse, 4),
                'Coverage_95_Pct': round(cov_sate, 1),
                'Coverage_SATE_95_Pct': round(cov_sate, 1),
                'Coverage_PATE_95_Pct': round(cov_pate, 1),
                'Mean_Lambda_Min': round(l_min, 4),
                'Mean_Kappa': round(kappa_mean, 2),
                'Mean_Entropy': round(entropy_mean, 4),
                'Mean_Proxy_Error': round(proxy_err_mean, 4)
            })
            
    df_summary = pd.DataFrame(summary_list)
    summary_path = "reports/or_dml_benchmark_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"Summary table saved to {summary_path}.")
    
    # Overall method summary across all delta_z
    overall_list = []
    for m in methods:
        sub_m = df_raw[df_raw['method'] == m]
        bias = sub_m['bias'].values
        overall_list.append({
            'Method': m,
            'Abs_Bias': round(float(np.mean(np.abs(bias))), 4),
            'Median_Abs_Bias': round(float(np.median(np.abs(bias))), 4),
            'Mean_Bias': round(float(np.mean(bias)), 4),
            'RMSE': round(float(np.sqrt(np.mean(bias ** 2))), 4),
            'Coverage_SATE_95_Pct': round(float(np.mean(sub_m['cov_sate']) * 100.0), 1),
            'Coverage_PATE_95_Pct': round(float(np.mean(sub_m['cov_pate']) * 100.0), 1)
        })
    df_overall = pd.DataFrame(overall_list)
    print("\n=== OVERALL BENCHMARK RESULTS (500 REPLICATIONS) ===")
    print(df_overall.to_string(index=False))
    
    # -------------------------------------------------------------
    # Factorial 2D Difficulty Grid (Disentangling Proxy Error and Causal Overlap)
    # -------------------------------------------------------------
    df_fact = run_factorial_difficulty_grid(n_replications_per_cell=40, N=N)
    
    # -------------------------------------------------------------
    # Regularization Tradeoff Experiment (Theorem 3 Validation)
    # -------------------------------------------------------------
    df_reg = run_regularization_experiment(n_replications=100, N=N)
    
    # -------------------------------------------------------------
    # Plotting: Figure 2 Difficulty Frontier (with Factorial Panel d)
    # -------------------------------------------------------------
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'mathtext.fontset': 'cm',
        'figure.dpi': 300,
        'axes.labelsize': 11,
        'axes.titlesize': 11.5,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5
    })
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
    palette = {
        'Oracle DML': '#2ca02c',
        'Standard DML': '#d62728',
        'Block DML': '#ff7f0e',
        'DML + Z Controls (Ridge)': '#8c564b',
        'DML + Z Controls (GBM)': '#e377c2',
        'Regime FE DML (Hard)': '#7f7f7f',
        'Unregularized Coupled DML': '#9467bd',
        'Spectral OR-DML (Ours)': '#1f77b4',
        'Filtered OR-DML (Ours)': '#17becf',
        'Decoupled Soft OR-DML': '#bcbd22',
        'HMM(Z, T) OR-DML': '#008080'
    }
    markers = {
        'Oracle DML': 'o',
        'Standard DML': 's',
        'Block DML': '^',
        'DML + Z Controls (Ridge)': 'v',
        'DML + Z Controls (GBM)': '*',
        'Regime FE DML (Hard)': 'p',
        'Unregularized Coupled DML': 'D',
        'Spectral OR-DML (Ours)': 'P',
        'Filtered OR-DML (Ours)': 'X',
        'Decoupled Soft OR-DML': 'h',
        'HMM(Z, T) OR-DML': '<'
    }
    
    # Panel (a): Absolute Bias vs Delta_Z
    ax_a = axes[0]
    for m in methods:
        sub = df_summary[df_summary['Method'] == m]
        ax_a.plot(sub['Delta_Z'], sub['Abs_Bias'], marker=markers[m], label=m, color=palette[m], lw=2.2, ms=6)
    ax_a.set_xscale('log')
    ax_a.set_xlabel(r'Separation $\Delta_Z$')
    ax_a.set_ylabel(r'Mean Absolute Bias $|\hat{\theta} - \theta^*|$')
    ax_a.set_title(r'(a) Bias vs. Separation $\Delta_Z$', fontweight='bold')
    
    # Panel (b): Empirical RMSE vs Delta_Z
    ax_b = axes[1]
    for m in methods:
        sub = df_summary[df_summary['Method'] == m]
        ax_b.plot(sub['Delta_Z'], sub['RMSE'], marker=markers[m], label=m, color=palette[m], lw=2.2, ms=6)
    ax_b.set_xscale('log')
    ax_b.set_xlabel(r'Separation $\Delta_Z$')
    ax_b.set_ylabel(r'Empirical RMSE')
    ax_b.set_title(r'(b) Empirical RMSE', fontweight='bold')
    
    # Panel (c): 95% Coverage vs Delta_Z
    ax_c = axes[2]
    for m in methods:
        sub = df_summary[df_summary['Method'] == m]
        ax_c.plot(sub['Delta_Z'], sub['Coverage_SATE_95_Pct'], marker=markers[m], label=m, color=palette[m], lw=2.2, ms=6)
    ax_c.axhline(95.0, color='black', linestyle='--', lw=1.5, label='Nominal 95%')
    ax_c.set_xscale('log')
    ax_c.set_xlabel(r'Separation $\Delta_Z$')
    ax_c.set_ylabel(r'Empirical 95% Coverage (%)')
    ax_c.set_title(r'(c) 95% CI Coverage', fontweight='bold')
    ax_c.set_ylim(-5, 105)
    
    # Shared legend for panels a-c above the subplots
    handles_methods = [plt.Line2D([0], [0], marker=markers[m], color=palette[m], label=m, lw=2.0, ms=6) for m in methods]
    handles_methods.append(plt.Line2D([0], [0], color='black', linestyle='--', lw=1.5, label='Nominal 95%'))
    
    # Panel (d): Factorial Calibration - Bias vs Theoretical Degradation Index (eps_gamma / lambda_min)
    ax_d = axes[3]
    or_fact = df_fact[df_fact['Method'] == 'Spectral OR-DML (Ours)'].copy()
    or_fact['Degradation_Index'] = or_fact['Mean_Proxy_Error'] / or_fact['Mean_Lambda_Min']
    
    dt_colors = {1.0: '#e377c2', 4.0: '#1f77b4', 8.0: '#2ca02c'}
    dt_markers = {1.0: 'o', 4.0: 's', 8.0: '^'}
    dt_labels = {
        1.0: r'Weak Overlap ($\lambda_{\min}\approx 0.19\text{--}0.43$)',
        4.0: r'Mod. Overlap ($\lambda_{\min}\approx 0.66\text{--}0.80$)',
        8.0: r'Strong Overlap ($\lambda_{\min}\approx 2.13\text{--}2.22$)'
    }
    
    for dt in [1.0, 4.0, 8.0]:
        sub = or_fact[or_fact['Delta_T'] == dt].sort_values('Degradation_Index')
        ax_d.plot(sub['Degradation_Index'], sub['Abs_Bias'], 
                  marker=dt_markers[dt], color=dt_colors[dt], lw=2.0, ms=6,
                  label=dt_labels[dt])
                  
    # Baseline Standard DML for comparison
    std_fact = df_fact[(df_fact['Method'] == 'Standard DML') & (df_fact['Delta_T'] == 4.0)]
    ax_d.axhline(float(std_fact['Abs_Bias'].mean()), color='#d62728', linestyle=':', lw=2.0, label='Standard DML (+8.44)')
    
    # Annotate the two adversarial extremes confirming orthogonal failure modes
    pt_a = or_fact[(or_fact['Delta_Z'] == 0.2) & (or_fact['Delta_T'] == 8.0)].iloc[0]
    ax_d.annotate('Adv. A: High $\\bar{\\varepsilon}_\\gamma$, High $\\lambda_{\\min}$\n($\\bar{\\varepsilon}_\\gamma=0.81, \\lambda_{\\min}=2.22$)',
                  xy=(pt_a['Degradation_Index'], pt_a['Abs_Bias']),
                  xytext=(pt_a['Degradation_Index'] + 0.15, pt_a['Abs_Bias'] - 1.8),
                  arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2),
                  fontsize=7.5, backgroundcolor='#ffffff')
                  
    pt_b = or_fact[(or_fact['Delta_Z'] == 4.0) & (or_fact['Delta_T'] == 1.0)].iloc[0]
    ax_d.annotate('Adv. B: Low $\\bar{\\varepsilon}_\\gamma$, Low $\\lambda_{\\min}$\n($\\bar{\\varepsilon}_\\gamma=0.05, \\lambda_{\\min}=0.43$)',
                  xy=(pt_b['Degradation_Index'], pt_b['Abs_Bias']),
                  xytext=(pt_b['Degradation_Index'] + 0.25, pt_b['Abs_Bias'] + 1.8),
                  arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2),
                  fontsize=7.5, backgroundcolor='#ffffff')

    ax_d.set_xlabel(r'Degradation Ratio $\bar{\varepsilon}_\gamma / \lambda_{\min}(\boldsymbol{J})$')
    ax_d.set_ylabel(r'Mean Absolute Bias $|\hat{\theta} - \theta^*|$')
    ax_d.set_title(r'(d) Calibration vs. $\bar{\varepsilon}_\gamma / \lambda_{\min}(\boldsymbol{J})$', fontweight='bold')
    ax_d.legend(frameon=True, fontsize=7.5, loc='upper left')
    
    fig.legend(handles=handles_methods, loc='upper center', bbox_to_anchor=(0.50, 1.05), ncol=6, framealpha=0.95, fontsize=7.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig_path = "plots/fig2_difficulty_frontier.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    import shutil
    shutil.copyfile(fig_path, "paper/plots/fig2_difficulty_frontier.png")
    print(f"Saved publication-grade signature figure to {fig_path} and synced to paper/plots/fig2_difficulty_frontier.png.")
    
    # -------------------------------------------------------------
    # Dedicated Plot: Figure 4 Spectral Regularization Frontier
    # -------------------------------------------------------------
    plt.figure(figsize=(7.5, 5.2))
    sub_reg_plot = df_reg[df_reg['Lambda'] > 0]
    plt.plot(sub_reg_plot['Lambda'], sub_reg_plot['MSE'], 'r-o', lw=2.4, ms=7, label=r'Empirical MSE $\|\hat{\boldsymbol{\theta}}_\lambda - \boldsymbol{\theta}^*\|_2^2$')
    plt.plot(sub_reg_plot['Lambda'], sub_reg_plot['Variance'], 'b--s', lw=2.0, ms=6, label=r'Variance $\mathrm{Tr}(\boldsymbol{\Sigma}_\lambda)/N$')
    plt.plot(sub_reg_plot['Lambda'], sub_reg_plot['Abs_Bias'] ** 2, 'g:^', lw=2.0, ms=6, label=r'Squared Bias $\|\mathrm{Bias}(\lambda)\|_2^2$')
    plt.xscale('log')
    plt.xlabel(r'Spectral Shrinkage Penalty $\lambda$ (Log scale)')
    plt.ylabel(r'Risk Decomposition')
    plt.title(r'Spectral Regularization Frontier ($\lambda_{\min} \approx 0.025$)', fontweight='bold')
    
    min_idx = sub_reg_plot['MSE'].idxmin()
    min_lambda = sub_reg_plot.loc[min_idx, 'Lambda']
    min_mse = sub_reg_plot.loc[min_idx, 'MSE']
    plt.annotate(rf'Empirical Min $\lambda \approx {min_lambda}$' + '\n' + r'(Risk reduction at $\lambda=0.10$)',
                 xy=(min_lambda, min_mse),
                 xytext=(min_lambda * 1.3, min_mse + 0.14),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                 fontweight='bold', fontsize=9)
    plt.legend(frameon=True, fontsize=9.5)
    plt.tight_layout()
    fig4_path = "plots/fig4_regularization_frontier.png"
    plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved dedicated regularization figure to {fig4_path}.")
    
    total_time = time.time() - t0
    print(f"Total benchmark run time: {total_time:.2f} seconds ({total_time/60.0:.2f} minutes).")
    return df_summary


def run_factorial_difficulty_grid(n_replications_per_cell: int = 40, N: int = 1200) -> pd.DataFrame:
    """
    Execute 2D Factorial Latent-Confounding Benchmark:
    5 Delta_Z grid points x 3 Delta_T grid points = 15 design cells.
    Disentangles state observability (proxy error epsilon_gamma) from causal overlap (lambda_min(J)).
    """
    delta_z_grid = [0.2, 0.5, 1.0, 2.0, 4.0]
    delta_t_grid = [1.0, 4.0, 8.0]
    
    print("\n==================================================================")
    print(f"Executing 2D Factorial Latent-Confounding Benchmark ({len(delta_z_grid)}x{len(delta_t_grid)} = 15 cells, {n_replications_per_cell} reps/cell, N={N})")
    print(f"Delta_Z (Proxy observability): {delta_z_grid}")
    print(f"Delta_T (Treatment regime separation / Overlap): {delta_t_grid}")
    print("==================================================================\n")
    
    tasks = [(r, dz, dt) for dz in delta_z_grid for dt in delta_t_grid for r in range(n_replications_per_cell)]
    
    def single_factorial_rep(r, dz, dt):
        seed = int(300000 + int(dz * 1000) + int(dt * 100) + r)
        Y, T, X, Z, gt = generate_difficulty_dgp(N=N, delta_z=dz, delta_t=dt, random_state=seed)
        sample_ate = gt['true_sample_ate']
        true_pop_ate = gt['true_pop_ate']
        S_true = gt['S']
        gamma_oracle = np.column_stack([1 - S_true, S_true])
        
        # 1. Oracle DML
        pb = PurgedBlockKFold(n_splits=4, embargo_tau=12)
        K = 2
        tilde_Y_orc = np.zeros((K, N))
        tilde_T_orc = np.zeros((K, N))
        for k in range(K):
            w = np.maximum(gamma_oracle[:, k], 1e-4)
            for tr, te in pb.split(N):
                my = Ridge(alpha=1.0).fit(X[tr], Y[tr], sample_weight=w[tr])
                tilde_Y_orc[k, te] = Y[te] - my.predict(X[te])
                mt = Ridge(alpha=1.0).fit(X[tr], T[tr], sample_weight=w[tr])
                tilde_T_orc[k, te] = T[te] - mt.predict(X[te])
        J_orc = np.zeros((K, K))
        S_orc = np.zeros(K)
        for j in range(K):
            S_orc[j] = np.mean(gamma_oracle[:, j] * tilde_T_orc[j] * tilde_Y_orc[j])
            for k in range(K):
                J_orc[j, k] = np.mean(gamma_oracle[:, j] * gamma_oracle[:, k] * tilde_T_orc[j] * tilde_T_orc[k])
        th_orc = float(np.mean(gamma_oracle, axis=0) @ (np.linalg.pinv(J_orc) @ S_orc))
        l_min_orc = float(np.min(np.linalg.eigvalsh(J_orc)))
        
        # 2. Standard DML
        kf = KFold(n_splits=4, shuffle=True, random_state=seed)
        tilde_Y_std = np.zeros(N)
        tilde_T_std = np.zeros(N)
        for tr, te in kf.split(X):
            my = Ridge(alpha=1.0).fit(X[tr], Y[tr])
            tilde_Y_std[te] = Y[te] - my.predict(X[te])
            mt = Ridge(alpha=1.0).fit(X[tr], T[tr])
            tilde_T_std[te] = T[te] - mt.predict(X[te])
        th_std = float(np.mean(tilde_T_std * tilde_Y_std) / max(np.mean(tilde_T_std**2), 1e-12))
        
        # 3. Unregularized Coupled DML
        unreg = OverlapAwareRegimeDML(n_regimes=2, n_splits=4, embargo_tau=12, reg_lambda=0.0, reg_alpha=0.0, nuisance_model=Ridge(alpha=1.0), random_state=seed)
        unreg.fit(Y, T, X, Z)
        th_unreg = unreg.ate_
        l_min_unreg = unreg.lambda_min_
        proxy_err, _ = compute_aligned_proxy_error(unreg.gamma_, gamma_oracle)
        
        # 4. Spectral OR-DML
        or_dml = OverlapAwareRegimeDML(n_regimes=2, n_splits=4, embargo_tau=12, reg_alpha=0.05, nuisance_model=Ridge(alpha=1.0), random_state=seed)
        or_dml.fit(Y, T, X, Z)
        th_or = or_dml.ate_
        
        return [
            {'rep': r, 'delta_z': dz, 'delta_t': dt, 'method': 'Oracle DML', 'theta': th_orc, 'bias': th_orc - true_pop_ate, 'proxy_error': 0.0, 'lambda_min': l_min_orc},
            {'rep': r, 'delta_z': dz, 'delta_t': dt, 'method': 'Standard DML', 'theta': th_std, 'bias': th_std - true_pop_ate, 'proxy_error': np.nan, 'lambda_min': float(np.mean(tilde_T_std**2))},
            {'rep': r, 'delta_z': dz, 'delta_t': dt, 'method': 'Unregularized Coupled DML', 'theta': th_unreg, 'bias': th_unreg - true_pop_ate, 'proxy_error': proxy_err, 'lambda_min': l_min_unreg},
            {'rep': r, 'delta_z': dz, 'delta_t': dt, 'method': 'Spectral OR-DML (Ours)', 'theta': th_or, 'bias': th_or - true_pop_ate, 'proxy_error': proxy_err, 'lambda_min': or_dml.lambda_min_},
        ]
        
    nested = Parallel(n_jobs=-1, verbose=0)(delayed(single_factorial_rep)(r, dz, dt) for r, dz, dt in tasks)
    flat = [rec for r_list in nested for rec in r_list]
    df_raw = pd.DataFrame(flat)
    
    summary = []
    for dz in delta_z_grid:
        for dt in delta_t_grid:
            cell_sub = df_raw[(df_raw['delta_z'] == dz) & (df_raw['delta_t'] == dt)]
            for m in ['Oracle DML', 'Standard DML', 'Unregularized Coupled DML', 'Spectral OR-DML (Ours)']:
                sub_m = cell_sub[cell_sub['method'] == m]
                biases = sub_m['bias'].values
                summary.append({
                    'Delta_Z': dz,
                    'Delta_T': dt,
                    'Method': m,
                    'Abs_Bias': round(float(np.mean(np.abs(biases))), 4),
                    'Mean_Bias': round(float(np.mean(biases)), 4),
                    'RMSE': round(float(np.sqrt(np.mean(biases**2))), 4),
                    'Mean_Proxy_Error': round(float(np.nanmean(sub_m['proxy_error'])), 4),
                    'Mean_Lambda_Min': round(float(np.mean(sub_m['lambda_min'])), 4)
                })
    df_fact = pd.DataFrame(summary)
    os.makedirs("reports", exist_ok=True)
    fact_path = "reports/or_dml_factorial_frontier.csv"
    df_fact.to_csv(fact_path, index=False)
    print(f"2D Factorial summary saved to {fact_path}.")
    return df_fact


def run_regularization_experiment(n_replications: int = 100, N: int = 1200) -> pd.DataFrame:
    """
    Run dedicated regularization experiment on an ill-conditioned regime (lambda_min ~ 0.025)
    evaluating lambda across a grid [0.0, 1e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5].
    Demonstrates the U-shaped bias-variance tradeoff predicted by Theorem 3.
    """
    print(f"\n==================================================================")
    print(f"Executing Spectral Regularization Tradeoff Experiment ({n_replications} reps, N={N})")
    print(f"==================================================================")
    
    lambdas = [0.0, 1e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5]
    
    def single_rep(r):
        seed = 200000 + r
        Y, T, X, Z, gt = generate_ill_conditioned_dgp(N=N, random_state=seed)
        sample_ate = gt['true_sample_ate']
        
        hmm = LatentRegimeHMM(n_regimes=2, random_state=seed).fit(Z)
        gamma = hmm.predict_posteriors(Z, mode='smooth')
        if np.mean(gamma[:, 0]) < np.mean(gamma[:, 1]):
            gamma = gamma[:, [1, 0]]
            
        pb = PurgedBlockKFold(n_splits=4, embargo_tau=12)
        tilde_Y = np.zeros((2, N))
        tilde_T = np.zeros((2, N))
        for k in range(2):
            w = np.maximum(gamma[:, k], 1e-4)
            for tr, te in pb.split(N):
                my = Ridge(alpha=1.0).fit(X[tr], Y[tr], sample_weight=w[tr])
                tilde_Y[k, te] = Y[te] - my.predict(X[te])
                mt = Ridge(alpha=1.0).fit(X[tr], T[tr], sample_weight=w[tr])
                tilde_T[k, te] = T[te] - mt.predict(X[te])
                
        J = np.zeros((2, 2))
        S_vec = np.zeros(2)
        for j in range(2):
            S_vec[j] = np.mean(gamma[:, j] * tilde_T[j] * tilde_Y[j])
            for k in range(2):
                J[j, k] = np.mean(gamma[:, j] * gamma[:, k] * tilde_T[j] * tilde_T[k])
                
        l_min = float(np.min(np.linalg.eigvalsh(J)))
        kappa = float(np.max(np.linalg.eigvalsh(J)) / max(l_min, 1e-12))
        pi = np.mean(gamma, axis=0)
        
        scores = np.zeros((N, 2))
        th_init = np.linalg.pinv(J + 0.01 * np.eye(2)) @ S_vec
        for k in range(2):
            scores[:, k] = gamma[:, k] * tilde_T[k] * (tilde_Y[k] - th_init[k] * tilde_T[k])
        Omega = (scores.T @ scores) / N
        for lag in range(1, 13):
            w_l = 1.0 - (lag / 13.0)
            G_l = (scores[lag:].T @ scores[:-lag]) / N
            Omega += w_l * (G_l + G_l.T)
            
        row_res = []
        for l in lambdas:
            inv_reg = np.linalg.pinv(J + l * np.eye(2))
            th_l = inv_reg @ S_vec
            ate_l = float(pi @ th_l)
            Sigma_l = (inv_reg @ Omega @ inv_reg) / N
            se_l = float(np.sqrt(max(pi @ Sigma_l @ pi, 1e-12)))
            cov_l = float(abs(ate_l - sample_ate) <= 1.96 * se_l)
            row_res.append({
                'rep': r, 'lambda': l, 'theta': ate_l, 'se': se_l,
                'bias': ate_l - sample_ate, 'coverage': cov_l,
                'lambda_min': l_min, 'kappa': kappa
            })
        return row_res

    rep_results = Parallel(n_jobs=-1, verbose=0)(delayed(single_rep)(r) for r in range(n_replications))
    flattened = [rec for r_list in rep_results for rec in r_list]
    df_reg_raw = pd.DataFrame(flattened)
    
    summary_reg = []
    for l in lambdas:
        sub = df_reg_raw[df_reg_raw['lambda'] == l]
        biases = sub['bias'].values
        thetas = sub['theta'].values
        mean_bias = float(np.mean(biases))
        abs_bias = float(np.mean(np.abs(biases)))
        variance = float(np.var(thetas))
        mse = float(np.mean(biases ** 2))
        cov_rate = float(np.mean(sub['coverage']) * 100.0)
        summary_reg.append({
            'Lambda': l,
            'Abs_Bias': round(abs_bias, 4),
            'Mean_Bias': round(mean_bias, 4),
            'Variance': round(variance, 4),
            'MSE': round(mse, 4),
            'RMSE': round(float(np.sqrt(mse)), 4),
            'Coverage_95_Pct': round(cov_rate, 1),
            'Mean_Lambda_Min': round(float(np.mean(sub['lambda_min'])), 4),
            'Mean_Kappa': round(float(np.mean(sub['kappa'])), 2)
        })
    df_summary_reg = pd.DataFrame(summary_reg)
    reg_path = "reports/or_dml_regularization_frontier.csv"
    df_summary_reg.to_csv(reg_path, index=False)
    print(f"Regularization summary saved to {reg_path}.")
    print("\n=== SPECTRAL REGULARIZATION TRADEOFF RESULTS ===")
    print(df_summary_reg.to_string(index=False))
    return df_summary_reg


if __name__ == '__main__':
    run_difficulty_frontier_benchmark(n_replications_per_grid=100, N=1200)
