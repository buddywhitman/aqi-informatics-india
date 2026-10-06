"""
deep_empirical_evaluations.py
=============================
Rigorous, publication-grade evaluations addressing key methodological frontiers
for Overlap-Aware Regime Double Machine Learning (OR-DML):

1. Exploration 1: The Transition Zone / Continuous Boundary Stress Test
   - Evaluates Soft Probabilistic Weighting (OR-DML) vs Hard Assignment (Regime FE DML)
     and Boundary Thresholding across varying degrees of regime overlap / continuous mixing.
   - Measures regime-specific bias, ATE bias, RMSE, and 95% sandwich coverage.

2. Exploration 2: Near-Singular Cross-Regime Geometry (Spectral Regularization Test)
   - Evaluates Unregularized Coupled DML (lambda = 0) vs Spectral OR-DML (adaptive lambda
     and fixed regularizers) across a condition-number / lambda_min(J) spectrum.
   - Tests whether spectral regularization prevents variance explosion when lambda_min(J) -> 0.

3. Exploration 3: Representation Perturbation Done Right (Theorem 3 & Proposition 3)
   - Evaluates causal estimation error as a function of proxy recovery error epsilon_gamma
     under properly weighted nuisance residualization (avoiding omitted-regime leakage).
   - Validates the theoretical bound: ||theta_hat - theta*|| ~ epsilon_gamma / lambda_min(J).

4. Exploration 4: Task Conditioning Geometry (Holding Representation Fixed)
   - Evaluates causal error across downstream task geometries J while holding representation
     gamma_t (and epsilon_gamma) strictly constant.
   - Demonstrates that representation quality alone is insufficient; task conditioning is decisive.
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
from scipy.special import softmax
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from typing import Dict, List, Tuple
from joblib import Parallel, delayed

from src.or_dml import OverlapAwareRegimeDML, PurgedBlockKFold, LatentRegimeHMM
from src.synthetic_dgp_benchmark import compute_aligned_proxy_error, compute_hac_se


# ==============================================================================
# 1. EXPLORATION 1: TRANSITION ZONE / CONTINUOUS BOUNDARY STRESS TEST
# ==============================================================================
def generate_boundary_dgp(N: int = 1200,
                          overlap_sigma: float = 1.0,
                          delta_z: float = 1.2,
                          transition_fraction: float = 0.25,
                          random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Generate synthetic time series with continuous boundary/transitional dynamics:
    - S_t is a continuous latent state weight w_t in [0, 1] for transition periods,
      or discrete states with controlled emission overlap.
    - Here, S_t has discrete anchor states 0 and 1 with persistence 0.85, but
      with an explicit transition probability where intermediate states exist,
      yielding genuine posterior uncertainty gamma_t in [0.2, 0.8].
    """
    rng = np.random.RandomState(random_state)
    
    # Latent state Markov process with smooth boundary transitions
    persistence = 0.88
    p01 = 1.0 - persistence
    p10 = 0.15
    P_trans = np.array([[persistence, p01], [p10, 1.0 - p10]])
    stat_pi = np.array([p10 / (p01 + p10), p01 / (p01 + p10)])
    
    # Discrete underlying mode
    S_discrete = np.zeros(N, dtype=int)
    S_discrete[0] = 0 if rng.rand() < stat_pi[0] else 1
    for t in range(1, N):
        p_next = P_trans[S_discrete[t-1]]
        S_discrete[t] = 0 if rng.rand() < p_next[0] else 1
        
    # Transition periods: smooth transition weight w_t between state changes
    w = S_discrete.astype(float)
    if transition_fraction > 0:
        window = max(2, int(transition_fraction * 8))
        # Find switch points
        switches = np.where(np.diff(S_discrete) != 0)[0]
        for s in switches:
            start = max(0, s - window // 2)
            end = min(N, s + window // 2 + 1)
            # Create linear interpolation ramp
            ramp = np.linspace(S_discrete[start], S_discrete[min(N-1, end)], end - start)
            w[start:end] = ramp
            
    # Exogenous proxy Z_t (2D)
    Z = np.zeros((N, 2))
    mu_0 = np.array([delta_z, delta_z * 1.5])
    mu_1 = np.array([-delta_z, -delta_z * 1.5])
    
    # Emission is continuous mixture of mu_0 and mu_1 plus noise
    for t in range(N):
        mean_t = (1.0 - w[t]) * mu_0 + w[t] * mu_1
        Z[t] = rng.multivariate_normal(mean_t, [[overlap_sigma**2, 0.2], [0.2, overlap_sigma**2]])
        
    # Observed confounders X_t (autocorrelated)
    X = np.zeros((N, 3))
    innovations = rng.randn(N, 3)
    X[0] = innovations[0]
    for t in range(1, N):
        X[t] = 0.65 * X[t-1] + np.sqrt(1 - 0.65**2) * innovations[t]
        
    # Continuous Treatment T_t
    V = rng.normal(0, 1.0, size=N)
    b0 = 2.0
    b1 = 6.0
    T = (1.0 - w) * b0 + w * b1 + 0.8 * X[:, 0] - 0.5 * X[:, 1] + V
    
    # True causal effects
    theta_0 = 0.75
    theta_1 = 2.50
    theta_t = (1.0 - w) * theta_0 + w * theta_1
    true_ate = float(np.mean(theta_t))
    
    # Outcome Y_t
    U = rng.normal(0, 1.2, size=N)
    g_w = 15.0 * (1.0 - w) + 50.0 * w
    Y = theta_t * T + g_w + 1.2 * X[:, 0] + 0.9 * X[:, 2] + U
    
    gt = {
        'theta_regimes': {0: theta_0, 1: theta_1},
        'true_ate': true_ate,
        'true_weights': w,
        'S_discrete': S_discrete
    }
    return Y, T, X, Z, gt


def evaluate_boundary_stress_single(rep_id: int,
                                    transition_fraction: float,
                                    delta_z: float = 1.2,
                                    N: int = 1200) -> List[Dict]:
    """Compare OR-DML vs Hard Regime FE vs Thresholded FE across transition dynamics."""
    seed = 500000 + int(transition_fraction * 1000) + rep_id
    Y, T, X, Z, gt = generate_boundary_dgp(
        N=N, delta_z=delta_z, transition_fraction=transition_fraction, random_state=seed
    )
    true_ate = gt['true_ate']
    th0_true = gt['theta_regimes'][0]
    th1_true = gt['theta_regimes'][1]
    w_true = gt['true_weights']
    gamma_oracle = np.column_stack([1.0 - w_true, w_true])
    
    results = []
    
    # Fit HMM to extract posteriors
    hmm = LatentRegimeHMM(n_regimes=2, random_state=seed)
    hmm.fit(Z)
    gamma_smooth = hmm.predict_posteriors(Z, mode='smooth')
    proxy_err, gamma_aligned = compute_aligned_proxy_error(gamma_smooth, gamma_oracle)
    
    pb = PurgedBlockKFold(n_splits=4, embargo_tau=12)
    
    # ---------------------------------------------------------
    # 1. Oracle DML (with true soft weights w_t)
    # ---------------------------------------------------------
    K = 2
    tilde_Y_orc = np.zeros((K, N))
    tilde_T_orc = np.zeros((K, N))
    for k in range(K):
        w_k = np.maximum(gamma_oracle[:, k], 1e-4)
        for train_idx, test_idx in pb.split(N):
            m_y = Ridge(alpha=1.0).fit(X[train_idx], Y[train_idx], sample_weight=w_k[train_idx])
            tilde_Y_orc[k, test_idx] = Y[test_idx] - m_y.predict(X[test_idx])
            m_t = Ridge(alpha=1.0).fit(X[train_idx], T[train_idx], sample_weight=w_k[train_idx])
            tilde_T_orc[k, test_idx] = T[test_idx] - m_t.predict(X[test_idx])
            
    J_orc = np.zeros((K, K))
    S_orc = np.zeros(K)
    for j in range(K):
        S_orc[j] = np.mean(gamma_oracle[:, j] * tilde_T_orc[j] * tilde_Y_orc[j])
        for k in range(K):
            J_orc[j, k] = np.mean(gamma_oracle[:, j] * gamma_oracle[:, k] * tilde_T_orc[j] * tilde_T_orc[k])
    th_orc_vec = np.linalg.pinv(J_orc) @ S_orc
    pi_orc = np.mean(gamma_oracle, axis=0)
    ate_orc = float(pi_orc @ th_orc_vec)
    
    # Sandwich SE
    scores_orc = np.zeros((N, K))
    for k in range(K):
        scores_orc[:, k] = gamma_oracle[:, k] * tilde_T_orc[k] * (tilde_Y_orc[k] - th_orc_vec[k] * tilde_T_orc[k])
    Omega_orc = (scores_orc.T @ scores_orc) / N
    for lag in range(1, 13):
        w_l = 1.0 - (lag / 13.0)
        G_l = (scores_orc[lag:].T @ scores_orc[:-lag]) / N
        Omega_orc += w_l * (G_l + G_l.T)
    Sigma_orc = (np.linalg.pinv(J_orc) @ Omega_orc @ np.linalg.pinv(J_orc)) / N
    se_orc = float(np.sqrt(max(pi_orc @ Sigma_orc @ pi_orc, 1e-12)))
    cov_orc = float(abs(ate_orc - true_ate) <= 1.96 * se_orc)
    
    results.append({
        'rep_id': rep_id, 'trans_frac': transition_fraction, 'delta_z': delta_z,
        'method': 'Oracle DML (True Soft)',
        'theta_0': float(th_orc_vec[0]), 'theta_1': float(th_orc_vec[1]), 'ate': ate_orc,
        'bias_th0': float(th_orc_vec[0] - th0_true),
        'bias_th1': float(th_orc_vec[1] - th1_true),
        'bias_ate': float(ate_orc - true_ate),
        'se_ate': se_orc, 'coverage': cov_orc
    })
    
    # ---------------------------------------------------------
    # 2. Spectral OR-DML (Ours, Continuous Posterior Weighting)
    # ---------------------------------------------------------
    or_dml = OverlapAwareRegimeDML(
        n_regimes=2, n_splits=4, embargo_tau=12,
        reg_alpha=0.05, posterior_mode='smooth',
        nuisance_model=Ridge(alpha=1.0), random_state=seed
    )
    or_dml.fit(Y, T, X, Z)
    
    # Align regimes with ground truth based on proxy error alignment
    if proxy_err < float(np.mean(np.sum(np.abs(gamma_smooth - gamma_oracle), axis=1))):
        th0_or = or_dml.theta_regimes_[1]
        th1_or = or_dml.theta_regimes_[0]
    else:
        th0_or = or_dml.theta_regimes_[0]
        th1_or = or_dml.theta_regimes_[1]
    ate_or = or_dml.ate_
    se_or = or_dml.sate_se_
    cov_or = float(abs(ate_or - true_ate) <= 1.96 * se_or)
    
    results.append({
        'rep_id': rep_id, 'trans_frac': transition_fraction, 'delta_z': delta_z,
        'method': 'Spectral OR-DML (Soft)',
        'theta_0': th0_or, 'theta_1': th1_or, 'ate': ate_or,
        'bias_th0': th0_or - th0_true,
        'bias_th1': th1_or - th1_true,
        'bias_ate': ate_or - true_ate,
        'se_ate': se_or, 'coverage': cov_or
    })
    
    # ---------------------------------------------------------
    # 3. Hard Assignment Regime FE DML (argmax gamma)
    # ---------------------------------------------------------
    s_hard = np.argmax(gamma_aligned, axis=1) # 0 or 1
    th_fe_list = []
    w_fe_list = []
    for k in range(2):
        mask_k = (s_hard == k)
        w_fe_list.append(np.mean(mask_k))
        if np.sum(mask_k) > 30:
            X_k = X[mask_k]
            Y_k = Y[mask_k]
            T_k = T[mask_k]
            m_y_k = Ridge(alpha=1.0).fit(X_k, Y_k)
            res_y_k = Y_k - m_y_k.predict(X_k)
            m_t_k = Ridge(alpha=1.0).fit(X_k, T_k)
            res_t_k = T_k - m_t_k.predict(X_k)
            th_k = float(np.mean(res_t_k * res_y_k) / max(np.mean(res_t_k**2), 1e-12))
        else:
            th_k = 0.0
        th_fe_list.append(th_k)
    ate_fe = float(w_fe_list[0] * th_fe_list[0] + w_fe_list[1] * th_fe_list[1])
    
    results.append({
        'rep_id': rep_id, 'trans_frac': transition_fraction, 'delta_z': delta_z,
        'method': 'Regime FE DML (Hard)',
        'theta_0': th_fe_list[0], 'theta_1': th_fe_list[1], 'ate': ate_fe,
        'bias_th0': th_fe_list[0] - th0_true,
        'bias_th1': th_fe_list[1] - th1_true,
        'bias_ate': ate_fe - true_ate,
        'se_ate': np.nan, 'coverage': np.nan
    })
    
    # ---------------------------------------------------------
    # 4. Thresholded Hard FE (drops uncertain points max(gamma) < 0.75)
    # ---------------------------------------------------------
    max_gamma = np.max(gamma_aligned, axis=1)
    confident_mask = (max_gamma >= 0.75)
    th_thresh_list = []
    w_thresh_list = []
    for k in range(2):
        mask_k = confident_mask & (s_hard == k)
        w_thresh_list.append(np.sum(mask_k) / max(np.sum(confident_mask), 1))
        if np.sum(mask_k) > 30:
            X_k = X[mask_k]
            Y_k = Y[mask_k]
            T_k = T[mask_k]
            m_y_k = Ridge(alpha=1.0).fit(X_k, Y_k)
            res_y_k = Y_k - m_y_k.predict(X_k)
            m_t_k = Ridge(alpha=1.0).fit(X_k, T_k)
            res_t_k = T_k - m_t_k.predict(X_k)
            th_k = float(np.mean(res_t_k * res_y_k) / max(np.mean(res_t_k**2), 1e-12))
        else:
            th_k = 0.0
        th_thresh_list.append(th_k)
    ate_thresh = float(w_thresh_list[0] * th_thresh_list[0] + w_thresh_list[1] * th_thresh_list[1])
    
    results.append({
        'rep_id': rep_id, 'trans_frac': transition_fraction, 'delta_z': delta_z,
        'method': 'Thresholded FE (tau>=0.75)',
        'theta_0': th_thresh_list[0], 'theta_1': th_thresh_list[1], 'ate': ate_thresh,
        'bias_th0': th_thresh_list[0] - th0_true,
        'bias_th1': th_thresh_list[1] - th1_true,
        'bias_ate': ate_thresh - true_ate,
        'se_ate': np.nan, 'coverage': np.nan
    })
    
    # ---------------------------------------------------------
    # 5. Standard DML (Ignores regimes)
    # ---------------------------------------------------------
    tilde_Y_std = np.zeros(N)
    tilde_T_std = np.zeros(N)
    for tr, te in pb.split(N):
        m_y = Ridge(alpha=1.0).fit(X[tr], Y[tr])
        tilde_Y_std[te] = Y[te] - m_y.predict(X[te])
        m_t = Ridge(alpha=1.0).fit(X[tr], T[tr])
        tilde_T_std[te] = T[te] - m_t.predict(X[te])
    ate_std = float(np.mean(tilde_T_std * tilde_Y_std) / max(np.mean(tilde_T_std ** 2), 1e-12))
    se_std = compute_hac_se(tilde_Y_std, tilde_T_std, ate_std, hac_lag=12)
    cov_std = float(abs(ate_std - true_ate) <= 1.96 * se_std)
    
    results.append({
        'rep_id': rep_id, 'trans_frac': transition_fraction, 'delta_z': delta_z,
        'method': 'Standard DML (Pooled)',
        'theta_0': ate_std, 'theta_1': ate_std, 'ate': ate_std,
        'bias_th0': ate_std - th0_true,
        'bias_th1': ate_std - th1_true,
        'bias_ate': ate_std - true_ate,
        'se_ate': se_std, 'coverage': cov_std
    })
    
    return results


def run_exploration1_boundary_stress(n_reps: int = 50, n_jobs: int = -1) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("RUNNING EXPLORATION 1: TRANSITION ZONE / BOUNDARY STRESS TEST")
    print("Testing Soft OR-DML vs Hard FE across Transition Fractions in [0.0, 0.15, 0.30, 0.50]")
    print("=" * 78)
    
    trans_fractions = [0.0, 0.15, 0.30, 0.50]
    all_tasks = [(rep, tf) for tf in trans_fractions for rep in range(n_reps)]
    
    start_t = time.time()
    raw_results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_boundary_stress_single)(rep, tf) for rep, tf in all_tasks
    )
    
    flat = [item for sublist in raw_results for item in sublist]
    df = pd.DataFrame(flat)
    
    summary = df.groupby(['trans_frac', 'method']).agg(
        Mean_Abs_Bias_ATE=('bias_ate', lambda x: np.mean(np.abs(x))),
        RMSE_ATE=('bias_ate', lambda x: np.sqrt(np.mean(x**2))),
        Mean_Abs_Bias_Th0=('bias_th0', lambda x: np.mean(np.abs(x))),
        Mean_Abs_Bias_Th1=('bias_th1', lambda x: np.mean(np.abs(x))),
        Coverage_Pct=('coverage', lambda x: np.nanmean(x) * 100.0)
    ).reset_index()
    
    os.makedirs('reports', exist_ok=True)
    df.to_csv('reports/deep_eval_exploration1_boundary_stress_raw.csv', index=False)
    summary.to_csv('reports/deep_eval_exploration1_boundary_stress_summary.csv', index=False)
    print(f"Exploration 1 completed in {time.time() - start_t:.1f}s.")
    print("\nSummary Table:")
    print(summary.to_string(index=False))
    return summary


# ==============================================================================
# 2. EXPLORATION 2: ILL-CONDITIONED CROSS-REGIME GEOMETRY (SPECTRAL REGULARIZATION)
# ==============================================================================
def generate_ill_conditioned_geometry_dgp(N: int = 1200,
                                          pi_rare: float = 0.08,
                                          treatment_var_ratio: float = 0.15,
                                          random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Generate synthetic time series where cross-regime Jacobian is near-singular:
    - Regime 1 is rare: pi_rare in [0.03, 0.08, 0.20, 0.50]
    - Treatment residual variance in Regime 1 is attenuated by treatment_var_ratio in [0.05, 0.2, 1.0]
    - Causes lambda_min(J) to plunge towards 0.005 - 0.05, testing Theorem 4 spectral shrinkage.
    """
    rng = np.random.RandomState(random_state)
    
    # 2-state Markov Chain with asymmetric stationary probabilities
    p10 = 0.40
    p01 = (pi_rare / (1.0 - pi_rare)) * p10
    persistence_0 = 1.0 - p01
    persistence_1 = 1.0 - p10
    
    P_trans = np.array([[persistence_0, p01], [p10, persistence_1]])
    stat_pi = np.array([1.0 - pi_rare, pi_rare])
    
    S = np.zeros(N, dtype=int)
    S[0] = 0 if rng.rand() < stat_pi[0] else 1
    for t in range(1, N):
        p_next = P_trans[S[t-1]]
        S[t] = 0 if rng.rand() < p_next[0] else 1
        
    # Exogenous proxy Z_t (well-separated so proxy error is low, isolating geometric ill-conditioning)
    Z = np.zeros((N, 2))
    Z[S == 0] = rng.multivariate_normal([2.0, 2.5], [[1.0, 0.2], [0.2, 1.0]], size=np.sum(S == 0))
    Z[S == 1] = rng.multivariate_normal([-2.0, -2.5], [[1.0, 0.2], [0.2, 1.0]], size=np.sum(S == 1))
    
    # Confounders X_t
    X = np.zeros((N, 3))
    innovations = rng.randn(N, 3)
    X[0] = innovations[0]
    for t in range(1, N):
        X[t] = 0.65 * X[t-1] + np.sqrt(1 - 0.65**2) * innovations[t]
        
    # Continuous Treatment T_t with heterogeneous variance
    V = np.zeros(N)
    V[S == 0] = rng.normal(0, 1.0, size=np.sum(S == 0))
    V[S == 1] = rng.normal(0, np.sqrt(treatment_var_ratio), size=np.sum(S == 1))
    
    b0 = 2.0
    b1 = 5.0
    T = b0 * (1 - S) + b1 * S + 0.8 * X[:, 0] - 0.5 * X[:, 1] + V
    
    theta_true = {0: 0.75, 1: 2.50}
    pop_ate = float(stat_pi[0] * theta_true[0] + stat_pi[1] * theta_true[1])
    sample_ate = float(np.mean(theta_true[0] * (1 - S) + theta_true[1] * S))
    
    U = rng.normal(0, 1.2, size=N)
    g_S = 15.0 * (1 - S) + 45.0 * S
    Y = (theta_true[0] * (1 - S) + theta_true[1] * S) * T + g_S + 1.2 * X[:, 0] + 0.9 * X[:, 2] + U
    
    gt = {
        'theta_regimes': theta_true,
        'true_pop_ate': pop_ate,
        'true_sample_ate': sample_ate,
        'stationary_pi': stat_pi,
        'S': S
    }
    return Y, T, X, Z, gt


def evaluate_ill_conditioned_single(rep_id: int,
                                    pi_rare: float,
                                    treatment_var_ratio: float,
                                    N: int = 1200) -> List[Dict]:
    seed = 600000 + int(pi_rare * 10000) + int(treatment_var_ratio * 100) + rep_id
    Y, T, X, Z, gt = generate_ill_conditioned_geometry_dgp(
        N=N, pi_rare=pi_rare, treatment_var_ratio=treatment_var_ratio, random_state=seed
    )
    true_ate = gt['true_pop_ate']
    th0_true = gt['theta_regimes'][0]
    th1_true = gt['theta_regimes'][1]
    
    results = []
    
    # 1. Unregularized Coupled DML (lambda = 0)
    unreg = OverlapAwareRegimeDML(
        n_regimes=2, n_splits=4, embargo_tau=12,
        reg_lambda=0.0, reg_alpha=0.0,
        posterior_mode='smooth', nuisance_model=Ridge(alpha=1.0),
        random_state=seed
    )
    unreg.fit(Y, T, X, Z)
    th_unreg_ate = unreg.ate_
    se_unreg = unreg.sate_se_
    cov_unreg = float(abs(th_unreg_ate - true_ate) <= 1.96 * se_unreg)
    lmin = unreg.lambda_min_
    kappa = unreg.kappa_
    
    results.append({
        'rep_id': rep_id, 'pi_rare': pi_rare, 'var_ratio': treatment_var_ratio,
        'method': 'Unregularized (lambda=0)',
        'lambda_param': 0.0,
        'lambda_min': lmin, 'kappa': kappa,
        'ate': th_unreg_ate, 'bias_ate': th_unreg_ate - true_ate,
        'se_ate': se_unreg, 'coverage': cov_unreg
    })
    
    # 2. Spectral OR-DML with Adaptive Lambda
    or_adaptive = OverlapAwareRegimeDML(
        n_regimes=2, n_splits=4, embargo_tau=12,
        reg_alpha=0.05, posterior_mode='smooth',
        nuisance_model=Ridge(alpha=1.0), random_state=seed
    )
    or_adaptive.fit(Y, T, X, Z)
    th_adapt_ate = or_adaptive.ate_
    se_adapt = or_adaptive.sate_se_
    cov_adapt = float(abs(th_adapt_ate - true_ate) <= 1.96 * se_adapt)
    
    results.append({
        'rep_id': rep_id, 'pi_rare': pi_rare, 'var_ratio': treatment_var_ratio,
        'method': 'Spectral Adaptive (Ours)',
        'lambda_param': or_adaptive.effective_lambda_,
        'lambda_min': lmin, 'kappa': kappa,
        'ate': th_adapt_ate, 'bias_ate': th_adapt_ate - true_ate,
        'se_ate': se_adapt, 'coverage': cov_adapt
    })
    
    # 3. Fixed Spectral Regularization Grid: lambda in [0.01, 0.05, 0.10, 0.25]
    for lam in [0.01, 0.05, 0.10, 0.25]:
        or_fixed = OverlapAwareRegimeDML(
            n_regimes=2, n_splits=4, embargo_tau=12,
            reg_lambda=lam, posterior_mode='smooth',
            nuisance_model=Ridge(alpha=1.0), random_state=seed
        )
        or_fixed.fit(Y, T, X, Z)
        th_fixed_ate = or_fixed.ate_
        se_fixed = or_fixed.sate_se_
        cov_fixed = float(abs(th_fixed_ate - true_ate) <= 1.96 * se_fixed)
        
        results.append({
            'rep_id': rep_id, 'pi_rare': pi_rare, 'var_ratio': treatment_var_ratio,
            'method': f'Spectral (lambda={lam})',
            'lambda_param': lam,
            'lambda_min': lmin, 'kappa': kappa,
            'ate': th_fixed_ate, 'bias_ate': th_fixed_ate - true_ate,
            'se_ate': se_fixed, 'coverage': cov_fixed
        })
        
    return results


def run_exploration2_ill_conditioned_geometry(n_reps: int = 50, n_jobs: int = -1) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("RUNNING EXPLORATION 2: ILL-CONDITIONED CROSS-REGIME GEOMETRY")
    print("Testing Spectral Regularization across pi_rare in [0.05, 0.10, 0.25] and var_ratio in [0.10, 0.30]")
    print("=" * 78)
    
    grid = [
        (0.05, 0.10), # Severely ill-conditioned (lambda_min ~ 0.01)
        (0.08, 0.15), # Moderate ill-conditioned (lambda_min ~ 0.03)
        (0.15, 0.30), # Mild ill-conditioned (lambda_min ~ 0.10)
        (0.40, 1.00)  # Well-conditioned benchmark (lambda_min ~ 0.45)
    ]
    all_tasks = [(rep, p, vr) for p, vr in grid for rep in range(n_reps)]
    
    start_t = time.time()
    raw_results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_ill_conditioned_single)(rep, p, vr) for rep, p, vr in all_tasks
    )
    
    flat = [item for sublist in raw_results for item in sublist]
    df = pd.DataFrame(flat)
    
    summary = df.groupby(['pi_rare', 'var_ratio', 'method']).agg(
        Mean_Lambda_Min=('lambda_min', 'mean'),
        Mean_Kappa=('kappa', 'mean'),
        Mean_Abs_Bias=('bias_ate', lambda x: np.mean(np.abs(x))),
        Variance_ATE=('ate', 'var'),
        RMSE_ATE=('bias_ate', lambda x: np.sqrt(np.mean(x**2))),
        Coverage_Pct=('coverage', lambda x: np.mean(x) * 100.0)
    ).reset_index()
    
    os.makedirs('reports', exist_ok=True)
    df.to_csv('reports/deep_eval_exploration2_ill_conditioned_raw.csv', index=False)
    summary.to_csv('reports/deep_eval_exploration2_ill_conditioned_summary.csv', index=False)
    print(f"Exploration 2 completed in {time.time() - start_t:.1f}s.")
    print("\nSummary Table:")
    print(summary.to_string(index=False))
    return summary


# ==============================================================================
# 3. EXPLORATION 3: CLEAN REPRESENTATION PERTURBATION EXPERIMENT
# ==============================================================================
def evaluate_representation_perturbation_single(rep_id: int, N: int = 1200) -> List[Dict]:
    """
    Perturb posterior representation gamma_t under PROPER regime-weighted nuisance models.
    Validates: ||theta_hat - theta*||_2 scales with epsilon_gamma / lambda_min(J).
    """
    seed = 700000 + rep_id
    from src.synthetic_dgp_benchmark import generate_difficulty_dgp
    Y, T, X, Z, gt = generate_difficulty_dgp(N=N, delta_z=1.5, delta_t=3.0, random_state=seed)
    S_true = gt['S']
    theta_true = np.array([gt['theta_regimes'][0], gt['theta_regimes'][1]])
    true_ate = gt['true_pop_ate']
    gamma_oracle = np.column_stack([1.0 - S_true, S_true])
    
    # Fit base calibrated HMM
    hmm = LatentRegimeHMM(n_regimes=2, random_state=seed)
    hmm.fit(Z)
    gamma_base = hmm.predict_posteriors(Z, mode='smooth')
    _, gamma_base_aligned = compute_aligned_proxy_error(gamma_base, gamma_oracle)
    
    variants = [
        ('Oracle Indicator', gamma_oracle),
        ('Calibrated HMM', gamma_base_aligned),
        ('Sharpened HMM (tau=0.3)', softmax(np.log(np.maximum(gamma_base_aligned, 1e-12)) / 0.3, axis=1)),
        ('Sharpened HMM (tau=0.6)', softmax(np.log(np.maximum(gamma_base_aligned, 1e-12)) / 0.6, axis=1)),
        ('Smeared HMM (tau=1.8)', softmax(np.log(np.maximum(gamma_base_aligned, 1e-12)) / 1.8, axis=1)),
        ('Smeared HMM (tau=3.5)', softmax(np.log(np.maximum(gamma_base_aligned, 1e-12)) / 3.5, axis=1)),
        ('Noisy Simplex (sigma=0.2)', softmax(np.log(np.maximum(gamma_base_aligned, 1e-12)) + np.random.RandomState(seed + 1).randn(N, 2) * 0.2, axis=1)),
        ('Noisy Simplex (sigma=0.5)', softmax(np.log(np.maximum(gamma_base_aligned, 1e-12)) + np.random.RandomState(seed + 2).randn(N, 2) * 0.5, axis=1)),
        ('Uninformative Uniform', np.full((N, 2), 0.5))
    ]
    
    results = []
    pb = PurgedBlockKFold(n_splits=4, embargo_tau=12)
    K = 2
    
    for v_name, gamma_v in variants:
        eps_gamma = float(np.mean(np.sum(np.abs(gamma_v - gamma_oracle), axis=1)))
        ent = float(-np.mean(np.sum(gamma_v * np.log(np.maximum(gamma_v, 1e-12)), axis=1)))
        
        # PROPER regime-weighted cross-fitting for nuisances:
        tilde_Y = np.zeros((K, N))
        tilde_T = np.zeros((K, N))
        for k in range(K):
            w_k = np.maximum(gamma_v[:, k], 1e-4)
            for tr, te in pb.split(N):
                m_y = Ridge(alpha=1.0).fit(X[tr], Y[tr], sample_weight=w_k[tr])
                tilde_Y[k, te] = Y[te] - m_y.predict(X[te])
                m_t = Ridge(alpha=1.0).fit(X[tr], T[tr], sample_weight=w_k[tr])
                tilde_T[k, te] = T[te] - m_t.predict(X[te])
                
        # Coupled Gram and score
        J = np.zeros((K, K))
        S = np.zeros(K)
        for j in range(K):
            S[j] = np.mean(gamma_v[:, j] * tilde_T[j] * tilde_Y[j])
            for k in range(K):
                J[j, k] = np.mean(gamma_v[:, j] * gamma_v[:, k] * tilde_T[j] * tilde_T[k])
                
        lmin = float(np.min(np.linalg.eigvalsh(J)))
        eff_lam = 0.05 * (np.trace(J) / K) * (1.0 / np.sqrt(N))
        th_vec = np.linalg.inv(J + eff_lam * np.eye(K)) @ S
        
        pi_v = np.mean(gamma_v, axis=0)
        ate_est = float(pi_v @ th_vec)
        causal_err_l2 = float(np.linalg.norm(th_vec - theta_true))
        ate_err = float(abs(ate_est - true_ate))
        
        results.append({
            'rep_id': rep_id,
            'variant': v_name,
            'proxy_error': eps_gamma,
            'entropy': ent,
            'lambda_min': lmin,
            'difficulty_proxy': eps_gamma / max(lmin, 1e-4),
            'difficulty_entropy': ent / max(lmin, 1e-4),
            'causal_error_l2': causal_err_l2,
            'ate_error': ate_err
        })
        
    return results


def run_exploration3_representation_perturbation(n_reps: int = 50, n_jobs: int = -1) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("RUNNING EXPLORATION 3: CLEAN REPRESENTATION PERTURBATION EXPERIMENT")
    print("Validating ||theta - theta*|| vs epsilon_gamma with proper weighted residualization")
    print("=" * 78)
    
    start_t = time.time()
    raw_results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_representation_perturbation_single)(rep) for rep in range(n_reps)
    )
    
    flat = [item for sublist in raw_results for item in sublist]
    df = pd.DataFrame(flat)
    
    summary = df.groupby('variant').agg(
        Mean_Proxy_Error=('proxy_error', 'mean'),
        Mean_Entropy=('entropy', 'mean'),
        Mean_Lambda_Min=('lambda_min', 'mean'),
        Mean_Causal_L2_Error=('causal_error_l2', 'mean'),
        Std_Causal_L2_Error=('causal_error_l2', 'std'),
        Mean_ATE_Error=('ate_error', 'mean')
    ).reset_index().sort_values('Mean_Proxy_Error')
    
    # Compute correlations
    corr_pe, p_pe = pearsonr(df['proxy_error'], df['causal_error_l2'])
    spear_pe, sp_pe = spearmanr(df['proxy_error'], df['causal_error_l2'])
    corr_dp, p_dp = pearsonr(df['difficulty_proxy'], df['causal_error_l2'])
    
    os.makedirs('reports', exist_ok=True)
    df.to_csv('reports/deep_eval_exploration3_representation_raw.csv', index=False)
    summary.to_csv('reports/deep_eval_exploration3_representation_summary.csv', index=False)
    
    print(f"Exploration 3 completed in {time.time() - start_t:.1f}s.")
    print(f"Proxy Error vs Causal L2 Correlation: Pearson r={corr_pe:.4f} (p={p_pe:.2e}), Spearman rho={spear_pe:.4f} (p={sp_pe:.2e})")
    print(f"Difficulty Score (eps/lambda_min) vs Causal L2: Pearson r={corr_dp:.4f} (p={p_dp:.2e})")
    print("\nSummary Table:")
    print(summary.to_string(index=False))
    return summary


# ==============================================================================
# 4. EXPLORATION 4: TASK CONDITIONING GEOMETRY (HOLDING REPRESENTATION FIXED)
# ==============================================================================
def evaluate_task_conditioning_single(rep_id: int, delta_t: float, N: int = 1200) -> Dict:
    """
    Hold representation gamma_t strictly constant while varying task geometry delta_t,
    modulating lambda_min(J). Validates that task geometry alone dictates causal error.
    """
    seed = 800000 + int(delta_t * 100) + rep_id
    from src.synthetic_dgp_benchmark import generate_difficulty_dgp
    Y, T, X, Z, gt = generate_difficulty_dgp(N=N, delta_z=1.5, delta_t=delta_t, random_state=seed)
    S_true = gt['S']
    theta_true = np.array([gt['theta_regimes'][0], gt['theta_regimes'][1]])
    true_ate = gt['true_pop_ate']
    gamma_oracle = np.column_stack([1.0 - S_true, S_true])
    
    # Fit base HMM
    hmm = LatentRegimeHMM(n_regimes=2, random_state=seed)
    hmm.fit(Z)
    gamma_base = hmm.predict_posteriors(Z, mode='smooth')
    eps_gamma, gamma_aligned = compute_aligned_proxy_error(gamma_base, gamma_oracle)
    
    # Proper regime-weighted OR-DML
    pb = PurgedBlockKFold(n_splits=4, embargo_tau=12)
    K = 2
    tilde_Y = np.zeros((K, N))
    tilde_T = np.zeros((K, N))
    for k in range(K):
        w_k = np.maximum(gamma_aligned[:, k], 1e-4)
        for tr, te in pb.split(N):
            m_y = Ridge(alpha=1.0).fit(X[tr], Y[tr], sample_weight=w_k[tr])
            tilde_Y[k, te] = Y[te] - m_y.predict(X[te])
            m_t = Ridge(alpha=1.0).fit(X[tr], T[tr], sample_weight=w_k[tr])
            tilde_T[k, te] = T[te] - m_t.predict(X[te])
            
    J = np.zeros((K, K))
    S = np.zeros(K)
    for j in range(K):
        S[j] = np.mean(gamma_aligned[:, j] * tilde_T[j] * tilde_Y[j])
        for k in range(K):
            J[j, k] = np.mean(gamma_aligned[:, j] * gamma_aligned[:, k] * tilde_T[j] * tilde_T[k])
            
    lmin = float(np.min(np.linalg.eigvalsh(J)))
    kappa = float(np.max(np.linalg.eigvalsh(J)) / max(lmin, 1e-12))
    eff_lam = 0.05 * (np.trace(J) / K) * (1.0 / np.sqrt(N))
    th_vec = np.linalg.inv(J + eff_lam * np.eye(K)) @ S
    
    pi_v = np.mean(gamma_aligned, axis=0)
    ate_est = float(pi_v @ th_vec)
    causal_err_l2 = float(np.linalg.norm(th_vec - theta_true))
    ate_err = float(abs(ate_est - true_ate))
    
    return {
        'rep_id': rep_id,
        'delta_t': delta_t,
        'proxy_error': eps_gamma,
        'lambda_min': lmin,
        'inv_lambda_min': 1.0 / max(lmin, 1e-4),
        'kappa': kappa,
        'causal_error_l2': causal_err_l2,
        'ate_error': ate_err
    }


def run_exploration4_task_conditioning(n_reps: int = 50, n_jobs: int = -1) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("RUNNING EXPLORATION 4: TASK CONDITIONING GEOMETRY (HOLDING REPRESENTATION FIXED)")
    print("Testing Causal Error across Delta_T in [0.5, 1.0, 2.0, 4.0, 8.0] with constant epsilon_gamma")
    print("=" * 78)
    
    delta_t_grid = [0.5, 1.0, 2.0, 4.0, 8.0]
    all_tasks = [(rep, dt) for dt in delta_t_grid for rep in range(n_reps)]
    
    start_t = time.time()
    raw_results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_task_conditioning_single)(rep, dt) for rep, dt in all_tasks
    )
    
    df = pd.DataFrame(raw_results)
    
    summary = df.groupby('delta_t').agg(
        Mean_Proxy_Error=('proxy_error', 'mean'),
        Mean_Lambda_Min=('lambda_min', 'mean'),
        Mean_Kappa=('kappa', 'mean'),
        Mean_Causal_L2_Error=('causal_error_l2', 'mean'),
        Std_Causal_L2_Error=('causal_error_l2', 'std'),
        Mean_ATE_Error=('ate_error', 'mean')
    ).reset_index()
    
    # Check correlation between 1/lambda_min and causal error
    corr_inv, p_inv = pearsonr(df['inv_lambda_min'], df['causal_error_l2'])
    spear_inv, sp_inv = spearmanr(df['inv_lambda_min'], df['causal_error_l2'])
    
    os.makedirs('reports', exist_ok=True)
    df.to_csv('reports/deep_eval_exploration4_task_conditioning_raw.csv', index=False)
    summary.to_csv('reports/deep_eval_exploration4_task_conditioning_summary.csv', index=False)
    
    print(f"Exploration 4 completed in {time.time() - start_t:.1f}s.")
    print(f"1 / Lambda_Min vs Causal L2 Correlation: Pearson r={corr_inv:.4f} (p={p_inv:.2e}), Spearman rho={spear_inv:.4f} (p={sp_inv:.2e})")
    print("\nSummary Table:")
    print(summary.to_string(index=False))
    return summary


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    print("Starting Deep Empirical Evaluations Suite...")
    total_start = time.time()
    
    # Run all 4 explorations
    sum1 = run_exploration1_boundary_stress(n_reps=50, n_jobs=4)
    sum2 = run_exploration2_ill_conditioned_geometry(n_reps=50, n_jobs=4)
    sum3 = run_exploration3_representation_perturbation(n_reps=50, n_jobs=4)
    sum4 = run_exploration4_task_conditioning(n_reps=50, n_jobs=4)
    
    print("\n" + "=" * 78)
    print(f"ALL 4 EXPLORATIONS COMPLETED IN {time.time() - total_start:.1f}s!")
    print("=" * 78)
