"""
advanced_methodological_frontiers.py
====================================
Rigorous investigations into three decisive frontiers:

1. Frontier A: The Selective Estimation Frontier (Risk-Coverage Trade-Off)
   - Evaluates abstention policies based on Coupled Difficulty D_t = H(gamma_t) / lambda_min(J_t)
     versus Raw Entropy H(gamma_t) versus Raw Gram Conditioning lambda_min(J_t).
   - Demonstrates why coupling state uncertainty with task geometry is necessary for selective causal inference.

2. Frontier B: Type I Error & Size Distortion (The Cost of Post-Clustering Selection)
   - Tests H_0: theta_1^* = 0 (no causal effect in Regime 1) under Markov confounding.
   - Evaluates whether Hard Regime FE suffers from severe Type I error inflation (false positives)
     due to post-clustering selection bias and broken temporal dependence,
     while OR-DML preserves nominal 5% size via its valid HAC sandwich covariance.

3. Frontier C: Low-Persistence / High-Switching Temporal Fragmentation
   - Sweeps Markov persistence rho in [0.30, 0.50, 0.70, 0.88].
   - Tests whether hard assignment shatters temporal contiguous blocks into micro-fragments,
     degrading nuisance fitting and HAC estimation under fast transitions.
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
from scipy.stats import norm
from sklearn.linear_model import Ridge
from joblib import Parallel, delayed
from typing import Dict, List, Tuple

from src.or_dml import OverlapAwareRegimeDML, PurgedBlockKFold, LatentRegimeHMM
from src.synthetic_dgp_benchmark import compute_aligned_proxy_error, compute_hac_se, generate_difficulty_dgp


# ==============================================================================
# FRONTIER A: SELECTIVE ESTIMATION FRONTIER (RISK-COVERAGE TRADE-OFF)
# ==============================================================================
def run_selective_estimation_single(rep_id: int, N: int = 1500) -> List[Dict]:
    """
    Test Risk-Coverage curve: evaluate causal estimation error as a function
    of coverage c in [0.50, 0.60, 0.70, 0.80, 0.90, 1.00] when abstaining on:
    1. Coupled Difficulty: D_t = H(gamma_t) / lambda_min(J_t)
    2. Raw Entropy: H_t = H(gamma_t)
    3. Inverse Conditioning: 1 / lambda_min(J_t)
    4. Random Abstention
    """
    seed = 110000 + rep_id
    # Use realistic mixed observability where difficult periods naturally arise
    Y, T, X, Z, gt = generate_difficulty_dgp(N=N, delta_z=1.0, delta_t=3.0, random_state=seed)
    true_ate = gt['true_pop_ate']
    S_true = gt['S']
    
    # Fit HMM
    hmm = LatentRegimeHMM(n_regimes=2, random_state=seed)
    hmm.fit(Z)
    gamma = hmm.predict_posteriors(Z, mode='smooth')
    
    # Nuisance residualization
    pb = PurgedBlockKFold(n_splits=4, embargo_tau=12)
    K = 2
    tilde_Y = np.zeros((K, N))
    tilde_T = np.zeros((K, N))
    for k in range(K):
        w_k = np.maximum(gamma[:, k], 1e-4)
        for tr, te in pb.split(N):
            m_y = Ridge(alpha=1.0).fit(X[tr], Y[tr], sample_weight=w_k[tr])
            tilde_Y[k, te] = Y[te] - m_y.predict(X[te])
            m_t = Ridge(alpha=1.0).fit(X[tr], T[tr], sample_weight=w_k[tr])
            tilde_T[k, te] = T[te] - m_t.predict(X[te])
            
    # Local rolling difficulty diagnostics (window = 48)
    window = 48
    H_t = -np.sum(gamma * np.log(np.maximum(gamma, 1e-12)), axis=1)
    
    lmin_t = np.zeros(N)
    for t in range(N):
        start = max(0, t - window // 2)
        end = min(N, t + window // 2 + 1)
        # Local Gram matrix
        J_loc = np.zeros((K, K))
        for j in range(K):
            for k in range(K):
                J_loc[j, k] = np.mean(gamma[start:end, j] * gamma[start:end, k] * tilde_T[j, start:end] * tilde_T[k, start:end])
        lmin_t[t] = max(float(np.min(np.linalg.eigvalsh(J_loc))), 1e-4)
        
    D_t = H_t / lmin_t
    inv_lmin_t = 1.0 / lmin_t
    rng = np.random.RandomState(seed + 5)
    rand_scores = rng.rand(N)
    
    coverage_levels = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    results = []
    
    rules = [
        ('Coupled Difficulty D_t', D_t),
        ('Raw Entropy H_t', H_t),
        ('Conditioning 1/lambda_min', inv_lmin_t),
        ('Random Baseline', rand_scores)
    ]
    
    for rule_name, score_arr in rules:
        for cov in coverage_levels:
            if cov < 1.00:
                cutoff = np.percentile(score_arr, cov * 100.0)
                retain_mask = (score_arr <= cutoff)
            else:
                retain_mask = np.ones(N, dtype=bool)
                
            n_retain = np.sum(retain_mask)
            if n_retain < 100:
                continue
                
            # Compute OR-DML on retained observations
            gamma_ret = gamma[retain_mask]
            tilde_Y_ret = tilde_Y[:, retain_mask]
            tilde_T_ret = tilde_T[:, retain_mask]
            
            J_ret = np.zeros((K, K))
            S_ret = np.zeros(K)
            for j in range(K):
                S_ret[j] = np.mean(gamma_ret[:, j] * tilde_T_ret[j] * tilde_Y_ret[j])
                for k in range(K):
                    J_ret[j, k] = np.mean(gamma_ret[:, j] * gamma_ret[:, k] * tilde_T_ret[j] * tilde_T_ret[k])
                    
            eff_lam = 0.05 * (np.trace(J_ret) / K) * (1.0 / np.sqrt(n_retain))
            th_ret = np.linalg.inv(J_ret + eff_lam * np.eye(K)) @ S_ret
            pi_ret = np.mean(gamma_ret, axis=0)
            ate_ret = float(pi_ret @ th_ret)
            
            # Ground truth ATE on the retained sample
            # Local sample ATE
            S_ret_true = S_true[retain_mask]
            sample_ate_ret = float(np.mean(gt['theta_regimes'][0] * (1 - S_ret_true) + gt['theta_regimes'][1] * S_ret_true))
            
            bias_ate = abs(ate_ret - sample_ate_ret)
            
            results.append({
                'rep_id': rep_id,
                'rule': rule_name,
                'coverage': cov,
                'n_retained': n_retain,
                'ate_est': ate_ret,
                'true_sample_ate': sample_ate_ret,
                'bias_ate': bias_ate,
                'sq_error': bias_ate ** 2
            })
            
    return results


def run_frontier_a_selective_estimation(n_reps: int = 40, n_jobs: int = 4) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("FRONTIER A: SELECTIVE ESTIMATION FRONTIER (RISK-COVERAGE TRADE-OFF)")
    print("Testing whether D_t = H / lambda_min produces a superior Risk-Coverage frontier")
    print("=" * 78)
    
    start_t = time.time()
    raw = Parallel(n_jobs=n_jobs)(delayed(run_selective_estimation_single)(rep) for rep in range(n_reps))
    flat = [item for sublist in raw for item in sublist]
    df = pd.DataFrame(flat)
    
    summary = df.groupby(['rule', 'coverage']).agg(
        Mean_Abs_Bias=('bias_ate', 'mean'),
        RMSE_ATE=('sq_error', lambda x: np.sqrt(np.mean(x))),
        Std_Bias=('bias_ate', 'std')
    ).reset_index()
    
    os.makedirs('reports', exist_ok=True)
    df.to_csv('reports/frontier_a_selective_estimation_raw.csv', index=False)
    summary.to_csv('reports/frontier_a_selective_estimation_summary.csv', index=False)
    print(f"Frontier A completed in {time.time() - start_t:.1f}s.")
    print("\nSummary Risk-Coverage Table:")
    print(summary.to_string(index=False))
    return summary


# ==============================================================================
# FRONTIER B: TYPE I ERROR & SIZE DISTORTION (POST-CLUSTERING SELECTION BIAS)
# ==============================================================================
def run_size_distortion_single(rep_id: int, N: int = 1200) -> Dict:
    """
    Test H_0: theta_1^* = 0.0 (True causal effect in Regime 1 is exactly ZERO).
    Evaluate empirical rejection rate (nominal alpha = 0.05).
    Compare:
    1. Spectral OR-DML (Joint HAC Sandwich)
    2. Hard Regime FE DML (Naive OLS/HAC within argmax cluster)
    3. Unregularized OR-DML (lambda = 0)
    """
    seed = 220000 + rep_id
    rng = np.random.RandomState(seed)
    
    # 2-state Markov chain
    persistence = 0.88
    p01 = 1.0 - persistence
    p10 = 0.15
    P_trans = np.array([[persistence, p01], [p10, 1.0 - p10]])
    stat_pi = np.array([p10 / (p01 + p10), p01 / (p01 + p10)])
    
    S = np.zeros(N, dtype=int)
    S[0] = 0 if rng.rand() < stat_pi[0] else 1
    for t in range(1, N):
        p_next = P_trans[S[t-1]]
        S[t] = 0 if rng.rand() < p_next[0] else 1
        
    # Exogenous proxy Z
    Z = np.zeros((N, 2))
    Z[S == 0] = rng.multivariate_normal([1.5, 2.0], [[1.0, 0.2], [0.2, 1.0]], size=np.sum(S == 0))
    Z[S == 1] = rng.multivariate_normal([-1.5, -2.0], [[1.0, 0.2], [0.2, 1.0]], size=np.sum(S == 1))
    
    # Confounders X
    X = rng.randn(N, 3)
    
    # Treatment T (confounded by S)
    V = rng.normal(0, 1.0, size=N)
    T = 2.0 * (1 - S) + 5.0 * S + 0.8 * X[:, 0] + V
    
    # Outcome Y:
    # Under H_0: theta_0 = 1.50, but theta_1 = 0.00!
    theta_0 = 1.50
    theta_1 = 0.00  # NULL HYPOTHESIS IS TRUE!
    
    g_S = 10.0 * (1 - S) + 30.0 * S
    U = rng.normal(0, 1.0, size=N)
    Y = (theta_0 * (1 - S) + theta_1 * S) * T + g_S + 1.2 * X[:, 0] + U
    
    # Fit models
    # 1. OR-DML
    or_dml = OverlapAwareRegimeDML(
        n_regimes=2, n_splits=4, embargo_tau=12,
        reg_alpha=0.05, posterior_mode='smooth',
        nuisance_model=Ridge(alpha=1.0), random_state=seed
    )
    or_dml.fit(Y, T, X, Z)
    
    # Match Regime corresponding to S=1 (emission mean is negative, -1.5)
    # or_dml.hmm_.means is sorted ascending by Z[:, 0], so index 0 has the negative mean
    target_k = 0 if or_dml.hmm_.means[0, 0] < or_dml.hmm_.means[1, 0] else 1
    
    th1_or = or_dml.theta_regimes_[target_k]
    se1_or = or_dml.se_regimes_[target_k]
    z_stat_or = abs(th1_or) / max(se1_or, 1e-12)
    reject_or = float(z_stat_or > 1.96)
    
    # 2. Hard Regime FE DML
    gamma = or_dml.gamma_
    s_hard = np.argmax(gamma, axis=1) # target_k corresponds to S=1
    
    mask_1 = (s_hard == target_k)
    if np.sum(mask_1) > 30:
        X_1 = X[mask_1]
        Y_1 = Y[mask_1]
        T_1 = T[mask_1]
        m_y_1 = Ridge(alpha=1.0).fit(X_1, Y_1)
        res_y_1 = Y_1 - m_y_1.predict(X_1)
        m_t_1 = Ridge(alpha=1.0).fit(X_1, T_1)
        res_t_1 = T_1 - m_t_1.predict(X_1)
        th1_hard = float(np.mean(res_t_1 * res_y_1) / max(np.mean(res_t_1**2), 1e-12))
        se1_hard = compute_hac_se(res_y_1, res_t_1, th1_hard, hac_lag=12)
    else:
        th1_hard = 0.0
        se1_hard = 1.0
        
    z_stat_hard = abs(th1_hard) / max(se1_hard, 1e-12)
    reject_hard = float(z_stat_hard > 1.96)
    
    return {
        'rep_id': rep_id,
        'th1_or': th1_or,
        'se1_or': se1_or,
        'reject_or': reject_or,
        'th1_hard': th1_hard,
        'se1_hard': se1_hard,
        'reject_hard': reject_hard
    }


def run_frontier_b_size_distortion(n_reps: int = 100, n_jobs: int = 4) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("FRONTIER B: TYPE I ERROR & SIZE DISTORTION (H_0: theta_1 = 0)")
    print("Testing whether Hard FE suffers from False Positive inflation vs OR-DML")
    print("=" * 78)
    
    start_t = time.time()
    raw = Parallel(n_jobs=n_jobs)(delayed(run_size_distortion_single)(rep) for rep in range(n_reps))
    df = pd.DataFrame(raw)
    
    rate_or = float(np.mean(df['reject_or'])) * 100.0
    rate_hard = float(np.mean(df['reject_hard'])) * 100.0
    
    bias_or = float(np.mean(np.abs(df['th1_or'])))
    bias_hard = float(np.mean(np.abs(df['th1_hard'])))
    
    summary = pd.DataFrame([
        {'Method': 'Spectral OR-DML (Sandwich)', 'Nominal_Size': '5.0%', 'Empirical_Type_I_Error': f'{rate_or:.1f}%', 'Mean_Abs_Bias': round(bias_or, 4)},
        {'Method': 'Hard Regime FE (Naive Post-Cluster)', 'Nominal_Size': '5.0%', 'Empirical_Type_I_Error': f'{rate_hard:.1f}%', 'Mean_Abs_Bias': round(bias_hard, 4)}
    ])
    
    os.makedirs('reports', exist_ok=True)
    df.to_csv('reports/frontier_b_size_distortion_raw.csv', index=False)
    summary.to_csv('reports/frontier_b_size_distortion_summary.csv', index=False)
    print(f"Frontier B completed in {time.time() - start_t:.1f}s.")
    print("\nSummary Type I Error Table:")
    print(summary.to_string(index=False))
    return summary


# ==============================================================================
# FRONTIER C: LOW-PERSISTENCE & TEMPORAL SHATTERING
# ==============================================================================
def run_persistence_stress_single(rep_id: int, persistence: float, N: int = 1200) -> List[Dict]:
    """
    Test how Markov persistence rho in [0.30, 0.50, 0.70, 0.88] affects estimators.
    When persistence is low, transitions occur frequently.
    """
    seed = 330000 + int(persistence * 1000) + rep_id
    rng = np.random.RandomState(seed)
    
    p01 = 1.0 - persistence
    p10 = 0.50 * (1.0 - persistence)
    P_trans = np.array([[persistence, p01], [p10, 1.0 - p10]])
    stat_pi = np.array([p10 / (p01 + p10), p01 / (p01 + p10)])
    
    S = np.zeros(N, dtype=int)
    S[0] = 0 if rng.rand() < stat_pi[0] else 1
    for t in range(1, N):
        p_next = P_trans[S[t-1]]
        S[t] = 0 if rng.rand() < p_next[0] else 1
        
    n_switches = int(np.sum(np.diff(S) != 0))
    
    # Exogenous proxy Z
    Z = np.zeros((N, 2))
    Z[S == 0] = rng.multivariate_normal([1.5, 2.0], [[1.0, 0.2], [0.2, 1.0]], size=np.sum(S == 0))
    Z[S == 1] = rng.multivariate_normal([-1.5, -2.0], [[1.0, 0.2], [0.2, 1.0]], size=np.sum(S == 1))
    
    X = rng.randn(N, 3)
    V = rng.normal(0, 1.0, size=N)
    T = 2.0 * (1 - S) + 5.0 * S + 0.8 * X[:, 0] + V
    
    theta_0 = 0.75
    theta_1 = 2.50
    true_ate = float(stat_pi[0] * theta_0 + stat_pi[1] * theta_1)
    
    g_S = 10.0 * (1 - S) + 40.0 * S
    U = rng.normal(0, 1.0, size=N)
    Y = (theta_0 * (1 - S) + theta_1 * S) * T + g_S + 1.2 * X[:, 0] + U
    
    # 1. OR-DML
    or_dml = OverlapAwareRegimeDML(
        n_regimes=2, n_splits=4, embargo_tau=12,
        reg_alpha=0.05, posterior_mode='smooth',
        nuisance_model=Ridge(alpha=1.0), random_state=seed
    )
    or_dml.fit(Y, T, X, Z)
    ate_or = or_dml.ate_
    cov_or = float(abs(ate_or - true_ate) <= 1.96 * or_dml.sate_se_)
    
    # 2. Hard Regime FE
    gamma = or_dml.gamma_
    s_hard = np.argmax(gamma, axis=1)
    th_k = []
    w_k = []
    for k in range(2):
        mask_k = (s_hard == k)
        w_k.append(np.mean(mask_k))
        if np.sum(mask_k) > 20:
            X_k = X[mask_k]
            Y_k = Y[mask_k]
            T_k = T[mask_k]
            m_y_k = Ridge(alpha=1.0).fit(X_k, Y_k)
            res_y = Y_k - m_y_k.predict(X_k)
            m_t_k = Ridge(alpha=1.0).fit(X_k, T_k)
            res_t = T_k - m_t_k.predict(X_k)
            th_k.append(float(np.mean(res_t * res_y) / max(np.mean(res_t**2), 1e-12)))
        else:
            th_k.append(0.0)
    ate_hard = float(w_k[0] * th_k[0] + w_k[1] * th_k[1])
    
    return [
        {
            'rep_id': rep_id, 'persistence': persistence, 'n_switches': n_switches,
            'method': 'Spectral OR-DML', 'bias_ate': ate_or - true_ate, 'cov': cov_or
        },
        {
            'rep_id': rep_id, 'persistence': persistence, 'n_switches': n_switches,
            'method': 'Hard Regime FE', 'bias_ate': ate_hard - true_ate, 'cov': np.nan
        }
    ]


def run_frontier_c_persistence_stress(n_reps: int = 40, n_jobs: int = 4) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("FRONTIER C: LOW-PERSISTENCE & TEMPORAL SHATTERING")
    print("Testing estimators across persistence rho in [0.30, 0.50, 0.70, 0.88]")
    print("=" * 78)
    
    grid = [0.30, 0.50, 0.70, 0.88]
    all_tasks = [(rep, p) for p in grid for rep in range(n_reps)]
    
    start_t = time.time()
    raw = Parallel(n_jobs=n_jobs)(delayed(run_persistence_stress_single)(rep, p) for rep, p in all_tasks)
    flat = [item for sublist in raw for item in sublist]
    df = pd.DataFrame(flat)
    
    summary = df.groupby(['persistence', 'method']).agg(
        Mean_Switches=('n_switches', 'mean'),
        Mean_Abs_Bias=('bias_ate', lambda x: np.mean(np.abs(x))),
        RMSE=('bias_ate', lambda x: np.sqrt(np.mean(x**2))),
        Coverage_Pct=('cov', lambda x: np.nanmean(x) * 100.0)
    ).reset_index()
    
    os.makedirs('reports', exist_ok=True)
    df.to_csv('reports/frontier_c_persistence_stress_raw.csv', index=False)
    summary.to_csv('reports/frontier_c_persistence_stress_summary.csv', index=False)
    print(f"Frontier C completed in {time.time() - start_t:.1f}s.")
    print("\nSummary Persistence Table:")
    print(summary.to_string(index=False))
    return summary


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == '__main__':
    t_start = time.time()
    sum_a = run_frontier_a_selective_estimation(n_reps=40, n_jobs=4)
    sum_b = run_frontier_b_size_distortion(n_reps=100, n_jobs=4)
    sum_c = run_frontier_c_persistence_stress(n_reps=40, n_jobs=4)
    print(f"\nALL ADVANCED FRONTIERS COMPLETED IN {time.time() - t_start:.1f}s!")
