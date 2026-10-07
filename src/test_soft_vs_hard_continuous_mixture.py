"""
test_soft_vs_hard_continuous_mixture.py
=======================================
Tests whether Soft Probabilistic Weighting (OR-DML) outperforms Hard Assignment
(Regime FE DML) when the latent state is a continuous mixture. (Result: it does not;
soft weighting is worse in every replication. Reported as a negative result.)

Setting: Continuous Latent Mixture / Partial Membership Dynamics
In atmospheric physics and financial regimes, environments rarely jump
discontinuously between pure extremes; they often exhibit continuous transitions
or partial regime membership (e.g., wind ventilation varying continuously from
stagnant to advective, w_t in [0, 1]).

DGP:
- Latent state w_t in [0, 1] is an autoregressive continuous process in Beta/Uniform distribution.
- Treatment: T_t = (b0 * (1 - w_t) + b1 * w_t) + X beta + V_t
- Outcome:   Y_t = (theta_0 * (1 - w_t) + theta_1 * w_t) * T_t + (g0 * (1 - w_t) + g1 * w_t) + X alpha + U_t
- True ATE:  E[theta(w_t)] = theta_0 * E[1 - w_t] + theta_1 * E[w_t]
- Proxies:   Z_t = mu_0 * (1 - w_t) + mu_1 * w_t + eps_Z

Comparison:
1. Soft OR-DML: Uses continuous posterior weights gamma_{t} = [1 - hat{w}_t, hat{w}_t].
2. Hard Regime FE: Discretizes into hat{S}_t = 1(hat{w}_t >= 0.5), forcing observations into 0 or 1.
3. Standard DML: Ignores latent mixture entirely.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from joblib import Parallel, delayed

from src.or_dml import PurgedBlockKFold


def run_continuous_mixture_rep(rep_id: int, N: int = 1500) -> dict:
    rng = np.random.RandomState(900000 + rep_id)
    
    # 1. Continuous autoregressive mixture weight w_t in [0, 1]
    # AR(1) in logit space
    z_lat = np.zeros(N)
    innov = rng.randn(N)
    z_lat[0] = innov[0]
    for t in range(1, N):
        z_lat[t] = 0.85 * z_lat[t-1] + np.sqrt(1 - 0.85**2) * innov[t]
        
    # Map to [0, 1] via sigmoid
    w = 1.0 / (1.0 + np.exp(-z_lat))
    
    # 2. Exogenous Proxy Z_t (informative about w_t)
    # Z_t provides a noisy observation of w_t
    Z = w + rng.normal(0, 0.20, size=N)
    # Estimated soft weight hat{w}_t via clipped linear regression or calibrated proxy
    # In practice, hat{w}_t is the continuous belief
    hat_w = np.clip(Z, 0.01, 0.99)
    gamma_soft = np.column_stack([1.0 - hat_w, hat_w])
    
    # Hard assignment discretizes hat{w} at threshold 0.5
    s_hard = (hat_w >= 0.5).astype(int)
    
    # 3. Confounders X_t
    X = rng.randn(N, 3)
    
    # 4. Continuous Treatment T_t
    b0, b1 = 2.0, 6.0
    V = rng.normal(0, 1.0, size=N)
    T = (1.0 - w) * b0 + w * b1 + 0.8 * X[:, 0] - 0.5 * X[:, 1] + V
    
    # 5. Outcome Y_t
    theta_0 = 0.75
    theta_1 = 2.50
    theta_t = (1.0 - w) * theta_0 + w * theta_1
    true_ate = float(np.mean(theta_t))
    
    g_w = 10.0 * (1.0 - w) + 40.0 * w
    U = rng.normal(0, 1.0, size=N)
    Y = theta_t * T + g_w + 1.2 * X[:, 0] + 0.9 * X[:, 2] + U
    
    pb = PurgedBlockKFold(n_splits=4, embargo_tau=12)
    
    # -------------------------------------------------------------
    # Method 1: Soft OR-DML (Continuous Weighting)
    # -------------------------------------------------------------
    K = 2
    tilde_Y_soft = np.zeros((K, N))
    tilde_T_soft = np.zeros((K, N))
    for k in range(K):
        w_k = np.maximum(gamma_soft[:, k], 1e-4)
        for tr, te in pb.split(N):
            m_y = Ridge(alpha=1.0).fit(X[tr], Y[tr], sample_weight=w_k[tr])
            tilde_Y_soft[k, te] = Y[te] - m_y.predict(X[te])
            m_t = Ridge(alpha=1.0).fit(X[tr], T[tr], sample_weight=w_k[tr])
            tilde_T_soft[k, te] = T[te] - m_t.predict(X[te])
            
    J_soft = np.zeros((K, K))
    S_soft = np.zeros(K)
    for j in range(K):
        S_soft[j] = np.mean(gamma_soft[:, j] * tilde_T_soft[j] * tilde_Y_soft[j])
        for k in range(K):
            J_soft[j, k] = np.mean(gamma_soft[:, j] * gamma_soft[:, k] * tilde_T_soft[j] * tilde_T_soft[k])
            
    eff_lam = 0.05 * (np.trace(J_soft) / K) * (1.0 / np.sqrt(N))
    th_soft = np.linalg.inv(J_soft + eff_lam * np.eye(K)) @ S_soft
    pi_soft = np.mean(gamma_soft, axis=0)
    ate_soft = float(pi_soft @ th_soft)
    
    # -------------------------------------------------------------
    # Method 2: Hard Regime FE (Discretized at 0.5)
    # -------------------------------------------------------------
    th_hard_list = []
    w_hard_list = []
    for k in range(2):
        mask_k = (s_hard == k)
        w_hard_list.append(np.mean(mask_k))
        if np.sum(mask_k) > 30:
            X_k = X[mask_k]
            Y_k = Y[mask_k]
            T_k = T[mask_k]
            m_y_k = Ridge(alpha=1.0).fit(X_k, Y_k)
            res_y = Y_k - m_y_k.predict(X_k)
            m_t_k = Ridge(alpha=1.0).fit(X_k, T_k)
            res_t = T_k - m_t_k.predict(X_k)
            th_k = float(np.mean(res_t * res_y) / max(np.mean(res_t**2), 1e-12))
        else:
            th_k = 0.0
        th_hard_list.append(th_k)
    ate_hard = float(w_hard_list[0] * th_hard_list[0] + w_hard_list[1] * th_hard_list[1])
    
    # -------------------------------------------------------------
    # Method 3: Standard DML (Pooled)
    # -------------------------------------------------------------
    tilde_Y_std = np.zeros(N)
    tilde_T_std = np.zeros(N)
    for tr, te in pb.split(N):
        m_y = Ridge(alpha=1.0).fit(X[tr], Y[tr])
        tilde_Y_std[te] = Y[te] - m_y.predict(X[te])
        m_t = Ridge(alpha=1.0).fit(X[tr], T[tr])
        tilde_T_std[te] = T[te] - m_t.predict(X[te])
    ate_std = float(np.mean(tilde_T_std * tilde_Y_std) / max(np.mean(tilde_T_std ** 2), 1e-12))
    
    # -------------------------------------------------------------
    # Method 4: OR-DML with posterior-adjusted nuisances (paper's estimator)
    # -------------------------------------------------------------
    from src.synthesis.estimators import coupled_posterior
    folds = [(tr, te) for tr, te in pb.split(N)]
    th_pa, sh_pa, _, _ = coupled_posterior(gamma_soft, X, T, Y, folds, lam_scale=0.05, L=12)
    ate_pa = float(sh_pa @ th_pa)

    return {
        'rep_id': rep_id,
        'ate_pa': ate_pa,
        'bias_pa': ate_pa - true_ate,
        'true_ate': true_ate,
        'ate_soft': ate_soft,
        'ate_hard': ate_hard,
        'ate_std': ate_std,
        'bias_soft': ate_soft - true_ate,
        'bias_hard': ate_hard - true_ate,
        'bias_std': ate_std - true_ate
    }


if __name__ == '__main__':
    print("Testing Soft OR-DML vs Hard Regime FE under Continuous Latent Mixture...")
    results = Parallel(n_jobs=4)(delayed(run_continuous_mixture_rep)(i) for i in range(100))
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("RESULTS ACROSS 100 MONTE CARLO REPLICATIONS:")
    print("=" * 60)
    print(f"Standard DML Mean Abs Bias:  {np.mean(np.abs(df['bias_std'])):.4f}  (RMSE: {np.sqrt(np.mean(df['bias_std']**2)):.4f})")
    print(f"Hard Regime FE Mean Abs Bias: {np.mean(np.abs(df['bias_hard'])):.4f}  (RMSE: {np.sqrt(np.mean(df['bias_hard']**2)):.4f})")
    print(f"Soft OR-DML Mean Abs Bias:    {np.mean(np.abs(df['bias_soft'])):.4f}  (RMSE: {np.sqrt(np.mean(df['bias_soft']**2)):.4f})")
    
    print(f"Posterior-adjusted OR-DML Mean Abs Bias: {np.mean(np.abs(df['bias_pa'])):.4f}")
    win_soft = np.mean(np.abs(df['bias_soft']) < np.abs(df['bias_hard']))
    print(f"\nSoft OR-DML achieves lower error in {win_soft * 100:.1f}% of replications.")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reports', 'synthesis')
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'soft_vs_hard_continuous_mixture_raw.csv'), index=False)
    pd.DataFrame([dict(mean_abs_bias_std=np.mean(np.abs(df['bias_std'])),
                       mean_abs_bias_hard=np.mean(np.abs(df['bias_hard'])),
                       mean_abs_bias_pa=np.mean(np.abs(df['bias_pa'])),
                       frac_pa_better_than_hard=np.mean(np.abs(df['bias_pa']) < np.abs(df['bias_hard'])),
                       mean_abs_bias_soft=np.mean(np.abs(df['bias_soft'])),
                       frac_soft_better=win_soft, n_reps=len(df))]).to_csv(
        os.path.join(out_dir, 'soft_vs_hard_continuous_mixture_summary.csv'), index=False)
