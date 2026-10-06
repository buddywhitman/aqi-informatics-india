"""
test_multiregime_scalability.py
===============================
Empirical verification of Theorem 5 and Appendix D.2:
State Space Scalability and Matrix Concentration for K > 2.

Evaluates OverlapAwareRegimeDML across state dimensions K in {2, 3, 4, 5}:
  - Generates dependent Markov switching processes with K states
  - Evaluates estimation bias, standard errors, condition number kappa(J),
    and empirical runtime scaling.
Saves results to reports/multiregime_scalability_summary.csv.
"""

import time
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pandas as pd
from src.or_dml import OverlapAwareRegimeDML

def run_multiregime_scalability_evaluation():
    print("=" * 75)
    print("EMPIRICAL VERIFICATION OF MULTI-REGIME SCALABILITY (K in {2, 3, 4, 5})")
    print("=" * 75)
    
    np.random.seed(42)
    N = 1200
    K_list = [2, 3, 4, 5]
    results = []

    for K in K_list:
        t0 = time.time()
        # True regime-specific causal effects
        theta_true = np.linspace(0.5, 2.0, K)
        
        # 1. Generate Markov chain with K states
        A = np.full((K, K), 0.1 / (K - 1))
        np.fill_diagonal(A, 0.90)
        
        states = np.zeros(N, dtype=int)
        for t in range(1, N):
            states[t] = np.random.choice(K, p=A[states[t-1]])
            
        # 2. Exogenous state proxy Z (K well-separated modes)
        Z = np.random.randn(N, 2)
        for k in range(K):
            mask = (states == k)
            Z[mask, 0] += k * 2.5
            
        # 3. Covariates X and Treatment T
        X = np.random.randn(N, 3)
        g_baseline = states * 2.0
        T = np.random.randn(N) + 0.5 * Z[:, 0] + 0.3 * X[:, 0]
        
        # 4. Outcome Y with heterogeneous treatment effects
        Y = g_baseline + theta_true[states] * T + np.sum(X, axis=1) + np.random.randn(N) * 0.5
        
        # 5. Fit OverlapAwareRegimeDML
        model = OverlapAwareRegimeDML(
            n_regimes=K,
            n_splits=2,
            embargo_tau=12,
            reg_alpha=0.1,
            random_state=42
        )
        model.fit(Y, T, X, Z)
        elapsed = time.time() - t0
        
        # Calculate regime-specific bias
        theta_est = np.array([model.theta_regimes_[k] for k in range(K)])
        se_est = np.array([model.se_regimes_[k] for k in range(K)])
        mean_abs_bias = float(np.mean(np.abs(theta_est - theta_true)))
        l2_error = float(np.linalg.norm(theta_est - theta_true))
        
        # Overall ATE
        weights_true = np.array([np.mean(states == k) for k in range(K)])
        ate_true = float(np.sum(weights_true * theta_true))
        ate_bias = float(abs(model.ate_ - ate_true))
        
        row = {
            "K": K,
            "N": N,
            "Mean_Abs_Bias": round(mean_abs_bias, 4),
            "L2_Error": round(l2_error, 4),
            "ATE_True": round(ate_true, 4),
            "ATE_Est": round(model.ate_, 4),
            "ATE_Bias": round(ate_bias, 4),
            "ATE_SE": round(model.ate_se_, 4),
            "Lambda_Min": round(model.lambda_min_, 4),
            "Lambda_Max": round(model.lambda_max_, 4),
            "Kappa_J": round(model.kappa_, 2),
            "Runtime_Sec": round(elapsed, 3)
        }
        results.append(row)
        print(f"[K={K}] Bias={mean_abs_bias:.4f}, L2={l2_error:.4f}, ATE_Bias={ate_bias:.4f}, "
              f"lambda_min={model.lambda_min_:.4f}, kappa={model.kappa_:.1f}, time={elapsed:.2f}s")

    df = pd.DataFrame(results)
    output_path = "reports/multiregime_scalability_summary.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved multi-regime scalability results to {output_path}")
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    run_multiregime_scalability_evaluation()
