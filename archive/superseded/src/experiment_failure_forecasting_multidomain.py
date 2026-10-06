"""
experiment_failure_forecasting_multidomain.py
=============================================
Unified Multi-Domain Downstream Failure Forecasting & Reliability Benchmark:
Evaluates whether task-conditioned difficulty D_t = H(gamma_t) / lambda_min(J_t)
serves as a calibrated, lead-time early warning predictor of future downstream model failure
across THREE diverse, non-stationary sequential domains:
  1. Real Indian Atmospheric Ground Sensor Networks (Delhi, Mumbai, Bengaluru, Kolkata; >14,000 hrs)
  2. Controlled Synthetic Benchmark Streams (LatentRegimeBench with varying Delta_Z)
  3. Real Federal Reserve Macro Market Sequences (FRED Daily S&P 500 & VIX; 2,510 trading days)

Metrics Computed:
  - Spearman rank correlation rho(D_t, L_{t+h}) vs rho(H_t, L_{t+h}) across lead times h
  - Failure ROC-AUC & PR-AUC (predicting top 10% extreme downstream losses)
  - Difficulty Calibration Error (DCE) across risk deciles
  - Three-Way Decision Regret Benchmark (Always Active vs Always Fallback vs Risk-Aware Adaptive)
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.regime_intelligence import GaussianHMMEncoder, RegimeIntelligenceEngine


def compute_dce(difficulty: np.ndarray, losses: np.ndarray, n_bins: int = 10) -> float:
    """
    Difficulty Calibration Error (DCE):
    Measures deviation between decile-partitioned predicted difficulty and empirical downstream loss.
    """
    bins = np.percentile(difficulty, np.linspace(0, 100, n_bins + 1))
    bins[0] -= 1e-6
    bins[-1] += 1e-6
    bin_losses = []
    weights = []
    for i in range(n_bins):
        mask = (difficulty >= bins[i]) & (difficulty < bins[i + 1])
        if np.any(mask):
            bin_losses.append(np.mean(losses[mask]))
            weights.append(np.sum(mask) / len(difficulty))
        else:
            bin_losses.append(0.0)
            weights.append(0.0)
    
    bin_losses = np.array(bin_losses)
    weights = np.array(weights)
    max_loss = np.max(bin_losses) + 1e-8
    norm_losses = bin_losses / max_loss
    ref = np.linspace(0.1, 1.0, n_bins)
    dce = np.sum(weights * np.abs(norm_losses - ref))
    return float(dce)


def run_failure_forecasting_domain(
    name: str,
    Z: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    lead_horizons: list = [1, 3, 6, 12, 24]
):
    print(f"\n--- Evaluating Domain: {name} (N={len(Z)}) ---")
    Z = np.nan_to_num(Z, nan=0.0)
    T = np.nan_to_num(T, nan=0.0)
    Y = np.nan_to_num(Y, nan=0.0)
    N = len(Z)
    engine = RegimeIntelligenceEngine(n_regimes=2, random_state=42)
    engine.fit_latent_regimes(Z)
    gamma = np.nan_to_num(engine.infer_regimes(Z), nan=0.5)

    # Compute rolling Gram matrix and lambda_min
    T_res = np.column_stack([T, T])
    diff_t, lambda_min_t = engine.compute_dynamic_difficulty(gamma, T_res, window=48)
    entropy_t = engine.compute_regime_entropy(gamma)

    # Downstream active prediction model (regime-weighted causal / predictive response)
    from sklearn.linear_model import Ridge
    w0 = np.clip(gamma[:N//2, 0], 1e-4, 1.0)
    w1 = np.clip(gamma[:N//2, 1], 1e-4, 1.0)
    m0 = Ridge().fit(T[:N//2, None], Y[:N//2], sample_weight=w0)
    m1 = Ridge().fit(T[:N//2, None], Y[:N//2], sample_weight=w1)

    y_pred_0 = m0.predict(T[:, None])
    y_pred_1 = m1.predict(T[:, None])
    y_pred_active = gamma[:, 0] * y_pred_0 + gamma[:, 1] * y_pred_1
    base_loss = (Y - y_pred_active) ** 2

    # Robust Fallback (conservative historical mean)
    y_fallback = np.full_like(Y, np.mean(Y[:N//2]))
    fallback_loss = (Y - y_fallback) ** 2

    rows = []
    for h in lead_horizons:
        if h >= N - 1:
            continue
        # Predict failure at time t+h using information at time t
        target_loss = base_loss[h:]
        D_lead = diff_t[:-h]
        H_lead = entropy_t[:-h]

        threshold_top10 = np.percentile(target_loss, 90)
        failure_bin = (target_loss > threshold_top10).astype(int)

        if np.unique(failure_bin).size < 2:
            continue

        rho_D, p_D = spearmanr(D_lead, target_loss)
        rho_H, p_H = spearmanr(H_lead, target_loss)

        auc_D = roc_auc_score(failure_bin, D_lead)
        auc_H = roc_auc_score(failure_bin, H_lead)

        prec_D, rec_D, _ = precision_recall_curve(failure_bin, D_lead)
        prec_H, rec_H, _ = precision_recall_curve(failure_bin, H_lead)
        pr_D = auc(rec_D, prec_D)
        pr_H = auc(rec_H, prec_H)

        dce_D = compute_dce(D_lead, target_loss)

        rows.append({
            "Domain": name,
            "Lead_Horizon_h": h,
            "Spearman_rho_Difficulty_D": round(rho_D, 4),
            "Spearman_rho_Entropy_H": round(rho_H, 4),
            "Failure_ROC_AUC_D": round(auc_D, 4),
            "Failure_ROC_AUC_H": round(auc_H, 4),
            "AUC_Advantage": round(auc_D - auc_H, 4),
            "Failure_PR_AUC_D": round(pr_D, 4),
            "Failure_PR_AUC_H": round(pr_H, 4),
            "DCE_D": round(dce_D, 4)
        })
        print(f"  h={h:2d} | ROC-AUC: D_t={auc_D:.4f} vs H_t={auc_H:.4f} (Diff: {auc_D - auc_H:+.4f}) | PR-AUC: {pr_D:.4f} vs {pr_H:.4f} | Spearman rho: {rho_D:.4f}")

    # Evaluate Decision Regret Benchmark on active vs fallback vs adaptive
    tau_sweep = [1.0, 1.5, 2.0, 3.0, 5.0]
    regret_rows = []
    oracle_min_loss = np.minimum(base_loss, fallback_loss)

    for tau in tau_sweep:
        abstain_mask = diff_t > tau
        y_policy = np.where(~abstain_mask, y_pred_active, y_fallback)
        policy_loss = (Y - y_policy) ** 2

        active_mean = float(np.mean(base_loss))
        fallback_mean = float(np.mean(fallback_loss))
        policy_mean = float(np.mean(policy_loss))

        active_tail = float(np.percentile(base_loss, 95))
        fallback_tail = float(np.percentile(fallback_loss, 95))
        policy_tail = float(np.percentile(policy_loss, 95))

        regret_policy = float(np.mean(policy_loss - oracle_min_loss))
        regret_active = float(np.mean(base_loss - oracle_min_loss))
        regret_fallback = float(np.mean(fallback_loss - oracle_min_loss))

        coverage = float(np.mean(~abstain_mask)) * 100.0

        regret_rows.append({
            "Domain": name,
            "Tau": tau,
            "Coverage_Pct": round(coverage, 2),
            "Active_MSE": round(active_mean, 4),
            "Fallback_MSE": round(fallback_mean, 4),
            "Policy_MSE": round(policy_mean, 4),
            "Active_Tail95": round(active_tail, 4),
            "Fallback_Tail95": round(fallback_tail, 4),
            "Policy_Tail95": round(policy_tail, 4),
            "Policy_Regret": round(regret_policy, 4),
            "Active_Regret": round(regret_active, 4),
            "Fallback_Regret": round(regret_fallback, 4),
            "Regret_Reduction_Pct": round((regret_active - regret_policy) / max(regret_active, 1e-6) * 100.0, 2)
        })

    return rows, regret_rows


def run_all_multidomain_experiments():
    print("=" * 80)
    print("RUNNING MULTI-DOMAIN DOWNSTREAM FAILURE FORECASTING BENCHMARK")
    print("=" * 80)

    all_lead_rows = []
    all_regret_rows = []

    # 1. Atmospheric Ground Sensor Networks from verified clean repository
    clean_path = "data/processed_clean/combined_hourly_clean.csv"
    if os.path.exists(clean_path):
        df_clean = pd.read_csv(clean_path)
        z_cols = ["temperature", "humidity", "wind_speed"]
        for city in ["Delhi", "Mumbai", "Bengaluru", "Kolkata"]:
            sub = df_clean[df_clean["city"] == city].dropna(subset=z_cols + ["no2", "pm25"])
            if len(sub) > 200:
                Z = sub[z_cols].values
                T = sub["no2"].values
                Y = sub["pm25"].values
                leads, regrets = run_failure_forecasting_domain(f"Atmospheric ({city})", Z, T, Y, lead_horizons=[1, 3, 6, 12, 24])
                all_lead_rows.extend(leads)
                all_regret_rows.extend(regrets)

    # 2. Controlled Synthetic Benchmark Streams (LatentRegimeBench)
    from src.train_real_representation_zoo import generate_regime_stream
    for delta in [0.5, 2.0]:
        d_name = f"Synthetic Benchmark (Delta_Z={delta})"
        Z_syn, S_syn, T_syn, Y_syn = generate_regime_stream(1500, Delta_Z=delta, shift_mode="shift_a", seed=77)
        leads, regrets = run_failure_forecasting_domain(d_name, Z_syn, T_syn, Y_syn, lead_horizons=[1, 2, 4, 8])
        all_lead_rows.extend(leads)
        all_regret_rows.extend(regrets)

    # 3. Real Federal Reserve Macro Market Sequence
    fred_path = "data/clean/fred_macro_market.csv"
    if os.path.exists(fred_path):
        df_fred = pd.read_csv(fred_path)
        # Z features: VIX, rolling volatility of returns
        ret = df_fred["Return"].values
        vix = df_fred["VIXCLS"].values
        roll_vol = pd.Series(ret).rolling(20, min_periods=1).std().bfill().fillna(0.01).values
        Z_fred = np.nan_to_num(np.column_stack([vix, roll_vol * 100.0]), nan=0.0)
        # T: VIX level, Y: absolute return (realized volatility target)
        T_fred = np.nan_to_num(vix, nan=15.0)
        Y_fred = np.nan_to_num(np.abs(ret) * 100.0, nan=0.5)
        leads, regrets = run_failure_forecasting_domain("Macro Finance (FRED S&P 500 & VIX)", Z_fred, T_fred, Y_fred, lead_horizons=[1, 3, 5, 10, 20])
        all_lead_rows.extend(leads)
        all_regret_rows.extend(regrets)

    df_leads = pd.DataFrame(all_lead_rows)
    df_regrets = pd.DataFrame(all_regret_rows)

    os.makedirs("reports", exist_ok=True)
    df_leads.to_csv("reports/multidomain_failure_forecasting.csv", index=False)
    df_regrets.to_csv("reports/multidomain_decision_regret.csv", index=False)

    print("\n" + "=" * 80)
    print("SUMMARY: MULTI-DOMAIN FAILURE PREDICTION (Lead h=1 step)")
    print("=" * 80)
    h1_df = df_leads[df_leads["Lead_Horizon_h"] == 1]
    print(h1_df[["Domain", "Failure_ROC_AUC_D", "Failure_ROC_AUC_H", "AUC_Advantage", "Failure_PR_AUC_D", "Failure_PR_AUC_H", "DCE_D"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("SUMMARY: DECISION REGRET BENCHMARK (Tau=1.5)")
    print("=" * 80)
    t15_df = df_regrets[df_regrets["Tau"] == 1.5]
    print(t15_df[["Domain", "Coverage_Pct", "Active_MSE", "Policy_MSE", "Active_Tail95", "Policy_Tail95", "Regret_Reduction_Pct"]].to_string(index=False))

    print("\nSaved artifacts to reports/multidomain_failure_forecasting.csv and reports/multidomain_decision_regret.csv")
    return df_leads, df_regrets


if __name__ == "__main__":
    run_all_multidomain_experiments()
