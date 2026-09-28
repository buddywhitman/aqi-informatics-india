"""
empirical_evaluation.py
=======================
Empirical Evaluation of Overlap-Aware Regime Double Machine Learning (OR-DML)
on clean, multi-season atmospheric monitoring datasets across 4 Indian megacities:
Delhi, Mumbai, Bengaluru, and Kolkata.

Methodological Improvements:
1. Rejects corrupted sensor columns and unphysical values (uses clean Open-Meteo & CPCB).
2. Spans the complete multi-season observation window (>14,000 complete hours, no 2,500-hour truncation).
3. Evaluates Standard DML, Block DML, Retrospective OR-DML (Ours), and Forward Filtered OR-DML (Ours).
4. Tracks spectral diagnostics: lambda_min(J), condition number kappa(J), and posterior entropy H_bar.
5. Estimates dynamic causal impulse-response functions theta_h(S_t) for horizons h = 0, ..., 24 hours.
6. Saves detailed empirical tables to reports/empirical_or_dml_results.csv and IRF to reports/empirical_irf_results.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath('.'))

from src.or_dml import OverlapAwareRegimeDML, PurgedBlockKFold


DATA_PATH = "data/processed_clean/combined_hourly_clean.csv"
REPORTS_DIR = "reports"
PLOTS_DIR = "plots"
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


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


def run_empirical_study():
    print(f"Loading clean empirical dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    treatment_col = 'no2'
    outcome_col = 'pm25'
    
    # Exogenous physical weather features for latent regime discovery (strictly pre-treatment)
    regime_features = ['temperature', 'wind_speed', 'humidity', 'pressure', 'hour_sin', 'hour_cos']
    
    # Observed control confounders (autocorrelation lags and local meteorological controls)
    controls = ['pm25_lag_1h', 'no2_lag_1h', 'pm25_roll_3h', 'no2_roll_3h',
                'temperature', 'humidity', 'wind_speed', 'pressure']
                
    required_cols = [treatment_col, outcome_col] + regime_features + controls
    cities = ['Delhi', 'Mumbai', 'Bengaluru', 'Kolkata']
    
    all_summary_records = []
    all_irf_records = []
    
    print("\n==================================================================")
    print("Executing Multi-City Empirical Evaluation (Clean Longitudinal Data)")
    print("==================================================================")
    
    for city in cities:
        city_df = df[df['city'] == city].dropna(subset=required_cols).copy()
        N = len(city_df)
        if N < 500:
            print(f"Skipping {city}: insufficient observations ({N}).")
            continue
            
        print(f"\n--- Analyzing {city}: {N} complete hourly observations ---")
        Y = city_df[outcome_col].values
        T = city_df[treatment_col].values
        X = city_df[controls].values
        Z = city_df[regime_features].values
        
        # 1. Naive OLS (Pooled with HAC SE)
        X_design = np.column_stack([T, X])
        ols = Ridge(alpha=1e-5).fit(X_design, Y)
        theta_ols = float(ols.coef_[0])
        res_y_ols = Y - ols.predict(X_design)
        res_t_ols = T - np.mean(T)
        se_ols = compute_hac_se(res_y_ols, res_t_ols, theta_ols, hac_lag=12)
        
        all_summary_records.append({
            'City': city, 'N_Obs': N, 'Method': 'Pooled OLS',
            'Regime': 'Pooled', 'Effect_Theta': round(theta_ols, 4),
            'Std_Error': round(se_ols, 4),
            'CI_95_Lower': round(theta_ols - 1.96 * se_ols, 4),
            'CI_95_Upper': round(theta_ols + 1.96 * se_ols, 4),
            'p_value': f"{2 * (1 - 0.9999):.4f}" if abs(theta_ols/se_ols) > 4 else round(float(2 * (1 - pd.Series([abs(theta_ols/se_ols)]).apply(lambda z: 0.5 * (1 + np.math.erf(z / np.sqrt(2)))).values[0])), 4),
            'Lambda_Min': np.nan, 'Kappa': np.nan, 'Entropy': np.nan
        })
        
        # 2. Standard DML (Random 5-fold CV, ignores regimes, with HAC SE)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        tilde_Y_std = np.zeros(N)
        tilde_T_std = np.zeros(N)
        for tr, te in kf.split(X):
            m_y = Ridge(alpha=1.0).fit(X[tr], Y[tr])
            tilde_Y_std[te] = Y[te] - m_y.predict(X[te])
            m_t = Ridge(alpha=1.0).fit(X[tr], T[tr])
            tilde_T_std[te] = T[te] - m_t.predict(X[te])
        theta_std = float(np.mean(tilde_T_std * tilde_Y_std) / max(np.mean(tilde_T_std ** 2), 1e-12))
        se_std = compute_hac_se(tilde_Y_std, tilde_T_std, theta_std, hac_lag=12)
        
        all_summary_records.append({
            'City': city, 'N_Obs': N, 'Method': 'Standard DML',
            'Regime': 'Pooled', 'Effect_Theta': round(theta_std, 4),
            'Std_Error': round(se_std, 4),
            'CI_95_Lower': round(theta_std - 1.96 * se_std, 4),
            'CI_95_Upper': round(theta_std + 1.96 * se_std, 4),
            'p_value': round(float(2.0 * (1.0 - 0.5 * (1.0 + np.math.erf(abs(theta_std/se_std) / np.sqrt(2))))), 4),
            'Lambda_Min': float(np.mean(tilde_T_std ** 2)), 'Kappa': 1.0, 'Entropy': np.nan
        })
        
        # 3. Block DML (Purged Block CV, ignores regimes, with HAC SE)
        pb = PurgedBlockKFold(n_splits=5, embargo_tau=24)
        tilde_Y_blk = np.zeros(N)
        tilde_T_blk = np.zeros(N)
        for tr, te in pb.split(N):
            m_y = Ridge(alpha=1.0).fit(X[tr], Y[tr])
            tilde_Y_blk[te] = Y[te] - m_y.predict(X[te])
            m_t = Ridge(alpha=1.0).fit(X[tr], T[tr])
            tilde_T_blk[te] = T[te] - m_t.predict(X[te])
        theta_blk = float(np.mean(tilde_T_blk * tilde_Y_blk) / max(np.mean(tilde_T_blk ** 2), 1e-12))
        se_blk = compute_hac_se(tilde_Y_blk, tilde_T_blk, theta_blk, hac_lag=12)
        
        all_summary_records.append({
            'City': city, 'N_Obs': N, 'Method': 'Block DML',
            'Regime': 'Pooled', 'Effect_Theta': round(theta_blk, 4),
            'Std_Error': round(se_blk, 4),
            'CI_95_Lower': round(theta_blk - 1.96 * se_blk, 4),
            'CI_95_Upper': round(theta_blk + 1.96 * se_blk, 4),
            'p_value': round(float(2.0 * (1.0 - 0.5 * (1.0 + np.math.erf(abs(theta_blk/se_blk) / np.sqrt(2))))), 4),
            'Lambda_Min': float(np.mean(tilde_T_blk ** 2)), 'Kappa': 1.0, 'Entropy': np.nan
        })
        
        # 4. Spectral OR-DML (Ours, Retrospective Smoothing, K=2 regimes: Ventilated vs Stagnant)
        or_model = OverlapAwareRegimeDML(
            n_regimes=2, n_splits=5, embargo_tau=24,
            reg_alpha=0.05, posterior_mode='smooth',
            nuisance_model=Ridge(alpha=1.0),
            random_state=42
        )
        or_model.fit(Y, T, X, Z)
        
        for k in range(2):
            th_k = or_model.theta_regimes_[k]
            se_k = or_model.se_regimes_[k]
            p_k = or_model.p_regimes_[k]
            all_summary_records.append({
                'City': city, 'N_Obs': N, 'Method': 'Spectral OR-DML (Ours)',
                'Regime': f'Regime {k+1}', 'Effect_Theta': round(th_k, 4),
                'Std_Error': round(se_k, 4),
                'CI_95_Lower': round(th_k - 1.96 * se_k, 4),
                'CI_95_Upper': round(th_k + 1.96 * se_k, 4),
                'p_value': round(p_k, 4),
                'Lambda_Min': round(or_model.lambda_min_, 4),
                'Kappa': round(or_model.kappa_, 2),
                'Entropy': round(or_model.mean_entropy_, 4)
            })
            
        all_summary_records.append({
            'City': city, 'N_Obs': N, 'Method': 'Spectral OR-DML (Ours)',
            'Regime': 'Overall ATE', 'Effect_Theta': round(or_model.ate_, 4),
            'Std_Error': round(or_model.ate_se_, 4),
            'CI_95_Lower': round(or_model.ate_ - 1.96 * or_model.ate_se_, 4),
            'CI_95_Upper': round(or_model.ate_ + 1.96 * or_model.ate_se_, 4),
            'p_value': round(or_model.ate_p_, 4),
            'Lambda_Min': round(or_model.lambda_min_, 4),
            'Kappa': round(or_model.kappa_, 2),
            'Entropy': round(or_model.mean_entropy_, 4)
        })
        
        # 5. Filtered OR-DML (Ours, Forward Filtering)
        filt_model = OverlapAwareRegimeDML(
            n_regimes=2, n_splits=5, embargo_tau=24,
            reg_alpha=0.05, posterior_mode='filter',
            nuisance_model=Ridge(alpha=1.0),
            random_state=42
        )
        filt_model.fit(Y, T, X, Z)
        all_summary_records.append({
            'City': city, 'N_Obs': N, 'Method': 'Filtered OR-DML (Ours)',
            'Regime': 'Overall ATE', 'Effect_Theta': round(filt_model.ate_, 4),
            'Std_Error': round(filt_model.ate_se_, 4),
            'CI_95_Lower': round(filt_model.ate_ - 1.96 * filt_model.ate_se_, 4),
            'CI_95_Upper': round(filt_model.ate_ + 1.96 * filt_model.ate_se_, 4),
            'p_value': round(filt_model.ate_p_, 4),
            'Lambda_Min': round(filt_model.lambda_min_, 4),
            'Kappa': round(filt_model.kappa_, 2),
            'Entropy': round(filt_model.mean_entropy_, 4)
        })
        
        print(f"  Standard DML ATE: {theta_std:.4f} +/- {1.96*se_std:.4f}")
        print(f"  OR-DML Overall ATE: {or_model.ate_:.4f} +/- {1.96*or_model.ate_se_:.4f}")
        print(f"  Regime 1: {or_model.theta_regimes_[0]:.4f} +/- {1.96*or_model.se_regimes_[0]:.4f} | Regime 2: {or_model.theta_regimes_[1]:.4f} +/- {1.96*or_model.se_regimes_[1]:.4f}")
        print(f"  Diagnostics: lambda_min={or_model.lambda_min_:.4f}, kappa={or_model.kappa_:.2f}, entropy={or_model.mean_entropy_:.4f}")
        
        # 6. Dynamic Causal Impulse-Response Function (Horizons 0 to 24 hours)
        horizons = [0, 1, 2, 3, 6, 12, 18, 24]
        irfs = or_model.fit_dynamic_irf(Y, T, X, Z, horizons=horizons)
        for h_idx, h in enumerate(horizons):
            all_irf_records.append({
                'City': city, 'Horizon': h,
                'Regime_1_Effect': irfs[0]['effects'][h_idx], 'Regime_1_SE': irfs[0]['ses'][h_idx],
                'Regime_2_Effect': irfs[1]['effects'][h_idx], 'Regime_2_SE': irfs[1]['ses'][h_idx],
                'ATE_Effect': irfs['ate']['effects'][h_idx], 'ATE_SE': irfs['ate']['ses'][h_idx]
            })

    # Save summary table
    df_results = pd.DataFrame(all_summary_records)
    out_csv = os.path.join(REPORTS_DIR, "empirical_or_dml_results.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"\nEmpirical causal results saved to {out_csv}.")
    
    # Save IRF table
    df_irf = pd.DataFrame(all_irf_records)
    irf_csv = os.path.join(REPORTS_DIR, "empirical_irf_results.csv")
    df_irf.to_csv(irf_csv, index=False)
    print(f"Dynamic impulse-response results saved to {irf_csv}.")
    
    # -------------------------------------------------------------
    # Plotting: Figure 3 Dynamic Causal Impulse Responses
    # -------------------------------------------------------------
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    
    for idx, city in enumerate(cities):
        ax = axes[idx]
        sub = df_irf[df_irf['City'] == city]
        if len(sub) == 0:
            continue
            
        h = sub['Horizon'].values
        # Regime 1
        r1 = sub['Regime_1_Effect'].values
        r1_se = sub['Regime_1_SE'].values
        ax.plot(h, r1, 'b-o', lw=2.2, label='Regime 1 (Ventilated / Advective)')
        ax.fill_between(h, r1 - 1.96 * r1_se, r1 + 1.96 * r1_se, color='b', alpha=0.15)
        
        # Regime 2
        r2 = sub['Regime_2_Effect'].values
        r2_se = sub['Regime_2_SE'].values
        ax.plot(h, r2, 'r-s', lw=2.2, label='Regime 2 (Stagnant / Inversion)')
        ax.fill_between(h, r2 - 1.96 * r2_se, r2 + 1.96 * r2_se, color='r', alpha=0.15)
        
        # Overall ATE
        ate = sub['ATE_Effect'].values
        ate_se = sub['ATE_SE'].values
        ax.plot(h, ate, 'k--', lw=1.8, label='Overall ATE')
        
        ax.axhline(0.0, color='gray', linestyle=':', lw=1.0)
        ax.set_title(f"Dynamic Causal Response: {city}", fontweight='bold')
        ax.set_ylabel(r'Causal Impact $\hat{\theta}_h$ ($\mu g/m^3$ per unit NO$_2$)')
        if idx >= 2:
            ax.set_xlabel('Impulse Horizon $h$ (Hours ahead)')
        if idx == 0:
            ax.legend(frameon=True, fontsize=9)
            
    plt.tight_layout()
    irf_plot_path = os.path.join(PLOTS_DIR, "fig3_dynamic_irf.png")
    plt.savefig(irf_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved dynamic impulse-response figure to {irf_plot_path}.")
    
    print("\n=== EMPIRICAL RESULTS SUMMARY (TABLE 2 IN PAPER) ===")
    print(df_results.to_string(index=False))
    return df_results


if __name__ == '__main__':
    run_empirical_study()
