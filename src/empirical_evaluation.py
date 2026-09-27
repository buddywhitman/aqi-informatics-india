"""
Empirical Evaluation of RC-DML on Real-World Atmospheric Sensor Networks
========================================================================
Applies Regime-Conditional Double Machine Learning (RC-DML) to real
hourly observations from Delhi, Mumbai, and Bengaluru.

Compares:
1. Naive DML (Standard Cross-Sectional DML, unmodeled regimes).
2. RC-DML (Ours: Regime-Conditional DML with Purged Block Cross-Fitting).

Saves results to reports/empirical_rc_dml_results.csv.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from src.rc_dml import RegimeConditionalDML, PurgedBlockKFold


DATA_PATH = "data/processed_hourly/combined_hourly_with_regimes.csv"
OUTPUT_PATH = "reports/empirical_rc_dml_results.csv"


def run_empirical_evaluation():
    print(f"Loading empirical dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    controls = ['temperature_x', 'humidity', 'pressure', 'pm25_lag_1h', 'no2_lag_1h']
    regime_features = ['wind_speed_y', 'temperature_y', 'relativehumidity']
    treatment_col = 'no2'
    outcome_col = 'pm25'
    
    all_results = []
    
    cities = ['Delhi', 'Mumbai', 'Bengaluru']
    
    for city in cities:
        print(f"\n--- Running Empirical Causal Analysis for {city} ---")
        city_df = df[df['city'] == city].dropna(subset=controls + regime_features + [treatment_col, outcome_col]).copy()
        
        # Subsample to 2000 contiguous hours for clean demonstration
        if len(city_df) > 2500:
            city_df = city_df.iloc[:2500]
            
        N = len(city_df)
        print(f"Sample size: {N} hourly observations")
        
        Y = city_df[outcome_col].values
        T = city_df[treatment_col].values
        X = city_df[controls].values
        Z = city_df[regime_features].values
        
        # 1. Naive OLS (Pooled OLS regression Y ~ T + X)
        from sklearn.linear_model import LinearRegression
        ols = LinearRegression()
        ols.fit(np.hstack([T.reshape(-1, 1), X]), Y)
        theta_ols = ols.coef_[0]
        res_ols = Y - ols.predict(np.hstack([T.reshape(-1, 1), X]))
        se_ols = np.sqrt(np.mean(res_ols**2) / np.sum((T - np.mean(T))**2))
        
        # 2. Random Forest Plug-in
        from sklearn.ensemble import RandomForestRegressor
        rf_y = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        rf_y.fit(np.hstack([T.reshape(-1, 1), X]), Y)
        # Numerical partial derivative dT
        eps = 1e-4
        pred_plus = rf_y.predict(np.hstack([(T + eps).reshape(-1, 1), X]))
        pred_minus = rf_y.predict(np.hstack([(T - eps).reshape(-1, 1), X]))
        theta_rf = np.mean((pred_plus - pred_minus) / (2 * eps))
        se_rf = np.std((pred_plus - pred_minus) / (2 * eps)) / np.sqrt(N)
        
        # 3. Naive Standard DML (Random 5-fold CV, ignoring regimes)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        tilde_Y_naive = np.zeros(N)
        tilde_T_naive = np.zeros(N)
        for train_idx, test_idx in kf.split(X):
            m_y = HistGradientBoostingRegressor(max_iter=100, min_samples_leaf=20, random_state=42)
            m_y.fit(X[train_idx], Y[train_idx])
            tilde_Y_naive[test_idx] = Y[test_idx] - m_y.predict(X[test_idx])
            
            m_t = HistGradientBoostingRegressor(max_iter=100, min_samples_leaf=20, random_state=42)
            m_t.fit(X[train_idx], T[train_idx])
            tilde_T_naive[test_idx] = T[test_idx] - m_t.predict(X[test_idx])
            
        theta_naive = np.sum(tilde_T_naive * tilde_Y_naive) / np.sum(tilde_T_naive ** 2)
        res_naive = tilde_Y_naive - theta_naive * tilde_T_naive
        se_naive = np.sqrt(np.mean(res_naive ** 2) / (np.sum(tilde_T_naive ** 2)))
        
        # 4. Block DML (Purged CV, no regimes)
        purged_cv = PurgedBlockKFold(n_splits=5, embargo_tau=24)
        tilde_Y_block = np.zeros(N)
        tilde_T_block = np.zeros(N)
        for train_idx, test_idx in purged_cv.split(N):
            m_y = HistGradientBoostingRegressor(max_iter=100, min_samples_leaf=20, random_state=42)
            m_y.fit(X[train_idx], Y[train_idx])
            tilde_Y_block[test_idx] = Y[test_idx] - m_y.predict(X[test_idx])
            
            m_t = HistGradientBoostingRegressor(max_iter=100, min_samples_leaf=20, random_state=42)
            m_t.fit(X[train_idx], T[train_idx])
            tilde_T_block[test_idx] = T[test_idx] - m_t.predict(X[test_idx])
            
        theta_block = np.sum(tilde_T_block * tilde_Y_block) / np.sum(tilde_T_block ** 2)
        res_block = tilde_Y_block - theta_block * tilde_T_block
        se_block = np.sqrt(np.mean(res_block ** 2) / (np.sum(tilde_T_block ** 2)))
        
        # 2. RC-DML (Ours: Latent Regimes + Purged Block Cross-Fitting)
        rc_model = RegimeConditionalDML(n_regimes=3, n_splits=5, embargo_tau=24)
        rc_model.fit(Y, T, X, Z)
        
        print(f"RC-DML Overall ATE: {rc_model.ate_:.4f} +/- {1.96*rc_model.ate_se_:.4f}")
        for k in range(3):
            th = rc_model.theta_regimes_[k]
            se = rc_model.se_regimes_[k]
            print(f"  Regime {k+1} Effect: {th:.4f} +/- {1.96*se:.4f}")
            
        all_results.append({
            'City': city,
            'Estimator': 'Naive_OLS',
            'Regime': 'Pooled_CrossSectional',
            'Effect_Theta': round(float(theta_ols), 4),
            'Std_Error': round(float(se_ols), 4),
            'CI_95_Lower': round(float(theta_ols - 1.96 * se_ols), 4),
            'CI_95_Upper': round(float(theta_ols + 1.96 * se_ols), 4),
            'Notes': 'Linear cross-sectional baseline'
        })

        all_results.append({
            'City': city,
            'Estimator': 'Random_Forest',
            'Regime': 'Pooled_CrossSectional',
            'Effect_Theta': round(float(theta_rf), 4),
            'Std_Error': round(float(se_rf), 4),
            'CI_95_Lower': round(float(theta_rf - 1.96 * se_rf), 4),
            'CI_95_Upper': round(float(theta_rf + 1.96 * se_rf), 4),
            'Notes': 'Non-linear plug-in estimator'
        })

        all_results.append({
            'City': city,
            'Estimator': 'Naive_DML',
            'Regime': 'Pooled_CrossSectional',
            'Effect_Theta': round(float(theta_naive), 4),
            'Std_Error': round(float(se_naive), 4),
            'CI_95_Lower': round(float(theta_naive - 1.96 * se_naive), 4),
            'CI_95_Upper': round(float(theta_naive + 1.96 * se_naive), 4),
            'Notes': 'Suffers from omitted regime bias & temporal leakage'
        })

        all_results.append({
            'City': city,
            'Estimator': 'Block_DML',
            'Regime': 'Pooled_CrossSectional',
            'Effect_Theta': round(float(theta_block), 4),
            'Std_Error': round(float(se_block), 4),
            'CI_95_Lower': round(float(theta_block - 1.96 * se_block), 4),
            'CI_95_Upper': round(float(theta_block + 1.96 * se_block), 4),
            'Notes': 'Purged CV without regime conditioning'
        })
        
        all_results.append({
            'City': city,
            'Estimator': 'RC_DML_Ours',
            'Regime': 'Overall_Weighted_ATE',
            'Effect_Theta': round(float(rc_model.ate_), 4),
            'Std_Error': round(float(rc_model.ate_se_), 4),
            'CI_95_Lower': round(float(rc_model.ate_ - 1.96 * rc_model.ate_se_), 4),
            'CI_95_Upper': round(float(rc_model.ate_ + 1.96 * rc_model.ate_se_), 4),
            'Notes': 'Consistent under latent atmospheric regimes'
        })
        
        for k in range(3):
            th = rc_model.theta_regimes_[k]
            se = rc_model.se_regimes_[k]
            all_results.append({
                'City': city,
                'Estimator': 'RC_DML_Ours',
                'Regime': f'Regime_{k+1}',
                'Effect_Theta': round(float(th), 4),
                'Std_Error': round(float(se), 4),
                'CI_95_Lower': round(float(th - 1.96 * se), 4),
                'CI_95_Upper': round(float(th + 1.96 * se), 4),
                'Notes': 'Regime-specific elasticity'
            })
            
    df_out = pd.DataFrame(all_results)
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nEmpirical results saved to {OUTPUT_PATH}")
    return df_out


if __name__ == "__main__":
    run_empirical_evaluation()
