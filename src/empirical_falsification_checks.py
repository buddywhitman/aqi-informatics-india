"""
empirical_falsification_checks.py
==================================
Empirical Placebo and Falsification Testing for Atmospheric Sensor Networks.

Tests the temporal identification assumption of Overlap-Aware Regime DML (OR-DML):
  Future atmospheric interventions cannot causally alter past air quality.

Evaluates pre-treatment lead horizons h in {-6, -3, -1} alongside contemporaneous (h = 0)
and dynamic lag horizons h in {1, 3, 6, 12, 24} across Delhi, Mumbai, Bengaluru, and Kolkata.

If causal identification holds and there is no anticipatory confounding:
  theta_h(S_t) for h in {-6, -3, -1} should be statistically indistinguishable from zero,
  providing a decisive falsification benchmark.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy.stats import norm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.or_dml import OverlapAwareRegimeDML


DATA_PATH = "data/processed_clean/combined_hourly_clean.csv"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


def run_placebo_falsification_checks():
    print(f"Loading clean atmospheric dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    treatment_col = "no2"
    outcome_col = "pm25"
    regime_features = ["temperature", "wind_speed", "humidity", "pressure", "hour_sin", "hour_cos"]
    controls = [
        "pm25_lag_1h", "no2_lag_1h", "pm25_roll_3h", "no2_roll_3h",
        "temperature", "humidity", "wind_speed", "pressure"
    ]

    required_cols = [treatment_col, outcome_col] + regime_features + controls
    cities = ["Delhi", "Mumbai", "Bengaluru", "Kolkata"]
    horizons = [-6, -3, -1, 0, 1, 3, 6, 12, 24]

    all_records = []

    print("\n==================================================================")
    print("Executing Multi-City Empirical Pre-Treatment Placebo Checks")
    print("==================================================================")

    for city in cities:
        city_df = df[df["city"] == city].dropna(subset=required_cols).copy()
        N = len(city_df)
        if N < 500:
            print(f"Skipping {city}: insufficient observations ({N}).")
            continue

        print(f"\n--- Placebo Falsification for {city} (N={N} hours) ---")
        Y = city_df[outcome_col].values
        T = city_df[treatment_col].values
        X = city_df[controls].values
        Z = city_df[regime_features].values

        model = OverlapAwareRegimeDML(
            n_regimes=2, n_splits=5, embargo_tau=24,
            reg_alpha=0.05, posterior_mode="smooth",
            nuisance_model=Ridge(alpha=1.0),
            random_state=42
        )

        irfs = model.fit_dynamic_irf(Y, T, X, Z, horizons=horizons)

        for h_idx, h in enumerate(horizons):
            ate = irfs["ate"]["effects"][h_idx]
            ate_se = irfs["ate"]["ses"][h_idx]
            r1 = irfs[0]["effects"][h_idx]
            r1_se = irfs[0]["ses"][h_idx]
            r2 = irfs[1]["effects"][h_idx]
            r2_se = irfs[1]["ses"][h_idx]

            status = "Pre-Treatment Placebo" if h < 0 else ("Contemporaneous" if h == 0 else "Post-Treatment Effect")
            pval = float(2.0 * norm.sf(abs(ate / max(ate_se, 1e-12))))

            all_records.append({
                "City": city,
                "Horizon": h,
                "Type": status,
                "ATE_Effect": round(ate, 4),
                "ATE_SE": round(ate_se, 4),
                "CI_95_Lower": round(ate - 1.96 * ate_se, 4),
                "CI_95_Upper": round(ate + 1.96 * ate_se, 4),
                "p_value": round(pval, 4),
                "Regime_1_Effect": round(r1, 4),
                "Regime_1_SE": round(r1_se, 4),
                "Regime_2_Effect": round(r2, 4),
                "Regime_2_SE": round(r2_se, 4)
            })

            print(f"  h={h:3d} ({status[:13]}): ATE={ate:+.4f} +/- {1.96*ate_se:.4f} (p={pval:.3f}) | R1={r1:+.4f}, R2={r2:+.4f}")

    df_out = pd.DataFrame(all_records)
    out_csv = os.path.join(REPORTS_DIR, "empirical_placebo_falsification.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"\nEmpirical placebo falsification results saved to {out_csv}.")
    return df_out


if __name__ == "__main__":
    run_placebo_falsification_checks()
