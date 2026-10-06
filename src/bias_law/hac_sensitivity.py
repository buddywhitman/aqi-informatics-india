"""
hac_sensitivity.py -- Newey-West lag sensitivity of the regime-specific OR-DML estimates (real data), on the corrected
hourly grid (ext_real_hourly_fix.build) with the frozen-reading screen (>=6 identical NO2 readings removed).

Replaces the unsupported "standard errors are stable across lags" statement of the previous draft with measured values.
    python src/bias_law/hac_sensitivity.py   ->  reports/bias_law/e6_hac_lag_sensitivity.csv
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.or_dml import OverlapAwareRegimeDML                       # noqa: E402
from src.bias_law.real_cities_sensitivity import REGIME_FEATURES, CONTROLS, ROOT, OUT  # noqa: E402
from src.bias_law import real_cities_sensitivity as R                                        # noqa: E402
from src.bias_law.ext_real_hourly_fix import build                                          # noqa: E402

warnings.filterwarnings('ignore')
LAGS = (12, 24, 72, 168)


def main():
    rows = []
    for city in ['Delhi', 'Mumbai', 'Bengaluru', 'Kolkata']:
        d = build(city).dropna(subset=['no2', 'pm25'] + REGIME_FEATURES + CONTROLS).reset_index(drop=True)
        d = d[~R.frozen_mask(d.no2.values, 6)].reset_index(drop=True)
        Y, T = d.pm25.values.astype(float), d.no2.values.astype(float)
        X, Z = d[CONTROLS].values.astype(float), d[REGIME_FEATURES].values.astype(float)
        for lag in LAGS:
            m = OverlapAwareRegimeDML(n_regimes=2, n_splits=5, embargo_tau=24, reg_alpha=0.05, posterior_mode='smooth',
                                      nuisance_model=Ridge(alpha=1.0), hac_lag=lag, random_state=42).fit(Y, T, X, Z)
            for k in range(2):
                th, se = float(m.theta_regimes_[k]), float(m.se_regimes_[k])
                rows.append(dict(city=city, lag=lag, regime=k + 1, theta=th, se=se, z=th / se,
                                 p=float(2 * norm.sf(abs(th / se)))))
            print(city, lag, [(round(float(m.theta_regimes_[k]), 4), round(float(m.se_regimes_[k]), 4)) for k in range(2)], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'e6_hac_lag_sensitivity.csv'), index=False)


if __name__ == '__main__':
    main()
