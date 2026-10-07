"""
city_stress_hourly.py
=====================
Four-city observational stress test on the CORRECTED hourly grid.

The legacy processed panel (data/processed_clean/combined_hourly_clean.csv) rounds OpenAQ hh:30Z timestamps
with ties-to-even, which collapses the series onto a 2-hour grid (see src/bias_law/ext_real_hourly_fix.py).
This script rebuilds a regular hourly grid from data/raw_hourly via ext_real_hourly_fix.build and reports,
for each city and for the unscreened and frozen-sensor-screened samples:
  * regime-specific OR-DML effects (posterior-adjusted nuisances; the legacy posterior-weighted variant is kept
    for comparison) with Newey-West HAC (lag 24 h), overall ATE, lambda_min (raw and
    standardised), condition number, mean posterior entropy, regime shares;
  * a pre-treatment placebo / dynamic-response curve with horizons measured in true hours: outcome leads and
    lags are taken on the regular hourly grid BEFORE rows with missing values are dropped;
  * sensor-quality audit statistics (distinct NO2 values, longest constant NO2 run in hours).

    python src/synthesis/city_stress_hourly.py
      -> reports/synthesis/city_ordml_hourly.csv, city_placebo_hourly.csv, city_audit_hourly.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.linear_model import Ridge

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from src.or_dml import OverlapAwareRegimeDML                                # noqa: E402
from src.bias_law import real_cities_sensitivity as R                       # noqa: E402
from src.bias_law.ext_real_hourly_fix import build                          # noqa: E402

warnings.filterwarnings('ignore')
OUT = os.path.join(ROOT, 'reports', 'synthesis')
os.makedirs(OUT, exist_ok=True)
CITIES = ['Delhi', 'Mumbai', 'Bengaluru', 'Kolkata']
HORIZONS = [-6, -3, -1, 0, 1, 3, 6, 12, 24]
REQ = ['no2', 'pm25'] + R.REGIME_FEATURES + R.CONTROLS


def longest_run_hours(index, x):
    best, start = 1, 0
    for i in range(1, len(x) + 1):
        if i == len(x) or x[i] != x[start] or (index[i] - index[i - 1]) != pd.Timedelta(hours=1):
            best = max(best, i - start)
            start = i
    return int(best)


def model(mode='posterior'):
    return OverlapAwareRegimeDML(n_regimes=2, n_splits=5, embargo_tau=24, reg_alpha=0.05, posterior_mode='smooth',
                                 nuisance_model=Ridge(alpha=1.0), hac_lag=24, random_state=42, nuisance_mode=mode)


def fit_city(city, screen):
    g = build(city)
    if screen:
        # screen on the grid: mark frozen runs among observed NO2 readings
        obs = g.no2.notna().values
        mask = np.zeros(len(g), bool)
        mask[obs] = R.frozen_mask(g.no2.values[obs], 6)
        g = g.copy()
        g.loc[mask, 'no2'] = np.nan
    d = g.dropna(subset=REQ)
    Y, T = d.pm25.values.astype(float), d.no2.values.astype(float)
    X, Z = d[R.CONTROLS].values.astype(float), d[R.REGIME_FEATURES].values.astype(float)
    m = model().fit(Y, T, X, Z)
    mw = model('weighted').fit(Y, T, X, Z)
    out = []
    shares = np.asarray(m.regime_weights_)
    base = dict(city=city, frozen_screen=screen, N=len(d), ate=m.ate_, ate_se=m.ate_se_, ate_p=m.ate_p_,
                lambda_min=m.lambda_min_, lambda_min_std=m.lambda_min_std_, kappa=m.kappa_,
                mean_entropy=m.mean_entropy_, unique_no2=int(d.no2.nunique()),
                ate_weighted=mw.ate_, ate_se_weighted=mw.ate_se_, lambda_min_weighted=mw.lambda_min_,
                lambda_min_std_weighted=mw.lambda_min_std_)
    for k in range(2):
        th, se = float(m.theta_regimes_[k]), float(m.se_regimes_[k])
        out.append(dict(base, regime=k + 1, share=float(shares[k]), theta=th, se=se,
                        p=float(2 * norm.sf(abs(th / max(se, 1e-12))))))
    # placebo / dynamic curve on the regular grid.
    #   h >= 0: outcome PM2.5_{t+h} on NO2_t with controls and regime proxies at t.
    #   h <  0 (lead placebo): outcome PM2.5_{t+h} (in the past) on NO2_t, conditioning on the information set at the
    #          OUTCOME time t+h (its own lagged pollutants, weather, and contemporaneous NO2_{t+h}).  Without this
    #          alignment the h=-1 placebo is mechanically zero because PM2.5_{t-1} is itself a control.
    curve = []
    for h in HORIZONS:
        gh = g.copy()
        gh['y_h'] = gh.pm25.shift(-h)
        ctrl = list(R.CONTROLS)
        if h < 0:
            for c in R.CONTROLS:
                gh[c + '_at_out'] = gh[c].shift(-h)
            gh['no2_at_out'] = gh.no2.shift(-h)
            ctrl = [c + '_at_out' for c in R.CONTROLS] + ['no2_at_out']
        dh = gh.dropna(subset=['no2', 'y_h'] + R.REGIME_FEATURES + ctrl)
        mh = model().fit(dh.y_h.values.astype(float), dh.no2.values.astype(float),
                         dh[ctrl].values.astype(float), dh[R.REGIME_FEATURES].values.astype(float))
        curve.append(dict(city=city, frozen_screen=screen, horizon_h=h, N=len(dh), ate=mh.ate_, ate_se=mh.ate_se_,
                          p=float(2 * norm.sf(abs(mh.ate_ / max(mh.ate_se_, 1e-12)))),
                          theta_r1=float(mh.theta_regimes_[0]), theta_r2=float(mh.theta_regimes_[1])))
    raw = build(city).dropna(subset=['no2'])
    audit = dict(city=city, frozen_screen=screen, N_model=len(d), unique_no2=int(raw.no2.nunique()),
                 longest_const_no2_h=longest_run_hours(raw.index, raw.no2.values),
                 first=str(raw.index.min()), last=str(raw.index.max()),
                 frac_hourly_steps=float((pd.Series(d.index).diff().dt.total_seconds() / 3600 == 1).mean()))
    print(city, screen, 'done', flush=True)
    return out, curve, audit


def main():
    jobs = [(c, s) for s in (False, True) for c in CITIES]
    res = Parallel(n_jobs=-1)(delayed(fit_city)(c, s) for c, s in jobs)
    pd.DataFrame([r for o, _, _ in res for r in o]).to_csv(os.path.join(OUT, 'city_ordml_hourly.csv'), index=False)
    pd.DataFrame([r for _, c, _ in res for r in c]).to_csv(os.path.join(OUT, 'city_placebo_hourly.csv'), index=False)
    pd.DataFrame([a for _, _, a in res]).to_csv(os.path.join(OUT, 'city_audit_hourly.csv'), index=False)


if __name__ == '__main__':
    main()
