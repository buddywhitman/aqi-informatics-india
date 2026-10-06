"""
ext_envreal.py -- law-regression on the four cities with calendar-month segments as environments (theta_e vs k_e = a_e v_e / Var(T~_e)).
Reports leverage (range of k_e) and the extrapolated intercept next to the segment-mean estimate.  Exploratory: invariance of b across months is untested.
    python src/bias_law/ext_envreal.py -> reports/bias_law/x11_envreal.csv, x11_envreal_segments.csv
"""
import os, sys, warnings
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.bias_law.real_cities_sensitivity import analyse, REGIME_FEATURES, CONTROLS, ROOT, OUT   # noqa: E402

df = pd.read_csv(os.path.join(ROOT, 'data', 'processed_clean', 'combined_hourly_clean.csv'))
df['month_id'] = pd.to_datetime(df.timestamp).dt.to_period('M').astype(str)
from sklearn.linear_model import Ridge
from src.or_dml import OverlapAwareRegimeDML, PurgedBlockKFold
from src.bias_law import bias_law as BL
segs, rows = [], []
for city in ['Delhi', 'Mumbai', 'Bengaluru', 'Kolkata']:
    d = df[df.city == city].dropna(subset=['no2', 'pm25'] + REGIME_FEATURES + CONTROLS).reset_index(drop=True)
    Y, T = d.pm25.values.astype(float), d.no2.values.astype(float)
    X, Z = d[CONTROLS].values.astype(float), d[REGIME_FEATURES].values.astype(float)
    N = len(d)
    m = OverlapAwareRegimeDML(n_regimes=2, n_splits=5, embargo_tau=24, reg_alpha=0.05, posterior_mode='smooth',
                              nuisance_model=Ridge(alpha=1.0), hac_lag=12, random_state=42).fit(Y, T, X, Z)
    g = m.gamma_; F = np.column_stack([X, g[:, 1]]); tT = np.zeros(N); tY = np.zeros(N)
    for tr, va in PurgedBlockKFold(5, 24).split(N):
        tT[va] = T[va] - Ridge(alpha=1.0).fit(F[tr], T[tr]).predict(F[va]); tY[va] = Y[va] - Ridge(alpha=1.0).fit(F[tr], Y[tr]).predict(F[va])
    for mth, idx in d.groupby('month_id').indices.items():
        if len(idx) < 150:
            continue
        TT, YY = tT[idx], tY[idx]
        th = float(TT @ YY / (TT @ TT)); v = float(np.mean(g[idx, 1] * (1 - g[idx, 1])))
        a = float(BL.treatment_shift(T[idx], X[idx], g[idx])[0]); vT = float(np.mean(TT ** 2))
        segs.append(dict(city=city, month=mth, N=len(idx), theta=th, k=a * v / vT, v=v, a=a, varT=vT))
S = pd.DataFrame(segs); S.to_csv(os.path.join(OUT, 'x11_envreal_segments.csv'), index=False)
for city, d in S.groupby('city'):
    d = d[np.isfinite(d.k) & np.isfinite(d.theta)]
    if len(d) < 4:
        continue
    A = np.column_stack([np.ones(len(d)), d.k]); w = d.N.values / d.N.sum()
    coef = np.linalg.lstsq(A * np.sqrt(w)[:, None], d.theta.values * np.sqrt(w), rcond=None)[0]
    rows.append(dict(city=city, n_seg=len(d), theta_mean=float(np.average(d.theta, weights=w)), intercept=float(coef[0]), slope_b=float(coef[1]),
                     k_min=float(d.k.min()), k_max=float(d.k.max()), k_sd=float(d.k.std()), theta_sd=float(d.theta.std())))
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, 'x11_envreal.csv'), index=False); print(R.round(4).to_string())
