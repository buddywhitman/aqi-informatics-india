"""ext_em_real.py -- EM convergence (H1) and predictability of the fitted regime posterior from the controls X (necessary for H3 to fail) on the real hourly series.
-> x26_em_real.csv"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from src.or_dml import LatentRegimeHMM, PurgedBlockKFold                         # noqa: E402
from src.bias_law import real_cities_sensitivity as R                            # noqa: E402
from src.bias_law.ext_real_hourly_fix import build                               # noqa: E402
from src.bias_law.ext_em_check import loglik, em                                 # noqa: E402
OUT = R.OUT
rows = []
for city in ['Delhi', 'Mumbai', 'Bengaluru', 'Kolkata']:
    d = build(city).dropna(subset=['no2', 'pm25'] + R.REGIME_FEATURES + R.CONTROLS).reset_index(drop=True)
    Z = d[R.REGIME_FEATURES].values.astype(float); X = d[R.CONTROLS].values.astype(float)
    Zs = (Z - Z.mean(0)) / Z.std(0)
    f25 = LatentRegimeHMM(2, n_iter=25, tol=1e-3).fit(Z); fc = LatentRegimeHMM(2, n_iter=600, tol=1e-9).fit(Z)
    l25 = loglik(Z, f25.means, f25.covs, f25.A, f25.pi)[0]; lc = loglik(Z, fc.means, fc.covs, fc.A, fc.pi)[0]
    g25 = f25.predict_posteriors(Z, 'smooth')[:, 1]; gc = fc.predict_posteriors(Z, 'smooth')[:, 1]
    lls = []; vs = []
    rng = np.random.default_rng(0)
    for r in range(5):
        m0 = Z[rng.choice(len(Z), 2, replace=False)].copy(); m, c, A, pi, ll = em(Z, m0, iters=300, tol=1e-6)
        lls.append(ll)
    # predictability of posterior logit-free gamma from X: purged block CV R^2 (ridge / gbm)
    r2 = {}
    for nm, mk in (('ridge', lambda: Ridge(1.0)), ('gbm', lambda: HistGradientBoostingRegressor(max_iter=100, random_state=0))):
        pred = np.zeros(len(gc))
        for tr, va in PurgedBlockKFold(5, 24).split(len(gc)):
            pred[va] = mk().fit(X[tr], gc[tr]).predict(X[va])
        r2[nm] = 1 - np.mean((gc - pred) ** 2) / np.var(gc)
    rows.append(dict(city=city, N=len(Z), ll25=l25, ll_conv=lc, gap=lc - l25, ll_restart_max=max(lls), ll_restart_min=min(lls),
                     restart_spread=max(lls) - min(lls), restart_vs_conv=max(lls) - lc, v25=float(np.mean(g25 * (1 - g25))), v_conv=float(np.mean(gc * (1 - gc))),
                     mean_abs_gamma_diff=float(np.mean(np.abs(g25 - gc))), r2_gamma_from_X_ridge=r2['ridge'], r2_gamma_from_X_gbm=r2['gbm']))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'x26_em_real.csv'), index=False)
