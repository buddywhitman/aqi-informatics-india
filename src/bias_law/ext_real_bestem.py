"""ext_real_bestem.py -- four-city analysis on the hourly grid with the HMM posterior from the best of 8 EM restarts (random means, run to convergence),
instead of the default 25-iteration EM.  Mumbai's likelihood is multimodal (x26).  -> x27_real_bestem.csv"""
import os, sys, warnings
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.bias_law import real_cities_sensitivity as R                            # noqa: E402
from src.bias_law.ext_real_hourly_fix import build                               # noqa: E402
from src.bias_law.ext_em_check import loglik, em                                 # noqa: E402
warnings.filterwarnings('ignore')

def best_posterior(Z, restarts=8, seed=0):
    rng = np.random.default_rng(seed); best = None
    for r in range(restarts):
        m0 = Z[rng.choice(len(Z), 2, replace=False)].copy()
        m, c, A, pi, ll = em(Z, m0, iters=300, tol=1e-6)
        if best is None or ll > best[0]: best = (ll, m, c, A, pi)
    ll, m, c, A, pi = best
    order = np.argsort(m[:, 0]); m, c, pi = m[order], c[order], pi[order]; A = A[order][:, order]
    from src.or_dml import LatentRegimeHMM
    h = LatentRegimeHMM(2); h.means, h.covs, h.A, h.pi = m, c, A, pi
    return h.predict_posteriors(Z, 'smooth'), ll

if __name__ == '__main__':
    rows = []
    for screen in (False, True):
        for city in ['Delhi', 'Mumbai', 'Bengaluru', 'Kolkata']:
            d = build(city).dropna(subset=['no2', 'pm25'] + R.REGIME_FEATURES + R.CONTROLS).reset_index(drop=True)
            if screen: d = d[~R.frozen_mask(d.no2.values, 6)].reset_index(drop=True)
            Z = d[R.REGIME_FEATURES].values.astype(float)
            g, ll = best_posterior(Z)
            r = R.analyse(d.pm25.values.astype(float), d.no2.values.astype(float), d[R.CONTROLS].values.astype(float), Z, gamma=g)
            r.update(city=city, frozen_screen=screen, n_raw=len(d), ll_best=ll); rows.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in ('city', 'frozen_screen', 'N', 'theta', 'theta_lo', 'theta_hi', 'v_hat', 'b_star_sd', 'b_star_sd_lo', 'b_star_sd_hi')}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(R.OUT, 'x27_real_bestem.csv'), index=False)
