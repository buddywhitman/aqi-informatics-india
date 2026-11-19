"""
deconv_semisynthetic.py -- rebuttal experiment E5: DC-DML on the real-data-calibrated semi-synthetic benchmark
(same generator, seeds and calibrated posteriors as src/synthesis/semisynthetic_hourly.py).

Caveat built into the benchmark: the treatment base series T0 is real NO2, which depends on the meteorological
proxies Z directly, so the exclusion restriction (T, Y independent of Z given X, S) behind DC-DML's
identification FAILS here.  Two variants are run:
  DC-DML (X basis)        assumes the exclusion restriction
  DC-DML (X + Z additive) allows an additive, regime-common quadratic effect of Z on every moment
SE: moving-block bootstrap of the second stage (block 24 h, B = 100), posterior held fixed.

    python src/rebuttal/deconv_semisynthetic.py -> reports/rebuttal/deconv_semisynthetic_{raw,summary}.csv
"""
import os, sys, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from src.synthesis.semisynthetic_hourly import calibrate, ffbs, CITIES, THETA, DT, DG, RHO_U, N_REPS
from src.rebuttal.deconv_estimator import basis
warnings.filterwarnings('ignore')
OUT = os.path.join(ROOT, 'reports', 'rebuttal'); os.makedirs(OUT, exist_ok=True)
B, BLK = 100, 24


def dc(g, X, T, Y, Zadd=None, idx=None):
    if idx is not None:
        g, X, T, Y = g[idx], X[idx], T[idx], Y[idx]
        Zadd = None if Zadd is None else Zadd[idx]
    Bx = basis(X); K = g.shape[1]; p = Bx.shape[1]
    R = np.column_stack([g[:, [k]] * Bx for k in range(K)])
    if Zadd is not None:
        R = np.column_stack([R, Zadd])
    th = np.empty(K); mom = {}
    for name, h in (('t', T), ('tt', T * T), ('y', Y), ('ty', T * Y)):
        c = np.linalg.lstsq(R, h, rcond=None)[0]
        zc = 0.0 if Zadd is None else Zadd @ c[K * p:]
        mom[name] = [Bx @ c[k * p:(k + 1) * p] + zc for k in range(K)]
    for k in range(K):
        w = g[:, k] / g[:, k].sum()
        th[k] = np.sum(w * (mom['ty'][k] - mom['t'][k] * mom['y'][k])) / np.sum(w * (mom['tt'][k] - mom['t'][k] ** 2))
    return th, float(g.mean(0) @ th)


def one(cal, rep):
    rng = np.random.RandomState(20261007 + 1000 * CITIES.index(cal['city']) + rep)   # identical to the paper
    Z, X, T0, gam = cal['Z'], cal['X'], cal['T0'], cal['gamma']
    n = len(T0)
    S = ffbs(rng, cal['logB'], cal['A'], cal['pi0'])
    U = np.zeros(n); e = rng.randn(n); U[0] = e[0]
    for t in range(1, n):
        U[t] = RHO_U * U[t - 1] + np.sqrt(1 - RHO_U ** 2) * e[t]
    beta = np.linspace(0.3, -0.3, X.shape[1])
    T = T0 + DT * S
    Y = THETA[S] * T + DG * S + X @ beta + U
    sate = float(THETA[S].mean())
    Zq = basis(Z)[:, 1:]                                  # linear + quadratic in Z, no intercept
    out = []
    brng = np.random.RandomState(rep)
    for name, Za in (('DC-DML (X basis)', None), ('DC-DML (X + Z additive)', Zq)):
        th, ate = dc(gam, X, T, Y, Za)
        nb = int(np.ceil(n / BLK)); bs = []
        for _ in range(B):
            st = brng.randint(0, n - BLK + 1, nb)
            idx = np.concatenate([np.arange(s, s + BLK) for s in st])[:n]
            bs.append(dc(gam, X, T, Y, Za, idx)[1])
        se = float(np.std(bs))
        out.append(dict(city=cal['city'], rep=rep, method=name, ate=ate, bias=ate - sate, se=se,
                        cover=float(abs(ate - sate) <= 1.96 * se), theta_err=float(np.linalg.norm(th - THETA))))
    return out


if __name__ == '__main__':
    cals = [calibrate(c) for c in CITIES]
    res = Parallel(n_jobs=-1)(delayed(one)(c, r) for c in cals for r in range(N_REPS))
    d = pd.DataFrame([x for rr in res for x in rr])
    d.to_csv(os.path.join(OUT, 'deconv_semisynthetic_raw.csv'), index=False)
    s = d.groupby(['city', 'method'], sort=False).agg(
        mean_abs_bias=('bias', lambda b: np.mean(np.abs(b))), mean_bias=('bias', 'mean'),
        rmse=('bias', lambda b: np.sqrt(np.mean(b ** 2))), coverage=('cover', 'mean'),
        theta_err_median=('theta_err', 'median')).reset_index()
    s.to_csv(os.path.join(OUT, 'deconv_semisynthetic_summary.csv'), index=False)
    print(s.round(3).to_string())
