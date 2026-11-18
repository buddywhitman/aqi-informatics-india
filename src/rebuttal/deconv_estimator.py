"""
deconv_estimator.py -- rebuttal prototype E4: moment-deconvolution regime DML ("DC-DML").

Identification idea. If (T_t, Y_t) are independent of the proxies Z_{1:N} given (X_t, S_t) and
gamma_t = P(S_t | X_t, Z_{1:N}) is calibrated, then for any function h,
    E[h(T_t, Y_t) | X_t, Z_{1:N}] = sum_k gamma_tk * E[h(T, Y) | X_t, S_t = k].
So every regime-conditional moment is recovered by regressing h on gamma (x) basis(X), provided
E[gamma gamma^T | X] is nonsingular -- i.e. the proxies are informative.  Theorem 2 is exactly the
singular case (gamma constant).  With Y = theta_S T + g_S(X) + U:
    Cov(T, Y | X, S=k) = theta_k Var(T | X, S=k)   =>   theta_k identified.
The estimator plugs in h in {T, T^2, Y, TY}, a quadratic basis in X, purged cross-fitting is unnecessary
for these parametric regressions.  SE: moving-block bootstrap of the second stage (posterior held fixed),
so HMM-estimation uncertainty is NOT included (same caveat as the paper's HAC intervals).

    python src/rebuttal/deconv_estimator.py -> reports/rebuttal/deconv_{raw,summary}.csv
"""
import os, sys, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from src.synthetic_dgp_benchmark import generate_difficulty_dgp
from src.synthesis.estimators import purged, coupled_posterior, single_index
from src.synthesis.frontier_designs import posteriors, N, L, THETA
warnings.filterwarnings('ignore')
OUT = os.path.join(ROOT, 'reports', 'rebuttal'); os.makedirs(OUT, exist_ok=True)
REPS, P00, B, BLK = 100, 0.88, 200, 24


def basis(X):
    n, p = X.shape
    cols = [np.ones(n)] + [X[:, j] for j in range(p)] + [X[:, i] * X[:, j] for i in range(p) for j in range(i, p)]
    return np.column_stack(cols)


def dc_dml(g, X, T, Y, idx=None):
    if idx is not None:
        g, X, T, Y = g[idx], X[idx], T[idx], Y[idx]
    Bx = basis(X); K = g.shape[1]
    R = np.column_stack([g[:, [k]] * Bx for k in range(K)])          # gamma (x) basis(X)
    p = Bx.shape[1]
    mom = {}
    for name, h in (('t', T), ('tt', T * T), ('y', Y), ('ty', T * Y)):
        c = np.linalg.lstsq(R, h, rcond=None)[0]
        mom[name] = [Bx @ c[k * p:(k + 1) * p] for k in range(K)]       # E[h | X_t, S=k] at every t
    th = np.empty(K)
    for k in range(K):
        w = g[:, k] / g[:, k].sum()                                     # average over X | S=k
        cov = np.sum(w * (mom['ty'][k] - mom['t'][k] * mom['y'][k]))
        var = np.sum(w * (mom['tt'][k] - mom['t'][k] ** 2))
        th[k] = cov / var
    share = g.mean(0)
    return th, float(share @ th)


def block_boot(g, X, T, Y, rng):
    n = len(T); nb = int(np.ceil(n / BLK))
    ates, ths = [], []
    for _ in range(B):
        starts = rng.randint(0, n - BLK + 1, nb)
        idx = np.concatenate([np.arange(s, s + BLK) for s in starts])[:n]
        th, ate = dc_dml(g, X, T, Y, idx)
        ates.append(ate); ths.append(th)
    return float(np.std(ates)), np.std(np.array(ths), axis=0)


def one(dz, rep):
    seed = 300000 + int(1000 * dz) + int(100 * P00) * 7 + rep          # same datasets as frontier_designs
    Y, T, X, Z, gt = generate_difficulty_dgp(N=N, delta_z=dz, persistence=P00, random_state=seed)
    S = gt['S']; H = np.c_[1 - S, S].astype(float)
    g = posteriors(Z, seed)
    if np.mean(np.abs(g - H).sum(1)) > 1.0:
        g = g[:, ::-1]
    sate = float(THETA[S].mean())
    rows = []
    th, ate = dc_dml(g, X, T, Y)
    se, se_th = block_boot(g, X, T, Y, np.random.RandomState(seed))
    rows.append(dict(method='DC-DML (deconvolution)', ate=ate, se=se, th=th, se_th=se_th))
    folds = purged(N, 5, L)
    th2, sh, se2, _ = coupled_posterior(g, X, T, Y, folds, lam_scale=0.05, L=L)
    rows.append(dict(method='OR-DML (posterior-adjusted)', ate=float(sh @ th2), se=se2, th=th2, se_th=None))
    tho, sho, seo, _ = coupled_posterior(H, X, T, Y, folds, L=L)
    rows.append(dict(method='Oracle-state', ate=float(sho @ tho), se=seo, th=tho, se_th=None))
    out = []
    for r in rows:
        d = dict(dz=dz, rep=rep, method=r['method'], ate=r['ate'], sate=sate, bias=r['ate'] - sate, se=r['se'],
                 cover=float(abs(r['ate'] - sate) <= 1.96 * r['se']),
                 theta_err=float(np.linalg.norm(r['th'] - THETA)),
                 eps=float(np.mean(np.abs(g - H).sum(1))))
        if r['se_th'] is not None:
            d['cover_theta'] = float(np.all(np.abs(r['th'] - THETA) <= 1.96 * r['se_th']))
        out.append(d)
    return out


if __name__ == '__main__':
    jobs = [(dz, r) for dz in (0.5, 1.0, 2.0, 4.0) for r in range(REPS)]
    res = Parallel(n_jobs=-1)(delayed(one)(*j) for j in jobs)
    d = pd.DataFrame([x for rr in res for x in rr])
    d.to_csv(os.path.join(OUT, 'deconv_raw.csv'), index=False)
    s = d.groupby(['dz', 'method'], sort=False).agg(
        mean_abs_bias=('bias', lambda b: np.mean(np.abs(b))), mean_bias=('bias', 'mean'),
        rmse=('bias', lambda b: np.sqrt(np.mean(b ** 2))), coverage=('cover', 'mean'),
        theta_err_median=('theta_err', 'median'), cover_theta=('cover_theta', 'mean'),
        eps=('eps', 'median')).reset_index()
    s.to_csv(os.path.join(OUT, 'deconv_summary.csv'), index=False)
    print(s.round(3).to_string())
