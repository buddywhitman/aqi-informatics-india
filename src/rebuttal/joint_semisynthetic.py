"""
joint_semisynthetic.py -- rebuttal experiment E7: joint Markov-switching MLE (the Table 10 competitor,
generalised to p controls) on the real-data-calibrated semi-synthetic benchmark, same datasets as the paper.
Here its Gaussian/linear emission model for (T, Y) is misspecified (real NO2 base series, AR(1) outcome noise,
regime-specific effects).  Two forms: common theta (as in Table 10) and regime-specific theta_k.  Reported against the regime-averaged SATE.

    python src/rebuttal/joint_semisynthetic.py -> reports/rebuttal/joint_semisynthetic_{raw,summary}.csv
"""
import os, sys, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from src.synthesis.semisynthetic_hourly import calibrate, ffbs, CITIES, THETA, DT, DG, RHO_U, N_REPS
from src.bias_law.ext_joint_proximal import _fwd_bwd
warnings.filterwarnings('ignore')
OUT = os.path.join(ROOT, 'reports', 'rebuttal')


def joint_em(Y, T, X, Z, g0, iters=60, hetero=False):
    """Markov-switching MLE. hetero=False: common theta (Table 10 form); hetero=True: regime-specific theta_k.
    Returns the share-weighted ATE."""
    N, p = X.shape; K = 2; g = g0.copy()
    A = np.array([[.9, .1], [.1, .9]]); pi = np.array([.5, .5]); ll_old = -np.inf
    for _ in range(iters):
        Nk = g.sum(0); mu = (g.T @ Z) / Nk[:, None]
        Sig = sum(((Z - mu[k]).T * g[:, k]) @ (Z - mu[k]) for k in range(K)) / N
        D = np.vstack([np.column_stack([X, np.eye(K)[k][None, :].repeat(N, 0)]) for k in range(K)])
        w = np.concatenate([g[:, k] for k in range(K)]); sw = np.sqrt(w)[:, None]
        bT = np.linalg.lstsq(D * sw, np.tile(T, K) * sw[:, 0], rcond=None)[0]
        sT2 = float(np.sum(w * (np.tile(T, K) - D @ bT) ** 2) / N)
        if hetero:
            TK = np.vstack([np.outer(T, np.eye(K)[k]) for k in range(K)])      # T x 1{regime k}
            DY = np.column_stack([TK, D])
        else:
            DY = np.column_stack([np.tile(T, K), D])
        bY = np.linalg.lstsq(DY * sw, np.tile(Y, K) * sw[:, 0], rcond=None)[0]
        sY2 = float(np.sum(w * (np.tile(Y, K) - DY @ bY) ** 2) / N)
        Si = np.linalg.inv(Sig); ld = np.linalg.slogdet(Sig)[1]
        logB = np.zeros((N, K))
        for k in range(K):
            dz = Z - mu[k]
            o = K if hetero else 1
            thk = bY[k] if hetero else bY[0]
            logB[:, k] = (-0.5 * np.einsum('ij,jk,ik->i', dz, Si, dz) - 0.5 * ld
                          - 0.5 * (T - X @ bT[:p] - bT[p + k]) ** 2 / sT2 - 0.5 * np.log(sT2)
                          - 0.5 * (Y - thk * T - X @ bY[o:o + p] - bY[o + p + k]) ** 2 / sY2 - 0.5 * np.log(sY2))
        g, xi, ll = _fwd_bwd(logB, A, pi)
        A = xi / xi.sum(1, keepdims=True); pi = g[0]
        if abs(ll - ll_old) < 1e-4:
            break
        ll_old = ll
    if hetero:
        return float(g.mean(0) @ bY[:K]), bY[:K].copy()
    return float(bY[0]), None


def one(cal, rep):
    rng = np.random.RandomState(20261007 + 1000 * CITIES.index(cal['city']) + rep)
    Z, X, T0, gam = cal['Z'], cal['X'], cal['T0'], cal['gamma']
    n = len(T0); S = ffbs(rng, cal['logB'], cal['A'], cal['pi0'])
    U = np.zeros(n); e = rng.randn(n); U[0] = e[0]
    for t in range(1, n):
        U[t] = RHO_U * U[t - 1] + np.sqrt(1 - RHO_U ** 2) * e[t]
    beta = np.linspace(0.3, -0.3, X.shape[1])
    T = T0 + DT * S; Y = THETA[S] * T + DG * S + X @ beta + U
    sate = float(THETA[S].mean())
    out = []
    for name, het in (('Joint MLE, common theta', False), ('Joint MLE, regime-specific theta', True)):
        ate, thk = joint_em(Y, T, X, Z, gam, hetero=het)
        out.append(dict(city=cal['city'], rep=rep, method=name, ate=ate, bias=ate - sate,
                        theta_err=float(np.linalg.norm(thk - THETA)) if thk is not None else np.nan))
    return out


if __name__ == '__main__':
    cals = [calibrate(c) for c in CITIES]
    res = Parallel(n_jobs=-1)(delayed(one)(c, r) for c in cals for r in range(N_REPS))
    d = pd.DataFrame([x for rr in res for x in rr])
    d.to_csv(os.path.join(OUT, 'joint_semisynthetic_raw.csv'), index=False)
    s = d.groupby(['city', 'method'], sort=False).agg(
        mean_abs_bias=('bias', lambda b: np.mean(np.abs(b))), mean_bias=('bias', 'mean'),
        rmse=('bias', lambda b: np.sqrt(np.mean(b ** 2))), median_abs=('bias', lambda b: np.median(np.abs(b))),
        theta_err_median=('theta_err', 'median')).reset_index()
    s.to_csv(os.path.join(OUT, 'joint_semisynthetic_summary.csv'), index=False)
    print(s.round(3).to_string())
