"""
learned_posterior_coupled.py
============================
Removes the synthetic-proxy circularity of the factorial experiment: posteriors are LEARNED by a
Gaussian HMM (EM, smoothing) from noisy proxies, nuisances are posterior-weighted and cross-fitted
with purged blocks, and every diagnostic uses the realised (proxy) Gram matrix.

Grid: proxy separation Delta_Z x regime-1 innovation SD sigma_1 x treatment shift Delta_m
(4 x 3 x 3 = 36 cells, 20 replications, N = 1200).  The DGP otherwise matches the factorial design.

Recorded per run: coupled OR-DML error ||theta_hat - theta|| (posterior-weighted nuisances), the error of OR-DML with
posterior-adjusted nuisances (err_pa), hard-FE error, eps_hat (evaluation only),
lambda_min of the proxy Gram, lambda_min of the oracle Gram, mean entropy, residual posterior variance,
and plug-in regime shifts of T and Y.

    python src/synthesis/learned_posterior_coupled.py  ->  reports/synthesis/learned_posterior_coupled_{raw,summary}.csv
"""
import itertools
import json
import os
import warnings

import numpy as np
import pandas as pd
from hmmlearn import hmm
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.synthesis.estimators import coupled_posterior  # noqa: E402

warnings.filterwarnings('ignore')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'reports', 'synthesis')
os.makedirs(OUT, exist_ok=True)

DZ_GRID = [0.75, 1.25, 2.0, 3.0]
SIG_GRID = [0.3, 0.75, 1.5]
DM_GRID = [1.5, 3.0, 6.0]
N_REPS, N = 20, 1200
THETA = np.array([0.75, 2.50])


def simulate(rng, dz, sig1, dm):
    S = np.zeros(N, dtype=int)
    S[0] = rng.random() < 0.444
    for t in range(1, N):
        S[t] = (rng.random() < 0.12) if S[t - 1] == 0 else (rng.random() >= 0.15)
    u = np.array([1.0, 1.0]) / np.sqrt(2.0)
    Z = dz * S[:, None] * u[None, :] + rng.randn(N, 2)
    X = rng.randn(N, 3)
    sig = np.where(S == 0, 1.0, sig1)
    T = 2.0 + dm * S + 0.8 * X[:, 0] - 0.5 * X[:, 1] + sig * rng.randn(N)
    Y = THETA[S] * T + 15.0 * (1 - S) + 50.0 * S + 1.2 * X[:, 0] + 0.9 * X[:, 2] + rng.normal(0, 1.2, N)
    return S, Z, X, T, Y


def posteriors(Z, S, seed):
    best, best_ll = None, -np.inf
    for i in range(3):
        m = hmm.GaussianHMM(n_components=2, covariance_type='full', n_iter=100, tol=1e-4, random_state=seed + 97 * i)
        try:
            m.fit(Z)
            ll = m.score(Z)
        except Exception:
            continue
        if ll > best_ll:
            best, best_ll = m, ll
    g = best.predict_proba(Z)
    # label alignment by proxy mean (evaluation of eps needs truth; ordering itself is observable)
    if best.means_[0].sum() > best.means_[1].sum():
        g = g[:, ::-1]
    return g


def purged_splits(n, k=5, emb=12):
    edges = np.linspace(0, n, k + 1).astype(int)
    for b in range(k):
        te = np.arange(edges[b], edges[b + 1])
        tr = np.r_[0:max(0, edges[b] - emb), min(n, edges[b + 1] + emb):n]
        yield tr, te


def coupled(W, X, T, Y):
    K = W.shape[1]
    tt, ty = np.zeros((K, N)), np.zeros((K, N))
    for k in range(K):
        w = np.maximum(W[:, k], 1e-4)
        for tr, te in purged_splits(N):
            tt[k, te] = T[te] - Ridge(alpha=1.0).fit(X[tr], T[tr], sample_weight=w[tr]).predict(X[te])
            ty[k, te] = Y[te] - Ridge(alpha=1.0).fit(X[tr], Y[tr], sample_weight=w[tr]).predict(X[te])
    J = np.array([[np.mean(W[:, j] * W[:, k] * tt[j] * tt[k]) for k in range(K)] for j in range(K)])
    Sv = np.array([np.mean(W[:, j] * tt[j] * ty[j]) for j in range(K)])
    return np.linalg.solve(J, Sv), float(np.linalg.eigvalsh(J).min())


def one(dz, sig1, dm, rep):
    rng = np.random.RandomState(910000 + rep)
    S, Z, X, T, Y = simulate(rng, dz, sig1, dm)
    H = np.c_[1 - S, S].astype(float)
    g = posteriors(Z, S, 1000 + rep)
    th, lam = coupled(g, X, T, Y)
    th_or, lam_or = coupled(H, X, T, Y)
    hard = np.eye(2)[g.argmax(1)]
    th_h, _ = coupled(hard, X, T, Y)
    th_pa, _, _, lam_pa = coupled_posterior(g, X, T, Y, list(purged_splits(N)))
    eps = float(np.mean(np.abs(g - H).sum(1)))
    ent = float(np.mean(-(g * np.log(g + 1e-12)).sum(1)))
    v = float(np.mean(g[:, 1] * (1 - g[:, 1])))
    w1 = g[:, 1] / g[:, 1].sum()
    w0 = g[:, 0] / g[:, 0].sum()
    dm_hat = float(abs(w1 @ T - w0 @ T))
    dy_hat = float(abs(w1 @ Y - w0 @ Y))
    return dict(dz=dz, sigma1=sig1, dm=dm, rep=rep, err=float(np.linalg.norm(th - THETA)),
                err_hard=float(np.linalg.norm(th_h - THETA)),
                err_pa=float(np.linalg.norm(th_pa - THETA)), lam_pa=lam_pa, err_oracle=float(np.linalg.norm(th_or - THETA)),
                eps_hat=eps, entropy=ent, v_hat=v, lam_proxy=lam, lam_oracle=lam_or, dm_hat=dm_hat, dy_hat=dy_hat)


def main():
    grid = list(itertools.product(DZ_GRID, SIG_GRID, DM_GRID, range(N_REPS)))
    rows = Parallel(n_jobs=-1, verbose=0)(delayed(one)(*a) for a in grid)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, 'learned_posterior_coupled_raw.csv'), index=False)
    lp = np.maximum(d.lam_proxy, 1e-12)
    scores = {
        'eps_over_lam_proxy': d.eps_hat / lp,
        'entropy_over_lam_proxy': d.entropy / lp,
        'eps_over_lam_oracle': d.eps_hat / np.maximum(d.lam_oracle, 1e-12),
        'eps_only': d.eps_hat,
        'inv_lam_proxy': 1 / lp,
        'entropy_only': d.entropy,
        'scaled_eps_shift_over_lam_proxy': d.eps_hat * d.dm_hat * d.dy_hat / lp,
        'scaled_entropy_shift_over_lam_proxy': d.entropy * d.dm_hat * d.dy_hat / lp,
    }
    res = []
    for name, s in scores.items():
        r_all = spearmanr(s, d.err)[0]
        r_within = np.mean([spearmanr(s[d.dm == m], d.err[d.dm == m])[0] for m in DM_GRID])
        cell = d.assign(score=s).groupby(['dz', 'sigma1', 'dm'])[['score', 'err']].median()
        res.append(dict(score=name, spearman_run_all=r_all, spearman_run_within_dm=r_within,
                        spearman_cell_all=spearmanr(cell.score, cell.err)[0]))
    res = pd.DataFrame(res)
    res.to_csv(os.path.join(OUT, 'learned_posterior_coupled_scores.csv'), index=False)
    summ = d.groupby(['dz', 'sigma1', 'dm']).agg(err=('err', 'median'), err_hard=('err_hard', 'median'),
                                                 err_pa=('err_pa', 'median'),
                                                 err_oracle=('err_oracle', 'median'), eps=('eps_hat', 'median'),
                                                 lam_proxy=('lam_proxy', 'median'), lam_oracle=('lam_oracle', 'median'),
                                                 entropy=('entropy', 'median')).reset_index()
    summ.to_csv(os.path.join(OUT, 'learned_posterior_coupled_summary.csv'), index=False)
    infl = (d.lam_proxy / d.lam_oracle)
    meta = dict(n_runs=int(len(d)), n_cells=int(len(summ)),
                frac_runs_lam_proxy_gt_oracle=float((infl > 1).mean()),
                median_lam_inflation=float(infl.median()),
                lam_inflation_q90=float(infl.quantile(.9)),
                frac_cells_hard_better=float((summ.err_hard < summ.err).mean()),
                median_err_ratio_coupled_over_hard=float((summ.err / summ.err_hard).median()),
                frac_cells_pa_better_than_hard=float((summ.err_pa < summ.err_hard).mean()),
                median_err_ratio_pa_over_hard=float((summ.err_pa / summ.err_hard).median()),
                frac_runs_pa_better_than_hard=float((d.err_pa < d.err_hard).mean()),
                median_err_ratio_pa_over_weighted=float((summ.err_pa / summ.err).median()))
    with open(os.path.join(OUT, 'learned_posterior_coupled_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(res.round(3).to_string())
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
