"""
frontier_designs.py
===================
Observability x persistence frontier for eight estimators that share identical inputs.

DGP: the difficulty-frontier generator of src/synthetic_dgp_benchmark.generate_difficulty_dgp (two regimes,
regime-specific effects theta = (0.75, 2.50), treatment shift 4, outcome offsets 15/50, autocorrelated controls,
2-d Gaussian proxies with separation Delta_Z, N = 1,200). Regime-0 persistence p00 is varied; p10 = 0.15.

Posteriors are LEARNED by a Gaussian HMM (EM, 3 restarts, smoothing) from the proxies Z alone.  Labels are aligned
to the truth only for evaluation (a permutation; the estimators never see S).

Estimators (purged 5-block cross-fitting, embargo 12, Ridge nuisances, HAC lag 12):
  Oracle-state      coupled regime DML with the true one-hot state
  Standard DML      single-index, random folds, nuisances on X
  Block DML         single-index, purged folds, nuisances on X
  DML + Z           single-index, nuisances on (X, Z)
  Posterior-feature single-index, nuisances on (X, gamma)
  Hard regime FE    coupled, one-hot argmax(gamma) weights, weighted nuisances
  OR-DML weighted   coupled, soft gamma weights, posterior-weighted nuisances (legacy design; ablation)
  OR-DML            coupled, soft gamma weights, posterior-adjusted nuisances on (X, gamma, gamma x X)

Grid: Delta_Z in {0.2, 0.5, 1, 2, 4} at p00 = 0.88 (100 replications each) and p00 in {0.70, 0.97} at
Delta_Z in {1, 2} (100 replications each).

    python src/synthesis/frontier_designs.py -> reports/synthesis/frontier_designs_{raw,summary}.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from hmmlearn import hmm
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from src.synthetic_dgp_benchmark import generate_difficulty_dgp                       # noqa: E402
from src.synthesis.estimators import (purged, single_index, coupled_weighted,          # noqa: E402
                                      coupled_posterior)

warnings.filterwarnings('ignore')
OUT = os.path.join(ROOT, 'reports', 'synthesis')
os.makedirs(OUT, exist_ok=True)
N, REPS, L = 1200, 100, 12
THETA = np.array([0.75, 2.50])
GRID = [(dz, 0.88) for dz in (0.2, 0.5, 1.0, 2.0, 4.0)] + [(dz, p) for p in (0.70, 0.97) for dz in (1.0, 2.0)]


def posteriors(Z, seed):
    best, ll = None, -np.inf
    for i in range(3):
        m = hmm.GaussianHMM(n_components=2, covariance_type='full', n_iter=100, tol=1e-4, random_state=seed + 97 * i)
        try:
            m.fit(Z)
            s = m.score(Z)
        except Exception:
            continue
        if s > ll:
            best, ll = m, s
    return best.predict_proba(Z)


def one(dz, p00, rep):
    seed = 300000 + int(1000 * dz) + int(100 * p00) * 7 + rep
    Y, T, X, Z, gt = generate_difficulty_dgp(N=N, delta_z=dz, persistence=p00, random_state=seed)
    S = gt['S']
    H = np.c_[1 - S, S].astype(float)
    g = posteriors(Z, seed)
    if np.mean(np.abs(g - H).sum(1)) > 1.0:          # evaluation-only label alignment (permutation)
        g = g[:, ::-1]
    sate = float(THETA[S].mean())
    folds = purged(N, 5, L)
    rows = []

    def add(name, ate, se, th=None, lam=np.nan):
        rows.append(dict(dz=dz, p00=p00, rep=rep, method=name, ate=ate, bias=ate - sate, se=se,
                         cover=float(abs(ate - sate) <= 1.96 * se),
                         theta_err=float(np.linalg.norm(th - THETA)) if th is not None else np.nan, lam=lam))

    th, sh, se, lam_or = coupled_weighted(H, X, T, Y, folds, L=L)
    add('Oracle-state', float(sh @ th), se, th, lam_or)
    kf = list(KFold(5, shuffle=True, random_state=rep).split(X))
    add('Standard DML', *single_index(X, T, Y, kf, L)[:2])
    add('Block DML', *single_index(X, T, Y, folds, L)[:2])
    add('DML + Z controls', *single_index(np.c_[X, Z], T, Y, folds, L)[:2])
    th_f, se_f, var_tt = single_index(np.c_[X, g[:, 1]], T, Y, folds, L)
    add('Posterior-feature DML', th_f, se_f)
    hard = np.eye(2)[g.argmax(1)]
    th, sh, se, lam = coupled_weighted(hard, X, T, Y, folds, L=L)
    add('Hard regime FE', float(sh @ th), se, th, lam)
    th, sh, se, lam_w = coupled_weighted(g, X, T, Y, folds, lam_scale=0.05, L=L)
    add('OR-DML, weighted nuisances', float(sh @ th), se, th, lam_w)
    th, sh, se, lam_p = coupled_posterior(g, X, T, Y, folds, lam_scale=0.05, L=L)
    add('OR-DML', float(sh @ th), se, th, lam_p)
    eps = float(np.mean(np.abs(g - H).sum(1)))
    ent = float(np.mean(-(g * np.log(g + 1e-12)).sum(1)))
    v_hat = float(np.mean(g[:, 1] * (1 - g[:, 1])))
    v_true = float(np.mean((S - g[:, 1]) ** 2))
    for r in rows:
        r.update(eps_hat=eps, entropy=ent, v_hat=v_hat, v_true=v_true, lam_oracle=lam_or,
                 switches=int(np.sum(S[1:] != S[:-1])))
    return rows


def main():
    jobs = [(dz, p, r) for dz, p in GRID for r in range(REPS)]
    res = Parallel(n_jobs=-1)(delayed(one)(*j) for j in jobs)
    d = pd.DataFrame([x for rr in res for x in rr])
    d.to_csv(os.path.join(OUT, 'frontier_designs_raw.csv'), index=False)
    s = d.groupby(['dz', 'p00', 'method'], sort=False).agg(
        mean_abs_bias=('bias', lambda b: np.mean(np.abs(b))), median_abs_bias=('bias', lambda b: np.median(np.abs(b))),
        rmse=('bias', lambda b: np.sqrt(np.mean(b ** 2))), coverage=('cover', 'mean'),
        theta_err=('theta_err', 'mean'), theta_err_median=('theta_err', 'median'), lam=('lam', 'median'),
        lam_oracle=('lam_oracle', 'median'), eps_hat=('eps_hat', 'median'), entropy=('entropy', 'median'),
        v_hat=('v_hat', 'median'), v_true=('v_true', 'median'), switches=('switches', 'mean'),
        n=('rep', 'size')).reset_index()
    s.to_csv(os.path.join(OUT, 'frontier_designs_summary.csv'), index=False)
    pd.set_option('display.width', 250)
    print(s[['dz', 'p00', 'method', 'mean_abs_bias', 'coverage', 'theta_err', 'lam', 'lam_oracle', 'eps_hat']].round(3).to_string())


if __name__ == '__main__':
    main()
