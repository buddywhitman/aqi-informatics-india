"""
bias_aware_ci.py -- rebuttal experiment E1.
Bias-aware intervals for posterior-feature DML on the frontier DGP of src/synthesis/frontier_designs.py
(same seeds, same HMM posteriors, same purged 5-block cross-fitting).

Interval for a sensitivity budget bbar on the composite outcome shift b_eff:
    theta_hat +/- ( |a_hat| * v_hat * bbar / Var(T~) + 1.96 * SE )
(the pooled residual-state law gives bias = a * v * b_eff / (a^2 v + sigma^2), and Var(T~) = a^2 v + sigma^2).
b_eff for this DGP is computed analytically: b + a(theta1 - theta_bar) + m_bar(theta1 - theta0).
Also records the per-run budget needed for coverage ('required budget') and its median.

    python src/rebuttal/bias_aware_ci.py -> reports/rebuttal/bias_aware_ci_{raw,summary}.csv
"""
import os, sys, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from src.synthetic_dgp_benchmark import generate_difficulty_dgp
from src.synthesis.estimators import purged, single_index
from src.synthesis.frontier_designs import posteriors, N, L, THETA
warnings.filterwarnings('ignore')
OUT = os.path.join(ROOT, 'reports', 'rebuttal'); os.makedirs(OUT, exist_ok=True)
REPS = 100
P00 = 0.88
DT = 4.0


def b_eff(p00=P00, dt=DT):
    p01, p10 = 1 - p00, 0.15
    pi1 = p01 / (p01 + p10); pi0 = 1 - pi1
    th0, th1 = THETA
    thbar = pi0 * th0 + pi1 * th1
    b = 50.0 - 15.0
    mbar = 4.0 - dt / 2.0
    a = dt
    return b + a * (th1 - thbar) + mbar * (th1 - th0)


def one(dz, rep):
    seed = 300000 + int(1000 * dz) + int(100 * P00) * 7 + rep       # identical to frontier_designs.one
    Y, T, X, Z, gt = generate_difficulty_dgp(N=N, delta_z=dz, persistence=P00, random_state=seed)
    S = gt['S']; H = np.c_[1 - S, S].astype(float)
    g = posteriors(Z, seed)
    if np.mean(np.abs(g - H).sum(1)) > 1.0:
        g = g[:, ::-1]
    sate = float(THETA[S].mean())
    folds = purged(N, 5, L)
    th, se, var_tt = single_index(np.c_[X, g[:, 1]], T, Y, folds, L)
    A = np.c_[np.ones(N), X, g[:, 1]]
    a_hat = float(np.linalg.lstsq(A, T, rcond=None)[0][-1])
    v_hat = float(np.mean(g[:, 1] * (1 - g[:, 1])))
    scale = abs(a_hat) * v_hat / var_tt                               # bias per unit of budget
    beff = b_eff()
    row = dict(dz=dz, rep=rep, theta=th, sate=sate, bias=th - sate, se=se, a_hat=a_hat, v_hat=v_hat,
               var_tt=var_tt, scale=scale, beff=beff, plugin_bias=a_hat * v_hat * beff / var_tt)
    gap = max(0.0, abs(th - sate) - 1.96 * se)
    row['b_required'] = gap / scale if scale > 0 else np.inf
    for k in (0.0, 0.5, 1.0, 2.0):
        row['cover_%gx' % k] = float(abs(th - sate) <= 1.96 * se + scale * k * beff)
    return row


if __name__ == '__main__':
    jobs = [(dz, r) for dz in (1.0, 2.0, 4.0) for r in range(REPS)]
    d = pd.DataFrame(Parallel(n_jobs=-1)(delayed(one)(*j) for j in jobs))
    d.to_csv(os.path.join(OUT, 'bias_aware_ci_raw.csv'), index=False)
    s = d.groupby('dz').agg(n=('rep', 'size'), mean_bias=('bias', 'mean'), mean_plugin=('plugin_bias', 'mean'),
                            median_b_required=('b_required', 'median'), beff=('beff', 'first'),
                            cov_0x=('cover_0x', 'mean'), cov_half=('cover_0.5x', 'mean'),
                            cov_1x=('cover_1x', 'mean'), cov_2x=('cover_2x', 'mean')).reset_index()
    s.to_csv(os.path.join(OUT, 'bias_aware_ci_summary.csv'), index=False)
    print(s.round(3).to_string())
