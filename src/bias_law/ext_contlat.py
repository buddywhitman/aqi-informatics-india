"""
ext_contlat.py -- latent confounder is a *continuous* AR(1) process (no regimes): HMM-based estimators are misspecified.
Compares posterior-DML (K=2 HMM), joint Markov-switching MLE, and proximal 2SLS (instruments: HMM posterior and/or raw Z).
    python src/bias_law/ext_contlat.py  -> reports/bias_law/x3_contlat.csv
"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.bias_law.sim import fit_posteriors, align_to_truth, partial_out          # noqa: E402
from src.bias_law.ext_joint_proximal import joint_em, _resid_X                      # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'bias_law')


def prox_general(Y, T, W, X, inst, seed=0):
    R = _resid_X(np.column_stack([Y, T, W, inst]), X, seed)
    Yt, Tt, Wt, It = R[:, 0], R[:, 1], R[:, 2], R[:, 3:]
    Rg = np.column_stack([Tt, Wt]); I = np.column_stack([Tt, It])
    P = I @ np.linalg.solve(I.T @ I, I.T @ Rg)
    b = np.linalg.solve(P.T @ Rg, P.T @ Yt)
    u = Yt - Rg @ b; Q = np.linalg.inv(P.T @ Rg)
    return float(b[0]), float(np.sqrt((Q @ (P.T * u ** 2) @ P @ Q.T)[0, 0]))


def one(a):
    dT, dg, snr, c, rep, N = a
    rng = np.random.default_rng(777 + 31 * rep + int(100 * dT))
    u = np.zeros(N)
    for t in range(1, N):
        u[t] = 0.97 * u[t - 1] + np.sqrt(1 - 0.97 ** 2) * rng.normal()
    X = rng.normal(size=(N, 2))
    Z = snr * np.column_stack([u, 1.5 * u]) + rng.multivariate_normal([0, 0], [[1, .2], [.2, 1]], size=N)
    V, U, E = rng.normal(size=N), rng.normal(size=N), rng.normal(size=N)
    T = 0.5 * X[:, 0] - 0.3 * X[:, 1] + dT * u + V
    Y = 1.0 * T + dg * u + 0.3 * X[:, 1] + U
    W = c * u + 0.3 * X[:, 0] + E
    Sb = (u > 0).astype(int)
    g = align_to_truth(fit_posteriors(Z, 2, rep)['smooth'], Sb)
    th_dml, _, _ = partial_out(Y, T, np.column_stack([X, g[:, 1]]), 'lin', rep)
    th_j, _ = joint_em(Y, T, X, Z, g)
    th_pg, se_pg = prox_general(Y, T, W, X, g[:, 1:2], rep)
    th_pz, se_pz = prox_general(Y, T, W, X, Z, rep)
    # oracle: condition on true u
    th_or, _, _ = partial_out(Y, T, np.column_stack([X, u]), 'lin', rep)
    return dict(dT=dT, snr=snr, rep=rep, dml=th_dml - 1, joint=th_j - 1, prox_gamma=th_pg - 1, prox_Z=th_pz - 1, oracle=th_or - 1, se_pz=se_pz)


if __name__ == '__main__':
    R = 12
    jobs = [(dT, 3.0, snr, 2.0, r, 3000) for dT in (1.0, 2.0) for snr in (0.5, 1.0) for r in range(R)]
    with Pool(2) as p:
        rows = list(p.imap_unordered(one, jobs))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, 'x3_contlat_raw.csv'), index=False)
    rm = lambda x: np.sqrt(np.mean(x ** 2))
    g = d.groupby(['dT', 'snr']).agg(**{f'{k}_bias': (k, 'mean') for k in ('dml', 'joint', 'prox_gamma', 'prox_Z', 'oracle')},
                                      **{f'{k}_rmse': (k, rm) for k in ('dml', 'joint', 'prox_gamma', 'prox_Z')})
    g.to_csv(os.path.join(OUT, 'x3_contlat.csv'))
    print(g.round(3).to_string())
