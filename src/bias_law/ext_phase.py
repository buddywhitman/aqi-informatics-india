"""
ext_phase.py -- phase diagram over (proxy strength dz, treatment shift dT): posterior-DML vs joint Markov-switching MLE vs proximal(W external).
Joint EM is started from (i) Z-only HMM posterior and (ii) k-means on (T,Y) residuals; best likelihood kept.
    python src/bias_law/ext_phase.py -> reports/bias_law/x5_phase.csv
"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.cluster import KMeans
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.bias_law.sim import gen_states_proxy, gen_TY, fit_posteriors, align_to_truth, partial_out   # noqa: E402
from src.bias_law.ext_joint_proximal import joint_em, proximal_2sls                                   # noqa: E402
OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'bias_law')


def one(a):
    dz, dT, dg, rep, N = a
    rng = np.random.default_rng(555 + 17 * rep + int(100 * dT) + int(1000 * dz))
    S, Z, X = gen_states_proxy(N, 2, max(dz, 1e-9), 0.9, rng)
    if dz == 0:
        Z = rng.multivariate_normal([0, 0], [[1, .2], [.2, 1]], size=N)
    V, U, E = rng.normal(size=N), rng.normal(size=N), rng.normal(size=N)
    T, Y = gen_TY(S, X, [0, dT], [0, dg], [1, 1], (V, U))
    W = 2.0 * S + 0.3 * X[:, 0] + E
    g = align_to_truth(fit_posteriors(Z, 2, rep)['smooth'], S)
    th_dml, tT, tY = partial_out(Y, T, np.column_stack([X, g[:, 1]]), 'lin', rep)
    # inits
    r = np.column_stack([T - np.polyfit(X[:, 0], T, 1)[0] * X[:, 0], Y])
    lab = KMeans(2, n_init=5, random_state=rep).fit_predict((r - r.mean(0)) / r.std(0))
    g2 = np.column_stack([1 - lab, lab]).astype(float) * 0.9 + 0.05
    best = None
    for gi in (g, g2):
        th, gg = joint_em(Y, T, X, Z, gi)
        if best is None or joint_em.last_ll > best[0]:
            best = (joint_em.last_ll, th)
    th_p, se_p, _ = proximal_2sls(Y, T, W, X, g[:, 1], rep)
    return dict(dz=dz, dT=dT, rep=rep, dml=th_dml - 1, joint=best[1] - 1, prox=th_p - 1, v_hat=float(np.mean(g[:, 1] * (1 - g[:, 1]))))


if __name__ == '__main__':
    R = 8
    jobs = [(dz, dT, 3.0, r, 3000) for dz in (0.0, 0.25, 0.5, 1.0) for dT in (0.5, 1.0, 2.0, 4.0) for r in range(R)]
    with Pool(2) as p:
        rows = list(p.imap_unordered(one, jobs))
    d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, 'x5_phase_raw.csv'), index=False)
    rm = lambda x: np.sqrt(np.mean(x ** 2))
    g = d.groupby(['dz', 'dT']).agg(v_hat=('v_hat', 'mean'), dml=('dml', 'mean'), joint=('joint', 'mean'), prox=('prox', 'mean'),
                                    dml_rmse=('dml', rm), joint_rmse=('joint', rm), prox_rmse=('prox', rm))
    g.to_csv(os.path.join(OUT, 'x5_phase.csv')); print(g.round(3).to_string())
