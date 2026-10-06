"""ext_em_check.py -- checks of the hypotheses behind Prop. (rate) for the EM fit actually used (src/or_dml.LatentRegimeHMM, NumPy fallback; hmmlearn absent).
H1: is the fitted HMM the MLE?  (i) 25-iteration/tol 1e-3 default vs run to convergence; (ii) random restarts (local optima).
H2: finite-difference sensitivity |d gamma_t / d theta| of the smoothing posterior at the truth vs N.
H3: on real data, how much of the fitted posterior is predictable from the controls X (necessary for H3 to fail).
-> x24_em_h1.csv, x25_h2_derivative.csv, x26_h3_real.csv"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.or_dml import LatentRegimeHMM                                           # noqa: E402
from src.bias_law.sim import gen_states_proxy, align_to_truth                    # noqa: E402
from src.bias_law.ext_joint_proximal import _fwd_bwd                             # noqa: E402
OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'bias_law')


def loglik(Z, means, covs, A, pi):
    var = np.maximum(covs, 1e-4)
    logB = np.stack([-0.5 * np.sum(np.log(2 * np.pi * var[k]) + (Z - means[k]) ** 2 / var[k], 1) for k in range(len(means))], 1)
    return _fwd_bwd(logB, A, pi)[2], logB


def em(Z, means0, iters=400, tol=1e-7):
    K = len(means0); N = len(Z)
    means = means0.copy(); covs = np.tile(Z.var(0), (K, 1)); A = np.full((K, K), 0.1 / (K - 1)); np.fill_diagonal(A, 0.9); pi = np.full(K, 1 / K)
    old = -np.inf
    for it in range(iters):
        ll, logB = loglik(Z, means, covs, A, pi)
        g, xi, _ = _fwd_bwd(logB, A, pi)
        A = xi / xi.sum(1, keepdims=True); pi = g[0]
        for k in range(K):
            w = g[:, k][:, None]; s = w.sum()
            means[k] = (w * Z).sum(0) / s; covs[k] = (w * (Z - means[k]) ** 2).sum(0) / s + 1e-4
        if abs(ll - old) < tol: break
        old = ll
    return means, covs, A, pi, ll


def h1(a):
    dz, rep, N = a
    rng = np.random.default_rng(777 + rep + int(10 * dz))
    S, Z, X = gen_states_proxy(N, 2, dz, 0.9, rng)
    out = dict(dz=dz, rep=rep)
    f25 = LatentRegimeHMM(2, n_iter=25, tol=1e-3).fit(Z); fc = LatentRegimeHMM(2, n_iter=600, tol=1e-9).fit(Z)
    ll25 = loglik(Z, f25.means, f25.covs, f25.A, f25.pi)[0]; llc = loglik(Z, fc.means, fc.covs, fc.A, fc.pi)[0]
    g25 = align_to_truth(f25.predict_posteriors(Z, 'smooth'), S)[:, 1]; gc = align_to_truth(fc.predict_posteriors(Z, 'smooth'), S)[:, 1]
    lls = []
    for r in range(5):
        m0 = Z[rng.choice(N, 2, replace=False)].copy()
        lls.append(em(Z, m0)[4])
    out.update(ll25=ll25, ll_conv=llc, ll_best_restart=max(lls), ll_restart_min=min(lls), n_distinct_restart_optima=int(len(set(np.round(lls, 1)))),
               gap_25_vs_conv=llc - ll25, v25=float(np.mean(g25 * (1 - g25))), v_conv=float(np.mean(gc * (1 - gc))),
               l1_gamma_25_vs_conv=float(np.mean(np.abs(g25 - gc))))
    return out


def h2(N, seed=0, dz=0.8, rho=0.9):
    rng = np.random.default_rng(seed)
    S, Z, X = gen_states_proxy(N, 2, dz, rho, rng)
    mu = np.array([[-dz, -1.5 * dz], [dz, 1.5 * dz]]); var = np.ones((2, 2)); A = np.array([[rho, 1 - rho], [1 - rho, rho]]); pi = np.array([.5, .5])
    def gam(mu, lv, lg):
        Aq = np.array([[1 / (1 + np.exp(-lg[0])), 1 - 1 / (1 + np.exp(-lg[0]))], [1 - 1 / (1 + np.exp(-lg[1])), 1 / (1 + np.exp(-lg[1]))]])
        ll, logB = loglik(Z, mu, np.exp(lv), Aq, pi)
        return _fwd_bwd(logB, Aq, pi)[0][:, 1]
    th = [('mu', i, j) for i in range(2) for j in range(2)] + [('lv', i, j) for i in range(2) for j in range(2)] + [('lg', i, None) for i in range(2)]
    lv0 = np.log(var); lg0 = np.log(np.array([rho / (1 - rho)] * 2))
    res = []
    for kind, i, j in th:
        e = 1e-4; P = [mu.copy(), lv0.copy(), lg0.copy()]; Mn = [mu.copy(), lv0.copy(), lg0.copy()]
        idx = {'mu': 0, 'lv': 1, 'lg': 2}[kind]
        if j is None: P[idx][i] += e; Mn[idx][i] -= e
        else: P[idx][i, j] += e; Mn[idx][i, j] -= e
        d = (gam(*P) - gam(*Mn)) / (2 * e)
        res.append(dict(N=N, param=f'{kind}{i}{"" if j is None else j}', max_abs=float(np.max(np.abs(d))), rms=float(np.sqrt(np.mean(d ** 2)))))
    return res


if __name__ == '__main__':
    with Pool(2) as p:
        r1 = pd.DataFrame(p.map(h1, [(dz, r, 3000) for dz in (0.5, 0.8, 1.2) for r in range(8)]))
    r1.to_csv(os.path.join(OUT, 'x24_em_h1.csv'), index=False)
    print(r1.groupby('dz')[['gap_25_vs_conv', 'v25', 'v_conv', 'l1_gamma_25_vs_conv', 'n_distinct_restart_optima']].mean().round(4).to_string())
    print('restarts: max (ll_best_restart - ll_conv) =', float((r1.ll_best_restart - r1.ll_conv).max()), ' min =', float((r1.ll_best_restart - r1.ll_conv).min()))
    rows = [x for N in (1000, 4000, 16000, 64000) for x in h2(N)]
    r2 = pd.DataFrame(rows); r2.to_csv(os.path.join(OUT, 'x25_h2_derivative.csv'), index=False)
    print(r2.pivot(index='param', columns='N', values='max_abs').round(2).to_string())
    print(r2.pivot(index='param', columns='N', values='rms').round(3).to_string())
