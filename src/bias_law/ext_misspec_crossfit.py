"""
ext_misspec_crossfit.py -- two closing checks.
 x29: misspecified (diagonal-Gaussian) HMM on correlated heavy-tailed Z.  Compares the in-sample plug-in v_hat with the
      pseudo-true v(theta_KL) (parameters from a large-N fit, applied to the evaluation sample) and the law bias using each.
 x30: HMM fitted on the training folds only (purged blocks) vs fitted on all of Z; theta_hat bias vs law.
    python src/bias_law/ext_misspec_crossfit.py -> reports/bias_law/x29_misspec_v.csv, x30_foldwise_hmm.csv
"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.bias_law import bias_law as BL
from src.bias_law.sim import partial_out, align_to_truth
from src.bias_law.ext_joint_proximal import _fwd_bwd

OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'bias_law')


def gen(N, rng, dz=0.5, rho=0.9, corr=0.7, heavy=True):
    S = np.zeros(N, int)
    for t in range(1, N):
        S[t] = S[t - 1] if rng.random() < rho else 1 - S[t - 1]
    mu = np.where(S[:, None] == 1, [dz, 1.5 * dz], [-dz, -1.5 * dz])
    L = np.linalg.cholesky(np.array([[1, corr], [corr, 1]]))
    e = rng.standard_t(5, size=(N, 2)) / np.sqrt(5 / 3) if heavy else rng.normal(size=(N, 2))
    return S, mu + e @ L.T, rng.normal(size=(N, 2))


def diag_em(Z, iters=25, init=None):
    N, d = Z.shape
    if init is None:
        s = Z @ np.ones(d); q = np.quantile(s, [.25, .75]); lab = (s > np.median(s)).astype(int)
        mu = np.array([Z[lab == k].mean(0) for k in range(2)]); var = np.array([Z[lab == k].var(0) for k in range(2)])
        A = np.array([[.9, .1], [.1, .9]]); pi = np.array([.5, .5])
    else:
        mu, var, A, pi = [np.copy(x) for x in init]
    for _ in range(iters):
        lb = np.stack([-0.5 * (((Z - mu[k]) ** 2) / var[k] + np.log(2 * np.pi * var[k])).sum(1) for k in range(2)], 1)
        g, xi = _fwd_bwd(lb, A, pi)[:2]
        w = g.sum(0)
        mu = (g.T @ Z) / w[:, None]
        var = np.array([(g[:, k:k + 1] * (Z - mu[k]) ** 2).sum(0) / w[k] for k in range(2)]) + 1e-6
        A = xi / xi.sum(1, keepdims=True); pi = g[0]
    return mu, var, A, pi


def smooth(Z, par):
    mu, var, A, pi = par
    lb = np.stack([-0.5 * (((Z - mu[k]) ** 2) / var[k] + np.log(2 * np.pi * var[k])).sum(1) for k in range(2)], 1)
    return _fwd_bwd(lb, A, pi)[0]


def run(R=30, N=3000, dT=1.0, dg=3.0, heavy=True, corr=0.7, tag=''):
    rows29, rows30 = [], []
    rng0 = np.random.default_rng(5)
    S_big, Z_big, _ = gen(60000, rng0, corr=corr, heavy=heavy)
    par_kl = diag_em(Z_big, iters=40)
    for rep in range(R):
        rng = np.random.default_rng(900 + rep)
        S, Z, X = gen(N, rng, corr=corr, heavy=heavy)
        V, U = rng.normal(size=N), rng.normal(size=N)
        T = 0.5 * X[:, 0] - 0.3 * X[:, 1] + dT * S + V
        Y = 1.0 * T + dg * S + 0.3 * X[:, 1] + U
        # --- x29
        par = diag_em(Z)
        g = align_to_truth(smooth(Z, par), S)
        gk = align_to_truth(smooth(Z, par_kl), S)
        v_hat, v_kl = float((g[:, 1] * (1 - g[:, 1])).mean()), float((gk[:, 1] * (1 - gk[:, 1])).mean())
        th, _, _ = partial_out(Y, T, np.column_stack([X, g[:, 1]]), 'lin', rep)
        v_true = float(((S - g[:, 1]) ** 2).mean()); ce = float(np.mean((S - g[:, 1]) ** 2) - np.mean(g[:, 1] * (1 - g[:, 1])))
        rows29.append(dict(rep=rep, v_true=v_true, law_vtrue=BL.law_bias(dT, dg, v_true), v_hat=v_hat, v_kl=v_kl, gap=v_hat - v_kl, bias=th - 1.0,
                           law_vhat=BL.law_bias(dT, dg, v_hat), law_vkl=BL.law_bias(dT, dg, v_kl)))
        # --- x30: fold-wise HMM (train on Z outside block +/- embargo), posterior on held-out block from smoothing whole series
        K, tau = 5, 24
        edges = np.linspace(0, N, K + 1).astype(int)
        gf = np.zeros(N)
        for k in range(K):
            a, b = edges[k], edges[k + 1]
            tr = np.r_[0:max(a - tau, 0), min(b + tau, N):N]
            p = diag_em(Z[tr])
            gk_ = align_to_truth(smooth(Z, p), S)[:, 1]
            gf[a:b] = gk_[a:b]
        thf, _, _ = partial_out(Y, T, np.column_stack([X, gf]), 'lin', rep)
        vf = float((gf * (1 - gf)).mean())
        rows30.append(dict(rep=rep, v_global=v_hat, v_fold=vf, bias_global=th - 1.0, bias_fold=thf - 1.0,
                           law_global=BL.law_bias(dT, dg, v_hat), law_fold=BL.law_bias(dT, dg, vf)))
    a, b = pd.DataFrame(rows29), pd.DataFrame(rows30)
    a.to_csv(os.path.join(OUT, f'x29_misspec_v{tag}.csv'), index=False); b.to_csv(os.path.join(OUT, f'x30_foldwise_hmm{tag}.csv'), index=False)
    print(a.mean().round(4).to_string()); print(a[['gap']].agg(['mean', 'std']).round(4).to_string())
    print(b.mean().round(4).to_string())
    print('sd bias global/fold', b.bias_global.std().round(4), b.bias_fold.std().round(4))
    print('corr(bias_global,bias_fold)', np.corrcoef(b.bias_global, b.bias_fold)[0, 1].round(3))


if __name__ == '__main__':
    run(R=24)
    run(R=24, heavy=False, corr=0.2, tag='_wellspec')
