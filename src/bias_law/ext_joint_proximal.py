"""
ext_joint_proximal.py -- two extensions of the residual-state bias law.

(A) Joint-likelihood competitor: Markov-switching regression (EM) with T,Y as emissions of the latent chain, common theta.
(B) Proximal ("negative-control") regime-DML: with a negative-control outcome W = c*S + e (not caused by T), 2SLS of Y on
    (T, W) using the regime posterior gamma_hat (built from Z only) as instrument for W identifies theta without knowing
    the outcome regime shift.  Cross-fitted partialling-out of (1, X) with purged blocks.

    python src/bias_law/ext_joint_proximal.py            -> reports/bias_law/x1_joint_proximal.csv
"""
import os
import sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.bias_law import bias_law as BL                                       # noqa: E402
from src.bias_law.sim import (gen_states_proxy, gen_TY, fit_posteriors, align_to_truth, partial_out)  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'bias_law')
os.makedirs(OUT, exist_ok=True)


# ----------------------------------------------------------------------------------------------- (A) joint EM
def _fwd_bwd(logB, A, pi):
    N, K = logB.shape
    m = logB.max(1, keepdims=True)
    B = np.exp(logB - m)
    al = np.zeros((N, K)); be = np.ones((N, K)); c = np.zeros(N)
    a = pi * B[0]; c[0] = a.sum(); al[0] = a / c[0]
    for t in range(1, N):
        a = (al[t - 1] @ A) * B[t]; c[t] = a.sum(); al[t] = a / c[t]
    for t in range(N - 2, -1, -1):
        be[t] = (A @ (B[t + 1] * be[t + 1])) / c[t + 1]
    g = al * be; g /= g.sum(1, keepdims=True)
    xi = np.zeros((K, K))
    for t in range(N - 1):
        x = al[t][:, None] * A * (B[t + 1] * be[t + 1])[None, :]
        xi += x / x.sum()
    return g, xi, float(np.sum(np.log(c)) + m.sum())


def joint_em(Y, T, X, Z, g0, iters=40):
    """Markov-switching regression; K=2.  Z|S~N(mu_S, Sig) ; T|X,S~N(X b + a_S, sT2) ; Y|T,X,S~N(theta T + X d + g_S, sY2)."""
    N = len(Y); K = 2
    g = g0.copy()
    A = np.array([[.9, .1], [.1, .9]]); pi = np.array([.5, .5])
    ll_old = -np.inf
    for it in range(iters):
        # M-step
        Nk = g.sum(0)
        mu = (g.T @ Z) / Nk[:, None]
        Sig = sum(((Z - mu[k]).T * g[:, k]) @ (Z - mu[k]) for k in range(K)) / N
        D = np.vstack([np.column_stack([X, np.eye(K)[k][None, :].repeat(N, 0)]) for k in range(K)])
        w = np.concatenate([g[:, k] for k in range(K)])
        sw = np.sqrt(w)[:, None]
        bT = np.linalg.lstsq(D * sw, np.tile(T, K) * sw[:, 0], rcond=None)[0]
        sT2 = float(np.sum(w * (np.tile(T, K) - D @ bT) ** 2) / N)
        DY = np.column_stack([np.tile(T, K), D])
        bY = np.linalg.lstsq(DY * sw, np.tile(Y, K) * sw[:, 0], rcond=None)[0]
        sY2 = float(np.sum(w * (np.tile(Y, K) - DY @ bY) ** 2) / N)
        # E-step
        Si = np.linalg.inv(Sig); ld = np.linalg.slogdet(Sig)[1]
        logB = np.zeros((N, K))
        for k in range(K):
            dz = Z - mu[k]
            logB[:, k] = (-0.5 * np.einsum('ij,jk,ik->i', dz, Si, dz) - 0.5 * ld
                          - 0.5 * (T - X @ bT[:2] - bT[2 + k]) ** 2 / sT2 - 0.5 * np.log(sT2)
                          - 0.5 * (Y - bY[0] * T - X @ bY[1:3] - bY[3 + k]) ** 2 / sY2 - 0.5 * np.log(sY2))
        g, xi, ll = _fwd_bwd(logB, A, pi)
        A = xi / xi.sum(1, keepdims=True); pi = g[0]
        if abs(ll - ll_old) < 1e-4:
            break
        ll_old = ll
    joint_em.last_ll = ll_old if np.isfinite(ll_old) else ll
    return float(bY[0]), g


# ------------------------------------------------------------------------------------------- (B) proximal 2SLS
def _resid_X(M, X, seed, nuis='lin'):
    """Cross-fitted linear partialling-out of (1,X) from columns of M, purged blocks."""
    from src.or_dml import PurgedBlockKFold
    from src.bias_law.sim import _mk
    M = np.atleast_2d(M.T).T
    out = np.zeros_like(M, float)
    for tr, va in PurgedBlockKFold(5, 24).split(len(M)):
        for j in range(M.shape[1]):
            out[va, j] = M[va, j] - _mk(nuis, seed).fit(X[tr], M[tr, j]).predict(X[va])
    return out


def proximal_2sls(Y, T, W, X, gam, seed=0, nuis='lin'):
    """theta from 2SLS of Y~ on [T~, W~] with instruments [T~, gam~]; gam = posterior prob of state 1 (from Z only)."""
    R = _resid_X(np.column_stack([Y, T, W, gam]), X, seed, nuis)
    Yt, Tt, Wt, Gt = R.T
    Rg = np.column_stack([Tt, Wt]); I = np.column_stack([Tt, Gt])
    P = I @ np.linalg.solve(I.T @ I, I.T @ Rg)
    beta = np.linalg.solve(P.T @ Rg, P.T @ Yt)
    u = Yt - Rg @ beta
    # sandwich SE (iid approx, HAC not needed for point-estimate comparison)
    Q = np.linalg.inv(P.T @ Rg)
    V = Q @ (P.T * u ** 2) @ P @ Q.T
    return float(beta[0]), float(np.sqrt(V[0, 0])), float(beta[1])


def one(args):
    scen, dT, dg, dz, c, rep, N = args
    rng = np.random.default_rng(10_000 + 97 * rep + int(100 * dT) + hash(scen) % 997)
    Ktrue = 3 if scen == 'K3true' else 2
    S, Z, X = gen_states_proxy(N, Ktrue, dz, 0.9, rng)
    if scen == 'heavyZ':                                              # heavy-tailed (t3) proxy noise, same location shifts
        Z = Z - (Z - np.where(S[:, None] == 1, 1, -1) * 0) * 0 + 0.0
        mu = np.where(S[:, None] == 1, [dz, 1.5 * dz], [-dz, -1.5 * dz])
        Z = mu + rng.standard_t(3, size=(N, 2)) / np.sqrt(3)
    V, U, E = rng.normal(size=N), rng.normal(size=N), rng.normal(size=N)
    Sb = (S == 1).astype(int) if Ktrue == 2 else (S > 0).astype(int)   # K3: states 1,2 share the shift
    fx = lambda X: (np.sin(2 * X[:, 0]) + 0.5 * X[:, 1] ** 2) if scen == 'nonlinX' else (0.5 * X[:, 0] - 0.3 * X[:, 1])
    T = fx(X) + dT * Sb + V
    th = np.where(Sb == 1, 2.0, 1.0) if scen == 'hetero' else np.ones(N)
    Y = th * T + dg * Sb + 0.3 * X[:, 1] + U
    W = c * Sb + 0.3 * X[:, 0] + E + (0.3 * T if scen == 'leakW' else 0.0)   # negative control (leak = exclusion violation)
    nuis = 'gbm' if scen == 'nonlinX' else 'lin'
    post = fit_posteriors(Z, 2, rep)
    g = align_to_truth(post['smooth'], Sb)
    th_dml, _, _ = partial_out(Y, T, np.column_stack([X, g[:, 1]]), nuis, rep)
    v_hat = float(np.mean(g[:, 1] * (1 - g[:, 1])))
    th_j, _ = joint_em(Y, T, X, Z, g)
    th_p, se_p, bw = proximal_2sls(Y, T, W, X, g[:, 1], rep, nuis)
    tgt = float(np.mean(th))
    return dict(scen=scen, dT=dT, dg=dg, dz=dz, c=c, rep=rep, v_hat=v_hat, law=BL.law_bias(dT, dg, v_hat),
                bias_dml=th_dml - tgt, bias_joint=th_j - tgt, bias_prox=th_p - tgt, se_prox=se_p)


def main(R=12, N=3000, dz=0.5, dg=3.0, c=2.0, scens=('base', 'nonlinX', 'K3true', 'heavyZ', 'hetero', 'leakW')):
    jobs = [(sc, dT, dg, dz, c, r, N) for sc in scens for dT in (1, 4) for r in range(R)]
    with Pool(2) as p:
        rows = []
        for i, r in enumerate(p.imap_unordered(one, jobs), 1):
            rows.append(r)
            if i % 12 == 0:
                print(i, '/', len(jobs), flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, 'x2_misspec_raw.csv'), index=False)
    rm = lambda x: np.sqrt(np.mean(x ** 2))
    g = d.groupby(['scen', 'dT']).agg(dml=('bias_dml', 'mean'), joint=('bias_joint', 'mean'), prox=('bias_prox', 'mean'),
                                      rmse_dml=('bias_dml', rm), rmse_joint=('bias_joint', rm), rmse_prox=('bias_prox', rm))
    g.to_csv(os.path.join(OUT, 'x2_misspec.csv'))
    print(g.round(3).to_string())


if __name__ == '__main__':
    main()
