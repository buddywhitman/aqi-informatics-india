"""
coupled_law_check.py
====================
Numerical checks of the two results that justify posterior-adjusted OR-DML (Section 3-4 of the paper).

(1) Lemma 1 (exact orthogonality). With nuisances E[T | X, gamma], E[Y | X, gamma] and score weights gamma,
    the derivative of the population coupled score in any nuisance direction delta(X, gamma) is zero; with
    posterior-weighted regime nuisances it is not.  We perturb the treatment nuisance by t * delta and report
    the finite-difference derivative of the mean score at the population target.

(2) Theorem 4 (residual-state law for posterior-adjusted designs).  For a calibrated posterior
    (S | X, gamma ~ Bernoulli(gamma)), T = m(X) + a S + V, Y = theta_S T + g(X) + b S + U with V exogenous,
    the population limit of OR-DML is theta* + J^{-1} b_gamma with
        J = E[w w^T (a^2 gamma(1-gamma) + sigma^2)],
        b_gamma = a E[w (b + dtheta (m(X) + a (1-gamma))) gamma(1-gamma)],   w = (1-gamma, gamma),
    and pooled posterior-feature DML is the case w = 1.  We compare the law with the estimator on N = 10^6
    draws for several (a, b, dtheta, sigma) configurations.

    python src/synthesis/coupled_law_check.py -> reports/synthesis/coupled_law_check.csv, orthogonality_check.csv
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'reports', 'synthesis')
os.makedirs(OUT, exist_ok=True)
N = 1_000_000
CONFIGS = [  # a, b, dtheta, sigma, m-slope, beta-shape
    (0.0, 3.0, 1.0, 1.0, 0.5, 0.6), (0.5, 3.0, 1.0, 1.0, 0.5, 0.6), (1.0, 3.0, 1.0, 1.0, 0.5, 0.6),
    (2.0, 3.0, 1.0, 1.0, 0.5, 0.6), (4.0, 3.0, 1.0, 1.0, 0.5, 0.6), (8.0, 3.0, 1.0, 1.0, 0.5, 0.6),
    (4.0, -2.0, 0.5, 0.7, 1.0, 0.4), (1.5, 3.0, 0.0, 1.0, 0.5, 0.6), (1.0, 3.0, 1.0, 1.0, 0.5, 2.0),
]


def draw(rng, a, b, dth, sig, mc, shape, th0=0.5):
    g = np.clip(rng.beta(shape, shape, N), 1e-6, 1 - 1e-6)
    S = (rng.random(N) < g).astype(float)                       # calibrated posterior
    X = rng.normal(size=N)
    T = mc * X + a * S + rng.normal(0, sig, N)
    Y = (th0 + dth * S) * T + 0.8 * X + b * S + rng.normal(size=N)
    return g, S, X, T, Y


def ols_resid(F, v):
    return v - F @ np.linalg.lstsq(F, v, rcond=None)[0]


def coupled(W, rt, ry):
    J = np.einsum('ni,nj,n->ij', W, W, rt ** 2) / len(rt)
    S = (W * (rt * ry)[:, None]).mean(0)
    return np.linalg.solve(J, S)


def main():
    rng = np.random.default_rng(20261007)
    rows = []
    for a, b, dth, sig, mc, shape in CONFIGS:
        g, S, X, T, Y = draw(rng, a, b, dth, sig, mc, shape)
        F = np.c_[np.ones(N), X, g, g * X]
        rt, ry = ols_resid(F, T), ols_resid(F, Y)
        W = np.c_[1 - g, g]
        est = coupled(W, rt, ry)
        v = g * (1 - g)
        c = b + dth * (mc * X + a * (1 - g))
        Jl = np.einsum('ni,nj,n->ij', W, W, a * a * v + sig ** 2) / N
        bl = (W * (a * c * v)[:, None]).mean(0)
        law = np.array([0.5, 0.5 + dth]) + np.linalg.solve(Jl, bl)
        pooled = float(rt @ ry / (rt @ rt))
        lam_pooled = (a * c * v).mean() / (a * a * v + sig ** 2).mean()
        theta_bar_w = float(((0.5 + dth * g) * (a * a * v + sig ** 2)).mean() / (a * a * v + sig ** 2).mean())
        rows.append(dict(a=a, b=b, dtheta=dth, sigma=sig, m_slope=mc, beta_shape=shape,
                         est_0=est[0], est_1=est[1], law_0=law[0], law_1=law[1],
                         max_abs_dev=float(np.abs(est - law).max()), bias_norm=float(np.linalg.norm(est - [0.5, 0.5 + dth])),
                         pooled_est=pooled, pooled_law=theta_bar_w + lam_pooled,
                         lam_min_J=float(np.linalg.eigvalsh(Jl).min()), lam_min_oracle=float(sig ** 2 * min(1 - g.mean(), g.mean()))))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, 'coupled_law_check.csv'), index=False)
    print(d.round(4).to_string())

    # (1) orthogonality: finite-difference derivative of the mean score in a nuisance direction
    a, b, dth, sig, mc, shape = 2.0, 3.0, 1.0, 1.0, 0.5, 0.6
    g, S, X, T, Y = draw(rng, a, b, dth, sig, mc, shape)
    W = np.c_[1 - g, g]
    delta = 1.0 + X ** 2 + g                                    # arbitrary direction in (X, gamma), non-zero mean
    out = []
    # posterior-adjusted: population nuisances by OLS on a rich basis, target theta_gamma from the coupled solve
    F = np.c_[np.ones(N), X, g, g * X]
    rt0, ry0 = ols_resid(F, T), ols_resid(F, Y)
    th = coupled(W, rt0, ry0)

    def score_pa(t):
        rt = rt0 - t * delta
        return (W * (rt * (ry0 - (W * rt[:, None]) @ th))[:, None]).mean(0)
    h = 1e-3
    Jpa = np.einsum('ni,nj,n->ij', W, W, rt0 ** 2) / N
    out.append(dict(design='posterior-adjusted', direction='treatment nuisance',
                    derivative_norm=float(np.linalg.norm((score_pa(h) - score_pa(-h)) / (2 * h))),
                    jacobian_norm=float(np.linalg.norm(Jpa, 2))))

    # posterior-weighted regime nuisances: m_k = weighted OLS of T on (1, X) with weights gamma_k
    Fx = np.c_[np.ones(N), X]
    tt0 = np.stack([T - Fx @ np.linalg.lstsq(Fx * np.sqrt(W[:, [k]]), T * np.sqrt(W[:, k]), rcond=None)[0] for k in range(2)])
    ty0 = np.stack([Y - Fx @ np.linalg.lstsq(Fx * np.sqrt(W[:, [k]]), Y * np.sqrt(W[:, k]), rcond=None)[0] for k in range(2)])
    J = np.einsum('ni,nj,in,jn->ij', W, W, tt0, tt0) / N
    Sv = np.array([(W[:, k] * tt0[k] * ty0[k]).mean() for k in range(2)])
    thw = np.linalg.solve(J, Sv)
    dX = 1.0 + X ** 2                                           # weighted nuisances are functions of X only

    def score_w(t):
        tt = tt0 - t * dX
        fit = (W * tt.T) @ thw
        return np.array([(W[:, k] * tt[k] * (ty0[k] - fit)).mean() for k in range(2)])
    out.append(dict(design='posterior-weighted', direction='treatment nuisance',
                    derivative_norm=float(np.linalg.norm((score_w(h) - score_w(-h)) / (2 * h))),
                    jacobian_norm=float(np.linalg.norm(J, 2))))
    o = pd.DataFrame(out)
    o['relative_to_jacobian'] = o.derivative_norm / o.jacobian_norm
    o.to_csv(os.path.join(OUT, 'orthogonality_check.csv'), index=False)
    print(o.to_string())


if __name__ == '__main__':
    main()
