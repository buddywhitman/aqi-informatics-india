"""
hetero_law_check.py -- rebuttal experiment E3.
Pooled residual-state law with regime-heteroscedastic treatment innovations (sigma_0 != sigma_1).
Derivation: with a calibrated posterior gamma, T~ = a(D-gamma) + V, E[V|D]=0, so
  E[T~ (b D~ + U)] = a b v,   E[T~^2] = a^2 v + E[sigma_S^2],   v = E[gamma(1-gamma)]
  => bias = a b v / (a^2 v + E[sigma_S^2]),   E[sigma_S^2] = E[gamma s1^2 + (1-gamma) s0^2].
Checks it by simulation (i.i.d. regimes, 1-d Gaussian proxy, exact posterior) and checks that the observable plug-in
Var(T~) - a_hat^2 v_hat recovers E[sigma_S^2].

    python src/rebuttal/hetero_law_check.py -> reports/rebuttal/hetero_law_check.csv
"""
import os
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'reports', 'rebuttal'); os.makedirs(OUT, exist_ok=True)


def run(s0, s1, sep, a=2.0, b=3.0, theta=1.0, N=300000, seed=0, pi1=0.2):
    rng = np.random.RandomState(seed)
    S = (rng.rand(N) < pi1).astype(int)
    Z = rng.randn(N) + sep * (2 * S - 1)
    X = rng.randn(N, 2)
    gam = 1 / (1 + np.exp(-(2 * sep * Z + np.log(pi1 / (1 - pi1)))))   # exact P(S=1|Z), unit-variance Gaussian, prior pi1
    V = rng.randn(N) * np.where(S == 1, s1, s0)
    T = 1.0 + 0.5 * X[:, 0] + a * S + V
    Y = theta * T + X[:, 1] + b * S + rng.randn(N)
    F = np.c_[X, gam]
    rt, ry = np.zeros(N), np.zeros(N)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(F):
        rt[te] = T[te] - Ridge(alpha=1e-6).fit(F[tr], T[tr]).predict(F[te])
        ry[te] = Y[te] - Ridge(alpha=1e-6).fit(F[tr], Y[tr]).predict(F[te])
    th = rt @ ry / (rt @ rt)
    v = np.mean(gam * (1 - gam))
    es2 = np.mean(gam * s1 ** 2 + (1 - gam) * s0 ** 2)
    law = a * b * v / (a * a * v + es2)
    homo = a * b * v / (a * a * v + 0.5 * (s0 ** 2 + s1 ** 2))  # naive: unweighted average of the two variances
    a_hat = LinearRegression().fit(np.c_[X, gam], T).coef_[-1]
    es2_hat = np.mean(rt ** 2) - a_hat ** 2 * v
    return dict(s0=s0, s1=s1, sep=sep, measured=th - theta, law_hetero=law, law_homo_avg_sigma=homo,
                Esig2_true=es2, Esig2_plugin=es2_hat)


if __name__ == '__main__':
    rows = []
    for (s0, s1) in [(1, 1), (0.5, 2), (0.3, 3)]:
        for sep in (0.5, 1.0):
            rows.append(run(s0, s1, sep, seed=int(10 * s1 + 100 * sep)))
    d = pd.DataFrame(rows)
    d['rel_err_hetero_%'] = 100 * (d.measured - d.law_hetero).abs() / d.measured.abs()
    d['rel_err_homo_%'] = 100 * (d.measured - d.law_homo_avg_sigma).abs() / d.measured.abs()
    d.to_csv(os.path.join(OUT, 'hetero_law_check.csv'), index=False)
    print(d.round(4).to_string())
