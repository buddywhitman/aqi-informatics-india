"""
ext_selfnc.py -- self-contained proximal regime-DML: the *lagged outcome* Y_{t-h} (adjusting for lagged treatments) serves as the
negative-control outcome, so no external variable is needed.  Time's arrow gives the exclusion (T_t cannot cause Y_{t-h}); relevance
comes from regime persistence.  Instruments: HMM posterior (and/or raw Z).  Stress: AR(1) idiosyncratic shocks, dynamic T_{t-1}->Y_t.
    python src/bias_law/ext_selfnc.py -> reports/bias_law/x4_selfnc.csv
"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.bias_law.sim import gen_states_proxy, fit_posteriors, align_to_truth, partial_out   # noqa: E402
from src.bias_law.ext_joint_proximal import joint_em                                          # noqa: E402
from src.bias_law.ext_contlat import prox_general                                              # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'bias_law')


def ar1(rng, N, phi):
    e = rng.normal(size=N); x = np.zeros(N)
    for t in range(1, N):
        x[t] = phi * x[t - 1] + np.sqrt(1 - phi ** 2) * e[t]
    return x


def lag(x, h):
    y = np.roll(x, h); y[:h] = np.nan; return y


def one(a):
    scen, dT, dg, dz, rho, h, rep, N = a
    rng = np.random.default_rng(4242 + 53 * rep + int(100 * dT) + hash(scen) % 991)
    S, Z, X = gen_states_proxy(N, 2, dz, rho, rng)
    phi = 0.5 if scen in ('ar', 'ar_dyn') else 0.0
    V, U = ar1(rng, N, phi), ar1(rng, N, phi)
    T = 0.5 * X[:, 0] - 0.3 * X[:, 1] + dT * S + V
    Y = 1.0 * T + dg * S + 0.3 * X[:, 1] + U
    if scen == 'ar_dyn':
        Y[1:] += 0.5 * T[:-1]
    W = lag(Y, h)
    ctrl = np.column_stack([X, lag(T, 1), lag(T, h), lag(T, h + 1), lag(Y, 1) * 0])
    ok = ~np.isnan(np.column_stack([W, ctrl])).any(1)
    Y, T, W, Z, X, ctrl, S = Y[ok], T[ok], W[ok], Z[ok], X[ok], ctrl[ok], S[ok]
    g = align_to_truth(fit_posteriors(Z, 2, rep)['smooth'], S)
    th_dml, _, _ = partial_out(Y, T, np.column_stack([ctrl, g[:, 1]]), 'lin', rep)
    th_j, _ = joint_em(Y, T, ctrl[:, :2], Z, g)
    th_p, se = prox_general(Y, T, W, ctrl, g[:, 1:2], rep)
    th_pz, _ = prox_general(Y, T, W, ctrl, Z, rep)
    return dict(scen=scen, dT=dT, rho=rho, h=h, rep=rep, dml=th_dml - 1, joint=th_j - 1, prox_gamma=th_p - 1, prox_Z=th_pz - 1, se=se)


if __name__ == '__main__':
    R = 10
    jobs = [(sc, dT, 3.0, 0.5, 0.95, h, r, 4000) for sc in ('iid', 'ar', 'ar_dyn') for dT in (1.0, 3.0) for h in (6, 24) for r in range(R)]
    with Pool(2) as p:
        rows = list(p.imap_unordered(one, jobs))
    d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, 'x4_selfnc_raw.csv'), index=False)
    rm = lambda x: np.sqrt(np.mean(x ** 2))
    g = d.groupby(['scen', 'dT', 'h']).agg(**{f'{k}_bias': (k, 'mean') for k in ('dml', 'joint', 'prox_gamma', 'prox_Z')},
                                           **{f'{k}_rmse': (k, rm) for k in ('dml', 'joint', 'prox_gamma', 'prox_Z')})
    g.to_csv(os.path.join(OUT, 'x4_selfnc.csv')); print(g.round(3).to_string())
