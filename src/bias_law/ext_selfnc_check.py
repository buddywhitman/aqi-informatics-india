"""ext_selfnc_check.py -- lagged-outcome negative control (Y_{t-h}) with instruments = smoothed HMM posterior vs raw Z_t, N=20000, h=4, dT=1.
Stores both columns that the paper's statement relies on.  -> x4c_selfnc_check.csv"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.bias_law.sim import gen_states_proxy, fit_posteriors, align_to_truth, partial_out    # noqa: E402
from src.bias_law.ext_selfnc import ar1, lag                                                   # noqa: E402
from src.bias_law.ext_contlat import prox_general                                               # noqa: E402
OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'bias_law')
rows = []
for scen in ('iid', 'ar'):
    for rep in range(4):
        N, h, dT, dg = 20000, 4, 1.0, 3.0
        rng = np.random.default_rng(9000 + rep)
        S, Z, X = gen_states_proxy(N, 2, 0.5, 0.95, rng)
        phi = 0.5 if scen == 'ar' else 0.0
        V, U = ar1(rng, N, phi), ar1(rng, N, phi)
        T = 0.5 * X[:, 0] - 0.3 * X[:, 1] + dT * S + V
        Y = T + dg * S + 0.3 * X[:, 1] + U
        W = lag(Y, h); ctrl = np.column_stack([X, lag(T, 1), lag(T, h), lag(T, h + 1)])
        ok = ~np.isnan(np.column_stack([W, ctrl])).any(1)
        Y, T, W, Z, ctrl, S = Y[ok], T[ok], W[ok], Z[ok], ctrl[ok], S[ok]
        g = align_to_truth(fit_posteriors(Z, 2, rep)['smooth'], S)
        th_dml, _, _ = partial_out(Y, T, np.column_stack([ctrl, g[:, 1]]), 'lin', rep)
        th_g, _ = prox_general(Y, T, W, ctrl, g[:, 1:2], rep)
        th_z, _ = prox_general(Y, T, W, ctrl, Z, rep)
        rows.append(dict(scen=scen, rep=rep, dml=th_dml - 1, prox_gamma=th_g - 1, prox_Z=th_z - 1)); print(rows[-1], flush=True)
d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, 'x4c_selfnc_check_raw.csv'), index=False)
g = d.groupby('scen').mean(numeric_only=True).drop(columns='rep'); g.to_csv(os.path.join(OUT, 'x4c_selfnc_check.csv')); print(g.round(3))
