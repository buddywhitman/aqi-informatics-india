"""ext_contlat_law.py -- law prediction for a continuous AR(1) latent confounder with a K=2 HMM posterior as the conditioning feature:
law(v) with v = Var(u | X, gamma_hat) (cross-fitted linear residual of the true latent; diagnostic only) vs observed DML bias.  -> x3b_contlat_law.csv"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.bias_law.sim import fit_posteriors, align_to_truth, partial_out       # noqa: E402
from src.bias_law import bias_law as BL                                          # noqa: E402
R = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'bias_law')
rows = []
for dT, snr in ((1.0, 0.5), (1.0, 1.0), (2.0, 0.5), (2.0, 1.0)):
    for rep in range(12):
        rng = np.random.default_rng(777 + 31 * rep + int(100 * dT)); N = 3000      # same seeds/DGP as ext_contlat.py
        u = np.zeros(N)
        for t in range(1, N):
            u[t] = 0.97 * u[t - 1] + np.sqrt(1 - 0.97 ** 2) * rng.normal()
        X = rng.normal(size=(N, 2))
        Z = snr * np.column_stack([u, 1.5 * u]) + rng.multivariate_normal([0, 0], [[1, .2], [.2, 1]], size=N)
        V, U, E = rng.normal(size=N), rng.normal(size=N), rng.normal(size=N)
        T = 0.5 * X[:, 0] - 0.3 * X[:, 1] + dT * u + V
        Y = T + 3.0 * u + 0.3 * X[:, 1] + U
        g = align_to_truth(fit_posteriors(Z, 2, rep)['smooth'], (u > 0).astype(int))
        F = np.column_stack([X, g[:, 1]])
        th, _, _ = partial_out(Y, T, F, 'lin', rep)
        Fc = np.column_stack([np.ones(N), F]); ut = u - Fc @ np.linalg.lstsq(Fc, u, rcond=None)[0]
        rows.append(dict(dT=dT, snr=snr, rep=rep, bias=th - 1, v_resid=float(np.mean(ut ** 2)), law=BL.law_bias(dT, 3.0, float(np.mean(ut ** 2)))))
d = pd.DataFrame(rows); d.to_csv(os.path.join(R, 'x3b_contlat_law_raw.csv'), index=False)
g = d.groupby(['dT', 'snr']).mean(numeric_only=True).drop(columns='rep'); g['rel_err'] = (g.law - g.bias) / g.bias
g.to_csv(os.path.join(R, 'x3b_contlat_law.csv')); print(g.round(4).to_string())
