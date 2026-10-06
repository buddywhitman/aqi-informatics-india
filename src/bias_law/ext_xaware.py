"""
ext_xaware.py -- (i) linear-Wiener upper bound on v for a finite-state chain vs exact HMM v (x8_finite_state.csv);
(ii) X-aware v: when the controls X also carry information about the regime, the plug-in v_hat (posterior Gini of the Z-only HMM)
is conservative; an HMM on (Z, X) gives a tighter v_hat (x9_xaware.csv).
"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.bias_law import bias_law as BL                                                    # noqa: E402
from src.bias_law.sim import gen_states_proxy, gen_TY, fit_posteriors, align_to_truth, partial_out, residual_state_cov  # noqa: E402
from src.bias_law.ext_joint_proximal import _fwd_bwd                                        # noqa: E402
OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'bias_law')
H = np.array([1, 1.5]); R = np.array([[1, .2], [.2, 1]]); Ri = np.linalg.inv(R)


def finite_state():
    rows = []
    for rho in (0.9, 0.97):
        for dz in (0.25, 0.5, 1.0, 2.0):
            vs = []
            for rep in range(3):
                rng = np.random.default_rng(rep)
                S, Z, X = gen_states_proxy(20000, 2, dz, rho, rng)
                mu = np.array([[-dz, -1.5 * dz], [dz, 1.5 * dz]])
                logB = np.stack([-0.5 * np.einsum('ij,jk,ik->i', Z - mu[k], Ri, Z - mu[k]) for k in range(2)], 1)
                g, _, _ = _fwd_bwd(logB, np.array([[rho, 1 - rho], [1 - rho, rho]]), np.array([.5, .5]))
                vs.append(np.mean(g[:, 1] * (1 - g[:, 1])))
            iota = dz ** 2 * float(H @ Ri @ H)
            rows.append(dict(rho=rho, dz=dz, v_exact=float(np.mean(vs)), v_wiener_bound=0.25 * BL.wiener_smoother_var(2 * rho - 1, iota)))
    d = pd.DataFrame(rows); d['ratio'] = d.v_wiener_bound / d.v_exact
    d.to_csv(os.path.join(OUT, 'x8_finite_state.csv'), index=False); print(d.round(4).to_string())


def xaware():
    rows = []
    for rep in range(8):
        for xs in (0.0, 1.0):
            rng = np.random.default_rng(900 + rep); N = 3000
            S, Z, X = gen_states_proxy(N, 2, 0.5, 0.9, rng)
            X = X.copy(); X[:, 0] = xs * S + rng.normal(size=N)                       # X0 informative about the regime
            V, U = rng.normal(size=N), rng.normal(size=N)
            T, Y = gen_TY(S, X, [0, 2.0], [0, 3.0], [1, 1], (V, U))
            g = align_to_truth(fit_posteriors(Z, 2, rep)['smooth'], S)
            F = np.column_stack([X, g[:, 1]])
            th, _, _ = partial_out(Y, T, F, 'lin', rep)
            v_hat = float(np.mean(g[:, 1] * (1 - g[:, 1])))
            v_true = float(residual_state_cov(S, F, 2, 'lin', rep)[0, 0])
            gx = align_to_truth(fit_posteriors(np.column_stack([Z, X[:, 0]]), 2, rep)['smooth'], S)
            v_hatX = float(np.mean(gx[:, 1] * (1 - gx[:, 1])))
            rows.append(dict(xs=xs, rep=rep, bias=th - 1, v_hat=v_hat, v_true=v_true, v_hatX=v_hatX,
                             law_vhat=BL.law_bias(2.0, 3.0, v_hat), law_vtrue=BL.law_bias(2.0, 3.0, v_true), law_vhatX=BL.law_bias(2.0, 3.0, v_hatX)))
    d = pd.DataFrame(rows).groupby('xs').mean(numeric_only=True).drop(columns='rep')
    d.to_csv(os.path.join(OUT, 'x9_xaware.csv')); print(d.round(3).to_string())


if __name__ == '__main__':
    finite_state(); xaware()
