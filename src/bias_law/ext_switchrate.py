"""
ext_switchrate.py -- for a symmetric persistent binary chain with Gaussian emissions (Mahalanobis separation d), the exact-HMM residual
variance v = E[gamma(1-gamma)] scales linearly with the switching rate (1-rho) when the signal is strong:  v ~ (1-rho) tau(d).
    python src/bias_law/ext_switchrate.py -> reports/bias_law/x12_switchrate.csv
"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.bias_law.sim import gen_states_proxy                           # noqa: E402
from src.bias_law.ext_joint_proximal import _fwd_bwd                    # noqa: E402
OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'bias_law')
H = np.array([1, 1.5]); R = np.array([[1, .2], [.2, 1]]); Ri = np.linalg.inv(R)
rows = []
for dz in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
    d2 = 4 * dz ** 2 * float(H @ Ri @ H)
    for rho in (0.8, 0.9, 0.95, 0.97, 0.99):
        vs = []
        for rep in range(3):
            rng = np.random.default_rng(rep); N = 40000
            S, Z, X = gen_states_proxy(N, 2, dz, rho, rng)
            mu = np.array([[-dz, -1.5 * dz], [dz, 1.5 * dz]])
            logB = np.stack([-0.5 * np.einsum('ij,jk,ik->i', Z - mu[k], Ri, Z - mu[k]) for k in range(2)], 1)
            g, _, _ = _fwd_bwd(logB, np.array([[rho, 1 - rho], [1 - rho, rho]]), np.array([.5, .5]))
            vs.append(np.mean(g[:, 1] * (1 - g[:, 1])))
        v = float(np.mean(vs)); rows.append(dict(dz=dz, d2=d2, rho=rho, v=v, tau=v / (1 - rho)))
d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, 'x12_switchrate.csv'), index=False)
print(d.pivot(index='dz', columns='rho', values='tau').round(4).to_string())
t = d[d.rho == 0.95].sort_values('d2'); print(np.polyfit(t.d2, np.log(t.tau), 1))
