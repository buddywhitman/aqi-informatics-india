"""ext_asym.py -- asymmetric two-state chains: is v ~ sqrt(8/pi) * (switch rate) * e^{-d^2/8}/d, switch rate = pi0*eps0 + pi1*eps1?  -> x14_asym.csv"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.bias_law.ext_joint_proximal import _fwd_bwd      # noqa: E402
OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'bias_law')
rows = []
for e0, e1 in ((0.05, 0.05), (0.02, 0.08), (0.01, 0.09), (0.03, 0.15), (0.005, 0.045)):
    pi1 = e0 / (e0 + e1); pi0 = 1 - pi1; rate = 2 * pi0 * e0
    for d2 in (6.2, 11.04, 17.25):
        d = np.sqrt(d2); vs = []
        for seed in range(3):
            rng = np.random.default_rng(seed); N = 60000
            S = np.zeros(N, int); S[0] = rng.random() < pi1
            u = rng.random(N)
            for t in range(1, N):
                S[t] = (u[t] < e0) if S[t - 1] == 0 else (u[t] >= e1)
            z = np.where(S == 1, d / 2, -d / 2) + rng.normal(size=N)
            logB = np.stack([-0.5 * (z + d / 2) ** 2, -0.5 * (z - d / 2) ** 2], 1)
            g, _, _ = _fwd_bwd(logB, np.array([[1 - e0, e0], [e1, 1 - e1]]), np.array([pi0, pi1]))
            vs.append(np.mean(g[:, 1] * (1 - g[:, 1])))
        v = float(np.mean(vs)); pred = np.sqrt(8 / np.pi) * rate * np.exp(-d2 / 8) / d
        rows.append(dict(e0=e0, e1=e1, pi1=pi1, switch_rate=rate, d2=d2, v=v, pred=pred, ratio=pred / v))
d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, 'x14_asym.csv'), index=False); print(d.round(4).to_string())
