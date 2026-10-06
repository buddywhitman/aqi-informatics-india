"""ext_oned_check.py -- 1-D Gaussian proxy (means +-d/2, unit variance) at the d^2 values of x12: formula / exact v.  -> x17_oned_check.csv"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.bias_law.ext_joint_proximal import _fwd_bwd      # noqa: E402
from src.bias_law import bias_law as BL                   # noqa: E402
OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'bias_law')
rows = []
for d2 in (2.76, 6.21, 11.04, 17.25, 24.84):
    d = np.sqrt(d2)
    for rho in (0.9, 0.95, 0.97):
        vs = []
        for seed in range(3):
            rng = np.random.default_rng(seed); N = 40000
            S = (np.cumsum(rng.random(N) < (1 - rho)) % 2).astype(int)
            z = np.where(S == 1, d / 2, -d / 2) + rng.normal(size=N)
            logB = np.stack([-0.5 * (z + d / 2) ** 2, -0.5 * (z - d / 2) ** 2], 1)
            g, _, _ = _fwd_bwd(logB, np.array([[rho, 1 - rho], [1 - rho, rho]]), np.array([.5, .5]))
            vs.append(np.mean(g[:, 1] * (1 - g[:, 1])))
        v = float(np.mean(vs)); rows.append(dict(d2=d2, rho=rho, v=v, formula=BL.v_changepoint(rho, d2), ratio=BL.v_changepoint(rho, d2) / v))
o = pd.DataFrame(rows); o.to_csv(os.path.join(OUT, 'x17_oned_check.csv'), index=False); print(o.round(4).to_string())
