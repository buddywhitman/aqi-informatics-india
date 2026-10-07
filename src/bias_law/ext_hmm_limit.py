"""ext_hmm_limit.py -- exact-HMM constant v/eps * d * exp(d^2/8) for a symmetric chain as eps -> 0 (1-D Gaussian emissions, means +-d/2).  -> x16_hmm_limit.csv
Derived asymptotic constant sqrt(pi/2) = 1.2533."""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.bias_law.ext_joint_proximal import _fwd_bwd      # noqa: E402
OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'bias_law')


def run(d, eps, N, seed=0):
    rng = np.random.default_rng(seed)
    S = (np.cumsum(rng.random(N) < eps) % 2).astype(int)
    z = np.where(S == 1, d / 2, -d / 2) + rng.normal(size=N)
    logB = np.stack([-0.5 * (z + d / 2) ** 2, -0.5 * (z - d / 2) ** 2], 1)
    g, _, _ = _fwd_bwd(logB, np.array([[1 - eps, eps], [eps, 1 - eps]]), np.array([.5, .5]))
    return np.mean(g[:, 1] * (1 - g[:, 1])) / eps * d * np.exp(d * d / 8)


if __name__ == '__main__':
    rows = []
    for d in (3.0, 4.0):
        for eps, N in ((0.1, 200000), (0.03, 300000), (0.01, 600000), (0.003, 1200000), (0.001, 1500000)):
            rows.append(dict(d=d, eps=eps, N=N, const_v_over_eps_d_exp=float(np.mean([run(d, eps, N, s) for s in range(2)]))))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'x16_hmm_limit.csv'), index=False)
