"""ext_switchrate_k3.py -- does v ~ (1-rho) tau(d^2) carry over to K=3 (nearest-pair d^2) and to heavy-tailed (t5) 1-D proxies with the true density?  -> x13_switchrate_ext.csv"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
from scipy.stats import t as student
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.bias_law.ext_joint_proximal import _fwd_bwd          # noqa: E402
OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'bias_law')


def run(K, rho, mus, noise, N=40000, seed=0):
    rng = np.random.default_rng(seed)
    S = np.zeros(N, int)
    for i in range(1, N):
        S[i] = S[i - 1] if rng.random() < rho else rng.choice([k for k in range(K) if k != S[i - 1]])
    if noise == 'gauss':
        z = mus[S] + rng.normal(size=N); logB = np.stack([-0.5 * (z - m) ** 2 for m in mus], 1)
    else:
        df = 5; sc = np.sqrt((df - 2) / df)                         # unit variance t5
        z = mus[S] + sc * rng.standard_t(df, size=N); logB = np.stack([student.logpdf((z - m) / sc, df) for m in mus], 1)
    A = np.full((K, K), (1 - rho) / (K - 1)); np.fill_diagonal(A, rho)
    g, _, _ = _fwd_bwd(logB, A, np.ones(K) / K)
    return float(np.mean(1 - (g ** 2).sum(1))) / 2 * (2 if K == 2 else 1)          # E[sum_k gamma_k(1-gamma_k)] / 2 = misclassification-mass proxy


rows = []
for noise in ('gauss', 't5'):
    for K, spacing in ((2, None), (3, None)):
        for dmin in (2.5, 3.3, 4.1):
            mus = np.array([-dmin / 2, dmin / 2]) if K == 2 else np.array([-dmin, 0.0, dmin])
            for rho in (0.9, 0.95):
                v = np.mean([run(K, rho, mus, noise, seed=s) for s in range(2)])
                rows.append(dict(noise=noise, K=K, d2_min=dmin ** 2, rho=rho, v=v, tau=v / (1 - rho)))
d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, 'x13_switchrate_ext.csv'), index=False)
print(d.round(4).to_string())
