"""ext_cond_decay.py -- decorrelation of the smoothed chain given all proxies (justifies the embargo argument for cross-fitting when the HMM is fit on all Z).
Given Z_{1:N}, S is an inhomogeneous Markov chain (backward sampling from the forward filter).  We measure corr(S_t - gamma_t, S_{t+h} - gamma_{t+h}) over
posterior draws, known parameters.  -> x22_cond_decay.csv"""
import os, numpy as np, pandas as pd
OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'bias_law')
def run(dz, rho, N=20000, ndraw=40, lags=(1, 2, 4, 8, 12, 24, 48), seed=0):
    rng = np.random.default_rng(seed)
    S = np.zeros(N, int)
    for t in range(1, N): S[t] = S[t - 1] if rng.random() < rho else 1 - S[t - 1]
    z = np.where(S == 1, dz, -dz) + rng.normal(size=N)
    A = np.array([[rho, 1 - rho], [1 - rho, rho]])
    B = np.exp(np.stack([-0.5 * (z + dz) ** 2, -0.5 * (z - dz) ** 2], 1))
    al = np.zeros((N, 2)); a = 0.5 * B[0]; al[0] = a / a.sum()
    for t in range(1, N): a = (al[t - 1] @ A) * B[t]; al[t] = a / a.sum()
    be = np.ones((N, 2))
    for t in range(N - 2, -1, -1): b = A @ (B[t + 1] * be[t + 1]); be[t] = b / b.sum()
    g = al * be; g /= g.sum(1, keepdims=True); g1 = g[:, 1]
    draws = np.zeros((ndraw, N))
    for m in range(ndraw):
        s = np.zeros(N, int); s[-1] = rng.random() < al[-1, 1]
        for t in range(N - 2, -1, -1):
            w = al[t] * A[:, s[t + 1]]; s[t] = rng.random() < w[1] / w.sum()
        draws[m] = s
    r = draws - g1[None, :]
    out = {}
    for h in lags:
        out[h] = float(np.mean(r[:, :-h] * r[:, h:]) / np.mean(r ** 2))
    return out
if __name__ == '__main__':
    rows = []
    for dz in (0.5, 0.8, 1.2):
        for rho in (0.9, 0.97):
            c = run(dz, rho); rows.append(dict(dz=dz, rho=rho, **{f'lag{h}': v for h, v in c.items()}))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'x22_cond_decay.csv'), index=False)
