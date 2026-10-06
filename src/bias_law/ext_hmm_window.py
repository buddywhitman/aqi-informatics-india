"""ext_hmm_window.py -- variance-reduced check of the isolated-change constant with the exact HMM posterior.
Windows of length 2L with exactly one switch (centre), emissions N(+-d/2,1), HMM prior with switch prob eps (known).
M = E sum_t gamma_t(1-gamma_t) per change; const = M d exp(d^2/8).  Compare nearest-neighbour quadrature and sqrt(pi/2).
Also reports standard error.  -> x19_hmm_window.csv"""
import os, numpy as np, pandas as pd
from scipy import integrate
OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'bias_law')

def nn_quad(d):
    f = lambda w: np.exp(-(w + d * d / 2) ** 2 / (2 * d * d)) / np.sqrt(2 * np.pi * d * d) / (4 * np.cosh(w / 2) ** 2)
    return 2 * integrate.quad(f, -80, 80, limit=400)[0] * d * np.exp(d * d / 8)

def batch_M(d, eps, R, L, rng):
    T = 2 * L
    S = (np.arange(T) >= L).astype(int)
    z = np.where(S == 1, d / 2, -d / 2)[None, :] + rng.normal(size=(R, T))
    B = np.stack([np.exp(-0.5 * (z + d / 2) ** 2), np.exp(-0.5 * (z - d / 2) ** 2)], 2)   # R,T,2
    A = np.array([[1 - eps, eps], [eps, 1 - eps]])
    al = np.zeros((R, T, 2)); be = np.ones((R, T, 2))
    a = 0.5 * B[:, 0]; al[:, 0] = a / a.sum(1, keepdims=True)
    for t in range(1, T):
        a = (al[:, t - 1] @ A) * B[:, t]; al[:, t] = a / a.sum(1, keepdims=True)
    for t in range(T - 2, -1, -1):
        b = (B[:, t + 1] * be[:, t + 1]) @ A.T; be[:, t] = b / b.sum(1, keepdims=True)
    g = al * be; g = g[:, :, 1] / g.sum(2)
    return (g * (1 - g)).sum(1)

if __name__ == '__main__':
    rng = np.random.default_rng(1); rows = []
    for d in (2.0, 3.0, 4.0, 5.0):
        for eps in (1e-1, 1e-2, 1e-3, 1e-4):
            R = 400000 if d < 5 else 1500000
            M = batch_M(d, eps, R, 12, rng)
            k = d * np.exp(d * d / 8)
            rows.append(dict(d=d, eps=eps, R=R, const=M.mean() * k, se=M.std() / np.sqrt(R) * k, nn_quad=nn_quad(d), sqrt_pi_2=np.sqrt(np.pi / 2)))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'x19_hmm_window.csv'), index=False)
