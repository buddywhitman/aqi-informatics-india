"""
ext_h2_lemma_check.py -- numerical check of the ingredients of Lemma (smoother derivative bound), App. proofs.
 (a) Doeblin: the backward-conditioned (smoothing) kernel K_s(i,j) = A_ij w_j / sum_l A_il w_l, w_j = f_j(Z_{s+1}) beta_{s+1}(j),
     has Dobrushin coefficient <= 1 - eps_A  (eps_A = min A_ij), whatever the emissions.
 (b) ||d gamma_t / d mu|| / sum_s rho^{|t-s|} (1+|Z_s|^2) stays bounded as N grows (rho = 1-eps_A).
    python src/bias_law/ext_h2_lemma_check.py -> reports/bias_law/x32_h2_lemma_check.csv
"""
import os
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'bias_law')


def gen(N, rng, dz, rho):
    S = np.zeros(N, int)
    for t in range(1, N):
        S[t] = S[t - 1] if rng.random() < rho else 1 - S[t - 1]
    mu = np.array([[-dz, -1.5 * dz], [dz, 1.5 * dz]])
    return S, mu[S] + rng.normal(size=(N, 2)), mu


def logf(Z, mu):
    return np.stack([-0.5 * ((Z - mu[k]) ** 2).sum(1) - np.log(2 * np.pi) for k in range(2)], 1)


def smooth(Z, mu, A, pi):
    N = len(Z); lf = logf(Z, mu); f = np.exp(lf - lf.max(1, keepdims=True))
    al = np.zeros((N, 2)); be = np.ones((N, 2)); c = np.zeros(N)
    a = pi * f[0]; c[0] = a.sum(); al[0] = a / c[0]
    for t in range(1, N):
        a = (al[t - 1] @ A) * f[t]; c[t] = a.sum(); al[t] = a / c[t]
    for t in range(N - 2, -1, -1):
        be[t] = (A @ (f[t + 1] * be[t + 1])) / c[t + 1]
    g = al * be; g /= g.sum(1, keepdims=True)
    # smoothing kernel coefficients
    dob = np.zeros(N - 1)
    for s in range(N - 1):
        w = f[s + 1] * be[s + 1]
        K = A * w[None, :]; K /= K.sum(1, keepdims=True)
        dob[s] = abs(K[0, 0] - K[1, 0])
    return g[:, 1], dob


def run(N, dz, rho, seed=0):
    rng = np.random.default_rng(seed)
    S, Z, mu = gen(N, rng, dz, rho)
    A = np.array([[rho, 1 - rho], [1 - rho, rho]]); pi = np.array([.5, .5])
    g, dob = smooth(Z, mu, A, pi)
    eps = 1 - rho
    h = 1e-4
    mu2 = mu.copy(); mu2[1, 0] += h
    g2, _ = smooth(Z, mu2, A, pi)
    dg = np.abs(g2 - g) / h
    r = 1 - eps
    # envelope  sum_s r^{|t-s|} (1+|Z_s|^2)   (computed by two-sided exponential filter)
    e = 1 + (Z ** 2).sum(1)
    L = np.zeros(N); R = np.zeros(N)
    for t in range(1, N): L[t] = r * (L[t - 1] + e[t - 1])
    for t in range(N - 2, -1, -1): R[t] = r * (R[t + 1] + e[t + 1])
    env = L + R + e
    return dict(N=N, dz=dz, rho=rho, max_dobrushin=dob.max(), one_minus_eps=1 - eps, doeblin_ok=bool(dob.max() <= 1 - eps + 1e-12),
                max_dgamma=dg.max(), mean_dgamma=dg.mean(), max_ratio=(dg / env).max(), mean_sq=float((dg ** 2).mean()))


if __name__ == '__main__':
    rows = [run(N, dz, rho) for dz in (0.5, 1.2) for rho in (0.9, 0.97) for N in (2000, 8000, 32000)]
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, 'x32_h2_lemma_check.csv'), index=False)
    print(df.round(4).to_string())
