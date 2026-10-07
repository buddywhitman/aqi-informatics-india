"""ext_singlechange.py -- isolated-change-point limit: posterior mass M(d) = E sum_t F_t(1-F_t), F_t = P(change <= t | window of obs), two-sided
random-walk posterior (uniform prior over change location).  Checks M(d) d e^{d^2/8} -> sqrt(8/pi) and the nearest-neighbour share.  -> x15_singlechange.csv"""
import os, numpy as np, pandas as pd
OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'bias_law')
rng = np.random.default_rng(0)
L = 14; T = np.arange(-L, L)                # sites t in [-L, L-1]; truth: state 0 for t<0, state 1 for t>=0
cands = np.arange(-L, L + 1)               # change location k: state 1 for t>=k  (k=-L: all 1; k=L: all 0)
rows = []
for d in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0):
    R = 200000 if d < 5 else 1000000
    z = np.where(T >= 0, d / 2, -d / 2)[None, :] + rng.normal(size=(R, len(T)))      # unit-variance emissions, means +-d/2
    ell = d * z                                                                      # LLR for means +-d/2: log p1/p0 = d*z
    # log-lik of hypothesis k: sum_{t>=k} ell_t/2 - sum_{t<k} ell_t/2  (up to const)
    cs = np.concatenate([np.zeros((R, 1)), np.cumsum(ell, axis=1)], axis=1)          # cs[:, j] = sum_{t<j} ell
    tot = cs[:, -1:]
    idx = cands + L                                                                  # number of sites before k
    logw = 0.5 * (tot - cs[:, idx]) - 0.5 * cs[:, idx]
    w = np.exp(logw - logw.max(1, keepdims=True)); p = w / w.sum(1, keepdims=True)
    F = np.stack([p[:, cands <= t].sum(1) for t in T], 1)
    contrib = F * (1 - F)
    M = contrib.sum(1).mean(); nn = (contrib[:, L - 1] + contrib[:, L]).mean()       # t=-1 and t=0
    rows.append(dict(d=d, M=M, M_d_exp=M * d * np.exp(d * d / 8), nn_share=nn / M, nn_const=nn * d * np.exp(d * d / 8)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'x15_singlechange.csv'), index=False)
print('sqrt(8/pi)=%.4f  sqrt(pi/2)=%.4f' % (np.sqrt(8 / np.pi), np.sqrt(np.pi / 2)))
