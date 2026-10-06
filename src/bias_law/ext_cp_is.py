"""ext_cp_is.py -- total isolated-change constant by importance sampling (closes the 'remainder not bounded' gap numerically).
Single switch at 0, unit-variance Gaussian emissions with means +-d/2, uniform prior on the switch location k in [-J,J].
Log-likelihood ratio of location k vs 0 is a sum of |k| iid N(-d^2/2, d^2) steps on each side (independent walks).
M(d)=E sum_t gamma_t(1-gamma_t); const(d)=M d e^{d^2/8}.  Proposal for steps: N(m, d^2) with m=-d^2*tilt (weights exact).
Reports: total const, nearest-neighbour const (quadrature), remainder ratio, and a plain-MC cross-check at d<=4. -> x28_cp_total.csv"""
import os, numpy as np, pandas as pd
from scipy import integrate
OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'bias_law')
J = 10

def nn_quad(d):
    f = lambda w: np.exp(-(w + d * d / 2) ** 2 / (2 * d * d)) / np.sqrt(2 * np.pi * d * d) / (4 * np.cosh(w / 2) ** 2)
    return 2 * integrate.quad(f, -120, 120, limit=500, points=[-d * d / 2, 0])[0] * d * np.exp(d * d / 8)

def sample(d, R, tilt, rng):
    mu = -d * d / 2; m = -d * d * tilt
    means = np.full(J, mu); means[:2] = m                              # proposal differs from truth only in the first two steps of each walk
    x = rng.normal(means, d, size=(R, 2, J))                           # side 0: right walk (k>0), side 1: left walk (k<0)
    logw = ((-((x - mu) ** 2) + (x - means) ** 2) / (2 * d * d)).sum((1, 2))
    lw = logw
    Wr = np.cumsum(x[:, 0], 1); Wl = np.cumsum(x[:, 1], 1)             # W_k for k=1..J and k=-1..-J
    logp = np.concatenate([Wl[:, ::-1], np.zeros((R, 1)), Wr], 1)      # index j -> k=j-J
    p = np.exp(logp - logp.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    gam = np.cumsum(p, 1)[:, :-1]                                      # gamma_t for t=-J..J-1  (P(c<=t))
    M = (gam * (1 - gam)).sum(1)
    return M, lw

def est(d, R, tilt, seed):
    rng = np.random.default_rng(seed); M, lw = sample(d, R, tilt, rng)
    w = np.exp(lw - lw.max()); k = d * np.exp(d * d / 8)
    # self-normalised would bias; use exact weights (no normalisation) in log-safe form
    ww = np.exp(lw); val = (ww * M).mean() * k
    se = (ww * M).std() / np.sqrt(R) * k
    ess = ww.sum() ** 2 / (ww ** 2).sum()
    return val, se, ess

def gh_const(d, n=44, Jq=2):
    # tensor Gauss-Hermite over the first Jq steps of each walk (steps ~ N(-d^2/2, d^2)); farther steps contribute O(e^{-d^2/4}) relative.
    x, w = np.polynomial.hermite_e.hermegauss(n); w = w / w.sum()
    mu = -d * d / 2
    steps = mu + d * x
    grids = np.meshgrid(*([steps] * (2 * Jq)), indexing='ij'); W = np.meshgrid(*([w] * (2 * Jq)), indexing='ij')
    Rg = [g.ravel() for g in grids]; ww = np.prod([q.ravel() for q in W], axis=0)
    Wr = np.cumsum(np.stack(Rg[:Jq], 1), 1); Wl = np.cumsum(np.stack(Rg[Jq:], 1), 1)
    logp = np.concatenate([Wl[:, ::-1], np.zeros((len(ww), 1)), Wr], 1)
    p = np.exp(logp - logp.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    gam = np.cumsum(p, 1)[:, :-1]
    return float((ww * (gam * (1 - gam)).sum(1)).sum() * d * np.exp(d * d / 8))


if __name__ == '__main__':
    rows = []
    for d in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 24.0):
        best = (np.nan, np.nan, np.nan, np.nan)
        best = None
        for tilt in (0.5, 0.35, 0.25, 0.15, 0.05):
            v, se, ess = est(d, 400000, tilt, 1)
            if tilt == 0.5: plain = (v, se)
            if ess > 0.2 * 400000 * 0.05 and (best is None or se < best[1]): best = (v, se, ess, tilt)
        if best is None: best = (np.nan, np.nan, np.nan, np.nan)
        nn = nn_quad(d); ghq = gh_const(d)
        rows.append(dict(d=d, const_gh=ghq, remainder_gh=ghq / nn - 1, const_total=best[0], se=best[1], tilt=best[3], ess=best[2], const_plain=plain[0], se_plain=plain[1], c_nn=nn,
                         remainder_ratio=best[0] / nn - 1, limit=np.sqrt(np.pi / 2)))
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'x28_cp_total.csv'), index=False)
