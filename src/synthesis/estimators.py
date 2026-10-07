"""
estimators.py -- compact reference implementations shared by the synthesis experiments.

All estimators take the same inputs (Y, T, X and a weight/posterior matrix W) and use purged contiguous
block cross-fitting with Ridge nuisances, so that differences between them are differences of design only.

  single_index(F, T, Y, folds)         pooled partially-linear DML with nuisance features F
                                       (F = X: standard/block DML; F = (X, Z): proxy controls;
                                        F = (X, gamma): posterior-feature DML)
  coupled_weighted(W, X, T, Y, folds)  legacy OR-DML: regime-k nuisances fitted on X with weights W[:, k]
  coupled_posterior(W, X, T, Y, folds) OR-DML with posterior-adjusted nuisances: E[T | X, W], E[Y | X, W]
                                       fitted unweighted on (X, W_{2:K}, W_{2:K} (x) X); exactly
                                       Neyman-orthogonal for any W (Section 3 of the paper)

Both coupled estimators solve (J + lam I) theta = S with
  J_jk = mean(W_j W_k T~_j T~_k),  S_j = mean(W_j T~_j Y~_j)
and return (theta, regime shares, HAC standard error of the share-weighted ATE, lambda_min(J)).
"""
import numpy as np
from sklearn.linear_model import Ridge


def purged(n, k=5, emb=24):
    """Contiguous blocks; training indices exclude an embargo of `emb` steps on each side of the test block."""
    e = np.linspace(0, n, k + 1).astype(int)
    out = []
    for i in range(k):
        te = np.arange(e[i], e[i + 1])
        tr = np.r_[0:max(0, e[i] - emb), min(n, e[i + 1] + emb):n]
        out.append((tr, te))
    return out


def hac(psi, L=24):
    """Newey-West long-run covariance of the rows of psi (n x d) with Bartlett weights."""
    psi = np.asarray(psi) - np.mean(psi, axis=0)
    n = len(psi)
    om = psi.T @ psi / n
    for l in range(1, min(L, n - 1) + 1):
        G = psi[l:].T @ psi[:-l] / n
        om += (1 - l / (L + 1)) * (G + G.T)
    return om


def _fit_predict(F, v, tr, te, w=None):
    m = Ridge(alpha=1.0).fit(F[tr], v[tr], sample_weight=None if w is None else w[tr])
    return v[te] - m.predict(F[te])


def single_index(F, T, Y, folds, L=24):
    rt, ry = np.zeros_like(T, dtype=float), np.zeros_like(Y, dtype=float)
    for tr, te in folds:
        rt[te] = _fit_predict(F, T, tr, te)
        ry[te] = _fit_predict(F, Y, tr, te)
    th = float(rt @ ry / (rt @ rt))
    psi = (rt * (ry - th * rt))[:, None]
    se = float(np.sqrt(hac(psi, L)[0, 0] / len(T)) / np.mean(rt ** 2))
    return th, se, float(np.mean(rt ** 2))


def _solve(W, tt, ty, lam_scale, L):
    n, K = W.shape
    J = np.array([[np.mean(W[:, j] * W[:, k] * tt[j] * tt[k]) for k in range(K)] for j in range(K)])
    Sv = np.array([np.mean(W[:, j] * tt[j] * ty[j]) for j in range(K)])
    lam = lam_scale * np.trace(J) / K / np.sqrt(n)
    Ji = np.linalg.pinv(J + lam * np.eye(K))
    th = Ji @ Sv
    fit = (W * tt.T) @ th
    psi = np.stack([W[:, k] * tt[k] * (ty[k] - fit) for k in range(K)], 1)
    Sig = Ji @ hac(psi, L) @ Ji / n
    share = W.mean(0)
    return th, share, float(np.sqrt(max(share @ Sig @ share, 1e-300))), float(np.linalg.eigvalsh(J).min()), Sig


def coupled_weighted(W, X, T, Y, folds, lam_scale=0.0, L=24, return_cov=False):
    n, K = W.shape
    tt, ty = np.zeros((K, n)), np.zeros((K, n))
    for k in range(K):
        w = np.maximum(W[:, k], 1e-4)
        for tr, te in folds:
            tt[k, te] = _fit_predict(X, T, tr, te, w)
            ty[k, te] = _fit_predict(X, Y, tr, te, w)
    out = _solve(W, tt, ty, lam_scale, L)
    return out if return_cov else out[:4]


def posterior_features(W, X):
    G = W[:, 1:]
    return np.column_stack([X, G] + [G[:, [k]] * X for k in range(G.shape[1])])


def coupled_posterior(W, X, T, Y, folds, lam_scale=0.0, L=24, return_cov=False):
    n, K = W.shape
    F = posterior_features(W, X)
    rt, ry = np.zeros(n), np.zeros(n)
    for tr, te in folds:
        rt[te] = _fit_predict(F, T, tr, te)
        ry[te] = _fit_predict(F, Y, tr, te)
    tt, ty = np.tile(rt, (K, 1)), np.tile(ry, (K, 1))
    out = _solve(W, tt, ty, lam_scale, L)
    return out if return_cov else out[:4]
