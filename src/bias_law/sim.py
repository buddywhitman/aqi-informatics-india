"""
sim.py -- simulation helpers for the residual-state bias law experiments.
Run from the repository root.
"""
import os
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for p in (ROOT, os.path.join(ROOT, 'src')):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.or_dml import LatentRegimeHMM, PurgedBlockKFold  # noqa: E402


def gen_states_proxy(N, K, dz, rho, rng):
    """Persistent K-state chain (stay prob rho, uniform exits), 2-d Gaussian proxy Z, 2-d AR-free controls X."""
    S = np.zeros(N, int)
    S[0] = rng.integers(K)
    for t in range(1, N):
        if rng.random() < rho:
            S[t] = S[t - 1]
        else:
            S[t] = rng.choice([k for k in range(K) if k != S[t - 1]])
    if K == 2:
        mu = np.array([[-dz, -1.5 * dz], [dz, 1.5 * dz]])
    else:
        ang = 2 * np.pi * np.arange(K) / K + np.pi / 2
        mu = 1.8 * dz * np.column_stack([np.cos(ang), np.sin(ang)])
    cov = np.array([[1.0, 0.2], [0.2, 1.0]])
    Z = mu[S] + rng.multivariate_normal([0, 0], cov, size=N)
    X = rng.normal(size=(N, 2))
    return S, Z, X


def gen_TY(S, X, dT, dg, theta, rng_noise):
    """T = 0.5 X0 - 0.3 X1 + dT[S] + V,  Y = theta[S] T + dg[S] + 0.3 X1 + U ; V,U ~ N(0,1)."""
    V, U = rng_noise
    dT = np.asarray(dT, float)
    dg = np.asarray(dg, float)
    theta = np.asarray(theta, float)
    T = 0.5 * X[:, 0] - 0.3 * X[:, 1] + dT[S] + V
    Y = theta[S] * T + dg[S] + 0.3 * X[:, 1] + U
    return T, Y


def fit_posteriors(Z, K, seed):
    """Fit a Gaussian HMM once; return dict of smoothing / filtering posteriors."""
    h = LatentRegimeHMM(n_regimes=K, random_state=seed).fit(Z)
    return {'smooth': h.predict_posteriors(Z, mode='smooth'), 'filter': h.predict_posteriors(Z, mode='filter')}


def align_to_truth(gamma, S):
    """Permute posterior columns to the true state labels (diagnostic only)."""
    K = gamma.shape[1]
    onehot = np.eye(K)[S]
    C = np.array([[np.corrcoef(gamma[:, j], onehot[:, k])[0, 1] if gamma[:, j].std() > 0 else 0 for k in range(K)]
                  for j in range(K)])
    row, col = linear_sum_assignment(-np.nan_to_num(C))
    out = np.zeros_like(gamma)
    out[:, col] = gamma[:, row]
    return out


def _mk(nuis, seed):
    if nuis == 'lin':
        return LinearRegression()
    if nuis == 'ridge':
        return Ridge(alpha=1.0)
    return HistGradientBoostingRegressor(max_iter=60, min_samples_leaf=20, random_state=seed)


def partial_out(Y, T, F, nuis='lin', seed=0, n_splits=5, tau=24):
    """Cross-fitted (purged block, embargo tau) partialling-out. Returns theta_hat, T_tilde, Y_tilde."""
    N = len(Y)
    tT = np.zeros(N)
    tY = np.zeros(N)
    for tr, va in PurgedBlockKFold(n_splits=n_splits, embargo_tau=tau).split(N):
        tT[va] = T[va] - _mk(nuis, seed).fit(F[tr], T[tr]).predict(F[va])
        tY[va] = Y[va] - _mk(nuis, seed).fit(F[tr], Y[tr]).predict(F[va])
    return float((tT @ tY) / (tT @ tT)), tT, tY


def residual_state_cov(S, F, K, nuis='lin', seed=0):
    """Realised v_N / Sigma_N: covariance of S_tilde = 1{S}[1:] - cross-fitted E[1{S}[1:] | F]."""
    N = len(S)
    oh = np.eye(K)[S][:, 1:]
    res = np.zeros_like(oh)
    for tr, va in PurgedBlockKFold(5, 24).split(N):
        for j in range(K - 1):
            res[va, j] = oh[va, j] - _mk(nuis, seed).fit(F[tr], oh[tr, j]).predict(F[va])
    return (res.T @ res) / N
