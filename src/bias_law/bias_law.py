"""
bias_law.py
===========
Residual-state bias law for soft conditioning on a latent-regime posterior.

Model (population, see paper Sec. 3):
    T = m(X) + a' S + V,      Y = theta(S) T + g0(X) + b' S + U
with S the (K-1)-dim indicator vector of the latent regime (state 0 is the baseline),
V independent of (S, Z, X), Var(V) = sigma2.  After partialling out E[. | X, gamma_hat]:

    theta_hat - theta  ->  a' Sigma b / (a' Sigma a + sigma2),
    Sigma = E Cov(S | X, gamma_hat)   (residual state covariance).

Equivalently  bias = a' Sigma b / Var(T_tilde)  where Var(T_tilde) is the measured residual
treatment variance (the scalar analogue of lambda_min(J)).  Only b ("Delta_g") is unidentified.
"""
import numpy as np


# ----------------------------------------------------------------------------- closed forms
def law_bias(dT, dg, v, sigma2=1.0):
    """Scalar (K=2) residual-state bias  dT*dg*v / (dT^2 v + sigma2)."""
    return dT * dg * v / (dT ** 2 * v + sigma2)


def law_bias_hetero(dT, dg, theta1, theta_bar, v, sigma2=1.0, dtheta=0.0, m_bar=0.0):
    """Excess bias over the state-averaged effect theta_bar = E theta(S) when theta(S) is heterogeneous (K=2).

    excess = a v [ b + a (theta1 - theta_bar) + m_bar * dtheta ] / (a^2 v + sigma2),   a = dT, b = dg,
    dtheta = theta1 - theta0, m_bar = E[T] - a E[S] is the state-free mean level of the treatment.
    (Assumes the X-part of T is uncorrelated with Var(S | X, gamma_hat).)
    """
    return dT * v * (dg + dT * (theta1 - theta_bar) + m_bar * dtheta) / (dT ** 2 * v + sigma2)


def law_bias_matrix(a, b, Sigma, sigma2=1.0):
    """K-state version: a'Sigma b / (a'Sigma a + sigma2)."""
    a = np.atleast_1d(a).astype(float)
    b = np.atleast_1d(b).astype(float)
    Sigma = np.atleast_2d(Sigma)
    return float(a @ Sigma @ b / (a @ Sigma @ a + sigma2))


def peak_dT(v, sigma2=1.0):
    """Treatment shift at which |bias| is maximal: sigma / sqrt(v)."""
    return float(np.sqrt(sigma2 / v))


def sup_bias(dg, v, sigma2=1.0):
    """sup over dT of |bias| = |dg| sqrt(v) / (2 sigma)."""
    return float(abs(dg) * np.sqrt(v / sigma2) / 2.0)


def calibration_gap_bound(ce):
    """|v_hat - v| <= sqrt(CE) + CE,  CE = E[(gamma_hat - E[S | gamma_hat])^2] (L2 calibration error)."""
    return float(np.sqrt(ce) + ce)


# ----------------------------------------------------------------------------- observable plug-ins
def posterior_residual_cov(gamma):
    """Plug-in Sigma_hat = mean_t (diag(g_t) - g_t g_t')[1:,1:] ((K-1)x(K-1)); equals mean g(1-g) for K=2."""
    g = np.asarray(gamma)
    K = g.shape[1]
    S = np.mean(np.einsum('tk,tl->tkl', -g, g), axis=0) + np.diag(g.mean(axis=0))
    return S[1:, 1:]


def treatment_shift(T, X, gamma):
    """OLS coefficient(s) of gamma[:,1:] in a regression of T on [1, X, gamma[:,1:]] (estimate of a)."""
    N = len(T)
    F = np.column_stack([np.ones(N), X, gamma[:, 1:]])
    coef = np.linalg.lstsq(F, T, rcond=None)[0]
    return coef[-(gamma.shape[1] - 1):]


def robustness_value(theta_hat, a_hat, Sigma_hat, var_Ttilde):
    """Smallest |b| (K=2: scalar Delta_g) for which the law's bias equals theta_hat (signed, same-sign direction).

    bias(b) = a' Sigma b / Var(T_tilde)  ->  b* = theta_hat * Var(T_tilde) / (a' Sigma 1)   (K=2).
    For K>2 the minimal-norm solution along direction Sigma a is returned as a scalar multiple c of Sigma a/||Sigma a||:
    |b*| = |theta_hat| Var(T_tilde) / ||Sigma a||.
    """
    a = np.atleast_1d(a_hat)
    Sigma = np.atleast_2d(Sigma_hat)
    s = Sigma @ a
    denom = float(np.linalg.norm(s))
    if denom < 1e-12:
        return np.inf
    return float(abs(theta_hat) * var_Ttilde / denom)


def wiener_smoother_var(phi, iota):
    """Steady-state smoothing error variance of a unit-variance AR(1) latent (coefficient phi) observed in white noise with
    per-step information iota = h' R^{-1} h:   v = (1-phi^2) / sqrt((1+phi^2+iota(1-phi^2))^2 - 4 phi^2).
    Slow-state limit: v ~ 0.5*sqrt((1-phi^2)/iota)  (so sup-bias ~ iota^{-1/4})."""
    import numpy as _np
    A = 1 + phi ** 2 + iota * (1 - phi ** 2)
    return (1 - phi ** 2) / _np.sqrt(A ** 2 - 4 * phi ** 2)


def sensor_information(h, R, snr=1.0):
    import numpy as _np
    h = snr * _np.asarray(h, float)
    return float(h @ _np.linalg.solve(_np.asarray(R, float), h))
