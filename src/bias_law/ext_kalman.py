"""
ext_kalman.py -- closed-form bias from *primitives* for a continuous latent confounder, and regime discretisation as a sieve.

Linear-Gaussian latent: u_t = phi u_{t-1} + sqrt(1-phi^2) w_t ; Z_t = h u_t + eps_t (h in R^2, eps ~ N(0,R)).
Conditioning on the Kalman-smoothed mean leaves residual variance P_s (smoother variance) => law with v = P_s:
    bias = dT*dg*P_s / (dT^2 P_s + sigma^2).
Discretising u into a K-state HMM posterior is a sieve: bias(K) follows the law with v_K = Var(u | X, gamma_hat_K) and decreases in K.
    python src/bias_law/ext_kalman.py -> reports/bias_law/x6_kalman.csv, x6_sieve.csv
"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.bias_law.sim import fit_posteriors, partial_out                  # noqa: E402
from src.bias_law import bias_law as BL                                    # noqa: E402
OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'bias_law')
H = np.array([1.0, 1.5]); R = np.array([[1, .2], [.2, 1]])


def kalman_smoother(Z, phi, snr):
    N = len(Z); h = snr * H; Ri = np.linalg.inv(R)
    m = np.zeros(N); P = np.zeros(N); mp = np.zeros(N); Pp = np.zeros(N)
    mm, PP = 0.0, 1.0
    for t in range(N):
        mp[t], Pp[t] = (phi * mm, phi ** 2 * PP + 1 - phi ** 2) if t else (0.0, 1.0)
        info = h @ Ri @ h
        Pt = 1.0 / (1.0 / Pp[t] + info)
        mm = Pt * (mp[t] / Pp[t] + h @ Ri @ Z[t]); PP = Pt
        m[t], P[t] = mm, PP
    ms, Ps = m.copy(), P.copy()
    for t in range(N - 2, -1, -1):
        J = P[t] * phi / Pp[t + 1]
        ms[t] = m[t] + J * (ms[t + 1] - mp[t + 1]); Ps[t] = P[t] + J ** 2 * (Ps[t + 1] - Pp[t + 1])
    return ms, Ps


def sim(rep, N, phi, snr, dT, dg):
    rng = np.random.default_rng(31000 + 13 * rep + int(100 * dT) + int(1000 * snr) + int(1000 * phi))
    u = np.zeros(N); u[0] = rng.normal()
    for t in range(1, N):
        u[t] = phi * u[t - 1] + np.sqrt(1 - phi ** 2) * rng.normal()
    X = rng.normal(size=(N, 2))
    Z = snr * np.outer(u, H) + rng.multivariate_normal([0, 0], R, size=N)
    T = 0.5 * X[:, 0] - 0.3 * X[:, 1] + dT * u + rng.normal(size=N)
    Y = T + dg * u + 0.3 * X[:, 1] + rng.normal(size=N)
    return u, X, Z, T, Y


def one_kal(a):
    phi, snr, dT, dg, rep, N = a
    u, X, Z, T, Y = sim(rep, N, phi, snr, dT, dg)
    ms, Ps = kalman_smoother(Z, phi, snr)
    th, _, _ = partial_out(Y, T, np.column_stack([X, ms]), 'lin', rep)
    v_theory = float(Ps.mean())
    v_real = float(np.mean((u - ms) ** 2))
    return dict(phi=phi, snr=snr, dT=dT, rep=rep, bias=th - 1, law_theory=BL.law_bias(dT, dg, v_theory), law_real=BL.law_bias(dT, dg, v_real), v_theory=v_theory)


def one_sieve(a):
    K, phi, snr, dT, dg, rep, N = a
    u, X, Z, T, Y = sim(rep, N, phi, snr, dT, dg)
    g = fit_posteriors(Z, K, rep)['smooth']
    F = np.column_stack([X, g[:, 1:]])
    th, _, _ = partial_out(Y, T, F, 'lin', rep)
    Fc = np.column_stack([np.ones(N), F]); ut = u - Fc @ np.linalg.lstsq(Fc, u, rcond=None)[0]
    v = float(np.mean(ut ** 2))
    return dict(K=K, rep=rep, bias=th - 1, v_resid=v, law=BL.law_bias(dT, dg, v))


if __name__ == '__main__':
    R_ = 8
    jobs = [(phi, snr, dT, 3.0, r, 3000) for phi in (0.9, 0.97) for snr in (0.5, 1.0, 2.0) for dT in (0.5, 1.0, 2.0) for r in range(R_)]
    with Pool(2) as p:
        d = pd.DataFrame(list(p.imap_unordered(one_kal, jobs)))
    g = d.groupby(['phi', 'snr', 'dT']).mean(numeric_only=True).drop(columns='rep'); g.to_csv(os.path.join(OUT, 'x6_kalman.csv'))
    print(g.round(3).to_string())
    print('corr(bias, law_theory)=%.4f  max|bias-law_theory|=%.3f' % (np.corrcoef(g.bias, g.law_theory)[0, 1], (g.bias - g.law_theory).abs().max()))
    jobs = [(K, 0.97, 1.0, 1.0, 3.0, r, 3000) for K in (2, 3, 4, 6) for r in range(6)]
    with Pool(2) as p:
        s = pd.DataFrame(list(p.imap_unordered(one_sieve, jobs)))
    s = s.groupby('K').mean(numeric_only=True).drop(columns='rep'); s.to_csv(os.path.join(OUT, 'x6_sieve.csv')); print(s.round(3))
