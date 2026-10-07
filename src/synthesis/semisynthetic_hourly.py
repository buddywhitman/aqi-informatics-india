"""
semisynthetic_hourly.py
=======================
Real-data-calibrated semi-synthetic benchmark on the corrected hourly grid of four Indian cities.

What is real (per city, from data/raw_hourly via ext_real_hourly_fix.build):
  * meteorological regime proxies Z_t (temperature, humidity, wind) and controls X_t (pressure, diurnal phase,
    lagged NO2): real
    seasonality, autocorrelation and cross-dependence;
  * the treatment's base series: observed NO2 (standardised);
  * the missingness pattern: rows where either real sensor (NO2 or PM2.5) is missing are absent from the analysis
    sample, so gaps, outages and the irregular sampling of the real panel are preserved;
  * regime persistence and overlap: a 2-state Gaussian HMM is calibrated on Z; in every replication the TRUE latent
    path S_{1:N} is drawn from that HMM's exact smoothing distribution by forward-filtering backward-sampling, so
    the regime durations are the calibrated ones and the true state is uncertain exactly where the calibrated
    posterior is uncertain (regime overlap is preserved, not removed).
What is injected (known ground truth):
  T_t = NO2std_t + dT * S_t,     Y_t = theta_{S_t} T_t + dG * S_t + X_t beta + U_t,   U_t AR(1) (rho = 0.8),
  theta = (0.5, 1.5), dT = 1.0, dG = 3.0 (outcome noise SD 1).
Estimators: oracle-state regime DML, standard DML (random folds), block DML (purged folds), DML + Z controls,
hard regime FE on argmax posteriors, posterior-feature DML (nuisances on (X, gamma)), spectral OR-DML with
posterior-weighted nuisances (legacy, ablation) and OR-DML with posterior-adjusted nuisances (the paper's estimator).
The posteriors handed to the estimators are the calibrated HMM's own smoothing posteriors, i.e. the most favourable
case for posterior-based methods (exact calibration).

    python src/synthesis/semisynthetic_hourly.py -> reports/synthesis/semisynthetic_hourly_{raw,summary}.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from hmmlearn import hmm
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from src.bias_law.ext_real_hourly_fix import build              # noqa: E402
from src.synthesis.estimators import coupled_posterior, hac            # noqa: E402

warnings.filterwarnings('ignore')
OUT = os.path.join(ROOT, 'reports', 'synthesis')
os.makedirs(OUT, exist_ok=True)
CITIES = ['Delhi', 'Mumbai', 'Bengaluru', 'Kolkata']
ZCOLS = ['temperature', 'humidity', 'wind_speed']                    # regime proxies (meteorology)
XCOLS = ['pressure', 'hour_sin', 'hour_cos', 'no2_lag_1h']           # observed controls
THETA = np.array([0.5, 1.5])
DT, DG, RHO_U = 1.0, 3.0, 0.8
N_REPS = 30


def std(a):
    return (a - a.mean(0)) / (a.std(0) + 1e-9)


def calibrate(city):
    g = build(city).dropna(subset=['no2', 'pm25'] + XCOLS + ZCOLS)
    Z, X = std(g[ZCOLS].values.astype(float)), std(g[XCOLS].values.astype(float))
    T0 = std(g.no2.values.astype(float))
    best, ll = None, -np.inf
    for i in range(5):
        m = hmm.GaussianHMM(n_components=2, covariance_type='diag', n_iter=200, tol=1e-4, random_state=42 + 101 * i)
        m.fit(Z)
        s = m.score(Z)
        if s > ll:
            best, ll = m, s
    gam = best.predict_proba(Z)
    order = np.argsort(best.means_[:, 1])[::-1]          # regime 1 = more humid (stagnant/monsoon) state
    gam = gam[:, order]
    logB = best._compute_log_likelihood(Z)[:, order]
    A = best.transmat_[order][:, order]
    pi0 = best.startprob_[order]
    return dict(city=city, Z=Z, X=X, T0=T0, gamma=gam, logB=logB, A=A, pi0=pi0, N=len(g),
                stay=np.diag(A).tolist())


def ffbs(rng, logB, A, pi0):
    N, K = logB.shape
    alpha = np.zeros((N, K))
    b = np.exp(logB - logB.max(1, keepdims=True))
    a = pi0 * b[0]
    alpha[0] = a / a.sum()
    for t in range(1, N):
        a = (alpha[t - 1] @ A) * b[t]
        alpha[t] = a / a.sum()
    S = np.zeros(N, dtype=int)
    S[-1] = rng.choice(K, p=alpha[-1])
    for t in range(N - 2, -1, -1):
        p = alpha[t] * A[:, S[t + 1]]
        S[t] = rng.choice(K, p=p / p.sum())
    return S


def purged(n, k=5, emb=24):
    e = np.linspace(0, n, k + 1).astype(int)
    for i in range(k):
        te = np.arange(e[i], e[i + 1])
        yield np.r_[0:max(0, e[i] - emb), min(n, e[i + 1] + emb):n], te


def single_index(X, T, Y, folds):
    rt, ry = np.zeros_like(T), np.zeros_like(Y)
    for tr, te in folds:
        rt[te] = T[te] - Ridge(alpha=1.0).fit(X[tr], T[tr]).predict(X[te])
        ry[te] = Y[te] - Ridge(alpha=1.0).fit(X[tr], Y[tr]).predict(X[te])
    th = (rt @ ry) / (rt @ rt)
    psi = (rt * (ry - th * rt))[:, None]
    se = float(np.sqrt(hac(psi)[0, 0] / len(T)) / np.mean(rt ** 2))
    single_index.var_rt = float(np.mean(rt ** 2))
    return th, se


def coupled(W, X, T, Y, lam_scale=0.0):
    n, K = W.shape
    tt, ty = np.zeros((K, n)), np.zeros((K, n))
    for k in range(K):
        w = np.maximum(W[:, k], 1e-4)
        for tr, te in purged(n):
            tt[k, te] = T[te] - Ridge(alpha=1.0).fit(X[tr], T[tr], sample_weight=w[tr]).predict(X[te])
            ty[k, te] = Y[te] - Ridge(alpha=1.0).fit(X[tr], Y[tr], sample_weight=w[tr]).predict(X[te])
    J = np.array([[np.mean(W[:, j] * W[:, k] * tt[j] * tt[k]) for k in range(K)] for j in range(K)])
    Sv = np.array([np.mean(W[:, j] * tt[j] * ty[j]) for j in range(K)])
    lam = lam_scale * np.trace(J) / K / np.sqrt(n)
    Ji = np.linalg.inv(J + lam * np.eye(K))
    th = Ji @ Sv
    psi = np.stack([W[:, k] * tt[k] * (ty[k] - (W * tt.T) @ th) for k in range(K)], 1)
    Sig = Ji @ hac(psi) @ Ji / n
    share = W.mean(0)
    return th, share, float(np.sqrt(share @ Sig @ share)), float(np.linalg.eigvalsh(J).min())


def one(cal, rep):
    rng = np.random.RandomState(20261007 + 1000 * CITIES.index(cal['city']) + rep)
    Z, X, T0, gam = cal['Z'], cal['X'], cal['T0'], cal['gamma']
    n = len(T0)
    S = ffbs(rng, cal['logB'], cal['A'], cal['pi0'])
    U = np.zeros(n)
    e = rng.randn(n)
    U[0] = e[0]
    for t in range(1, n):
        U[t] = RHO_U * U[t - 1] + np.sqrt(1 - RHO_U ** 2) * e[t]
    beta = np.linspace(0.3, -0.3, X.shape[1])
    T = T0 + DT * S
    Y = THETA[S] * T + DG * S + X @ beta + U
    sate = float(THETA[S].mean())
    H = np.c_[1 - S, S].astype(float)
    rows = []

    def add(name, ate, se, th=None):
        rows.append(dict(city=cal['city'], rep=rep, method=name, ate=ate, bias=ate - sate, se=se,
                         cover=float(abs(ate - sate) <= 1.96 * se) if se == se else np.nan,
                         theta_err=float(np.linalg.norm(th - THETA)) if th is not None else np.nan))

    th, sh, se, lam_or = coupled(H, X, T, Y)
    add('Oracle-state regime DML', float(sh @ th), se, th)
    add('Standard DML (random folds)', *single_index(X, T, Y, list(KFold(5, shuffle=True, random_state=rep).split(X))))
    add('Block DML (purged folds)', *single_index(X, T, Y, list(purged(n))))
    add('DML + Z controls', *single_index(np.c_[X, Z], T, Y, list(purged(n))))
    add('Posterior-feature DML', *single_index(np.c_[X, gam], T, Y, list(purged(n))))
    # residual-state bias law (heterogeneous-effects form, K=2) evaluated with observable plug-ins:
    #   theta_hat -> theta_bar + a v [b + a(theta_1 - theta_bar) + m_bar(theta_1 - theta_0)] / Var(T~)
    v_hat = float(np.mean(gam[:, 1] * (1 - gam[:, 1])))
    var_tt = single_index.var_rt
    m_bar = float(T0.mean())
    law = DT * v_hat * (DG + DT * (THETA[1] - sate) + m_bar * (THETA[1] - THETA[0])) / var_tt
    rows[-1].update(law_bias=law, v_hat=v_hat, var_Ttilde=var_tt)
    hard = np.eye(2)[gam.argmax(1)]
    th, sh, se, _ = coupled(hard, X, T, Y)
    add('Hard regime FE DML', float(sh @ th), se, th)
    th, sh, se, lam_px = coupled(gam, X, T, Y, lam_scale=0.05)
    add('Spectral OR-DML (soft, coupled)', float(sh @ th), se, th)
    th, sh, se, lam_pa = coupled_posterior(gam, X, T, Y, list(purged(n)), lam_scale=0.05)
    add('OR-DML (posterior-adjusted)', float(sh @ th), se, th)
    rows[-1].update(lam_pa=lam_pa)
    eps = float(np.mean(np.abs(gam - H).sum(1)))
    for r in rows:
        r.update(eps_hat=eps, lam_proxy=lam_px, lam_oracle=lam_or, N=n, share_regime1=float(S.mean()))
    return rows


def main():
    cals = [calibrate(c) for c in CITIES]
    meta = pd.DataFrame([dict(city=c['city'], N=c['N'], stay0=c['stay'][0], stay1=c['stay'][1],
                              mean_entropy=float(np.mean(-(c['gamma'] * np.log(c['gamma'] + 1e-12)).sum(1))),
                              two_gini=float(np.mean(2 * (1 - (c['gamma'] ** 2).sum(1)))))
                         for c in cals])
    meta.to_csv(os.path.join(OUT, 'semisynthetic_hourly_calibration.csv'), index=False)
    rows = Parallel(n_jobs=-1)(delayed(one)(c, r) for c in cals for r in range(N_REPS))
    d = pd.DataFrame([x for rr in rows for x in rr])
    d.to_csv(os.path.join(OUT, 'semisynthetic_hourly_raw.csv'), index=False)
    s = d.groupby(['city', 'method'], sort=False).agg(
        mean_abs_bias=('bias', lambda b: np.mean(np.abs(b))), rmse=('bias', lambda b: np.sqrt(np.mean(b ** 2))),
        coverage=('cover', 'mean'), theta_err=('theta_err', 'median'), eps_hat=('eps_hat', 'median'),
        lam_proxy=('lam_proxy', 'median'), lam_oracle=('lam_oracle', 'median'), N=('N', 'first'),
        mean_bias=('bias', 'mean'), law_bias=('law_bias', 'mean')).reset_index()
    s.to_csv(os.path.join(OUT, 'semisynthetic_hourly_summary.csv'), index=False)
    print(meta.round(3).to_string())
    print(s.round(3).to_string())


if __name__ == '__main__':
    main()
