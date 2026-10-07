"""
closed_form_factorial.py
========================
Exact population law for the coupled regime score under a label-mixing proxy, and its
verification against the committed 7x7 factorial experiment (4,900 runs).

Setting (two regimes, oracle regime-wise nuisances, exogenous innovations):
    T = m_S + sigma_S V,      Y = theta_S T + g_S + U,      gamma_j(s) = 1-alpha if j == s else alpha,
with alpha = eps/2 so that eps = E||gamma - H||_1.  Writing mu_s = theta_s m_s + g_s,

    J_jk = sum_s pi_s gamma_j(s) gamma_k(s) [ (m_s - m_j)(m_s - m_k) + sigma_s^2 ]
    S_j  = sum_s pi_s gamma_j(s)            [ (m_s - m_j)(mu_s - mu_j) + theta_s sigma_s^2 ]

and the population coupled estimand is theta_gamma = J^{-1} S (Proposition 3 of the manuscript).
The script
  1. evaluates the law on the factorial design and compares it with the committed run-level data;
  2. verifies the symmetric-case eigen-structure and the hump location symbolically (sympy);
  3. writes reports/synthesis/closed_form_factorial_cells.csv and closed_form_factorial_summary.json.

    python src/synthesis/closed_form_factorial.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'reports', 'synthesis')
os.makedirs(OUT, exist_ok=True)

# Factorial design constants (src/factorial_reliability_experiment.py)
M = np.array([2.0, 5.0])
G = np.array([15.0, 50.0])
THETA = np.array([0.75, 2.50])
P01, P10 = 0.12, 0.15                      # persistence 0.88 in state 0, exit prob 0.15 from state 1
PI = np.array([P10, P01]) / (P01 + P10)    # stationary distribution


def population_JS(sigma1, eps, m=M, g=G, theta=THETA, pi=PI, sigma0=1.0):
    sig = np.array([sigma0, sigma1])
    mu = theta * m + g
    a = eps / 2.0
    gam = np.array([[1 - a, a], [a, 1 - a]])   # gam[s, j] = gamma_j(s)
    J = np.zeros((2, 2))
    S = np.zeros(2)
    for s in range(2):
        for j in range(2):
            S[j] += pi[s] * gam[s, j] * ((m[s] - m[j]) * (mu[s] - mu[j]) + theta[s] * sig[s] ** 2)
            for k in range(2):
                J[j, k] += pi[s] * gam[s, j] * gam[s, k] * ((m[s] - m[j]) * (m[s] - m[k]) + sig[s] ** 2)
    return J, S


def population_error(sigma1, eps, **kw):
    J, S = population_JS(sigma1, eps, **kw)
    th = np.linalg.solve(J, S)
    return float(np.linalg.norm(th - kw.get('theta', THETA))), float(np.linalg.eigvalsh(J).min())


def symbolic_check():
    """Symmetric case (pi = 1/2, common sigma): eigenvalues and hump location."""
    import sympy as sp
    a, s, D, e = sp.symbols('alpha sigma Delta epsilon', positive=True)
    b = 1 - a  # weight on the correct regime
    # regimes 0,1 with m_1 - m_0 = D; residual of regime-j nuisance evaluated in regime s
    J = sp.zeros(2, 2)
    gam = [[b, a], [a, b]]
    dm = [[0, -D], [D, 0]]                      # dm[s][j] = m_s - m_j
    for st in range(2):
        for j in range(2):
            for k in range(2):
                J[j, k] += sp.Rational(1, 2) * gam[st][j] * gam[st][k] * (dm[st][j] * dm[st][k] + s ** 2)
    J = sp.simplify(J)
    lam_plus = sp.simplify(J[0, 0] + J[0, 1])
    lam_minus = sp.simplify(J[0, 0] - J[0, 1])
    ok1 = sp.simplify(lam_plus - (s ** 2 / 2 + a ** 2 * D ** 2 / 2)) == 0
    ok2 = sp.simplify(lam_minus - (s ** 2 * (1 - 2 * a) ** 2 / 2 + a ** 2 * D ** 2 / 2)) == 0
    # leading-order error along (1,1): proportional to eps / (sigma^2/2 + eps^2 D^2 / 8); maximise over eps
    f = e / (s ** 2 / 2 + e ** 2 * D ** 2 / 8)
    crit = sp.solve(sp.diff(f, e), e)
    ok3 = any(sp.simplify(c - 2 * s / D) == 0 for c in crit)
    # full symmetric-case displacement theta_gamma - theta* (Proposition 3 of the manuscript)
    t0, t1, m0, g0, g1 = sp.symbols('theta0 theta1 m0 g0 g1', real=True)
    m = [m0, m0 + D]
    th = [t0, t1]
    mu = [t0 * m[0] + g0, t1 * m[1] + g1]
    Jg = sp.zeros(2, 2)
    Sg = sp.zeros(2, 1)
    for st in range(2):
        for j in range(2):
            Sg[j] += sp.Rational(1, 2) * gam[st][j] * ((m[st] - m[j]) * (mu[st] - mu[j]) + th[st] * s ** 2)
            for k in range(2):
                Jg[j, k] += sp.Rational(1, 2) * gam[st][j] * gam[st][k] * ((m[st] - m[j]) * (m[st] - m[k]) + s ** 2)
    disp = sp.simplify(Jg.LUsolve(Sg) - sp.Matrix(th))
    G = mu[1] - mu[0]
    lp = s ** 2 / 2 + a ** 2 * D ** 2 / 2
    lm = s ** 2 * (1 - 2 * a) ** 2 / 2 + a ** 2 * D ** 2 / 2
    c_plus = (a * D * G - a ** 2 * D ** 2 * (t0 + t1) / 2) / (2 * lp)          # coefficient on (1, 1)
    c_minus = (t0 - t1) * (a * (1 - 2 * a) * s ** 2 - a ** 2 * D ** 2 / 2) / (2 * lm)   # coefficient on (1, -1)
    claim = sp.Matrix([c_plus + c_minus, c_plus - c_minus])
    ok4 = all(sp.simplify(x) == 0 for x in (disp - claim))
    return dict(eig_plus_ok=bool(ok1), eig_minus_ok=bool(ok2), hump_at_2sigma_over_Delta=bool(ok3),
                displacement_formula_ok=bool(ok4), lam_plus=str(lam_plus), lam_minus=str(lam_minus))


def main():
    raw = pd.read_csv(os.path.join(ROOT, 'reports', 'factorial_reliability_raw.csv'))
    raw = raw.rename(columns={'Residual_SD_State1': 'sigma1', 'Proxy_Error_Eps': 'eps', 'Lambda_Min_J': 'lam',
                              'Causal_Error_L2': 'err'})
    cells = raw.groupby(['sigma1', 'eps']).agg(err_median=('err', 'median'), err_mean=('err', 'mean'),
                                               lam_mean=('lam', 'mean'), n=('err', 'size')).reset_index()
    pred = [population_error(r.sigma1, r.eps) for r in cells.itertuples()]
    cells['err_population'] = [p[0] for p in pred]
    cells['lam_population'] = [p[1] for p in pred]
    cells['ratio_obs_over_pop'] = cells.err_median / cells.err_population
    cells['lam_rel_diff'] = (cells.lam_population - cells.lam_mean).abs() / cells.lam_mean
    cells.to_csv(os.path.join(OUT, 'closed_form_factorial_cells.csv'), index=False)

    nz = cells[cells.eps > 0]
    peaks = []
    for sd, g in nz.groupby('sigma1'):
        peaks.append(dict(sigma1=float(sd), eps_peak_observed=float(g.loc[g.err_median.idxmax(), 'eps']),
                          eps_peak_population=float(g.loc[g.err_population.idxmax(), 'eps'])))
    # lambda inflation at the smallest sigma
    g = cells[cells.sigma1 == cells.sigma1.min()].sort_values('eps')
    infl = float(g.lam_mean.iloc[-1] / g.lam_mean.iloc[0])
    # rank correlations at cell level: proxy-lambda ratio vs design-sigma ratio
    nz = nz.assign(D_proxy=nz.eps / nz.lam_mean, D_design=nz.eps / nz.sigma1 ** 2)
    summary = dict(
        n_cells_compared=int(len(nz)),
        ratio_median=float(nz.ratio_obs_over_pop.median()),
        ratio_q25=float(nz.ratio_obs_over_pop.quantile(.25)),
        ratio_q75=float(nz.ratio_obs_over_pop.quantile(.75)),
        ratio_min=float(nz.ratio_obs_over_pop.min()), ratio_max=float(nz.ratio_obs_over_pop.max()),
        spearman_population_vs_observed=float(spearmanr(nz.err_population, nz.err_median)[0]),
        lambda_max_rel_diff=float(nz.lam_rel_diff.max()),
        peaks=peaks,
        n_sigma_with_interior_hump=int(sum(p['eps_peak_observed'] < nz.eps.max() for p in peaks)),
        lambda_inflation_smallest_sigma=infl,
        lambda_smallest_sigma_eps0=float(g.lam_mean.iloc[0]),
        lambda_smallest_sigma_epsmax=float(g.lam_mean.iloc[-1]),
        err_smallest_sigma_peak=float(g.err_median.max()),
        err_smallest_sigma_epsmax=float(g.err_median.iloc[-1]),
        spearman_cell_D_proxy=float(spearmanr(nz.D_proxy, nz.err_median)[0]),
        spearman_cell_D_design=float(spearmanr(nz.D_design, nz.err_median)[0]),
        symbolic=symbolic_check(),
    )
    with open(os.path.join(OUT, 'closed_form_factorial_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
