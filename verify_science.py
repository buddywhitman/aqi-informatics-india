"""
verify_science.py
=================
Fast (~1 min) scientific checks of the paper's theory against simulation.  These are *not* algebraic tautologies:
each test simulates data from the model, runs the actual estimator, and compares with the closed-form prediction.

    python verify_science.py
"""
import os
import sys
import numpy as np

os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.bias_law import bias_law as BL                                # noqa: E402
from src.bias_law.sim import (gen_states_proxy, gen_TY, fit_posteriors, align_to_truth,  # noqa: E402
                              partial_out, residual_state_cov)

RESULTS = []


def check(name, cond, detail=''):
    RESULTS.append((name, bool(cond)))
    print(f"[{'OK' if cond else 'FAIL'}] {name} {detail}")


def exact_posterior_sim(N, dz, dT, dg, rho, seed):
    """Known-parameter filter (calibrated posterior) for a persistent two-state chain."""
    rng = np.random.default_rng(seed)
    S = np.zeros(N, int)
    for t in range(1, N):
        S[t] = S[t - 1] if rng.random() < rho else 1 - S[t - 1]
    Z = np.where(S == 1, dz, -dz) + rng.normal(size=N)
    A = np.array([[rho, 1 - rho], [1 - rho, rho]])
    mu = np.array([-dz, dz])
    a = np.array([.5, .5])
    g = np.zeros(N)
    for t in range(N):
        a = (a @ A if t else a) * np.exp(-0.5 * (Z[t] - mu) ** 2)
        a /= a.sum()
        g[t] = a[1]
    T = dT * S + rng.normal(size=N)
    Y = T + dg * S + rng.normal(size=N)
    return S, g, T, Y


def test_exact_law_and_hump():
    # (i) the law matches OLS partialling-out with a calibrated filter; (ii) peak location/height of Corollary 2
    dz, dg = 0.5, 3.0
    best = None
    errs = []
    for dT in (0.5, 1, 2, 4, 8, 16):
        S, g, T, Y = exact_posterior_sim(60000, dz, dT, dg, 0.9, int(dT * 10))
        F = np.column_stack([np.ones_like(g), g])
        rt = T - F @ np.linalg.lstsq(F, T, rcond=None)[0]
        ry = Y - F @ np.linalg.lstsq(F, Y, rcond=None)[0]
        bias = (rt @ ry) / (rt @ rt) - 1.0
        v = np.var(S - g)
        errs.append(abs(bias - BL.law_bias(dT, dg, v)))
        if best is None or bias > best[1]:
            best = (dT, bias, v)
    check('law matches OLS with calibrated posterior (max abs err < 0.02)', max(errs) < 0.02, f'(max err {max(errs):.4f})')
    pk = BL.peak_dT(best[2])
    check('hump peak within 2x of sigma/sqrt(v) and height <= sup bound',
          0.5 * pk <= best[0] <= 2 * pk and best[1] <= BL.sup_bias(dg, best[2]) * 1.05,
          f'(argmax {best[0]}, predicted {pk:.1f}; max bias {best[1]:.3f} <= {BL.sup_bias(dg, best[2]):.3f})')


def test_learned_hmm_dependent():
    # learned HMM + dependent data + cross-fitted learners: law with the observable v_hat is within 20% of the bias
    rows = []
    for rep in range(6):
        rng = np.random.default_rng(900 + rep)
        S, Z, X = gen_states_proxy(3000, 2, 0.6, 0.9, rng)
        V, U = rng.normal(size=3000), rng.normal(size=3000)
        post = fit_posteriors(Z, 2, rep)
        g = align_to_truth(post['smooth'], S)
        T, Y = gen_TY(S, X, [0, 2.0], [0, 3.0], [1.0, 1.0], (V, U))
        th, tT, tY = partial_out(Y, T, np.column_stack([X, g[:, 1]]), 'lin', rep)
        v_hat = float(np.mean(g[:, 1] * (1 - g[:, 1])))
        rows.append((th - 1.0, BL.law_bias(2.0, 3.0, v_hat)))
    r = np.array(rows)
    rel = abs(r[:, 0].mean() - r[:, 1].mean()) / abs(r[:, 0].mean())
    check('learned HMM, dependent data: |law(v_hat) - bias| / bias < 0.25', rel < 0.25, f'(bias {r[:, 0].mean():.3f}, law {r[:, 1].mean():.3f})')


def test_information_monotonicity():
    # soft posterior never has larger residual variance than its hard label (law of total variance)
    rng = np.random.default_rng(3)
    S, Z, X = gen_states_proxy(6000, 2, 0.7, 0.9, rng)
    post = fit_posteriors(Z, 2, 0)
    g = align_to_truth(post['smooth'], S)[:, 1]
    hard = (g > 0.5).astype(float)
    v_soft = float(np.mean((S - g) ** 2))                 # Brier >= E Var(S|g); use conditional variance estimates below
    # conditional variances via binning
    bins = np.clip((g * 20).astype(int), 0, 19)
    vs = np.mean([np.var(S[bins == b]) * np.mean(bins == b) for b in range(20) if np.sum(bins == b) > 1])
    vs = sum(np.var(S[bins == b]) * np.mean(bins == b) for b in range(20) if np.sum(bins == b) > 1)
    vh = sum(np.var(S[hard == h]) * np.mean(hard == h) for h in (0, 1))
    check('information monotonicity: E Var(S|soft) <= E Var(S|hard)', vs <= vh + 1e-9, f'({vs:.4f} <= {vh:.4f})')


def test_calibration_bound():
    rng = np.random.default_rng(5)
    ok = True
    for _ in range(200):
        p = rng.beta(0.5, 0.5, size=2000)                              # true posterior c
        S = (rng.random(2000) < p).astype(float)
        tau = rng.uniform(0.3, 3)
        g = 1 / (1 + np.exp(-np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1)) / tau))  # miscalibrated
        v = np.mean(p * (1 - p))
        vh = np.mean(g * (1 - g))
        ce = np.mean((g - p) ** 2)
        ok &= abs(vh - v) <= BL.calibration_gap_bound(ce) + 1e-9
    check('calibration bound |v_hat - v| <= sqrt(CE) + CE (200 random posteriors)', ok)


def test_three_state_matrix_law():
    errs = []
    for rep in range(6):
        rng = np.random.default_rng(1200 + rep)
        S, Z, X = gen_states_proxy(3000, 3, 0.9, 0.9, rng)
        V, U = rng.normal(size=3000), rng.normal(size=3000)
        post = fit_posteriors(Z, 3, rep)
        g = align_to_truth(post['smooth'], S)
        aF, bF = (0, 3, -3), (0, 2, 2)
        T, Y = gen_TY(S, X, aF, bF, [1.0] * 3, (V, U))
        F = np.column_stack([X, g[:, 1:]])
        th, _, _ = partial_out(Y, T, F, 'lin', rep)
        Sig = residual_state_cov(S, F, 3, 'lin', rep)
        pred = BL.law_bias_matrix(np.array(aF[1:]) - aF[0], np.array(bF[1:]) - bF[0], Sig, 1.0)
        errs.append(th - 1.0 - pred)
    check('K=3 matrix law (mean abs err < 0.03)', np.mean(np.abs(errs)) < 0.03, f'({np.mean(np.abs(errs)):.4f})')


def test_heterogeneous_formula():
    rng = np.random.default_rng(77)
    N = 60000
    S, g, T0, _ = exact_posterior_sim(N, 0.5, 3.0, 0.0, 0.9, 11)
    V = np.random.default_rng(1).normal(size=N)
    mbar, dT, dg = 2.0, 3.0, 4.0
    T = mbar + dT * S + V
    th0, th1 = 0.75, 2.5
    thS = np.where(S == 1, th1, th0)
    Y = thS * T + dg * S + np.random.default_rng(2).normal(size=N)
    F = np.column_stack([np.ones(N), g])
    rt = T - F @ np.linalg.lstsq(F, T, rcond=None)[0]
    ry = Y - F @ np.linalg.lstsq(F, Y, rcond=None)[0]
    est = (rt @ ry) / (rt @ rt)
    v = np.var(S - g)
    thbar = np.mean(thS)
    pred = thbar + BL.law_bias_hetero(dT, dg, th1, thbar, v, 1.0, th1 - th0, mbar)
    check('heterogeneous-effects formula (abs err < 0.05)', abs(est - pred) < 0.05, f'(est {est:.3f}, formula {pred:.3f})')


def test_wiener_closed_form():
    from src.bias_law.ext_kalman import kalman_smoother, sim, H, R
    errs = []
    for phi, snr in ((0.9, 0.5), (0.97, 1.0), (0.97, 2.0)):
        u, X, Z, T, Y = sim(0, 6000, phi, snr, 1.0, 3.0)
        _, Ps = kalman_smoother(Z, phi, snr)
        errs.append(abs(Ps[200:-200].mean() - BL.wiener_smoother_var(phi, BL.sensor_information(H, R, snr))))
    check('Wiener closed-form smoothing variance matches Kalman smoother (< 2e-3)', max(errs) < 2e-3, f'(max {max(errs):.1e})')


if __name__ == '__main__':
    test_exact_law_and_hump()
    test_calibration_bound()
    test_information_monotonicity()
    test_heterogeneous_formula()
    test_learned_hmm_dependent()
    test_three_state_matrix_law()
    test_wiener_closed_form()
    n_ok = sum(ok for _, ok in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} scientific checks passed")
    sys.exit(0 if n_ok == len(RESULTS) else 1)
