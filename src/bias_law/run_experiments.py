"""
run_experiments.py -- verification experiments for the residual-state bias law.

    python src/bias_law/run_experiments.py            # all experiments
    python src/bias_law/run_experiments.py e1 e3      # selected ones

Outputs: reports/bias_law/*.csv
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys
import warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.bias_law.sim import *            # noqa: E402,F401,F403
from src.bias_law.sim import _mk                  # noqa: E402
from src.bias_law import bias_law as BL   # noqa: E402

warnings.filterwarnings('ignore')
OUT = os.path.join(ROOT, 'reports', 'bias_law')
os.makedirs(OUT, exist_ok=True)

DG = 3.0
DT_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
DZ_GRID = [0.5, 0.8, 1.2]


# ------------------------------------------------------------------ E1: K=2 grid, learned HMM, dependent data
def _e1_cell(args):
    dz, rep = args
    rng = np.random.default_rng(10_000 * int(dz * 10) + rep)
    N = 3000
    S, Z, X = gen_states_proxy(N, 2, dz, 0.9, rng)
    V, U = rng.normal(size=N), rng.normal(size=N)
    post = fit_posteriors(Z, 2, rep)
    rows = []
    for dT in DT_GRID:
        T, Y = gen_TY(S, X, [0, dT], [0, DG], [1.0, 1.0], (V, U))
        for mode in ('smooth', 'filter'):
            g = align_to_truth(post[mode], S)
            F = np.column_stack([X, g[:, 1]])
            v_hat = float(np.mean(g[:, 1] * (1 - g[:, 1])))
            brier = float(np.mean((S - g[:, 1]) ** 2))
            for nuis in ('lin', 'gbm'):
                if mode == 'filter' and nuis == 'gbm':
                    continue
                th, tT, tY = partial_out(Y, T, F, nuis, rep)
                vN = float(residual_state_cov(S, F, 2, nuis, rep)[0, 0])
                dT_hat = float(BL.treatment_shift(T, X, g)[0])
                vT = float(np.mean(tT ** 2))
                rows.append(dict(dz=dz, rep=rep, dT=dT, mode=mode, nuis=nuis, bias=th - 1.0, v_hat=v_hat, v_N=vN,
                                 brier=brier, dT_hat=dT_hat, var_Ttilde=vT,
                                 law_true=BL.law_bias(dT, DG, vN, 1.0),
                                 law_hat=BL.law_bias(dT, DG, v_hat, 1.0),
                                 law_obs=dT_hat * DG * v_hat / vT))
        # baselines (smoother, linear nuisance)
        g = align_to_truth(post['smooth'], S)
        oh = np.eye(2)[S][:, 1:]
        hard = np.eye(2)[g.argmax(1)][:, 1:]
        for name, F in (('standard_dml', X), ('oracle_state', np.column_stack([X, oh])),
                        ('hard_fe', np.column_stack([X, hard]))):
            th, tT, tY = partial_out(Y, T, F, 'lin', rep)
            rows.append(dict(dz=dz, rep=rep, dT=dT, mode='smooth', nuis=name, bias=th - 1.0, v_hat=np.nan, v_N=np.nan,
                             brier=np.nan, dT_hat=np.nan, var_Ttilde=float(np.mean(tT ** 2)), law_true=np.nan,
                             law_hat=np.nan, law_obs=np.nan))
    return rows


def e1(R=12):
    out = []
    with Pool(2) as p:
        for i, o in enumerate(p.imap_unordered(_e1_cell, [(dz, r) for dz in DZ_GRID for r in range(R)])):
            out.append(o)
            print('  e1 cell', i + 1, flush=True)
    df = pd.DataFrame([r for o in out for r in o])
    df.to_csv(os.path.join(OUT, 'e1_learned_hmm_raw.csv'), index=False)
    main = df[df.nuis.isin(['lin', 'gbm'])]
    s = main.groupby(['dz', 'dT', 'mode', 'nuis']).agg(bias=('bias', 'mean'), bias_se=('bias', lambda x: x.std() / np.sqrt(len(x))),
                                                       v_hat=('v_hat', 'mean'), v_N=('v_N', 'mean'),
                                                       law_true=('law_true', 'mean'), law_hat=('law_hat', 'mean'),
                                                       law_obs=('law_obs', 'mean'), var_Ttilde=('var_Ttilde', 'mean')).reset_index()
    s.to_csv(os.path.join(OUT, 'e1_learned_hmm_summary.csv'), index=False)
    b = df[~df.nuis.isin(['lin', 'gbm'])].groupby(['dz', 'dT', 'nuis']).agg(bias=('bias', 'mean'), var_Ttilde=('var_Ttilde', 'mean')).reset_index()
    b.to_csv(os.path.join(OUT, 'e1_baselines_summary.csv'), index=False)
    for c in ('law_true', 'law_hat', 'law_obs'):
        r = np.corrcoef(s.bias, s[c])[0, 1]
        rmse = np.sqrt(np.mean((s.bias - s[c]) ** 2))
        print(f"E1 corr(bias, {c}) = {r:.4f}  RMSE = {rmse:.4f}  (n={len(s)} cells)")
    cl = np.corrcoef(s.bias, s.dz.map(lambda z: 0) + 0)[0, 1] if False else None
    return s


# ------------------------------------------------------------------ E2: K=3 states, matrix law
def _e2_cell(args):
    dz, cfg, rep = args
    a_full, b_full = cfg
    rng = np.random.default_rng(777 + 100 * rep + int(dz * 10))
    N = 3000
    S, Z, X = gen_states_proxy(N, 3, dz, 0.9, rng)
    V, U = rng.normal(size=N), rng.normal(size=N)
    post = fit_posteriors(Z, 3, rep)
    g = align_to_truth(post['smooth'], S)
    T, Y = gen_TY(S, X, a_full, b_full, [1.0] * 3, (V, U))
    F = np.column_stack([X, g[:, 1:]])
    th, tT, tY = partial_out(Y, T, F, 'lin', rep)
    SigN = residual_state_cov(S, F, 3, 'lin', rep)
    SigH = BL.posterior_residual_cov(g)
    a = np.array(a_full[1:]) - a_full[0]
    b = np.array(b_full[1:]) - b_full[0]
    ah = BL.treatment_shift(T, X, g)
    vT = float(np.mean(tT ** 2))
    return dict(dz=dz, cfg=str(cfg), rep=rep, bias=th - 1.0, law_true=BL.law_bias_matrix(a, b, SigN, 1.0),
                law_hat=BL.law_bias_matrix(a, b, SigH, 1.0), law_obs=float(ah @ SigH @ b / vT))


def e2(R=20):
    cfgs = [((0, 2, 5), (0, 3, 1)), ((0, 3, -3), (0, 2, 2)), ((0, 1, 1), (0, 3, -3)), ((0, 6, 3), (0, 3, 3))]
    with Pool(2) as p:
        out = p.map(_e2_cell, [(dz, c, r) for dz in (0.8, 1.2) for c in cfgs for r in range(R)])
    df = pd.DataFrame(out)
    s = df.groupby(['dz', 'cfg']).mean(numeric_only=True).reset_index().drop(columns='rep')
    s.to_csv(os.path.join(OUT, 'e2_three_state_summary.csv'), index=False)
    for c in ('law_true', 'law_hat', 'law_obs'):
        print(f"E2 (K=3) corr(bias,{c}) = {np.corrcoef(s.bias, s[c])[0, 1]:.4f}  RMSE = {np.sqrt(np.mean((s.bias - s[c]) ** 2)):.4f}")
    return s


# ------------------------------------------------------------------ E3: the paper's Table-1 DGP (heterogeneous theta, X-confounding)
def _e3_cell(args):
    dz, rep = args
    from src.synthetic_dgp_benchmark import generate_difficulty_dgp
    Y, T, X, Z, gt = generate_difficulty_dgp(N=1200, delta_z=dz, delta_t=4.0, random_state=5000 + rep)
    S = gt['S']
    post = fit_posteriors(Z, 2, rep)
    g = align_to_truth(post['smooth'], S)
    Fsoft = np.column_stack([X, g[:, 1]])
    Foracle = np.column_stack([X, S])
    th_soft, tT, tY = partial_out(Y, T, Fsoft, 'lin', rep)
    th_or, _, _ = partial_out(Y, T, Foracle, 'lin', rep)
    th_std, _, _ = partial_out(Y, T, X, 'lin', rep)
    vN = float(residual_state_cov(S, Fsoft, 2, 'lin', rep)[0, 0])
    v_hat = float(np.mean(g[:, 1] * (1 - g[:, 1])))
    t0, t1 = gt['theta_regimes'][0], gt['theta_regimes'][1]
    dg, dT = 35.0, 4.0
    sig2 = 1.0
    excess = th_soft - th_or
    return dict(dz=dz, rep=rep, th_std=th_std, th_soft=th_soft, th_oracle=th_or, excess=excess, v_hat=v_hat, v_N=vN,
                law_true=BL.law_bias_hetero(dT, dg, t1, th_or, vN, sig2, t1 - t0, 2.0), law_hat=BL.law_bias_hetero(dT, dg, t1, th_or, v_hat, sig2, t1 - t0, 2.0),
                var_Ttilde=float(np.mean(tT ** 2)))


def e3(R=30):
    with Pool(2) as p:
        out = p.map(_e3_cell, [(dz, r) for dz in (0.2, 0.5, 1.0, 1.5, 2.0, 4.0) for r in range(R)])
    df = pd.DataFrame(out)
    s = df.groupby('dz').mean().reset_index().drop(columns='rep')
    s.to_csv(os.path.join(OUT, 'e3_table1_dgp_summary.csv'), index=False)
    print("E3 (paper Table-1 DGP: theta=(0.75,2.5), dT=4, dg=35):")
    print(s[['dz', 'th_std', 'th_soft', 'th_oracle', 'excess', 'law_true', 'law_hat', 'v_hat', 'v_N']].round(3).to_string(index=False))
    return s


# ------------------------------------------------------------------ E4: calibration-gap experiment
def _logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def _e4_cell(args):
    tau, rep = args
    from sklearn.isotonic import IsotonicRegression
    rng = np.random.default_rng(31_000 + rep)
    N = 3000
    S, Z, X = gen_states_proxy(N, 2, 0.8, 0.9, rng)
    V, U = rng.normal(size=N), rng.normal(size=N)
    post = fit_posteriors(Z, 2, rep)
    g1 = align_to_truth(post['smooth'], S)[:, 1]
    gt_ = 1 / (1 + np.exp(-_logit(g1) / tau))           # temperature-scaled posterior (tau<1 over-confident)
    dT, dg = 2.0, DG
    T, Y = gen_TY(S, X, [0, dT], [0, dg], [1, 1], (V, U))
    F = np.column_stack([X, gt_])
    th, tT, tY = partial_out(Y, T, F, 'gbm', rep)
    vN = float(residual_state_cov(S, F, 2, 'gbm', rep)[0, 0])
    v_hat = float(np.mean(gt_ * (1 - gt_)))
    c = IsotonicRegression(out_of_bounds='clip').fit(gt_, S).predict(gt_)
    ce = float(np.mean((gt_ - c) ** 2))
    return dict(tau=tau, rep=rep, bias=th - 1.0, v_hat=v_hat, v_N=vN, ce=ce,
                law_hat=BL.law_bias(dT, dg, v_hat), law_true=BL.law_bias(dT, dg, vN),
                gap=abs(v_hat - vN), bound=BL.calibration_gap_bound(ce))


def e4(R=20):
    with Pool(2) as p:
        out = p.map(_e4_cell, [(t, r) for t in (0.3, 0.5, 0.75, 1.0, 1.5, 2.5) for r in range(R)])
    df = pd.DataFrame(out)
    s = df.groupby('tau').mean().reset_index().drop(columns='rep')
    s['bound_holds_frac'] = df.assign(h=df.gap <= df.bound + 1e-9).groupby('tau').h.mean().values
    s.to_csv(os.path.join(OUT, 'e4_calibration_summary.csv'), index=False)
    print("E4 calibration (tau<1 over-confident):")
    print(s[['tau', 'bias', 'law_hat', 'law_true', 'v_hat', 'v_N', 'ce', 'gap', 'bound', 'bound_holds_frac']].round(4).to_string(index=False))
    return s


# ------------------------------------------------------------------ E7: corrected "task conditioning" experiment (replaces Table 10 Panels A/B)
def _e7_cell(args):
    dT, rep = args
    from src.synthetic_dgp_benchmark import generate_difficulty_dgp
    Y, T, X, Z, gt = generate_difficulty_dgp(N=1200, delta_z=1.5, delta_t=dT, random_state=500_000 + rep)
    S = gt['S']
    g = np.column_stack([1 - S, S]).astype(float)
    th = np.array([gt['theta_regimes'][0], gt['theta_regimes'][1]])
    rows = []
    for name, F in (('X_only_nuisance(paper)', X), ('X+regime_nuisance', np.column_stack([X, g]))):
        N = len(Y)
        tT = np.zeros(N)
        tY = np.zeros(N)
        for tr, va in PurgedBlockKFold(4, 12).split(N):
            tT[va] = T[va] - _mk('gbm', rep).fit(F[tr], T[tr]).predict(F[va])
            tY[va] = Y[va] - _mk('gbm', rep).fit(F[tr], Y[tr]).predict(F[va])
        J = np.array([[np.mean(g[:, j] * g[:, k] * tT ** 2) for k in range(2)] for j in range(2)])
        Sv = np.array([np.mean(g[:, j] * tT * tY) for j in range(2)])
        est = np.linalg.solve(J + 1e-9 * np.eye(2), Sv)
        rows.append(dict(dT=dT, rep=rep, nuisance=name, err_l2=float(np.linalg.norm(est - th)),
                         lam_min=float(np.linalg.eigvalsh(J).min())))
    return rows


def e7(R=12):
    with Pool(2) as p:
        out = p.map(_e7_cell, [(d, r) for d in (0.4, 0.8, 1.5, 3.0, 6.0, 10.0) for r in range(R)])
    df = pd.DataFrame([r for o in out for r in o])
    s = df.groupby(['nuisance', 'dT']).mean().reset_index().drop(columns='rep')
    s.to_csv(os.path.join(OUT, 'e7_corrected_conditioning.csv'), index=False)
    print("E7 corrected conditioning experiment:")
    print(s.round(3).to_string(index=False))
    return s


if __name__ == '__main__':
    which = sys.argv[1:] or ['e1', 'e2', 'e3', 'e4', 'e7']
    for w in which:
        print('=' * 30, w)
        globals()[w]()
