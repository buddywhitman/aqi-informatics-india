"""
ext_real_joint.py -- model-based cross-check on the four cities.  Fits the joint Markov-switching regression (K=2) with a common NO2 effect,
and compares theta_joint with posterior-DML theta and with the law-adjusted DML estimate theta - a*v*b_hat/Var(T~) where b_hat is the joint
model's regime shift in PM2.5 (not identified by DML alone).  Real data are non-Gaussian: treat as a consistency check, not a proof.
    python src/bias_law/ext_real_joint.py -> reports/bias_law/x7_real_joint.csv
"""
import os, sys, warnings
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.bias_law.ext_joint_proximal import _fwd_bwd                                           # noqa: E402
from src.bias_law.real_cities_sensitivity import analyse, REGIME_FEATURES, CONTROLS, ROOT, OUT, frozen_mask  # noqa: E402
from src.or_dml import OverlapAwareRegimeDML                                                    # noqa: E402
from sklearn.linear_model import Ridge                                                          # noqa: E402


def joint_em_p(Y, T, X, Z, g0, iters=60):
    N, p = X.shape; K = 2; g = g0.copy()
    A = np.array([[.95, .05], [.05, .95]]); pi = np.array([.5, .5]); ll_old = -np.inf
    for it in range(iters):
        Nk = g.sum(0); mu = (g.T @ Z) / Nk[:, None]
        Sig = sum(((Z - mu[k]).T * g[:, k]) @ (Z - mu[k]) for k in range(K)) / N + 1e-6 * np.eye(Z.shape[1])
        D = np.vstack([np.column_stack([X, np.tile(np.eye(K)[k], (N, 1))]) for k in range(K)])
        w = np.concatenate([g[:, k] for k in range(K)]); sw = np.sqrt(w)[:, None]
        bT = np.linalg.lstsq(D * sw, np.tile(T, K) * sw[:, 0], rcond=None)[0]
        sT2 = float(np.sum(w * (np.tile(T, K) - D @ bT) ** 2) / N)
        DY = np.column_stack([np.tile(T, K), D])
        bY = np.linalg.lstsq(DY * sw, np.tile(Y, K) * sw[:, 0], rcond=None)[0]
        sY2 = float(np.sum(w * (np.tile(Y, K) - DY @ bY) ** 2) / N)
        Si = np.linalg.inv(Sig); ld = np.linalg.slogdet(Sig)[1]; logB = np.zeros((N, K))
        for k in range(K):
            dz = Z - mu[k]
            logB[:, k] = (-0.5 * np.einsum('ij,jk,ik->i', dz, Si, dz) - 0.5 * ld
                          - 0.5 * (T - X @ bT[:p] - bT[p + k]) ** 2 / sT2 - 0.5 * np.log(sT2)
                          - 0.5 * (Y - bY[0] * T - X @ bY[1:1 + p] - bY[1 + p + k]) ** 2 / sY2 - 0.5 * np.log(sY2))
        g, xi, ll = _fwd_bwd(logB, A, pi)
        A = xi / xi.sum(1, keepdims=True); pi = g[0]
        if abs(ll - ll_old) < 1e-3:
            break
        ll_old = ll
    return dict(theta=float(bY[0]), dT=float(bT[p + 1] - bT[p]), dg=float(bY[1 + p + 1] - bY[1 + p]), ll=ll, g=g)


def main():
    df = pd.read_csv(os.path.join(ROOT, 'data', 'processed_clean', 'combined_hourly_clean.csv'))
    rows = []
    for city in ['Delhi', 'Mumbai', 'Bengaluru', 'Kolkata']:
        d = df[df.city == city].dropna(subset=['no2', 'pm25'] + REGIME_FEATURES + CONTROLS).reset_index(drop=True)
        Y, T = d.pm25.values.astype(float), d.no2.values.astype(float)
        X, Z = d[CONTROLS].values.astype(float), d[REGIME_FEATURES].values.astype(float)
        Xs, Zs = (X - X.mean(0)) / X.std(0), (Z - Z.mean(0)) / Z.std(0)
        r = analyse(Y, T, X, Z, n_boot=20)
        m = OverlapAwareRegimeDML(n_regimes=2, n_splits=5, embargo_tau=24, reg_alpha=0.05, posterior_mode='smooth',
                                  nuisance_model=Ridge(alpha=1.0), hac_lag=12, random_state=42).fit(Y, T, X, Z)
        g0 = m.gamma_
        best = None
        for gi in (g0, g0[:, ::-1]):
            j = joint_em_p(Y, T, Xs, Zs, gi)
            if best is None or j['ll'] > best['ll']:
                best = j
        # sign of regime orientation: dT of the joint fit defines state 1; DML uses its own orientation -> align via a_hat sign
        sgn = np.sign(best['dT']) * np.sign(r['a_hat']) if r['a_hat'] != 0 else 1.0
        b_hat = best['dg'] * (1 if sgn >= 0 else -1) * (1 if best['dT'] >= 0 else 1)
        # law-adjusted: theta - a*v*b/VarT~ with b expressed in the DML's orientation (b has the sign of dg*sign(dT)*sign(a))
        b_or = best['dg'] * np.sign(best['dT']) * np.sign(r['a_hat'])
        adj = r['theta'] - r['a_hat'] * r['v_hat'] * b_or / r['var_Ttilde']
        rows.append(dict(city=city, N=len(d), theta_dml=r['theta'], theta_joint=best['theta'], theta_law_adj=adj,
                         a_dml=r['a_hat'], dT_joint=best['dT'], dg_joint=best['dg'], v_hat=r['v_hat'], var_Ttilde=r['var_Ttilde']))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'x7_real_joint.csv'), index=False)


if __name__ == '__main__':
    main()
