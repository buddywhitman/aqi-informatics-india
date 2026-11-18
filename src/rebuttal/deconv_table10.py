"""
deconv_table10.py -- rebuttal experiment E6: DC-DML added to the Table 10 scenarios
(src/bias_law/ext_joint_proximal.py: same generator, seeds, posteriors; N = 3000, Delta_Z = 0.5, stay 0.9,
Delta_g = 3, 12 replications, Delta_T in {1, 4}).  DC-DML uses a cubic polynomial basis in X.

    python src/rebuttal/deconv_table10.py -> reports/rebuttal/deconv_table10_{raw,summary}.csv
"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
from multiprocessing import Pool
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from src.bias_law.sim import gen_states_proxy, fit_posteriors, align_to_truth, partial_out
from src.bias_law.ext_joint_proximal import joint_em
OUT = os.path.join(ROOT, 'reports', 'rebuttal'); os.makedirs(OUT, exist_ok=True)


def basis3(X):
    from sklearn.preprocessing import PolynomialFeatures
    return PolynomialFeatures(3).fit_transform(X)


def dc(g, X, T, Y):
    Bx = basis3(X); K = g.shape[1]; p = Bx.shape[1]
    R = np.column_stack([g[:, [k]] * Bx for k in range(K)])
    mom = {}
    for name, h in (('t', T), ('tt', T * T), ('y', Y), ('ty', T * Y)):
        c = np.linalg.lstsq(R, h, rcond=None)[0]
        mom[name] = [Bx @ c[k * p:(k + 1) * p] for k in range(K)]
    th = np.empty(K)
    for k in range(K):
        w = g[:, k] / g[:, k].sum()
        th[k] = np.sum(w * (mom['ty'][k] - mom['t'][k] * mom['y'][k])) / np.sum(w * (mom['tt'][k] - mom['t'][k] ** 2))
    return th, float(g.mean(0) @ th)


def one(args):
    scen, dT, dg, dz, rep, N = args
    rng = np.random.default_rng(10_000 + 97 * rep + int(100 * dT) + hash(scen) % 997)
    Ktrue = 3 if scen == 'K3true' else 2
    S, Z, X = gen_states_proxy(N, Ktrue, dz, 0.9, rng)
    if scen == 'heavyZ':
        mu = np.where(S[:, None] == 1, [dz, 1.5 * dz], [-dz, -1.5 * dz])
        Z = mu + rng.standard_t(3, size=(N, 2)) / np.sqrt(3)
    V, U, E = rng.normal(size=N), rng.normal(size=N), rng.normal(size=N)
    Sb = (S == 1).astype(int) if Ktrue == 2 else (S > 0).astype(int)
    fx = (lambda X: np.sin(2 * X[:, 0]) + 0.5 * X[:, 1] ** 2) if scen == 'nonlinX' else (lambda X: 0.5 * X[:, 0] - 0.3 * X[:, 1])
    T = fx(X) + dT * Sb + V
    th = np.where(Sb == 1, 2.0, 1.0) if scen == 'hetero' else np.ones(N)
    Y = th * T + dg * Sb + 0.3 * X[:, 1] + U
    nuis = 'gbm' if scen == 'nonlinX' else 'lin'
    post = fit_posteriors(Z, 2, rep)
    g = align_to_truth(post['smooth'], Sb)
    th_dml, _, _ = partial_out(Y, T, np.column_stack([X, g[:, 1]]), nuis, rep)
    th_j, _ = joint_em(Y, T, X, Z, g)
    _, ate_dc = dc(g, X, T, Y)
    tgt = float(np.mean(th))
    return dict(scen=scen, dT=dT, rep=rep, bias_dml=th_dml - tgt, bias_joint=th_j - tgt, bias_dc=ate_dc - tgt)


if __name__ == '__main__':
    scens = ('base', 'nonlinX', 'K3true', 'heavyZ', 'hetero')
    jobs = [(sc, dT, 3.0, 0.5, r, 3000) for sc in scens for dT in (1, 4) for r in range(12)]
    with Pool(4) as p:
        rows = p.map(one, jobs)
    d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, 'deconv_table10_raw.csv'), index=False)
    rm = lambda x: np.sqrt(np.mean(x ** 2))
    s = d.groupby(['scen', 'dT']).agg(dml=('bias_dml', 'mean'), joint=('bias_joint', 'mean'), dc=('bias_dc', 'mean'),
                                     rmse_dml=('bias_dml', rm), rmse_joint=('bias_joint', rm), rmse_dc=('bias_dc', rm),
                                     medabs_dc=('bias_dc', lambda x: np.median(np.abs(x))))
    s.to_csv(os.path.join(OUT, 'deconv_table10_summary.csv'))
    print(s.round(3).to_string())
