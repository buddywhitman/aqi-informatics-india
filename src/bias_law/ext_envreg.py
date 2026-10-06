"""
ext_envreg.py -- 'law-regression': identification of theta from cross-environment variation in the treatment shift.
Per environment e, posterior-DML gives theta_e = theta + b_e * k_e, k_e = a_e v_e / Var(T~_e) (observable plug-ins).  If the outcome shift b_e is
invariant (or independent of k_e) across environments, regressing theta_e on k_e identifies theta as the intercept -- no negative control, no joint model.
    python src/bias_law/ext_envreg.py -> reports/bias_law/x10_envreg.csv
"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.bias_law import bias_law as BL                                                 # noqa: E402
from src.bias_law.sim import gen_states_proxy, gen_TY, fit_posteriors, align_to_truth, partial_out   # noqa: E402
OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'bias_law')


def env(args):
    dT, b, rep, e, N = args
    rng = np.random.default_rng(55000 + 1000 * rep + e)
    S, Z, X = gen_states_proxy(N, 2, 0.5, 0.9, rng)
    T, Y = gen_TY(S, X, [0, dT], [0, b], [1, 1], (rng.normal(size=N), rng.normal(size=N)))
    g = align_to_truth(fit_posteriors(Z, 2, rep * 100 + e)['smooth'], S)
    F = np.column_stack([X, g[:, 1]])
    th, tT, tY = partial_out(Y, T, F, 'lin', rep)
    v = float(np.mean(g[:, 1] * (1 - g[:, 1])))
    a = float(BL.treatment_shift(T, X, g)[0])
    return th, a * v / float(np.mean(tT ** 2)), dT


def run_design(label, rep, E=10, N=1500, bfun=None):
    rng = np.random.default_rng(rep)
    dTs = np.linspace(0.3, 4.0, E)
    bs = [bfun(dT, rng) for dT in dTs]
    res = [env((dT, b, rep, e, N)) for e, (dT, b) in enumerate(zip(dTs, bs))]
    th = np.array([r[0] for r in res]); k = np.array([r[1] for r in res])
    A = np.column_stack([np.ones(E), k]); coef = np.linalg.lstsq(A, th, rcond=None)[0]
    return dict(design=label, rep=rep, pooled=float(th.mean()) - 1, envreg=float(coef[0]) - 1, b_hat=float(coef[1]), k_range=float(k.max() - k.min()))


def job(a):
    label, rep = a
    f = {'b_const': lambda dT, r: 3.0,
         'b_random': lambda dT, r: 3.0 + 1.0 * r.normal(),
         'b_corr_with_a': lambda dT, r: 1.0 + 1.0 * dT,            # violation: outcome shift grows with the treatment shift
         'b_const_hetero_sign': lambda dT, r: 3.0}[label]
    return run_design(label, rep, bfun=f)


if __name__ == '__main__':
    jobs = [(l, r) for l in ('b_const', 'b_random', 'b_corr_with_a') for r in range(12)]
    with Pool(2) as p:
        d = pd.DataFrame(list(p.imap_unordered(job, jobs)))
    d.to_csv(os.path.join(OUT, 'x10_envreg_raw.csv'), index=False)
    rm = lambda x: np.sqrt(np.mean(x ** 2))
    g = d.groupby('design').agg(pooled_bias=('pooled', 'mean'), pooled_rmse=('pooled', rm), envreg_bias=('envreg', 'mean'), envreg_rmse=('envreg', rm), b_hat=('b_hat', 'mean'))
    g.to_csv(os.path.join(OUT, 'x10_envreg.csv')); print(g.round(3).to_string())
