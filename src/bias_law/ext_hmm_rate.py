"""ext_hmm_rate.py -- empirical rate of the learned posterior: E|gamma_hat - gamma*| and |v_hat - v*| vs N (well-specified 2-state HMM, EM fit
via src.or_dml.LatentRegimeHMM, known-parameter posterior gamma* as reference).  -> x21_hmm_rate.csv  (log-log slope reported)."""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.bias_law.sim import gen_states_proxy, fit_posteriors, align_to_truth   # noqa: E402
from src.bias_law.ext_joint_proximal import _fwd_bwd                            # noqa: E402
OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'bias_law')
rho, dz = 0.9, 0.8
def one(a):
    N, rep, spec = a
    rng = np.random.default_rng(5000 + 31 * rep + N)
    S, Z, X = gen_states_proxy(N, 2, dz, rho, rng)
    C = np.array([[1, .2], [.2, 1]]) if spec == 'misspec_diag' else np.eye(2)
    if spec != 'misspec_diag':
        Z = np.where(S[:, None] == 1, [dz, 1.5 * dz], [-dz, -1.5 * dz]) + rng.normal(size=(N, 2))
    mu = np.array([[-dz, -1.5 * dz], [dz, 1.5 * dz]]); Ri = np.linalg.inv(C)
    logB = np.stack([-0.5 * np.einsum('ij,jk,ik->i', Z - mu[k], Ri, Z - mu[k]) for k in range(2)], 1)
    gs = _fwd_bwd(logB, np.array([[rho, 1 - rho], [1 - rho, rho]]), np.array([.5, .5]))[0][:, 1]
    gh = align_to_truth(fit_posteriors(Z, 2, rep)['smooth'], S)[:, 1]
    return dict(N=N, rep=rep, spec=spec, l1=float(np.mean(np.abs(gh - gs))), l2=float(np.sqrt(np.mean((gh - gs) ** 2))),
                dv=float(abs(np.mean(gh * (1 - gh)) - np.mean(gs * (1 - gs)))), v_star=float(np.mean(gs * (1 - gs))))
if __name__ == '__main__':
    jobs = [(N, r, sp) for sp in ('well_specified', 'misspec_diag') for N in (500, 1000, 2000, 4000, 8000, 16000) for r in range(24)]
    with Pool(2) as p: d = pd.DataFrame(p.map(one, jobs))
    g = d.groupby(['spec', 'N']).agg(l1=('l1', 'mean'), l2=('l2', 'mean'), dv=('dv', 'mean'), dv_med=('dv', 'median'), v_star=('v_star', 'mean')).reset_index()
    g.to_csv(os.path.join(OUT, 'x21_hmm_rate.csv'), index=False)
    for sp, h in g.groupby('spec'):
        for c in ('l1', 'l2', 'dv_med'):
            print(sp, c, 'log-log slope vs N: %.3f' % np.polyfit(np.log(h.N), np.log(h[c]), 1)[0])
    print(g.round(5).to_string())
