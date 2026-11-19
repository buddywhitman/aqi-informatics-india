"""
joint_frontier.py -- joint Markov-switching MLE with regime-specific effects (generalised Table-10 competitor)
on the frontier DGP of src/synthesis/frontier_designs.py: same datasets (seeds), same EM posteriors used for
initialisation, 100 datasets per cell.  Reports ATE bias against the sample ATE and regime-effect error.

    python src/rebuttal/joint_frontier.py -> reports/rebuttal/joint_frontier_{raw,summary}.csv
"""
import os, sys, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from src.synthetic_dgp_benchmark import generate_difficulty_dgp
from src.synthesis.frontier_designs import posteriors, N, THETA
from src.rebuttal.joint_semisynthetic import joint_em
warnings.filterwarnings('ignore')
OUT = os.path.join(ROOT, 'reports', 'rebuttal'); os.makedirs(OUT, exist_ok=True)
REPS, P00 = 100, 0.88


def one(dz, rep):
    seed = 300000 + int(1000 * dz) + int(100 * P00) * 7 + rep
    Y, T, X, Z, gt = generate_difficulty_dgp(N=N, delta_z=dz, persistence=P00, random_state=seed)
    S = gt['S']; H = np.c_[1 - S, S].astype(float)
    g = posteriors(Z, seed)
    if np.mean(np.abs(g - H).sum(1)) > 1.0:
        g = g[:, ::-1]
    sate = float(THETA[S].mean())
    ate, thk = joint_em(Y, T, X, Z, g, hetero=True)
    if np.linalg.norm(thk - THETA) > np.linalg.norm(thk[::-1] - THETA):     # evaluation-only label alignment
        thk = thk[::-1]
    return dict(dz=dz, rep=rep, ate=ate, bias=ate - sate, theta_err=float(np.linalg.norm(thk - THETA)))


if __name__ == '__main__':
    d = pd.DataFrame(Parallel(n_jobs=-1)(delayed(one)(dz, r) for dz in (0.5, 1.0, 2.0, 4.0) for r in range(REPS)))
    d.to_csv(os.path.join(OUT, 'joint_frontier_raw.csv'), index=False)
    s = d.groupby('dz').agg(mean_abs_bias=('bias', lambda b: np.mean(np.abs(b))), mean_bias=('bias', 'mean'),
                            rmse=('bias', lambda b: np.sqrt(np.mean(b ** 2))), theta_err_median=('theta_err', 'median')).reset_index()
    s.to_csv(os.path.join(OUT, 'joint_frontier_summary.csv'), index=False)
    print(s.round(3).to_string())
