"""
lambda_rule_check.py -- size of the fixed spectral penalty lambda = 0.05 tr(J)/(K sqrt(N)) used in the frontier and
semi-synthetic sweeps (src/synthesis/estimators._solve), measured on the first 10 frontier datasets at Delta_Z in {1, 2}.
Reports lambda, sqrt(N) lambda, lambda_min(J) and the shrinkage-bias bound lambda M_theta / (lambda_min + lambda), M_theta = 2.6.

    python src/rebuttal/lambda_rule_check.py -> reports/rebuttal/lambda_rule_check.csv
"""
import os, sys, warnings
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from src.synthetic_dgp_benchmark import generate_difficulty_dgp
from src.synthesis.estimators import purged, posterior_features, _fit_predict
from src.synthesis.frontier_designs import posteriors, N, L
warnings.filterwarnings('ignore')
rows = []
for dz in (1.0, 2.0):
    for rep in range(10):
        seed = 300000 + int(1000 * dz) + int(100 * 0.88) * 7 + rep
        Y, T, X, Z, gt = generate_difficulty_dgp(N=N, delta_z=dz, persistence=0.88, random_state=seed)
        S = gt['S']; H = np.c_[1 - S, S].astype(float); g = posteriors(Z, seed)
        if np.mean(np.abs(g - H).sum(1)) > 1:
            g = g[:, ::-1]
        F = posterior_features(g, X); rt = np.zeros(N)
        for tr, te in purged(N, 5, L):
            rt[te] = _fit_predict(F, T, tr, te)
        J = np.array([[np.mean(g[:, j] * g[:, k] * rt * rt) for k in range(2)] for j in range(2)])
        lam = 0.05 * np.trace(J) / 2 / np.sqrt(N); lmin = np.linalg.eigvalsh(J).min()
        rows.append((dz, lam, np.sqrt(N) * lam, lmin, lam * 2.6 / (lmin + lam)))
d = pd.DataFrame(rows, columns=['dz', 'lambda', 'sqrtN_lambda', 'lmin', 'shrink_bias_bound'])
d.to_csv(os.path.join(ROOT, 'reports', 'rebuttal', 'lambda_rule_check.csv'), index=False)
print(d.groupby('dz').median().round(5))
