"""ext_apriori_check.py -- a priori v (change-point formula) plugged into the law vs observed learned-HMM bias (E1 cells with dz in {0.8,1.2}), all learner/mode groups,
for the practical constant sqrt(8/pi) (fitted on simulated exact-HMM v from the same generator) and the derived asymptotic sqrt(pi/2).  -> x18_apriori_check.csv"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.bias_law import bias_law as BL      # noqa: E402
R = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'bias_law')
d = pd.read_csv(os.path.join(R, 'e1_learned_hmm_summary.csv')); d = d[d.dz >= 0.8].copy()
d2 = 11.04 * d.dz ** 2     # Mahalanobis separation for the generator's means +-dz*(1,1.5), proxy covariance [[1,.2],[.2,1]]
rows = []
for c, name in ((np.sqrt(8 / np.pi), 'sqrt(8/pi)'), (np.sqrt(np.pi / 2), 'sqrt(pi/2)')):
    d['law_prior'] = [BL.law_bias(r.dT, 3.0, BL.v_changepoint(0.9, x, c=c)) for r, x in zip(d.itertuples(), d2)]
    for (mode, nuis), g in d.groupby(['mode', 'nuis']):
        rows.append(dict(constant=name, mode=mode, nuis=nuis, n_cells=len(g), rmse_prior=float(np.sqrt(np.mean((g.bias - g.law_prior) ** 2))),
                         rmse_law_vhat=float(np.sqrt(np.mean((g.bias - g.law_hat) ** 2)))))
o = pd.DataFrame(rows); o.to_csv(os.path.join(R, 'x18_apriori_check.csv'), index=False); print(o.round(4).to_string())
