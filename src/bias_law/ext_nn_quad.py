"""ext_nn_quad.py -- nearest-neighbour constant of the isolated-change formula by quadrature:
c_NN(d) = 2 d e^{d^2/8} E[sigma(W) sigma(-W)], W ~ N(-d^2/2, d^2); -> sqrt(pi/2) with relative error ~ pi^2/(2 d^2).  -> x20_nn_quad.csv"""
import os, numpy as np, pandas as pd
from scipy import integrate
OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'bias_law')
def c(d):
    f = lambda w: np.exp(-(w + d * d / 2) ** 2 / (2 * d * d)) / np.sqrt(2 * np.pi * d * d) / (4 * np.cosh(w / 2) ** 2)
    return 2 * integrate.quad(f, -120, 120, limit=500, points=[-d * d / 2, 0])[0] * d * np.exp(d * d / 8)
rows = [dict(d=d, c_nn=c(d), limit=np.sqrt(np.pi / 2), rel_gap=1 - c(d) / np.sqrt(np.pi / 2), series=np.pi ** 2 / (2 * d * d)) for d in (2, 3, 4, 5, 6, 8, 12, 20, 40)]
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'x20_nn_quad.csv'), index=False); print(pd.DataFrame(rows).round(4).to_string())
