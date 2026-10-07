"""make_ext_tables.py -> paper/generated/tab_ext.tex from x2/x3 CSVs (bias, RMSE in parentheses)."""
import os
import pandas as pd
R = os.path.join(os.path.dirname(__file__), '..', '..')
x2 = pd.read_csv(os.path.join(R, 'reports/bias_law/x2_misspec.csv'))
x3 = pd.read_csv(os.path.join(R, 'reports/bias_law/x3_contlat.csv'))
names = {'base': 'HMM correctly specified', 'nonlinX': 'nonlinear $f(X)$ (GBM nuisances)', 'heavyZ': 'heavy-tailed ($t_3$) proxy noise',
         'K3true': 'true $K{=}3$, fitted $K{=}2$', 'hetero': 'heterogeneous $\\theta(S)$ (ATE target)', 'leakW': 'negative control leaks ($W\\ni0.3T$)'}
f = lambda b, r: f'{b:+.2f} ({r:.2f})'
rows = []
for sc in ['base', 'nonlinX', 'heavyZ', 'K3true', 'hetero', 'leakW']:
    for dT in (1, 4):
        r = x2[(x2.scen == sc) & (x2.dT == dT)].iloc[0]
        rows.append(f"{names[sc]} & {dT} & {f(r.dml, r.rmse_dml)} & {f(r.joint, r.rmse_joint)} & {f(r.prox, r.rmse_prox)} \\\\")
for _, r in x3.iterrows():
    rows.append(f"continuous AR(1) latent, SNR {r.snr:g} & {r.dT:g} & {f(r.dml_bias, r.dml_rmse)} & {f(r.joint_bias, r.joint_rmse)} & {f(r.prox_gamma_bias, r.prox_gamma_rmse)} \\\\")
tab = ("\\begin{tabular}{llccc}\n\\toprule\nScenario & $\\Delta_T$ & Posterior-DML & Joint MLE & Proximal 2SLS \\\\\n\\midrule\n"
       + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
open(os.path.join(R, 'paper/generated/tab_ext.tex'), 'w').write(tab)
print(tab)

# --- Kalman / Wiener primitives table
import sys
sys.path.insert(0, os.path.abspath(R))
from src.bias_law import bias_law as BL
k = pd.read_csv(os.path.join(R, 'reports/bias_law/x6_kalman.csv'))
H, RR = [1, 1.5], [[1, .2], [.2, 1]]
rows = []
for _, r in k[k.dT == 1.0].iterrows():
    io = BL.sensor_information(H, RR, r.snr)
    vc = BL.wiener_smoother_var(r.phi, io)
    rows.append(f"{r.phi:g} & {io:.2f} & {vc:.3f} & {BL.law_bias(1.0, 3.0, vc):.3f} & {r.bias:.3f} \\\\")
tab2 = ("\\begin{tabular}{ccccc}\n\\toprule\n$\\phi$ & $\\iota$ & closed-form $v$ & predicted bias & observed bias \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
open(os.path.join(R, 'paper/generated/tab_kalman.tex'), 'w').write(tab2)
print(tab2)
