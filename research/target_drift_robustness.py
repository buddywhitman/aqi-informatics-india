"""Cost of target-aware specialization under target drift.

Regularize/suppress directions deemed irrelevant for target a0, then evaluate a rotated
future target a(phi). Quantifies robustness-specialization frontier.
"""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
theta=np.array([1.,2.]);eps=1e-3;J=np.diag([1.,eps]);S=J@theta
rows=[]
for lam in [.001,.01,.1,1.]:
 iso=np.linalg.solve(J+lam*np.eye(2),S);sel=np.linalg.solve(J+np.diag([0.,lam]),S)
 for deg in [0,1,2,5,10,20,45,90]:
  p=np.deg2rad(deg);a=np.array([np.cos(p),np.sin(p)]);truth=a@theta
  rows.append((lam,deg,abs(a@iso-truth),abs(a@sel-truth)))
pd.DataFrame(rows,columns=["lambda","target_rotation_deg","isotropic_abs_bias","specialized_abs_bias"]).to_csv(os.path.join(OUT,"target_drift_robustness.csv"),index=False)
