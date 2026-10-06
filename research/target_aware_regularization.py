"""Target-aware spectral regularization.

J diagonal with one scientific direction and one irrelevant weak direction. Compare
isotropic ridge chosen to stabilize global inverse against selective spectral shrinkage
only in irrelevant direction. Quantify scientific-target bias/variance.
"""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
rows=[]
theta=np.array([1.,2.]);N=2000
for eps in [1e-1,1e-2,1e-3,1e-4]:
 J=np.diag([1.,eps]);S=J@theta
 for lam in [0,.001,.01,.1,1.]:
  iso=np.linalg.solve(J+lam*np.eye(2),S)
  selective=np.linalg.solve(J+np.diag([0.,lam]),S)
  rows.append((eps,lam,abs(iso[0]-theta[0]),abs(selective[0]-theta[0]),np.linalg.cond(J+lam*np.eye(2)),np.linalg.cond(J+np.diag([0.,lam]))))
pd.DataFrame(rows,columns=["weak_eigenvalue","lambda","isotropic_target_bias","selective_target_bias","isotropic_condition","selective_condition"]).to_csv(os.path.join(OUT,"target_aware_regularization.csv"),index=False)
