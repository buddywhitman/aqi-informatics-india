"""Decompose total local target error into representation and sampling components.

Gaussian linearized experiment: theta_hat-theta = J^-1(b_gamma + xi/sqrt(N)).
For target a, compare representation bias |a'J^-1 b| and sampling SD
sqrt(a'J^-1 Sigma J^-1 a/N). Map dominance regions over representation error scale,
N, and direction.
"""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
J=np.diag([1.,.1,.01]);Sigma=J
a=np.array([1.,0.,0.]);rows=[]
for direction,b0 in {"strong":np.array([1.,0,0]),"medium":np.array([0,1.,0]),"weak":np.array([0,0,1.]),"mixed":np.ones(3)/np.sqrt(3)}.items():
 for eps in [1e-4,1e-3,1e-2,1e-1]:
  b=eps*b0;rb=abs(a@np.linalg.solve(J,b))
  for N in [100,1000,10000,100000]:
   Ji=np.linalg.inv(J);sd=np.sqrt(a@Ji@Sigma@Ji@a/N);rows.append((direction,eps,N,rb,sd,rb/sd if sd else np.inf))
pd.DataFrame(rows,columns=["perturbation_direction","epsilon","N","target_representation_bias","target_sampling_sd","bias_to_sampling_ratio"]).to_csv(os.path.join(OUT,"reliability_budget_decomposition.csv"),index=False)
