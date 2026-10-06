"""Robust target-family regularization.

Instead of specializing to one target a0, declare a cone of plausible future targets
a(phi), phi in [-Phi,Phi]. Choose selective penalty lambda2 on weak direction to
minimize worst-case target bias plus a stability penalty proportional to inverse weak
regularized information. Demonstrates interpolation between confirmatory specialization
and exploratory robustness.
"""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
theta=np.array([1.,2.]);eps=1e-3;J=np.diag([1.,eps]);S=J@theta
rows=[]
for Phi in [0,2,5,10,20,45,90]:
 phis=np.deg2rad(np.linspace(-Phi,Phi,max(3,2*Phi+1)))
 for stability_weight in [.0001,.001,.01,.1]:
  best=None
  for lam in np.r_[0,np.logspace(-5,1,121)]:
   est=np.linalg.solve(J+np.diag([0.,lam]),S);bias=max(abs(np.array([np.cos(p),np.sin(p)])@(est-theta)) for p in phis);stability=stability_weight/(eps+lam);obj=bias+bias+stability # emphasize target bias, penalize instability
   if best is None or obj<best[0]:best=(obj,lam,bias,stability)
  rows.append((Phi,stability_weight,best[1],best[2],best[3],best[0]))
pd.DataFrame(rows,columns=["target_cone_deg","stability_weight","selected_lambda","worst_target_bias","stability_penalty","objective"]).to_csv(os.path.join(OUT,"robust_target_family_regularization.csv"),index=False)
