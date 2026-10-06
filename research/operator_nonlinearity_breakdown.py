"""Adversarial breakdown test: when does first-order J^-1 b cease to predict nonlinear target error?

Solve nonlinear moment F(theta)=J(theta-theta0)+c*(theta-theta0)^2-b=0 componentwise.
Compare exact root near theta0 with linear prediction J^-1 b as perturbation grows and
information weakens.
"""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
rows=[]
for lam in [1.,.2,.05,.01]:
 for curvature in [.1,1.,5.,20.]:
  for b in np.logspace(-5,-.3,30):
   # solve c*d^2 + lam*d - b=0, positive local root
   if curvature==0:d=b/lam
   else:d=(-lam+np.sqrt(lam*lam+4*curvature*b))/(2*curvature)
   lin=b/lam;rel=abs(lin-d)/max(abs(d),1e-15)
   rows.append((lam,curvature,b,d,lin,rel))
pd.DataFrame(rows,columns=["information","curvature","perturbation","exact_displacement","linear_prediction","relative_linearization_error"]).to_csv(os.path.join(OUT,"operator_nonlinearity_breakdown.csv"),index=False)
