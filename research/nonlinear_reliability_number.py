"""Collapse nonlinear operator error by dimensionless curvature/weak-information number.

For scalar moment lambda*d + c*d^2=b, linear approximation d_lin=b/lambda.
Dimensionless nonlinearity z=c*b/lambda^2 controls breakdown exactly after rescaling.
Compute relative error versus z across many lambda,c,b and verify collapse.
"""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
rows=[]
for lam in [1.,.3,.1,.03,.01]:
 for c in [.1,.3,1.,3.,10.]:
  for b in np.logspace(-7,-.2,80):
   d=(-lam+np.sqrt(lam*lam+4*c*b))/(2*c);lin=b/lam;z=c*b/(lam*lam);rel=abs(lin-d)/abs(d)
   rows.append((lam,c,b,z,rel))
pd.DataFrame(rows,columns=["information","curvature","perturbation","nonlinearity_number","relative_linearization_error"]).to_csv(os.path.join(OUT,"nonlinear_reliability_number.csv"),index=False)
