"""First- versus second-order sensitivity to representation perturbation.

Compare target error when score is:
- non-orthogonal to representation: bias ~ epsilon;
- representation-orthogonal by construction: leading bias ~ epsilon^2.
Map rate requirements for nominal root-N inference.
"""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
rows=[]
for r in [.2,.25,.3,.4,.5,.6]:
 for N in [400,1600,6400,25600,102400]:
  eps=N**(-r);first=eps;second=eps**2;se=N**-.5
  rows.append((r,N,eps,first/se,second/se))
pd.DataFrame(rows,columns=["representation_rate","N","epsilon","first_order_bias_over_se","second_order_bias_over_se"]).to_csv(os.path.join(OUT,"orthogonality_to_representation.csv"),index=False)
