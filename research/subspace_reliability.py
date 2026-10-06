"""Subspace reliability: scientific targets often span only a low-dimensional contrast subspace.

Compare global worst-case 1/lambda_min(J) with operator norm of A J^-1, where rows
of A define scientifically relevant contrasts. Weak eigen-directions outside the
target subspace should not determine reliability.
"""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
r=np.random.default_rng(211000000);rows=[]
for K in [5,10,20,50]:
 lam=np.geomspace(1,1e-4,K);Q,_=np.linalg.qr(r.normal(size=(K,K)));J=Q@np.diag(lam)@Q.T;global_amp=1/lam[-1]
 for d in [1,2,3,5]:
  if d>K:continue
  # strong subspace, weak subspace, random subspace
  for typ in ["strong","weak","random"]:
   if typ=="strong":A=Q[:,:d].T
   elif typ=="weak":A=Q[:,-d:].T
   else:
    A,_=np.linalg.qr(r.normal(size=(K,d)));A=A.T
   sub=np.linalg.svd(A@np.linalg.inv(J),compute_uv=False)[0];rows.append((K,d,typ,global_amp,sub,global_amp/sub))
pd.DataFrame(rows,columns=["K","target_dim","subspace_type","global_amplification","target_subspace_amplification","global_over_subspace"]).to_csv(os.path.join(OUT,"subspace_reliability.csv"),index=False)
