"""Contrast-specific sensitivity versus global lambda_min bound.

For random SPD J and scientific contrasts a, exact worst-case amplification from
unit-norm score perturbation to scalar target a'theta is ||J^{-1}a||, while the
generic bound is ||a||/lambda_min(J). Quantify looseness.
"""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
r=np.random.default_rng(191000000);rows=[]
for K in [3,5,10,20,50]:
 for cond in [10,100,1000,10000]:
  lam=np.geomspace(1,1/cond,K);Q,_=np.linalg.qr(r.normal(size=(K,K)));J=Q@np.diag(lam)@Q.T;lmin=lam[-1];rat=[]
  for _ in range(10000):
   a=r.normal(size=K);a/=np.linalg.norm(a);exact=np.linalg.norm(np.linalg.solve(J,a));rat.append((1/lmin)/exact)
  q=np.quantile(rat,[.1,.5,.9,.99]);rows.append((K,cond,np.mean(rat),*q))
pd.DataFrame(rows,columns=["K","condition_number","mean_global_over_contrast_bound","q10","median","q90","q99"]).to_csv(os.path.join(OUT,"contrast_specific_bound.csv"),index=False)
