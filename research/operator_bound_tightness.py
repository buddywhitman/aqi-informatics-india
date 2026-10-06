"""Distribution of tightness of lambda_min worst-case reliability bound in high dimension.

J eigenvalues span condition numbers; random perturbation orientations are sampled.
Compute worst-case bound / exact ||J^-1 b||. Shows lambda_min bound can become
increasingly loose for typical perturbations as dimension/anisotropy grow.
"""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
r=np.random.default_rng(181000000);rows=[]
for K in [3,5,10,20,50]:
 for cond in [10,100,1000,10000]:
  lam=np.geomspace(1,1/cond,K);lmin=lam[-1]
  ratios=[]
  for _ in range(20000):
   b=r.normal(size=K);b/=np.linalg.norm(b);exact=np.sqrt(np.sum((b/lam)**2));bound=1/lmin;ratios.append(bound/exact)
  q=np.quantile(ratios,[.1,.5,.9,.99]);rows.append((K,cond,np.mean(ratios),*q))
pd.DataFrame(rows,columns=["K","condition_number","mean_bound_over_exact","q10","median","q90","q99"]).to_csv(os.path.join(OUT,"operator_bound_tightness.csv"),index=False)
