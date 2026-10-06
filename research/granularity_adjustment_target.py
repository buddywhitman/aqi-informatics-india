"""Fine adjustment versus coarse causal target."""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
rows=[]
for N in [1000,5000,20000,100000]:
 for K in [2,5,10,20,50,100,200]:
  # fine state-specific effect SE versus pooled target SE, healthy overlap
  fine=np.sqrt(K/N);pooled=1/np.sqrt(N)
  rows.append((N,K,fine,pooled,fine/pooled))
pd.DataFrame(rows,columns=["N","K","fine_effect_se_scale","pooled_target_se_scale","fine_over_pooled"]).to_csv(os.path.join(OUT,"granularity_adjustment_target.csv"),index=False)
