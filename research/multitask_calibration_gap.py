"""Same representation errors, incompatible task-induced calibration measures.

Two downstream tasks put leverage on disjoint regions. Construct two representations
whose errors are concentrated respectively on those regions. Shows ranking reversal:
representation A is safer for task B and worse for task A; representation B vice versa.
"""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
rows=[]
for N in [2000,10000,50000]:
 for rep in range(300):
  r=np.random.default_rng(990000000+N*10+rep);x=r.random(N);S=r.integers(0,2,N);H=np.eye(2)[S]
  wA=np.where(x<.2,100.,1.);wB=np.where(x>.8,100.,1.)
  for name,region in [("err_left",x<.2),("err_right",x>.8)]:
   g=H.copy();a=.2;g[region]=(1-a)*H[region]+a*H[region][:,::-1];e=np.sum((g-H)**2,1)
   rows.append((N,rep,name,e.mean(),np.sum(wA*e)/np.sum(wA),np.sum(wB*e)/np.sum(wB)))
d=pd.DataFrame(rows,columns="N rep representation global_error taskA_error taskB_error".split());d.to_csv(os.path.join(OUT,"multitask_calibration_gap_raw.csv"),index=False);d.groupby(["N","representation"]).mean(numeric_only=True).reset_index().drop(columns=["rep"]).to_csv(os.path.join(OUT,"multitask_calibration_gap_summary.csv"),index=False)
