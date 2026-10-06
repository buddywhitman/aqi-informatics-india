"""Finite-task impossibility for any task-agnostic scalar metric.

For a fixed scalar global score ordering A above B, construct downstream leverage tasks
that concentrate on regions where B has lower error. Quantify regret as leverage ratio
grows. This complements the Pareto experiment with an explicit adversarial theorem-like
construction.
"""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
r=np.random.default_rng(161000000);N=500000;x=r.random(N)
# A globally better: small error almost everywhere, large on rare 5% region.
# B globally worse: moderate everywhere, tiny on rare region.
A=np.where(x<.05,.08,.002);B=np.where(x<.05,.0005,.008)
rows=[]
for L in [1,2,5,10,20,50,100,500,1000,10000]:
 w=np.where(x<.05,L,1.);ra=np.sum(w*A)/np.sum(w);rb=np.sum(w*B)/np.sum(w)
 rows.append((L,A.mean(),B.mean(),ra,rb,ra/rb))
pd.DataFrame(rows,columns=["leverage_ratio","global_A","global_B","task_A","task_B","regret_A_over_B"]).to_csv(os.path.join(OUT,"representation_metric_impossibility.csv"),index=False)
