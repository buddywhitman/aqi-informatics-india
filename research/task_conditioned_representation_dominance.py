"""Pareto dominance of representations across a family of downstream leverage tasks.

Representations have error profiles e_r(x). Tasks are leverage functions w_t(x).
Compute risk matrix R[r,t]=E[w_t e_r]/E[w_t], identify Pareto-dominated representations,
and show that a representation with best global calibration need not be on the
task-reliability frontier.
"""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
r=np.random.default_rng(120300000);N=200000;x=r.random(N)
# error profiles; 'global_best' has lower average error but concentrated in central high-leverage region for one task
profiles={
 "uniform":np.full(N,.01),
 "left_specialist":np.where(x<.2,.002,.014),
 "right_specialist":np.where(x>.8,.002,.014),
 "center_specialist":np.where((x>.4)&(x<.6),.002,.014),
 "global_best":np.where((x>.4)&(x<.6),.03,.003)
}
tasks={"uniform":np.ones(N),"left":np.where(x<.2,100.,1.),"right":np.where(x>.8,100.,1.),"center":np.where((x>.4)&(x<.6),100.,1.)}
rows=[]
for name,e in profiles.items():
 row={"representation":name,"global_error":e.mean()}
 for t,w in tasks.items():row[t]=np.sum(w*e)/np.sum(w)
 rows.append(row)
d=pd.DataFrame(rows);d.to_csv(os.path.join(OUT,"task_conditioned_representation_risk.csv"),index=False)
# Pareto dominance across task columns (excluding global metric)
T=list(tasks);dom=[]
for i,a in d.iterrows():
 dominated=False
 for j,b in d.iterrows():
  if i==j:continue
  if all(b[t]<=a[t] for t in T) and any(b[t]<a[t] for t in T):dominated=True;break
 dom.append(dominated)
d["pareto_dominated"]=dom;d.to_csv(os.path.join(OUT,"task_conditioned_representation_pareto.csv"),index=False)
