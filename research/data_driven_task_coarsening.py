"""Data-driven task-aware coarsening of four well-recovered microstates.

Fit K=4 Gaussian mixture to emissions, estimate state-specific residual slopes and SEs,
then choose the pairing of four states into two macro-states minimizing within-pair
Wald discrepancy. This uses downstream effect similarity, not the true grouping.
"""
import os,numpy as np,pandas as pd
from itertools import combinations
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));OUT=os.path.join(ROOT,"research","results");os.makedirs(OUT,exist_ok=True)
rows=[]
for N in [400,800,1600]:
 for rep in range(500):
  r=np.random.default_rng(700000+N*1000+rep); s=r.integers(0,4,N); macro=s//2; theta=np.array([.5,2.5]); sig=np.array([1.,.1,1.,.1])
  v=r.normal(size=N)*sig[s]; u=r.normal(size=N); y=theta[macro]*v+u; z=r.normal(np.array([-3.,-1.,1.,3.])[s],.35).reshape(-1,1)
  gm=GaussianMixture(4,random_state=rep,n_init=3).fit(z); pred=gm.predict(z); order=np.argsort(gm.means_.ravel()); inv=np.empty(4,int);inv[order]=np.arange(4); shat=inv[pred]
  b=[];se=[];info=[];counts=[]
  for k in range(4):
   m=shat==k; den=np.sum(v[m]**2); bk=np.sum(v[m]*y[m])/den; resid=y[m]-bk*v[m]; s2=np.sum(resid**2)/max(m.sum()-1,1)
   b.append(bk);se.append(np.sqrt(s2/den));info.append(den);counts.append(m.sum())
  b=np.array(b);se=np.array(se);counts=np.array(counts)
  refined=np.array([np.average(b[:2],weights=counts[:2]),np.average(b[2:],weights=counts[2:])])
  pairings=[((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2))]
  scores=[sum((b[a]-b[bb])**2/(se[a]**2+se[bb]**2) for a,bb in p) for p in pairings]; pairing=pairings[int(np.argmin(scores))]
  pooled=[]
  for grp in pairing:
   m=np.isin(shat,grp);pooled.append(np.sum(v[m]*y[m])/np.sum(v[m]**2))
  correct=set(tuple(sorted(x)) for x in pairing)=={(0,1),(2,3)}
  if correct:
   mp={tuple(sorted(g)):val for g,val in zip(pairing,pooled)}; task=np.array([mp[(0,1)],mp[(2,3)]])
  else:
   p=np.array(pooled); task=p if np.linalg.norm(p-theta)<=np.linalg.norm(p[::-1]-theta) else p[::-1]
  om=[]
  for k in range(2):
   m=macro==k;om.append(np.sum(v[m]*y[m])/np.sum(v[m]**2))
  rows.append((N,rep,adjusted_rand_score(s,shat),np.linalg.norm(refined-theta),np.linalg.norm(task-theta),np.linalg.norm(np.array(om)-theta),correct))
d=pd.DataFrame(rows,columns=["N","rep","microstate_ARI","refined_error","task_coarsened_error","oracle_macro_error","correct_pairing"])
d.to_csv(os.path.join(OUT,"data_driven_coarsening_raw.csv"),index=False)
s=d.groupby("N").agg(microstate_ARI=("microstate_ARI","mean"),refined_error=("refined_error","mean"),task_coarsened_error=("task_coarsened_error","mean"),oracle_macro_error=("oracle_macro_error","mean"),correct_pairing_rate=("correct_pairing","mean")).reset_index()
s["task_coarsening_win_rate"]=d.assign(win=d.task_coarsened_error<d.refined_error).groupby("N").win.mean().values
s.to_csv(os.path.join(OUT,"data_driven_coarsening_summary.csv"),index=False);print(s.to_string(index=False))
