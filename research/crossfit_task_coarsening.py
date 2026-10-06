"""Sample-split task-aware coarsening to avoid using outcome information twice.

Train half: fit K=4 GMM, estimate microstate slopes/SEs, choose pairing by minimum
within-pair Wald discrepancy. Test half: predict microstates and estimate both refined
and selected-coarsened effects. The merge decision never sees test outcomes.
"""
import os,numpy as np,pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));OUT=os.path.join(ROOT,"research","results");os.makedirs(OUT,exist_ok=True)
rows=[]
for N in [800,1600,3200]:
 for rep in range(300):
  r=np.random.default_rng(910000+N*1000+rep); s=r.integers(0,4,N); macro=s//2; theta=np.array([.5,2.5]);sig=np.array([1.,.1,1.,.1])
  v=r.normal(size=N)*sig[s];u=r.normal(size=N);y=theta[macro]*v+u;z=r.normal(np.array([-3.,-1.,1.,3.])[s],.35).reshape(-1,1)
  idx=np.arange(N);r.shuffle(idx);tr=idx[:N//2];te=idx[N//2:]
  gm=GaussianMixture(4,random_state=rep,n_init=3).fit(z[tr]);order=np.argsort(gm.means_.ravel());inv=np.empty(4,int);inv[order]=np.arange(4);a=inv[gm.predict(z[tr])];bte=inv[gm.predict(z[te])]
  bh=[];se=[]
  for k in range(4):
   m=a==k;den=np.sum(v[tr][m]**2);bk=np.sum(v[tr][m]*y[tr][m])/den;res=y[tr][m]-bk*v[tr][m];s2=np.sum(res**2)/max(m.sum()-1,1);bh.append(bk);se.append(np.sqrt(s2/den))
  bh=np.array(bh);se=np.array(se);pairings=[((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2))];score=[sum((bh[x]-bh[q])**2/(se[x]**2+se[q]**2) for x,q in p) for p in pairings];pairing=pairings[int(np.argmin(score))]
  bt=[];cnt=[]
  for k in range(4):
   m=bte==k;den=np.sum(v[te][m]**2);bt.append(np.sum(v[te][m]*y[te][m])/den);cnt.append(m.sum())
  bt=np.array(bt);cnt=np.array(cnt);ref=np.array([np.average(bt[:2],weights=cnt[:2]),np.average(bt[2:],weights=cnt[2:])])
  pool=[]
  for grp in pairing:
   m=np.isin(bte,grp);pool.append(np.sum(v[te][m]*y[te][m])/np.sum(v[te][m]**2))
  correct=set(tuple(sorted(g)) for g in pairing)=={(0,1),(2,3)};p=np.array(pool)
  if correct:
   mp={tuple(sorted(g)):val for g,val in zip(pairing,p)};task=np.array([mp[(0,1)],mp[(2,3)]])
  else: task=p if np.linalg.norm(p-theta)<=np.linalg.norm(p[::-1]-theta) else p[::-1]
  rows.append((N,rep,adjusted_rand_score(s[te],bte),np.linalg.norm(ref-theta),np.linalg.norm(task-theta),correct,np.linalg.norm(task-theta)<np.linalg.norm(ref-theta)))
d=pd.DataFrame(rows,columns=["N","rep","test_microstate_ARI","refined_test_error","coarsened_test_error","correct_pairing","coarsening_wins"])
d.to_csv(os.path.join(OUT,"crossfit_coarsening_raw.csv"),index=False)
s=d.groupby("N").agg(test_microstate_ARI=("test_microstate_ARI","mean"),refined_test_error=("refined_test_error","mean"),coarsened_test_error=("coarsened_test_error","mean"),correct_pairing_rate=("correct_pairing","mean"),coarsening_win_rate=("coarsening_wins","mean")).reset_index()
s.to_csv(os.path.join(OUT,"crossfit_coarsening_summary.csv"),index=False);print(s.to_string(index=False))
