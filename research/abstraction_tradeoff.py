"""Cross-fitted task-abstraction bias-variance frontier.

Four observational microstates are learned on a training split. A two-group partition
is selected on training outcomes by minimum within-group Wald heterogeneity. Held-out
evaluation scores the resulting pooled coefficient assigned back to *each microstate*
against its true microstate effect, so causal heterogeneity creates approximation bias.
This is the relevant tradeoff: coarsening lowers variance but can erase real task signal.
"""
import os,numpy as np,pandas as pd
from sklearn.mixture import GaussianMixture
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));OUT=os.path.join(ROOT,"research","results");os.makedirs(OUT,exist_ok=True)
PAIR=[((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2))]
REPS=int(os.getenv("ABSTRACTION_REPS","50"));rows=[]
for delta in [0.,.1,.25,.5,1.,2.]:
 for weak_sd in [.05,.1,.25,.5]:
  for N in [800,1600]:
   for rep in range(REPS):
    r=np.random.default_rng(44_000_000+int(delta*100)*100000+int(weak_sd*100)*1000+N+rep)
    s=r.integers(0,4,N); th=np.array([.5-delta/2,.5+delta/2,2.5-delta/2,2.5+delta/2]);sig=np.array([1.,weak_sd,1.,weak_sd])
    v=r.normal(size=N)*sig[s];u=r.normal(size=N);y=th[s]*v+u;z=r.normal(np.array([-3.,-1.,1.,3.])[s],.35).reshape(-1,1)
    idx=r.permutation(N);tr=idx[:N//2];te=idx[N//2:]
    gm=GaussianMixture(4,random_state=rep,n_init=2).fit(z[tr]);order=np.argsort(gm.means_.ravel());inv=np.empty(4,int);inv[order]=np.arange(4);st=inv[gm.predict(z[tr])];se=inv[gm.predict(z[te])]
    bh=[];ses=[]
    for k in range(4):
      m=st==k;den=np.sum(v[tr][m]**2);bk=np.sum(v[tr][m]*y[tr][m])/den;res=y[tr][m]-bk*v[tr][m];s2=np.sum(res**2)/max(m.sum()-1,1);bh.append(bk);ses.append(np.sqrt(s2/den))
    bh=np.array(bh);ses=np.array(ses);score=[sum((bh[a]-bh[b])**2/(ses[a]**2+ses[b]**2) for a,b in p) for p in PAIR];p=PAIR[int(np.argmin(score))]
    bt=[]
    for k in range(4):
      m=se==k;bt.append(np.sum(v[te][m]*y[te][m])/np.sum(v[te][m]**2))
    refined=np.linalg.norm(np.array(bt)-th);pred=np.zeros(4)
    for g in p:
      m=np.isin(se,g);pooled=np.sum(v[te][m]*y[te][m])/np.sum(v[te][m]**2);pred[list(g)]=pooled
    coarse=np.linalg.norm(pred-th)
    rows.append((delta,weak_sd,N,rep,refined,coarse,set(tuple(sorted(x)) for x in p)=={(0,1),(2,3)}))
d=pd.DataFrame(rows,columns=["delta","weak_sd","N","rep","refined_error","coarsened_total_error","canonical_pairing"])
d.to_csv(os.path.join(OUT,"abstraction_tradeoff_raw.csv"),index=False)
s=d.groupby("delta").agg(refined_error=("refined_error","mean"),coarsened_error=("coarsened_total_error","mean"),canonical_rate=("canonical_pairing","mean")).reset_index()
s["coarsening_gain_pct"]=100*(s.refined_error-s.coarsened_error)/s.refined_error
s.to_csv(os.path.join(OUT,"abstraction_tradeoff_summary.csv"),index=False);print(s.to_string(index=False))
