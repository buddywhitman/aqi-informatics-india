"""Cross-fitted task-aware state abstraction with heterogeneity penalty.

Train split learns K=4 emission states and state-specific causal slopes. For each
partition of four states into two groups, choose the partition minimizing
  within-group Wald heterogeneity + eta * sum_g 1 / information_g.
Evaluate the frozen partition on held-out outcomes. The experiment sweeps true
within-pair causal heterogeneity delta and weak-state treatment information.
"""
import os,numpy as np,pandas as pd
from sklearn.mixture import GaussianMixture
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));OUT=os.path.join(ROOT,"research","results");os.makedirs(OUT,exist_ok=True)
PAIR=[((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2))]
ETAS=[0.,.01,.1,1.,10.]
rows=[]
for delta in [0.,.1,.25,.5,1.]:
 for weak_sd in [.05,.1,.25,.5]:
  for N in [800,1600]:
   for rep in range(150):
    r=np.random.default_rng(33_000_000+int(delta*100)*100000+int(weak_sd*100)*1000+N+rep)
    s=r.integers(0,4,N); true_theta=np.array([.5-delta/2,.5+delta/2,2.5-delta/2,2.5+delta/2]); sig=np.array([1.,weak_sd,1.,weak_sd])
    v=r.normal(size=N)*sig[s];u=r.normal(size=N);y=true_theta[s]*v+u;z=r.normal(np.array([-3.,-1.,1.,3.])[s],.35).reshape(-1,1)
    idx=r.permutation(N);tr=idx[:N//2];te=idx[N//2:]
    gm=GaussianMixture(4,random_state=rep,n_init=3).fit(z[tr]);order=np.argsort(gm.means_.ravel());inv=np.empty(4,int);inv[order]=np.arange(4);st=inv[gm.predict(z[tr])];se=inv[gm.predict(z[te])]
    bh=[];ses=[];infos=[]
    for k in range(4):
      m=st==k;den=np.sum(v[tr][m]**2);bk=np.sum(v[tr][m]*y[tr][m])/den;res=y[tr][m]-bk*v[tr][m];s2=np.sum(res**2)/max(m.sum()-1,1)
      bh.append(bk);ses.append(np.sqrt(s2/den));infos.append(den)
    bh=np.array(bh);ses=np.array(ses);infos=np.array(infos)
    # refined held-out state slopes
    bt=[]
    for k in range(4):
      m=se==k;bt.append(np.sum(v[te][m]*y[te][m])/np.sum(v[te][m]**2))
    bt=np.array(bt); refined=np.linalg.norm(bt-true_theta)
    for eta in ETAS:
      scores=[]
      for p in PAIR:
        heter=sum((bh[a]-bh[b])**2/(ses[a]**2+ses[b]**2) for a,b in p)
        info_pen=sum(1/max(infos[a]+infos[b],1e-12) for a,b in p)
        scores.append(heter+eta*info_pen*N)
      p=PAIR[int(np.argmin(scores))]
      # held-out group effects; compare to occupancy-weighted true group effects for selected grouping
      est=[];target=[]
      for g in p:
        m=np.isin(se,g);est.append(np.sum(v[te][m]*y[te][m])/np.sum(v[te][m]**2))
        # treatment-information weighted estimand for pooling heterogeneous slopes
        inf=np.array([np.sum(v[te][se==k]**2) for k in g]); target.append(np.sum(inf*true_theta[list(g)])/np.sum(inf))
      coarse=np.linalg.norm(np.array(est)-np.array(target))
      canonical=set(tuple(sorted(x)) for x in p)=={(0,1),(2,3)}
      rows.append((delta,weak_sd,N,rep,eta,refined,coarse,canonical))
d=pd.DataFrame(rows,columns=["delta","weak_sd","N","rep","eta","refined_error","coarsened_estimation_error","canonical_pairing"])
d.to_csv(os.path.join(OUT,"abstraction_tradeoff_raw.csv"),index=False)
s=d.groupby(["delta","weak_sd","N","eta"]).agg(refined_error=("refined_error","mean"),coarsened_error=("coarsened_estimation_error","mean"),canonical_rate=("canonical_pairing","mean")).reset_index()
s["gain_pct"]=100*(s.refined_error-s.coarsened_error)/s.refined_error
s.to_csv(os.path.join(OUT,"abstraction_tradeoff_summary.csv"),index=False)
print(s.to_string(index=False))
