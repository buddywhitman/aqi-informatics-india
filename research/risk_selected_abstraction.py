"""Cross-fitted task-aware partition selection with a train-only risk criterion.

Candidate partitions: K=4 refinement and all 3 pairings into two groups. Training half
estimates state slopes/SEs and predicts held-out microstate-vector MSE as:
  approximation bias^2 + estimation variance.
The selected partition is frozen and evaluated on held-out outcomes.
"""
import os,numpy as np,pandas as pd
from sklearn.mixture import GaussianMixture
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));OUT=os.path.join(ROOT,"research","results");os.makedirs(OUT,exist_ok=True)
PAIR=[((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2))];REPS=int(os.getenv("RISK_SELECTION_REPS","100"));rows=[]
for delta in [0.,.1,.25,.5,1.,2.]:
 for weak in [.05,.1,.25,.5]:
  for N in [800,1600,3200]:
   for rep in range(REPS):
    r=np.random.default_rng(61_000_000+int(delta*100)*100000+int(weak*100)*1000+N+rep);s=r.integers(0,4,N);th=np.array([.5-delta/2,.5+delta/2,2.5-delta/2,2.5+delta/2]);sig=np.array([1.,weak,1.,weak]);v=r.normal(size=N)*sig[s];u=r.normal(size=N);y=th[s]*v+u;z=r.normal(np.array([-3.,-1.,1.,3.])[s],.35).reshape(-1,1)
    idx=r.permutation(N);tr=idx[:N//2];te=idx[N//2:];gm=GaussianMixture(4,random_state=rep,n_init=2).fit(z[tr]);order=np.argsort(gm.means_.ravel());inv=np.empty(4,int);inv[order]=np.arange(4);st=inv[gm.predict(z[tr])];se=inv[gm.predict(z[te])]
    bh=[];var=[]
    for k in range(4):
      m=st==k;den=np.sum(v[tr][m]**2);bk=np.sum(v[tr][m]*y[tr][m])/den;res=y[tr][m]-bk*v[tr][m];s2=np.sum(res**2)/max(m.sum()-1,1);bh.append(bk);var.append(s2/den)
    bh=np.array(bh);var=np.array(var)
    # candidate risk: refined=sum variances; pooled group uses information weights and estimated heterogeneity around pooled slope
    candidates=[tuple((k,) for k in range(4))]+PAIR; risks=[]
    for p in candidates:
      rr=0
      for g in p:
        g=list(g)
        if len(g)==1: rr+=var[g[0]]
        else:
          w=1/np.maximum(var[g],1e-12); mu=np.sum(w*bh[g])/np.sum(w); approx=np.sum((bh[g]-mu)**2); pooled_var=1/np.sum(w); rr+=approx+len(g)*pooled_var
      risks.append(rr)
    pick=candidates[int(np.argmin(risks))]
    def evalpart(p):
      pred=np.zeros(4)
      for g in p:
        m=np.isin(se,list(g));po=np.sum(v[te][m]*y[te][m])/np.sum(v[te][m]**2);pred[list(g)]=po
      return np.linalg.norm(pred-th)
    selected=evalpart(pick);refined=evalpart(candidates[0]);oracle=min(evalpart(p) for p in candidates)
    rows.append((delta,weak,N,rep,str(pick),selected,refined,oracle,selected<=refined))
d=pd.DataFrame(rows,columns=["delta","weak_sd","N","rep","selected_partition","selected_error","refined_error","oracle_partition_error","selected_beats_refined"]);d.to_csv(os.path.join(OUT,"risk_selected_abstraction_raw.csv"),index=False)
s=d.groupby(["delta","N"]).agg(selected_error=("selected_error","mean"),refined_error=("refined_error","mean"),oracle_error=("oracle_partition_error","mean"),win_rate=("selected_beats_refined","mean")).reset_index();s["regret_to_oracle"]=s.selected_error-s.oracle_error;s.to_csv(os.path.join(OUT,"risk_selected_abstraction_summary.csv"),index=False);print(s.to_string(index=False))
