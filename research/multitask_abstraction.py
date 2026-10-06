"""Same latent microstates, different downstream tasks, different optimal abstractions."""
import os,numpy as np,pandas as pd
from sklearn.mixture import GaussianMixture
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));OUT=os.path.join(ROOT,"research","results");os.makedirs(OUT,exist_ok=True)
PAIR=[((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2))];rows=[]
for N in [800,1600,3200]:
 for rep in range(100):
  r=np.random.default_rng(223000+N*100+rep);s=r.integers(0,4,N);v=r.normal(size=N);z=r.normal(np.array([-3.,-1.,1.,3.])[s],.35).reshape(-1,1)
  ths=[np.array([.5,.5,2.5,2.5]),np.array([.5,2.5,.5,2.5])];ys=[th[s]*v+r.normal(size=N) for th in ths]
  idx=r.permutation(N);tr=idx[:N//2];te=idx[N//2:];gm=GaussianMixture(4,random_state=rep,n_init=2).fit(z[tr]);order=np.argsort(gm.means_.ravel());inv=np.empty(4,int);inv[order]=np.arange(4);st=inv[gm.predict(z[tr])];se=inv[gm.predict(z[te])]
  out=[]
  for y,th,truep in zip(ys,ths,[{(0,1),(2,3)},{(0,2),(1,3)}]):
   bh=[];ses=[]
   for k in range(4):
    m=st==k;den=np.sum(v[tr][m]**2);bk=np.sum(v[tr][m]*y[tr][m])/den;res=y[tr][m]-bk*v[tr][m];s2=np.sum(res**2)/max(m.sum()-1,1);bh.append(bk);ses.append(np.sqrt(s2/den))
   p=PAIR[int(np.argmin([sum((bh[a]-bh[b])**2/(ses[a]**2+ses[b]**2) for a,b in q) for q in PAIR]))];pred=np.zeros(4)
   for g in p:
    m=np.isin(se,g);po=np.sum(v[te][m]*y[te][m])/np.sum(v[te][m]**2);pred[list(g)]=po
   out.append((set(tuple(sorted(x)) for x in p)==truep,np.linalg.norm(pred-th),str(p)))
  rows.append((N,rep,out[0][0],out[1][0],out[0][1],out[1][1],out[0][2],out[1][2]))
d=pd.DataFrame(rows,columns=["N","rep","taskA_correct","taskB_correct","taskA_error","taskB_error","taskA_partition","taskB_partition"]);d.to_csv(os.path.join(OUT,"multitask_abstraction_raw.csv"),index=False)
s=d.groupby("N").agg(taskA_correct=("taskA_correct","mean"),taskB_correct=("taskB_correct","mean"),taskA_error=("taskA_error","mean"),taskB_error=("taskB_error","mean")).reset_index();s.to_csv(os.path.join(OUT,"multitask_abstraction_summary.csv"),index=False);print(s.to_string(index=False))
