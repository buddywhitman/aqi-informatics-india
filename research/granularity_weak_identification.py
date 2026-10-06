"""Simulation of granularity-induced weak identification with healthy within-state overlap.

K_N approximately N^kappa perfectly observed states. Every state has residual treatment
variance 1 and the same causal effect. Compare RMS error of separately estimated
state effects with pooled common-effect error.
"""
import os,numpy as np,pandas as pd
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));OUT=os.path.join(ROOT,"research","results");os.makedirs(OUT,exist_ok=True)
REPS=int(os.getenv("GRANULARITY_REPS","200"));rows=[]
for kappa in [0.,.25,.5,.75,1.]:
 for N in [400,1600,6400,25600]:
  K=max(2,int(round(N**kappa)))
  for rep in range(REPS):
   r=np.random.default_rng(311_000_000+int(kappa*100)*100000+N+rep);S=r.integers(0,K,N);T=r.normal(size=N);theta=.75;Y=theta*T+r.normal(size=N)
   est=[];weights=[]
   for k in range(K):
    m=S==k
    if m.sum()>=2 and np.sum(T[m]**2)>0:
      est.append(np.sum(T[m]*Y[m])/np.sum(T[m]**2));weights.append(m.sum())
   fine_rmse=np.sqrt(np.mean((np.array(est)-theta)**2)) if est else np.nan
   pooled=np.sum(T*Y)/np.sum(T*T)
   rows.append((kappa,N,K,rep,len(est),fine_rmse,abs(pooled-theta)))
d=pd.DataFrame(rows,columns=["kappa","N","K","rep","estimable_states","fine_state_rmse","pooled_abs_error"]);d.to_csv(os.path.join(OUT,"granularity_weak_id_raw.csv"),index=False)
s=d.groupby(["kappa","N","K"]).agg(fine_state_rmse=("fine_state_rmse","mean"),pooled_abs_error=("pooled_abs_error","mean"),estimable_fraction=("estimable_states",lambda x:np.mean(x)/x.name[2] if False else np.nan)).reset_index()
# compute fraction separately
frac=d.groupby(["kappa","N","K"]).apply(lambda g:g.estimable_states.mean()/g.K.iloc[0],include_groups=False).reset_index(name="estimable_fraction")
s=s.drop(columns=["estimable_fraction"]).merge(frac,on=["kappa","N","K"]);s.to_csv(os.path.join(OUT,"granularity_weak_id_summary.csv"),index=False);print(s.to_string(index=False))
