"""Posterior second-moment completion experiment."""
import os,numpy as np,pandas as pd
from scipy.special import logsumexp
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));OUT=os.path.join(ROOT,"research","results");os.makedirs(OUT,exist_ok=True)
REPS=int(os.getenv("MOMENT_COMPLETION_REPS","150"));rows=[]
for N in [600,1200,2400,4800]:
 for sep in [.5,1.,2.]:
  for rep in range(REPS):
   r=np.random.default_rng(900000000+N*1000+int(sep*10)*100+rep);S=r.integers(0,3,N);mus=np.array([-sep,0.,sep]);sig=np.array([1.,.3,.08]);Z=r.normal(mus[S],1);T=r.normal(0,sig[S]);H=np.eye(3)[S]
   lz=-.5*(Z[:,None]-mus[None,:])**2;g=np.exp(lz-logsumexp(lz,axis=1,keepdims=True))
   lt=-np.log(sig)[None,:]-.5*(T[:,None]/sig[None,:])**2;q=np.exp(lz+lt-logsumexp(lz+lt,axis=1,keepdims=True))
   true=np.linalg.eigvalsh((H.T*(T*T))@H/N)[0]
   outer=np.linalg.eigvalsh((g.T*(T*T))@g/N)[0]
   zg=np.min(np.mean(g*(T*T)[:,None],axis=0));complete=np.min(np.mean(q*(T*T)[:,None],axis=0))
   rows.append((N,sep,rep,true,outer,zg,complete))
d=pd.DataFrame(rows,columns=["N","separation","rep","oracle_lmin","posterior_outer_lmin","Z_only_completed_lmin","ZT_completed_lmin"]);d.to_csv(os.path.join(OUT,"posterior_second_moment_completion_raw.csv"),index=False)
s=d.groupby(["N","separation"]).mean(numeric_only=True).reset_index().drop(columns=["rep"]);s["outer_ratio"]=s.posterior_outer_lmin/s.oracle_lmin;s["Z_only_ratio"]=s.Z_only_completed_lmin/s.oracle_lmin;s["ZT_completed_ratio"]=s.ZT_completed_lmin/s.oracle_lmin;s.to_csv(os.path.join(OUT,"posterior_second_moment_completion_summary.csv"),index=False)
