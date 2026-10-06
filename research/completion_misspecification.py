"""Misspecification stress test for posterior second-moment completion.

True K=3 iid states, Gaussian Z and state-specific treatment SD (1,.3,.08).
Diagnostic posterior q(S|Z,T) is computed under misspecified treatment SDs:
- correct;
- common variance (ignores state information in T);
- weak variance 2x/4x too large;
- strong variance compressed.
Evaluate oracle weak information recovery.
"""
import os,numpy as np,pandas as pd
from scipy.special import logsumexp
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));OUT=os.path.join(ROOT,"research","results");os.makedirs(OUT,exist_ok=True)
REPS=int(os.getenv("COMPLETION_MISSPEC_REPS","300"));rows=[]
models={"correct":np.array([1.,.3,.08]),"common":np.array([.5,.5,.5]),"weak_x2":np.array([1.,.3,.16]),"weak_x4":np.array([1.,.3,.32]),"compressed":np.array([.6,.3,.15])}
for sep in [.5,1.,2.]:
 for N in [600,2400,9600]:
  for rep in range(REPS):
   r=np.random.default_rng(620000000+N*1000+int(sep*10)*100+rep);S=r.integers(0,3,N);mu=np.array([-sep,0,sep]);true_sig=np.array([1.,.3,.08]);Z=r.normal(mu[S],1);T=r.normal(0,true_sig[S]);H=np.eye(3)[S];oracle=np.min(np.mean(H*(T*T)[:,None],axis=0));lz=-.5*(Z[:,None]-mu[None,:])**2
   for name,sig in models.items():
    lt=-np.log(sig)[None,:]-.5*(T[:,None]/sig[None,:])**2;q=np.exp(lz+lt-logsumexp(lz+lt,axis=1,keepdims=True));est=np.min(np.mean(q*(T*T)[:,None],axis=0))
    rows.append((sep,N,rep,name,oracle,est,est/oracle))
d=pd.DataFrame(rows,columns=["separation","N","rep","model","oracle_lmin","completed_lmin","ratio"]);d.to_csv(os.path.join(OUT,"completion_misspecification_raw.csv"),index=False)
s=d.groupby(["separation","N","model"]).agg(mean_ratio=("ratio","mean"),sd_ratio=("ratio","std")).reset_index();s.to_csv(os.path.join(OUT,"completion_misspecification_summary.csv"),index=False)
