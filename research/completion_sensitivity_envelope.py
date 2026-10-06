"""Sensitivity envelope for completed weak information under uncertain weak-state treatment SD.

True K=3 model. Diagnostic model varies assumed weak-state SD over a multiplicative
grid around a fitted/nominal value. Report min/max completed lambda_min ratio.
"""
import os,numpy as np,pandas as pd
from scipy.special import logsumexp
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));OUT=os.path.join(ROOT,"research","results");os.makedirs(OUT,exist_ok=True)
REPS=int(os.getenv("SENS_ENVELOPE_REPS","200"));rows=[]
mults=np.array([.5,.75,1.,1.25,1.5,2.,3.,4.])
for sep in [.5,1.,2.]:
 for rep in range(REPS):
  N=2400;r=np.random.default_rng(730000000+int(sep*10)*1000+rep);S=r.integers(0,3,N);mu=np.array([-sep,0,sep]);ts=np.array([1.,.3,.08]);Z=r.normal(mu[S],1);T=r.normal(0,ts[S]);H=np.eye(3)[S];oracle=np.min(np.mean(H*(T*T)[:,None],axis=0));lz=-.5*(Z[:,None]-mu[None,:])**2
  vals=[]
  for mult in mults:
   sig=np.array([1.,.3,.08*mult]);lt=-np.log(sig)[None,:]-.5*(T[:,None]/sig[None,:])**2;q=np.exp(lz+lt-logsumexp(lz+lt,axis=1,keepdims=True));vals.append(np.min(np.mean(q*(T*T)[:,None],axis=0)))
  rows.append((sep,rep,oracle,min(vals),max(vals),vals[2],min(vals)/oracle,max(vals)/oracle))
d=pd.DataFrame(rows,columns=["separation","rep","oracle_lmin","envelope_min","envelope_max","nominal","min_ratio","max_ratio"]);d.to_csv(os.path.join(OUT,"completion_sensitivity_envelope_raw.csv"),index=False)
s=d.groupby("separation").mean(numeric_only=True).reset_index().drop(columns=["rep"]);s.to_csv(os.path.join(OUT,"completion_sensitivity_envelope_summary.csv"),index=False)
