"""Concentration of phantom proxy lambda_min around an optimistic pseudo-geometry."""
import os,numpy as np,pandas as pd
from sklearn.mixture import GaussianMixture
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));OUT=os.path.join(ROOT,"research","results");os.makedirs(OUT,exist_ok=True)
REPS=int(os.getenv("PHANTOM_CONC_REPS","120"));rows=[]
for N in [300,600,1200,2400,4800]:
 for rep in range(REPS):
  r=np.random.default_rng(777000000+N*1000+rep);S=r.integers(0,3,N);Z=r.normal(np.array([-.5,0,.5])[S],1).reshape(-1,1);T=r.normal(size=N)*np.array([1.,.3,.08])[S];H=np.eye(3)[S]
  gm=GaussianMixture(3,random_state=rep,n_init=2).fit(Z);g=gm.predict_proba(Z)
  rows.append((N,rep,np.linalg.eigvalsh((H.T*(T*T))@H/N)[0],np.linalg.eigvalsh((g.T*(T*T))@g/N)[0]))
d=pd.DataFrame(rows,columns=["N","rep","oracle_lmin","proxy_lmin"]);d.to_csv(os.path.join(OUT,"phantom_resolution_concentration_raw.csv"),index=False)
s=d.groupby("N").agg(oracle_mean=("oracle_lmin","mean"),proxy_mean=("proxy_lmin","mean"),proxy_sd=("proxy_lmin","std")).reset_index();s["bias"]=s.proxy_mean-s.oracle_mean;s["bias_in_proxy_sd"]=s.bias/s.proxy_sd;s.to_csv(os.path.join(OUT,"phantom_resolution_concentration_summary.csv"),index=False)
