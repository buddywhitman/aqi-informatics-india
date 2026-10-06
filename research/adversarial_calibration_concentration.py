"""Same global calibration error, different causal-geometry damage.

Construct posterior errors concentrated either on low-leverage or high-leverage
observations while matching mean squared posterior error. Demonstrates that global
calibration magnitude alone cannot determine downstream causal-geometry error.
"""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
rows=[]
for N in [1000,5000,20000]:
 for rep in range(300):
  r=np.random.default_rng(970000000+N*100+rep);S=r.integers(0,2,N);H=np.eye(2)[S];T=r.normal(size=N)*np.where(S==0,1.,.1);lev=T*T
  for mode in ["low_leverage","high_leverage"]:
   idx=np.argsort(lev);chosen=idx[:N//5] if mode=="low_leverage" else idx[-N//5:]
   g=H.copy();a=.2;g[chosen]=(1-a)*H[chosen]+a*H[chosen][:,::-1]
   global_err=np.mean(np.sum((g-H)**2,1));weighted=np.sum(lev*np.sum((g-H)**2,1))/np.sum(lev)
   oracle=np.min(np.mean(H*lev[:,None],0));proxy=np.linalg.eigvalsh((g.T*lev)@g/N)[0];rows.append((N,rep,mode,global_err,weighted,abs(np.log(max(proxy,1e-12)/max(oracle,1e-12)))))
d=pd.DataFrame(rows,columns="N rep mode global_error task_weighted_error geometry_log_error".split());d.to_csv(os.path.join(OUT,"adversarial_calibration_concentration_raw.csv"),index=False);d.groupby(["N","mode"]).mean(numeric_only=True).reset_index().drop(columns=["rep"]).to_csv(os.path.join(OUT,"adversarial_calibration_concentration_summary.csv"),index=False)
