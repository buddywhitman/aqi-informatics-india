"""Task-weighted calibration versus ordinary calibration.

Synthetic K=3 posterior distortions preserve/alter global calibration differently.
Evaluate ordinary multiclass Brier/ECE-like error versus treatment-information-weighted
posterior moment error and oracle causal-geometry error.
"""
import os,numpy as np,pandas as pd
from scipy.special import softmax
from scipy.stats import spearmanr
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
rows=[]
for sep in [.5,1.,2.]:
 for weak in [.04,.08,.16]:
  for temp in [.5,.75,1.,1.5,2.,3.]:
   for rep in range(120):
    N=2000;r=np.random.default_rng(930000000+int(sep*10)*100000+int(weak*100)*1000+int(temp*10)*100+rep);S=r.integers(0,3,N);mu=np.array([-sep,0,sep]);sig=np.array([1.,.3,weak]);Z=r.normal(mu[S],1);T=r.normal(0,sig[S]);H=np.eye(3)[S];logit=-.5*(Z[:,None]-mu)**2;g=softmax(logit/temp,axis=1)
    brier=np.mean(np.sum((g-H)**2,1));weighted=np.sum(T**2*np.sum((g-H)**2,1))/np.sum(T**2)
    oracle=np.min(np.mean(H*T[:,None]**2,0));proxy=np.linalg.eigvalsh((g.T*T**2)@g/N)[0];geom=abs(np.log(max(proxy,1e-12)/max(oracle,1e-12)))
    rows.append((sep,weak,temp,rep,brier,weighted,geom))
d=pd.DataFrame(rows,columns="separation weak_sd temperature rep brier task_weighted_brier geometry_log_error".split());d.to_csv(os.path.join(OUT,"task_weighted_calibration_raw.csv"),index=False)
pd.DataFrame([{"metric":"ordinary_brier","rho_geometry_error":spearmanr(d.brier,d.geometry_log_error).statistic},{"metric":"task_weighted_brier","rho_geometry_error":spearmanr(d.task_weighted_brier,d.geometry_log_error).statistic}]).to_csv(os.path.join(OUT,"task_weighted_calibration_summary.csv"),index=False)
