"""Instability of correcting phantom causal resolution by confusion-matrix inversion."""
import os,numpy as np,pandas as pd
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));OUT=os.path.join(ROOT,"research","results");os.makedirs(OUT,exist_ok=True)
REPS=int(os.getenv("DECONFUSION_REPS","1000"));rows=[]
for N in [1000,5000,20000]:
 for p in [0.,.1,.2,.3,.4,.45,.49]:
  for rep in range(REPS):
   r=np.random.default_rng(123000+N+int(p*1000)*100+rep);S=r.integers(0,2,N);T=r.normal(size=N)*np.where(S==0,1.,.1);W=S.copy();flip=r.random(N)<p;W=np.where(flip,1-W,W)
   true=np.array([np.sum(T[S==s]**2)/N for s in [0,1]]);obs=np.array([np.sum(T[W==w]**2)/N for w in [0,1]])
   C=np.array([[1-p,p],[p,1-p]]);corr=np.linalg.solve(C.T,obs)
   rows.append((N,p,rep,true.min(),obs.min(),corr[1],true[1],obs.min()/true.min(),corr[1]-true[1],corr[1]<0,1/max(1-2*p,1e-12)))
d=pd.DataFrame(rows,columns=["N","misclass_p","rep","true_lmin","naive_proxy_lmin","corrected_weak_info","true_weak_info","naive_lmin_inflation","correction_error","corrected_negative","confusion_inverse_norm"]);d.to_csv(os.path.join(OUT,"confusion_deconvolution_raw.csv"),index=False)
s=d.groupby(["N","misclass_p"]).agg(naive_lmin_inflation=("naive_lmin_inflation","mean"),corrected_weak_rmse=("correction_error",lambda x:np.sqrt(np.mean(x*x))),negative_rate=("corrected_negative","mean"),confusion_inverse_norm=("confusion_inverse_norm","mean")).reset_index();s.to_csv(os.path.join(OUT,"confusion_deconvolution_summary.csv"),index=False);print(s.to_string(index=False))
