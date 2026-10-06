"""Simulation of opposing observational and causal resolution with increasing N."""
import os,numpy as np,pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));OUT=os.path.join(ROOT,"research","results");os.makedirs(OUT,exist_ok=True)
REPS=int(os.getenv("DIVERGENCE_REPS","200"));rows=[]
for alpha in [.25,.5,.75,1.]:
 for N in [400,800,1600,3200,6400,12800]:
  for rep in range(REPS):
   r=np.random.default_rng(150_000_000+int(alpha*100)*100000+N+rep);S=r.integers(0,2,N);Z=r.normal(np.where(S==0,-1.5,1.5),1.).reshape(-1,1)
   b1=GaussianMixture(1,random_state=rep).fit(Z).bic(Z);b2=GaussianMixture(2,random_state=rep,n_init=2).fit(Z).bic(Z)
   # causal contrast theta1-theta0=1, both state treatment residual SD N^-alpha
   sd=N**(-alpha);T=r.normal(0,sd,N);theta=np.where(S==0,.5,1.5);Y=theta*T+r.normal(size=N)
   hats=[];vars=[]
   for s in [0,1]:
    m=S==s;den=np.sum(T[m]**2);hat=np.sum(T[m]*Y[m])/den;hats.append(hat);vars.append(1/den)
   z=(hats[1]-hats[0])/np.sqrt(vars[0]+vars[1]);power_event=abs(z)>norm.ppf(.975)
   # Bayes state classification error under known equal-prior emissions threshold 0
   class_err=np.mean((Z.ravel()>0).astype(int)!=S)
   rows.append((alpha,N,rep,b1-b2,class_err,z,power_event))
d=pd.DataFrame(rows,columns=["alpha","N","rep","BIC_advantage_K2","state_class_error","causal_contrast_z","causal_reject"]);d.to_csv(os.path.join(OUT,"resolution_divergence_raw.csv"),index=False)
s=d.groupby(["alpha","N"]).agg(BIC_advantage=("BIC_advantage_K2","mean"),state_error=("state_class_error","mean"),causal_power=("causal_reject","mean"),mean_abs_z=("causal_contrast_z",lambda x:np.mean(np.abs(x)))).reset_index();s.to_csv(os.path.join(OUT,"resolution_divergence_summary.csv"),index=False);print(s.to_string(index=False))
