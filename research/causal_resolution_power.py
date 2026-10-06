"""Monte Carlo validation of the causal heterogeneity detection boundary alpha+beta=1/2."""
import os,numpy as np,pandas as pd
from scipy.stats import norm
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));OUT=os.path.join(ROOT,"research","results");os.makedirs(OUT,exist_ok=True)
REPS=int(os.getenv("RESOLUTION_REPS","2000"));rows=[]
theta0=.5;pi=.5;su=1.
for alpha in [0.,.15,.3,.45,.6,.75]:
 for beta in [0.,.1,.25,.4]:
  for N in [400,1600,6400,25600]:
   delta=N**(-beta); reject=0; zvals=[]; kls=[]
   for rep in range(REPS):
    r=np.random.default_rng(121000000+int(alpha*100)*1000000+int(beta*100)*10000+N+rep);n=int(pi*N);sd=N**(-alpha);t=r.normal(0,sd,n);u=r.normal(0,su,n);y=(theta0+delta)*t+u
    den=np.sum(t*t);hat=np.sum(t*y)/den;se=su/np.sqrt(den);z=(hat-theta0)/se;reject += abs(z)>norm.ppf(.975);zvals.append(z);kls.append(delta*delta*den/(2*su*su))
   rows.append((alpha,beta,N,alpha+beta,delta,np.mean(kls),reject/REPS,np.mean(zvals)))
d=pd.DataFrame(rows,columns=["alpha","beta","N","alpha_plus_beta","delta","mean_KL","wald_power","mean_z"]);d.to_csv(os.path.join(OUT,"causal_resolution_power_raw.csv"),index=False)
s=d[d.N==25600].copy();s["regime"]=np.where(s.alpha_plus_beta<.5,"detectable",np.where(s.alpha_plus_beta>.5,"undetectable","boundary"));s.to_csv(os.path.join(OUT,"causal_resolution_power_N25600.csv"),index=False);print(s.to_string(index=False))
