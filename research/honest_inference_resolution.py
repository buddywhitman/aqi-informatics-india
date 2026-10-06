"""Coverage under a vanishing-information causal contrast.

One-dimensional Gaussian residual-slope experiment:
 T_i ~ N(0,N^-2alpha), Y_i=theta*T_i+U_i.
Compare honest information-aware CI, a naive root-N CI calibrated at alpha=0,
and ridge CI that ignores shrinkage bias.
"""
import os,numpy as np,pandas as pd
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));OUT=os.path.join(ROOT,"research","results");os.makedirs(OUT,exist_ok=True)
REPS=int(os.getenv("HONEST_CI_REPS","5000"));rows=[]
for theta in [0.,.5,1.]:
 for alpha in [.25,.5,.75,1.]:
  for N in [400,1600,6400,25600]:
   cov_h=cov_n=cov_r=0; wh=[];wn=[];wr=[]
   lam_r=.1
   for rep in range(REPS):
    r=np.random.default_rng(201_000_000+int(theta*10)*10000000+int(alpha*100)*100000+N+rep);t=r.normal(0,N**(-alpha),N);y=theta*t+r.normal(size=N);den=np.sum(t*t);hat=np.sum(t*y)/den;se=1/np.sqrt(den)
    lo,hi=hat-1.96*se,hat+1.96*se;cov_h+=lo<=theta<=hi;wh.append(hi-lo)
    # naive fixed-information root-N interval
    sen=1/np.sqrt(N);lo,hi=hat-1.96*sen,hat+1.96*sen;cov_n+=lo<=theta<=hi;wn.append(hi-lo)
    # ridge slope, variance around shrunken target but CI incorrectly treated as theta CI
    rh=np.sum(t*y)/(den+lam_r*N);ser=np.sqrt(den)/(den+lam_r*N);lo,hi=rh-1.96*ser,rh+1.96*ser;cov_r+=lo<=theta<=hi;wr.append(hi-lo)
   rows.append((theta,alpha,N,cov_h/REPS,cov_n/REPS,cov_r/REPS,np.mean(wh),np.mean(wn),np.mean(wr)))
d=pd.DataFrame(rows,columns=["theta","alpha","N","honest_coverage","naive_rootN_coverage","naive_ridge_coverage","honest_width","naive_rootN_width","naive_ridge_width"]);d.to_csv(os.path.join(OUT,"honest_inference_resolution_raw.csv"),index=False);print(d.to_string(index=False))
