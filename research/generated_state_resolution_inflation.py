"""Generated-state inflation of downstream causal information.

Balanced K=3 true states with treatment residual SDs (1,.3,.08). Emissions have means
(-sep,0,+sep). Fit K=3 Gaussian mixture; compare eigenvalues of true one-hot Gram
E[HH'T^2] with soft-posterior and hard-MAP proxy Grams.
"""
import os,numpy as np,pandas as pd
from sklearn.mixture import GaussianMixture
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));OUT=os.path.join(ROOT,"research","results");os.makedirs(OUT,exist_ok=True)
REPS=int(os.getenv("PHANTOM_RES_REPS","100"));rows=[]
for N in [600,1200,2400]:
 for sep in [.5,1.,2.,3.]:
  for rep in range(REPS):
   r=np.random.default_rng(91_000_000+N*100+int(sep*10)*10000+rep);S=r.integers(0,3,N);Z=r.normal(np.array([-sep,0.,sep])[S],1.).reshape(-1,1);T=r.normal(size=N)*np.array([1.,.3,.08])[S];H=np.eye(3)[S]
   gm=GaussianMixture(3,random_state=rep,n_init=3).fit(Z);g=gm.predict_proba(Z);hard=np.eye(3)[g.argmax(1)]
   et=np.linalg.eigvalsh((H.T*(T*T))@H/N)
   for mode,A in [("soft",g),("hard",hard)]:
    ep=np.linalg.eigvalsh((A.T*(T*T))@A/N)
    rows.append((N,sep,rep,mode,*et,*ep,ep[0]/et[0],et[-1]/et[0],ep[-1]/ep[0]))
d=pd.DataFrame(rows,columns=["N","separation","rep","mode","true_lmin","true_lmid","true_lmax","proxy_lmin","proxy_lmid","proxy_lmax","lmin_inflation","true_condition","proxy_condition"]);d.to_csv(os.path.join(OUT,"generated_state_resolution_inflation_raw.csv"),index=False)
s=d.groupby(["N","separation","mode"]).mean(numeric_only=True).reset_index().drop(columns=["rep"]);s.to_csv(os.path.join(OUT,"generated_state_resolution_inflation_summary.csv"),index=False);print(s.to_string(index=False))
