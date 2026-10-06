"""Cross-fitting does not remove systematic representation bias.

Stylized simulation: representation-induced score perturbation b_N=N^-r is estimated
on an independent training fold, then treated as fixed on evaluation fold. Compare
coverage of nominal CI that ignores b_N across r. Demonstrates cross-fitting removes
reuse/overfit dependence but not first-order bias unless b_N vanishes sufficiently fast.
"""
import os,numpy as np,pandas as pd
from scipy.stats import norm
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
rgen=np.random.default_rng(231000000);rows=[]
for rate in [.25,.5,.75,1.]:
 for N in [400,1600,6400,25600]:
  R=10000;b=N**(-rate);cover=0
  for _ in range(R):
   # independent eval score noise; representation perturbation comes from separate training fold
   est=b+rgen.normal()/np.sqrt(N);se=1/np.sqrt(N);cover += (est-1.96*se<=0<=est+1.96*se)
  rows.append((rate,N,b,cover/R,b*np.sqrt(N)))
pd.DataFrame(rows,columns=["representation_rate","N","bias","nominal_coverage","bias_over_se"]).to_csv(os.path.join(OUT,"crossfit_representation_rate.csv"),index=False)
