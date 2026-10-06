"""How much labeled-state validation is needed for useful perturbation bounds?

Stylized K=3 setting. A validation sample observes true state H and proxy posterior g,
allowing estimation of coordinatewise score perturbation contributions. Construct
normal-approximation simultaneous coordinate bounds and propagate through target
sensitivity c=a'J^-1. Measure bound width and coverage versus validation size.
"""
import os,numpy as np,pandas as pd
from scipy.stats import norm
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
r=np.random.default_rng(301000000);K=3;J=np.diag([1.,.1,.01]);a=np.ones(K)/np.sqrt(K);c=a@np.linalg.inv(J)
# population perturbation distribution: directional state-posterior error times score-like leverage
M=2000000;S=r.integers(0,K,M);H=np.eye(K)[S];noise=r.normal(size=(M,K))*np.array([.01,.005,.002]);X=noise # score perturbation observations with anisotropy
btrue=X.mean(0);truth=abs(c@btrue)
rows=[]
for m in [50,100,250,500,1000,2500,5000]:
 cover=0;width=[]
 for rep in range(2000):
  ix=r.integers(0,M,m);x=X[ix];bh=x.mean(0);se=x.std(0,ddof=1)/np.sqrt(m);z=norm.ppf(1-.05/(2*K));rad=z*se
  lo=c@bh-np.sum(np.abs(c)*rad);hi=c@bh+np.sum(np.abs(c)*rad)
  # bound absolute target perturbation by |center|+radius
  B=abs(c@bh)+np.sum(np.abs(c)*rad);cover+=truth<=B;width.append(B)
 rows.append((m,truth,cover/2000,np.mean(width),np.median(width)))
pd.DataFrame(rows,columns=["validation_n","true_abs_target_bias","coverage","mean_upper_bound","median_upper_bound"]).to_csv(os.path.join(OUT,"validation_sample_bounds.csv"),index=False)
