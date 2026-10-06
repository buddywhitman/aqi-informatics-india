"""Local-weak-overlap state-refinement paradox.

Four microstates are perfectly observed and observationally distinct, but states (0,1)
share causal effect theta_A and states (2,3) share theta_B. States 1 and 3 have
residual treatment SD N^{-alpha}. Estimating each microstate slope separately and
occupancy-averaging has SD order N^{alpha-1/2}; pooling causally equivalent states
retains O(N) information and root-N error.
"""
import os,numpy as np,pandas as pd
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));OUT=os.path.join(ROOT,"research","results");os.makedirs(OUT,exist_ok=True)
rows=[]
for alpha in [.25,.50,.75,1.00]:
  for N in [400,800,1600,3200,6400]:
    for rep in range(300):
      r=np.random.default_rng(1_000_000+int(alpha*100)*100000+N*10+rep); s=r.integers(0,4,N); macro=s//2
      theta=np.array([.5,2.5]); sig=np.array([1.,N**(-alpha),1.,N**(-alpha)])
      v=r.normal(size=N)*sig[s]; u=r.normal(size=N); y=theta[macro]*v+u
      b=[]
      for k in range(4):
        m=s==k; b.append(np.sum(v[m]*y[m])/np.sum(v[m]**2))
      b=np.array(b); n=np.array([(s==k).sum() for k in range(4)])
      refined=np.array([np.average(b[:2],weights=n[:2]),np.average(b[2:],weights=n[2:])])
      pooled=[]
      for k in range(2):
        m=macro==k; pooled.append(np.sum(v[m]*y[m])/np.sum(v[m]**2))
      rows.append((alpha,N,rep,np.linalg.norm(refined-theta),np.linalg.norm(np.array(pooled)-theta)))
d=pd.DataFrame(rows,columns=["alpha","N","rep","refined_error","coarsened_error"])
d.to_csv(os.path.join(OUT,"weak_overlap_refinement_paradox_raw.csv"),index=False)
s=d.groupby(["alpha","N"]).agg(refined_mean=("refined_error","mean"),refined_median=("refined_error","median"),coarsened_mean=("coarsened_error","mean"),coarsened_median=("coarsened_error","median")).reset_index()
s["error_ratio"]=s.refined_mean/s.coarsened_mean
s.to_csv(os.path.join(OUT,"weak_overlap_refinement_paradox_summary.csv"),index=False)
print(s.to_string(index=False))
