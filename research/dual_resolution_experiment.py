"""Dual-resolution causal estimation experiment.

Within each macrostate, two microstates have the SAME causal slope but different
treatment means and outcome baselines. One microstate has local weak residual
treatment SD N^{-alpha}.

Compare:
  1. fine adjustment + fine target, then occupancy-average;
  2. fine adjustment + coarse target (pool residual scores);
  3. coarse adjustment + coarse target (omit microstate in nuisances).

Method 3 has analytic asymptotic omitted-microstate bias; method 1 has the local
weak-overlap rate penalty; method 2 avoids both when within-macro slopes are equal.
"""
import os,numpy as np,pandas as pd
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));OUT=os.path.join(ROOT,"research","results");os.makedirs(OUT,exist_ok=True)
rows=[]; REPS=int(os.getenv("DUAL_RES_REPS","500"))
theta=np.array([.5,2.5])
# each macrostate has two microstates; nuisance means/baselines covary within macrostate
tr_mean=np.array([-1.,1.,-1.,1.]); out_base=np.array([-2.,2.,-2.,2.])
for alpha in [.25,.5,.75,1.]:
 for N in [400,800,1600,3200,6400]:
  for rep in range(REPS):
   r=np.random.default_rng(88_000_000+int(alpha*100)*100000+N*10+rep);s=r.integers(0,4,N);A=s//2;weak=(s%2==1);sig=np.where(weak,N**(-alpha),1.)
   v=r.normal(size=N);u=r.normal(size=N);T=tr_mean[s]+sig*v;Y=theta[A]*T+out_base[s]+u
   est_fineavg=[];est_dual=[];est_coarse=[]
   for a in [0,1]:
    idx=np.where(A==a)[0]
    # fine-state nuisance residuals use true conditional means
    rt=T[idx]-tr_mean[s[idx]]
    ry=Y[idx]-(theta[a]*tr_mean[s[idx]]+out_base[s[idx]]) # equals theta*rt+u
    # fine target: estimate each micro slope, occupancy-average
    vals=[];w=[]
    for sm in [2*a,2*a+1]:
      j=np.where(s==sm)[0];rT=T[j]-tr_mean[sm];rY=Y[j]-(theta[a]*tr_mean[sm]+out_base[sm])
      vals.append(np.sum(rT*rY)/np.sum(rT*rT));w.append(len(j))
    est_fineavg.append(np.average(vals,weights=w))
    # dual resolution: fine nuisance, one pooled target slope
    est_dual.append(np.sum(rt*ry)/np.sum(rt*rt))
    # coarse nuisance: residualize only by macrostate sample means
    Tc=T[idx]-T[idx].mean();Yc=Y[idx]-Y[idx].mean();est_coarse.append(np.sum(Tc*Yc)/np.sum(Tc*Tc))
   for method,est in [("fine_adjust_fine_target",est_fineavg),("fine_adjust_coarse_target",est_dual),("coarse_adjust_coarse_target",est_coarse)]:
    rows.append((alpha,N,rep,method,np.linalg.norm(np.array(est)-theta)))
d=pd.DataFrame(rows,columns=["alpha","N","rep","method","l2_error"]);d.to_csv(os.path.join(OUT,"dual_resolution_raw.csv"),index=False)
s=d.groupby(["alpha","N","method"]).agg(mean_error=("l2_error","mean"),median_error=("l2_error","median")).reset_index();s.to_csv(os.path.join(OUT,"dual_resolution_summary.csv"),index=False);print(s.to_string(index=False))
