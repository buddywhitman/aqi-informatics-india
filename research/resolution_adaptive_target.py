"""Resolution-adaptive spectral target experiment.

K=6 Gaussian score experiment with heterogeneous information eigenvalues. Compare:
- unregularized fine estimator;
- oracle ridge over a grid;
- hard spectral projection retaining directions whose nominal resolution
  1/sqrt(N lambda_j) <= delta_science;
- oracle hard projection minimizing true MSE (benchmark only).

The scientific-resolution projection deliberately changes the estimand by suppressing
effects below a user-specified resolution. We report both estimation error to the
projected target and total error to the full theta so the approximation cost is visible.
"""
import os,numpy as np,pandas as pd
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));OUT=os.path.join(ROOT,"research","results");os.makedirs(OUT,exist_ok=True)
REPS=int(os.getenv("RESOLUTION_TARGET_REPS","500"));K=6;L=np.r_[0,np.logspace(-5,1,25)];rows=[]
for N in [400,1600,6400]:
 for alpha in [.25,.5,.75]:
  lam=np.array([1.,.5,.2,.08,N**(-2*alpha),.02])
  # fixed random orthogonal geometry
  rr=np.random.default_rng(991+int(alpha*100));Q,_=np.linalg.qr(rr.normal(size=(K,K)));J=Q@np.diag(lam)@Q.T
  for signal in ["aligned_strong","aligned_weak","diffuse"]:
   if signal=="aligned_strong": coef=np.array([1.,.6,.3,.1,0.,0.])
   elif signal=="aligned_weak": coef=np.array([1.,.6,.3,.1,.8,0.])
   else: coef=np.array([1.,.6,.3,.1,.25,.15])
   theta=Q@coef
   for delta in [.05,.1,.25,.5]:
    keep=(1/np.sqrt(N*lam)<=delta)
    for rep in range(REPS):
      r=np.random.default_rng(73_000_000+N*1000+int(alpha*100)*100+rep);noise_eig=r.normal(size=K)*np.sqrt(lam/N);S=J@theta+Q@noise_eig
      un=np.linalg.solve(J,S)
      ridge=[np.linalg.solve(J+l*np.eye(K),S) for l in L];ridge_err=[np.linalg.norm(x-theta)**2 for x in ridge];best_r=ridge[int(np.argmin(ridge_err))]
      # spectral projection estimate and its projected estimand
      sh=Q.T@S; estcoef=np.where(keep,sh/lam,0.);proj=Q@estcoef;target=Q@(coef*keep)
      # oracle hard projection among all eigen directions independently: keep j iff realized squared estimation error <= omitted signal^2
      rawcoef=sh/lam; ok=np.square(rawcoef-coef)<=np.square(coef); oracle=Q@(rawcoef*ok)
      rows.append((N,alpha,signal,delta,keep.sum(),np.linalg.norm(un-theta)**2,np.linalg.norm(best_r-theta)**2,np.linalg.norm(proj-theta)**2,np.linalg.norm(proj-target)**2,np.linalg.norm(target-theta)**2,np.linalg.norm(oracle-theta)**2))
d=pd.DataFrame(rows,columns=["N","alpha","signal","delta_science","effective_rank","mse_unregularized","mse_oracle_ridge","mse_resolution_total","mse_resolution_estimation","mse_resolution_approximation","mse_oracle_projection"])
d.to_csv(os.path.join(OUT,"resolution_adaptive_target_raw.csv"),index=False)
s=d.groupby(["N","alpha","signal","delta_science"]).mean(numeric_only=True).reset_index();s.to_csv(os.path.join(OUT,"resolution_adaptive_target_summary.csv"),index=False);print(s.to_string(index=False))
