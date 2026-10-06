"""Integrated stress test for the directional reliability operator.

Synthetic K=3 local experiment crosses:
- information spectrum / weak overlap;
- representation perturbation magnitude;
- perturbation orientation;
- scientific target orientation;
- sample size.
Data are generated directly in the linearized score experiment:
    theta_hat-theta = J^-1(b + xi/sqrt(N)).
Compare actual target error over Monte Carlo with predicted representation displacement
|a'J^-1 b| and predicted sampling SD sqrt(a'J^-1 Sigma J^-1 a/N).
This is the first integrated falsification test of the proposed local operator, rather
than another isolated toy phenomenon.
"""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
r=np.random.default_rng(261000000);rows=[]
for weak in [.2,.05,.01]:
 J=np.diag([1.,.2,weak]);Ji=np.linalg.inv(J);Sigma=J
 for eps in [.001,.01,.05]:
  for bn,bdir in {"strong":np.array([1.,0,0]),"weak":np.array([0,0,1.]),"mixed":np.ones(3)/np.sqrt(3)}.items():
   b=eps*bdir
   for an,a in {"strong":np.array([1.,0,0]),"weak":np.array([0,0,1.]),"average":np.ones(3)/np.sqrt(3)}.items():
    pred_bias=abs(a@Ji@b)
    for N in [200,2000,20000]:
     pred_sd=np.sqrt(a@Ji@Sigma@Ji@a/N);errs=[]
     for _ in range(3000):
      xi=r.multivariate_normal(np.zeros(3),Sigma);d=Ji@(b+xi/np.sqrt(N));errs.append(a@d)
     arr=np.array(errs);rmse=np.sqrt(np.mean(arr**2));pred_rmse=np.sqrt(pred_bias**2+pred_sd**2)
     rows.append((weak,eps,bn,an,N,pred_bias,pred_sd,rmse,pred_rmse,rmse/pred_rmse))
pd.DataFrame(rows,columns=["weak_info","epsilon","bias_direction","target_direction","N","pred_bias","pred_sd","empirical_rmse","pred_rmse","rmse_ratio"]).to_csv(os.path.join(OUT,"integrated_operator_stress_test.csv"),index=False)
