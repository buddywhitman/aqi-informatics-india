"""Global ill-conditioning can worsen while a fixed scientific target becomes easier.

Triangular sequence: add nuisance/irrelevant effect directions with eigenvalues shrinking
as N^-alpha, while scientific target A selects a fixed strong direction. Compare global
condition/lambda_min diagnostic with target-specific standard error/amplification.
"""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
rows=[]
for alpha in [.25,.5,.75,1.]:
 for N in [100,400,1600,6400,25600]:
  # two-dimensional: scientific direction info fixed 1; irrelevant direction shrinks
  J=np.diag([1.,N**(-2*alpha)]);global_amp=1/np.linalg.eigvalsh(J)[0];target_amp=np.linalg.norm(np.array([[1.,0.]])@np.linalg.inv(J));target_se=target_amp/np.sqrt(N);global_nominal_se=global_amp/np.sqrt(N)
  rows.append((alpha,N,np.linalg.eigvalsh(J)[0],global_amp,target_amp,global_nominal_se,target_se))
pd.DataFrame(rows,columns=["alpha","N","lambda_min","global_amplification","target_amplification","global_nominal_se_scale","target_se_scale"]).to_csv(os.path.join(OUT,"target_subspace_resolution_paradox.csv"),index=False)
