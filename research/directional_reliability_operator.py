"""Directional reliability operator experiment.

Construct K=3 score geometry J=diag(pi_k sigma_k^2). Inject equal-norm score
perturbations b along strong and weak eigendirections. Compare:
- scalar worst-case bound ||b||/lambda_min(J);
- exact target displacement ||J^-1 b||;
- direction-specific normalized perturbation.
Shows scalar bound cannot rank orientation when perturbation norm is fixed.
"""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
pi=np.ones(3)/3;sig=np.array([1.,.3,.08]);J=np.diag(pi*sig**2);lmin=np.linalg.eigvalsh(J)[0]
rows=[]
dirs={"strong":np.array([1.,0,0]),"medium":np.array([0,1.,0]),"weak":np.array([0,0,1.]),"mixed":np.ones(3)/np.sqrt(3)}
for mag in [.001,.005,.01,.05]:
 for name,v in dirs.items():
  b=mag*v/np.linalg.norm(v);disp=np.linalg.norm(np.linalg.solve(J,b));bound=np.linalg.norm(b)/lmin;ray=(v@J@v);dirscale=np.linalg.norm(b)/ray
  rows.append((mag,name,np.linalg.norm(b),lmin,ray,disp,bound,dirscale,bound/disp))
pd.DataFrame(rows,columns=["magnitude","direction","b_norm","lambda_min","directional_information","exact_displacement","scalar_bound","directional_scale","bound_over_exact"]).to_csv(os.path.join(OUT,"directional_reliability_operator.csv"),index=False)
