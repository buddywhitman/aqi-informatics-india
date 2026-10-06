"""Target-aware versus global-conditioning representation/model selection.

Candidate models have information matrices whose weak direction may be irrelevant or
relevant to a declared scientific contrast. Compare selection by lambda_min(J) against
target amplification ||A J^-1|| and realized contrast sensitivity.
"""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
rows=[]
# Model G: globally healthy but only moderate information in scientific direction.
# Model T: terrible irrelevant direction but excellent information in scientific direction.
for eps in [1e-1,1e-2,1e-3,1e-4,1e-6]:
 models={"global_healthy":np.diag([.2,.2]),"target_healthy":np.diag([1.,eps])}
 for target,a in {"first":np.array([1.,0.]),"second":np.array([0.,1.]),"average":np.ones(2)/np.sqrt(2)}.items():
  vals=[]
  for name,J in models.items():
   lmin=np.linalg.eigvalsh(J)[0];amp=np.linalg.norm(a@np.linalg.inv(J));vals.append((name,lmin,amp))
  global_pick=max(vals,key=lambda x:x[1])[0];target_pick=min(vals,key=lambda x:x[2])[0]
  gp=[x for x in vals if x[0]==global_pick][0];tp=[x for x in vals if x[0]==target_pick][0]
  rows.append((eps,target,global_pick,target_pick,gp[2],tp[2],gp[2]/tp[2]))
pd.DataFrame(rows,columns=["irrelevant_eigenvalue","target","lambda_min_pick","target_aware_pick","lambda_pick_target_amp","target_pick_amp","selection_regret"]).to_csv(os.path.join(OUT,"target_aware_model_selection.csv"),index=False)
