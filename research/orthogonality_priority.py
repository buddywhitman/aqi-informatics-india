"""Which representation perturbation directions should be corrected first?

Given limited correction budget d, compare choosing tangent directions by:
- largest raw representation-error magnitude;
- weakest information eigenvalue;
- largest target-amplified contribution |a_j b_j/lambda_j| (oracle priority).
Construct cases where naive priorities fail.
"""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
lam=np.array([1.,.1,.01,.001]);a=np.array([1.,1.,1.,1.])/2
cases={"raw_misleading":np.array([.1,.02,.005,.001]),"weak_misleading":np.array([.1,.02,0.,0.]),"mixed":np.array([.03,.02,.01,.005])}
rows=[]
for case,b in cases.items():
 contrib=np.abs(a*b/lam);base=contrib.sum()
 orders={"raw_error":np.argsort(-np.abs(b)),"weak_info":np.argsort(lam),"target_amplified":np.argsort(-contrib)}
 for d in [1,2,3]:
  for rule,order in orders.items():
   keep=np.ones(4,bool);keep[order[:d]]=False;res=np.sum(contrib[keep]);rows.append((case,d,rule,base,res,res/base if base else 0))
pd.DataFrame(rows,columns=["case","budget","priority_rule","baseline_abs_component_sum","residual_abs_component_sum","residual_fraction"]).to_csv(os.path.join(OUT,"orthogonality_priority.csv"),index=False)
