"""Phase diagram for representation accuracy rate required for valid target inference.

Let target-projected representation perturbation scale N^-r and target information
scale lambda_N=N^-s. Then induced target bias scales N^(s-r), sampling SE under
score information scales approximately N^((s-1)/2). Ratio bias/SE scales
N^((s+1)/2-r). Root-N/standard inference requires r>(s+1)/2 in this stylized model.
"""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
rows=[]
for s in [0,.25,.5,.75,1.]:
 threshold=(1+s)/2
 for r in [.25,.5,.625,.75,.875,1.,1.25]:
  expo=(1+s)/2-r
  rows.append((s,r,threshold,expo,"bias_dominates" if expo>0 else ("boundary" if abs(expo)<1e-12 else "bias_negligible")))
pd.DataFrame(rows,columns=["information_decay_s","representation_rate_r","required_rate","bias_to_se_exponent","regime"]).to_csv(os.path.join(OUT,"representation_accuracy_rate_requirement.csv"),index=False)
