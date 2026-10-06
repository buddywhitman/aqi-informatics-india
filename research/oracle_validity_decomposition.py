"""Decompose scientific error into proxy-oracle and oracle-target components."""
import os,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
rows=[]
for ob in [-2,-1,-.5,.5,1,2]:
 for pg in [-2,-1,-.5,0,.5,1,2]:
  total=ob+pg;rows.append((ob,pg,total,abs(pg),abs(total),abs(total)<abs(ob)))
pd.DataFrame(rows,columns=["oracle_minus_target","proxy_minus_oracle","proxy_minus_target","proxy_oracle_error","scientific_error","proxy_accidentally_improves"]).to_csv(os.path.join(OUT,"oracle_validity_decomposition.csv"),index=False)
