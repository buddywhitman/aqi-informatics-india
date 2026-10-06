"""Compute empirical causal-resolution spectra as latent state count K increases.

Exploratory diagnostic only. J is the OR-DML score Gram matrix. For effect scale delta,
we report the number of eigen-directions with N * delta^2 * lambda_j(J) >= 3.84,
an approximate one-df 5% information threshold, not a calibrated power guarantee.
"""
import os,sys,numpy as np,pandas as pd
from sklearn.linear_model import Ridge
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));sys.path.insert(0,ROOT)
from src.or_dml import OverlapAwareRegimeDML
d=pd.read_csv(os.path.join(ROOT,"data","processed_clean","combined_hourly_clean.csv"))
feat=["temperature","wind_speed","humidity","pressure","hour_sin","hour_cos"]
controls=["pm25_lag_1h","no2_lag_1h","pm25_roll_3h","no2_roll_3h","temperature","humidity","wind_speed","pressure"];req=["no2","pm25"]+feat+controls
rows=[]
for city in ["Delhi","Mumbai","Bengaluru","Kolkata"]:
 x=d[d.city==city].dropna(subset=req)
 for K in [2,3,4,5]:
  m=OverlapAwareRegimeDML(n_regimes=K,n_splits=5,embargo_tau=24,reg_alpha=.05,posterior_mode="smooth",nuisance_model=Ridge(alpha=1),random_state=42)
  m.fit(x.pm25.values,x.no2.values,x[controls].values,x[feat].values)
  ev=np.linalg.eigvalsh(m.J_mat_)
  row={"city":city,"K":K,"N":len(x),"lambda_min":ev[0],"lambda_max":ev[-1],"eigenvalues":";".join(f"{v:.8g}" for v in ev)}
  for delta in [.1,.25,.5,1.]: row[f"effective_rank_delta_{delta}"]=int(np.sum(len(x)*delta*delta*ev>=3.84))
  rows.append(row)
out=pd.DataFrame(rows);os.makedirs(os.path.join(ROOT,"research","results"),exist_ok=True);out.to_csv(os.path.join(ROOT,"research","results","real_causal_resolution_spectrum.csv"),index=False);print(out.to_string(index=False))
