"""Sensitivity to latent state count K in the observational stress test.

Exploratory only. Reports OR-DML estimates/conditioning and HMM BIC for K=2..5.
"""
import os,sys,pandas as pd
from sklearn.linear_model import Ridge
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));sys.path.insert(0,ROOT)
from src.or_dml import OverlapAwareRegimeDML,LatentRegimeHMM
d=pd.read_csv(os.path.join(ROOT,"data","processed_clean","combined_hourly_clean.csv"))
feat=["temperature","wind_speed","humidity","pressure","hour_sin","hour_cos"]
controls=["pm25_lag_1h","no2_lag_1h","pm25_roll_3h","no2_roll_3h","temperature","humidity","wind_speed","pressure"]
req=["no2","pm25"]+feat+controls; rows=[]; bic=[]
for city in ["Delhi","Mumbai","Bengaluru","Kolkata"]:
    x=d[d.city==city].dropna(subset=req); Z=x[feat].values
    for K in [2,3,4]:
        m=OverlapAwareRegimeDML(n_regimes=K,n_splits=5,embargo_tau=24,reg_alpha=.05,posterior_mode="smooth",nuisance_model=Ridge(alpha=1),random_state=42)
        m.fit(x.pm25.values,x.no2.values,x[controls].values,Z)
        rows.append((city,K,m.ate_,m.ate_se_,m.lambda_min_,m.kappa_,m.mean_entropy_,";".join(f"{v:.4f}" for v in m.theta_regimes_.values())))
    for K in [2,3,4,5]:
        h=LatentRegimeHMM(K,n_iter=25,random_state=42).fit(Z); bic.append((city,K,h.compute_bic(Z)))
out=os.path.join(ROOT,"research","results");os.makedirs(out,exist_ok=True)
pd.DataFrame(rows,columns=["city","K","ate","ate_se","lmin","kappa","entropy","theta"]).to_csv(os.path.join(out,"latent_state_count_sensitivity.csv"),index=False)
pd.DataFrame(bic,columns=["city","K","bic"]).to_csv(os.path.join(out,"latent_state_count_bic.csv"),index=False)
