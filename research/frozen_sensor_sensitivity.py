"""Sensitivity of the observational OR-DML fit to long constant-NO2 sensor runs.

Exploratory only; does not modify canonical processed data or submission artifacts.
"""
import os,sys,numpy as np,pandas as pd
from sklearn.linear_model import Ridge
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT)
from src.or_dml import OverlapAwareRegimeDML
p=os.path.join(ROOT,"data","processed_clean","combined_hourly_clean.csv")
d=pd.read_csv(p)
feat=["temperature","wind_speed","humidity","pressure","hour_sin","hour_cos"]
controls=["pm25_lag_1h","no2_lag_1h","pm25_roll_3h","no2_roll_3h","temperature","humidity","wind_speed","pressure"]
req=["no2","pm25"]+feat+controls; rows=[]
for city in ["Delhi","Mumbai","Bengaluru","Kolkata"]:
    x=d[d.city==city].dropna(subset=req).copy().reset_index(drop=True); v=x.no2.values
    grp=np.r_[0,np.cumsum(v[1:]!=v[:-1])]; counts=pd.Series(grp).map(pd.Series(grp).value_counts()).values
    for label,z in [("original",x),("remove_frozen_ge6",x[counts<6].copy())]:
        m=OverlapAwareRegimeDML(n_regimes=2,n_splits=5,embargo_tau=24,reg_alpha=.05,posterior_mode="smooth",nuisance_model=Ridge(alpha=1),random_state=42)
        m.fit(z.pm25.values,z.no2.values,z[controls].values,z[feat].values)
        rows.append((city,label,len(z),m.ate_,m.ate_se_,m.theta_regimes_[0],m.theta_regimes_[1],m.lambda_min_,m.kappa_,m.mean_entropy_))
out=pd.DataFrame(rows,columns=["city","sample","N","ate","ate_se","theta1","theta2","lmin","kappa","entropy"])
os.makedirs(os.path.join(ROOT,"research","results"),exist_ok=True)
out.to_csv(os.path.join(ROOT,"research","results","frozen_sensor_sensitivity.csv"),index=False)
print(out.to_string(index=False))
