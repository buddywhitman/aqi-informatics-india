"""Real-data diagnostic panel from committed empirical/sensitivity reports."""
import os,pandas as pd,numpy as np
ROOT=os.path.dirname(os.path.dirname(__file__));emp=pd.read_csv(os.path.join(ROOT,"reports","empirical_or_dml_results.csv"));fr=pd.read_csv(os.path.join(ROOT,"research","results","frozen_sensor_sensitivity.csv"));pl=pd.read_csv(os.path.join(ROOT,"reports","empirical_placebo_falsification.csv"))
rows=[]
for city in emp.City.unique():
 e=emp[(emp.City==city)&(emp.Method=="Spectral OR-DML (Ours)")&(emp.Regime=="Overall ATE")].iloc[0]
 f0=fr[(fr.city==city)&(fr["sample"]=="original")].iloc[0];f1=fr[(fr.city==city)&(fr["sample"]=="remove_frozen_ge6")].iloc[0]
 pre=pl[(pl.City==city)&(pl.Type=="Pre-Treatment Placebo")]
 rows.append((city,int(e.N_Obs),e.Effect_Theta,e.Std_Error,e.Lambda_Min,e.Kappa,e.Entropy,int((pre.p_value<.05).sum()),len(pre),f1.ate-f0.ate,f1.ate_se-f0.ate_se,f1.lmin/f0.lmin if f0.lmin else np.nan,f1.entropy-f0.entropy))
pd.DataFrame(rows,columns=["city","N","ate","se","lambda_min","kappa","entropy","significant_pre_placebos","num_pre_placebos","frozen_filter_delta_ate","frozen_filter_delta_se","frozen_filter_lmin_ratio","frozen_filter_delta_entropy"]).to_csv(os.path.join(ROOT,"research","results","real_data_collision_panel.csv"),index=False)
