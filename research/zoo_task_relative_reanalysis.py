"""Aggregate zoo diagnostics that can be computed from committed world table."""
import os,pandas as pd,numpy as np
ROOT=os.path.dirname(os.path.dirname(__file__));d=pd.read_csv(os.path.join(ROOT,"reports","representation_zoo_world_evaluations.csv"));out=[]
for arch,g in d.groupby("Architecture"):
 out.append((arch,len(g),g.F1.mean(),g.ECE.mean(),g.NLL.mean(),g.Reliability_AUC_D.mean(),g.Reliability_AUC_H.mean(),(g.Reliability_AUC_D-g.Reliability_AUC_H).mean(),g.F1.corr(g.Reliability_AUC_D,method="spearman"),g.ECE.corr(g.Reliability_AUC_D,method="spearman"),g.NLL.corr(g.Reliability_AUC_D,method="spearman")))
pd.DataFrame(out,columns=["Architecture","N_worlds","mean_F1","mean_ECE","mean_NLL","mean_AUC_D","mean_AUC_H","mean_D_minus_H","across_world_rho_F1_D","across_world_rho_ECE_D","across_world_rho_NLL_D"]).to_csv(os.path.join(ROOT,"research","results","zoo_task_relative_aggregate.csv"),index=False)
