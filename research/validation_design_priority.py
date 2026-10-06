"""Optimal validation allocation across latent directions/states.

Given target sensitivity c_j and per-validation-observation perturbation variance v_j,
allocate total labeled validation budget m across coordinates to minimize propagated
variance sum c_j^2 v_j/m_j. Neyman allocation m_j proportional |c_j| sqrt(v_j).
Compare equal, state-frequency, raw-error, and target-sensitive allocation.
"""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
c=np.array([1.,10.,100.])/np.sqrt(3);v=np.array([.01,.0025,.0004]);freq=np.array([.6,.3,.1]);M=600
weights={"equal":np.ones(3),"frequency":freq,"raw_sd":np.sqrt(v),"target_sensitive":np.abs(c)*np.sqrt(v)}
rows=[]
for name,w in weights.items():
 alloc=np.maximum(1,np.round(M*w/w.sum())).astype(int);var=np.sum(c*c*v/alloc);rows.append((name,*alloc,np.sqrt(var)))
pd.DataFrame(rows,columns=["allocation_rule","m1","m2","m3","propagated_target_se"]).to_csv(os.path.join(OUT,"validation_design_priority.csv"),index=False)
