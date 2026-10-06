"""Multiplicity cost for simultaneous target-family inference."""
import os,numpy as np,pandas as pd
from scipy.stats import norm
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
rows=[]
for M in [1,2,5,10,20,50,100,500,1000]:
 z=norm.ppf(1-.05/(2*M));rows.append((M,1.96,z,z/1.96))
pd.DataFrame(rows,columns=["num_targets","pointwise_z","bonferroni_z","width_inflation"]).to_csv(os.path.join(OUT,"simultaneous_target_family_inference.csv"),index=False)
