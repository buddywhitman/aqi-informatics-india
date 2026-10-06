"""Partial representation orthogonality when only some perturbation directions are correctable.

J diagonal; representation bias b has components in known tangent subspace U and unknown
orthogonal remainder. Project/correct U component, quantify residual target bias.
"""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
J=np.diag([1.,.1,.01]);Ji=np.linalg.inv(J);a=np.array([1.,1.,1.])/np.sqrt(3);rows=[]
for known_dim in [0,1,2,3]:
 P=np.diag([1. if j<known_dim else 0. for j in range(3)])
 for bname,b in {"strong":np.array([.01,0,0]),"weak":np.array([0,0,.01]),"mixed":np.array([.01,.01,.01])}.items():
  before=abs(a@Ji@b);after=abs(a@Ji@((np.eye(3)-P)@b));rows.append((known_dim,bname,before,after,after/before if before else 0))
pd.DataFrame(rows,columns=["correctable_tangent_dim","bias_direction","before","after","residual_fraction"]).to_csv(os.path.join(OUT,"partial_orthogonality.csv"),index=False)
