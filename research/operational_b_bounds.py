"""Conservative target-bias bounds from uncertain representation perturbation.

Given target sensitivity row c=a'J^-1 and uncertainty sets for b:
- L2 ball ||b||<=rho -> |c b|<=rho||c||;
- coordinate box |b_j|<=rho_j -> sum |c_j|rho_j;
- ellipsoid b'W^-1 b<=1 -> sqrt(c W c').
Compare tightness to realized bias for random b and anisotropic uncertainty.
"""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
r=np.random.default_rng(291000000);J=np.diag([1.,.1,.01]);a=np.ones(3)/np.sqrt(3);c=a@np.linalg.inv(J);rows=[]
for case,rhos in {"isotropic":np.array([.01,.01,.01]),"strong_uncertain":np.array([.05,.005,.001]),"weak_uncertain":np.array([.001,.005,.05])}.items():
 l2=np.linalg.norm(rhos);l2bound=l2*np.linalg.norm(c);box=np.sum(np.abs(c)*rhos);W=np.diag(rhos**2);ell=np.sqrt(c@W@c)
 for rep in range(5000):
  u=r.uniform(-1,1,3);b=rhos*u;actual=abs(c@b);rows.append((case,actual,l2bound,box,ell,l2bound/max(actual,1e-15),box/max(actual,1e-15),ell/max(actual,1e-15)))
pd.DataFrame(rows,columns=["case","actual_bias","l2_bound","box_bound","ellipsoid_bound","l2_over_actual","box_over_actual","ellipsoid_over_actual"]).to_csv(os.path.join(OUT,"operational_b_bounds.csv"),index=False)
