"""Directional task-calibration spectrum.

K=3 posterior errors and a family of causal contrast directions v. For each v define
a leverage-weighted representation error using score leverage (v'S_tilde)^2 in a
diagonal state-score model. Demonstrates that one representation can be calibrated
for one causal contrast and poor for another even within the same causal task.
"""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
r=np.random.default_rng(171000000);N=300000;S=r.integers(0,3,N);H=np.eye(3)[S];T=r.normal(size=N)*np.array([1.,.3,.08])[S]
# two equal-global-error representations, corrupt different true states
reps={}
for target,name in [(0,"err_state0"),(2,"err_state2")]:
 g=H.copy();m=S==target;a=.15;g[m]=(1-a)*H[m]+a*np.roll(H[m],1,axis=1);reps[name]=g
dirs={"state0":np.array([1.,0,0]),"state2":np.array([0.,0,1.]),"contrast02":np.array([1.,0,-1.])/np.sqrt(2),"average":np.ones(3)/np.sqrt(3)}
rows=[]
for name,g in reps.items():
 e=np.sum((g-H)**2,1)
 row={"representation":name,"global_error":e.mean()}
 for dn,v in dirs.items():
  # per-observation directional causal score leverage: T^2*(v_state)^2
  w=T*T*(v[S]**2);row[dn]=np.sum(w*e)/max(np.sum(w),1e-12)
 rows.append(row)
pd.DataFrame(rows).to_csv(os.path.join(OUT,"directional_calibration_spectrum.csv"),index=False)
