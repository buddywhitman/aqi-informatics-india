"""Score dependence, not state persistence alone, controls temporal information."""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
r=np.random.default_rng(311000000);rows=[]
for rhoT in [0,.3,.6,.9]:
 for rhoU in [0,.3,.6,.9]:
  N=200000;T=np.empty(N);U=np.empty(N);T[0]=r.normal();U[0]=r.normal()
  for t in range(1,N):
   T[t]=rhoT*T[t-1]+np.sqrt(1-rhoT**2)*r.normal();U[t]=rhoU*U[t-1]+np.sqrt(1-rhoU**2)*r.normal()
  sc=T*U;ac=np.corrcoef(sc[:-1],sc[1:])[0,1];theory=rhoT*rhoU;lrv=(1+theory)/(1-theory) if theory<1 else np.inf
  rows.append((rhoT,rhoU,ac,theory,lrv))
pd.DataFrame(rows,columns=["rho_T","rho_U","score_lag1_empirical","score_lag1_theory","score_lrv_inflation"]).to_csv(os.path.join(OUT,"sequential_score_dependence.csv"),index=False)
