"""Task-aware state coarsening: emission-optimal K can be causally suboptimal."""
import os,numpy as np,pandas as pd
from sklearn.mixture import GaussianMixture
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));OUT=os.path.join(ROOT,"research","results");os.makedirs(OUT,exist_ok=True)
rows=[];bic=[]
for N in [400,800,1600]:
  for rep in range(1000):
    r=np.random.default_rng(N*10000+rep); s=r.integers(0,4,N); macro=s//2; theta=np.array([.5,2.5]); sigma=np.array([1.,.1,1.,.1]); v=r.normal(size=N)*sigma[s]; u=r.normal(size=N); y=theta[macro]*v+u
    micro=[]
    for k in range(4):
        m=s==k; micro.append(np.sum(v[m]*y[m])/np.sum(v[m]**2))
    micro=np.array(micro)
    refined=np.array([np.average(micro[:2],weights=[np.sum(s==0),np.sum(s==1)]),np.average(micro[2:],weights=[np.sum(s==2),np.sum(s==3)])])
    merged=[]
    for k in range(2):
        m=macro==k; merged.append(np.sum(v[m]*y[m])/np.sum(v[m]**2))
    rows.append((N,rep,np.linalg.norm(refined-theta),np.linalg.norm(np.array(merged)-theta)))
  for rep in range(100):
    r=np.random.default_rng(800000+N*100+rep); s=r.integers(0,4,N); z=r.normal(np.array([-3.,-1.,1.,3.])[s],.35).reshape(-1,1)
    b2=GaussianMixture(2,random_state=0).fit(z).bic(z); b4=GaussianMixture(4,random_state=0).fit(z).bic(z); bic.append((N,rep,b2,b4,b2-b4,b4<b2))
d=pd.DataFrame(rows,columns=["N","rep","micro_refined_error","merged_error"]);d["merged_wins"]=d.merged_error<d.micro_refined_error
b=pd.DataFrame(bic,columns=["N","rep","BIC_K2","BIC_K4","BIC_advantage_K4","BIC_prefers_K4"])
d.to_csv(os.path.join(OUT,"task_aware_coarsening_raw.csv"),index=False);b.to_csv(os.path.join(OUT,"task_aware_coarsening_bic_raw.csv"),index=False)
s=d.groupby("N").agg(micro_error=("micro_refined_error","mean"),merged_error=("merged_error","mean"),merged_win_rate=("merged_wins","mean")).reset_index()
sb=b.groupby("N").agg(mean_BIC_advantage_K4=("BIC_advantage_K4","mean"),BIC_prefers_K4_rate=("BIC_prefers_K4","mean")).reset_index();s.merge(sb,on="N").to_csv(os.path.join(OUT,"task_aware_coarsening_summary.csv"),index=False)
print(s.merge(sb,on="N").to_string(index=False))
