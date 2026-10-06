"""Test whether proxy conditioning can cancel entropy warnings."""
import os,numpy as np,pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
OUT=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"research","results");os.makedirs(OUT,exist_ok=True)
rows=[]
for sep in [.35,.5,.75,1.,1.5,2.,3.]:
 for weak in [.04,.06,.08,.12,.2]:
  for rep in range(100):
   N=1200;r=np.random.default_rng(501000000+int(sep*100)*100000+int(weak*1000)*100+rep);S=r.integers(0,3,N);Z=r.normal(np.array([-sep,0,sep])[S],1).reshape(-1,1);T=r.normal(size=N)*np.array([1.,.3,weak])[S];H=np.eye(3)[S]
   gm=GaussianMixture(3,random_state=rep,n_init=2).fit(Z);g0=gm.predict_proba(Z);order=np.argsort(gm.means_.ravel());g=g0[:,order]
   ent=-np.mean(np.sum(g*np.log(np.clip(g,1e-12,1)),axis=1));eps=np.mean(np.abs(g-H).sum(1))
   lo=max(np.linalg.eigvalsh((H.T*(T*T))@H/N)[0],1e-12);lp=max(np.linalg.eigvalsh((g.T*(T*T))@g/N)[0],1e-12);target=eps/lo
   rows.append((sep,weak,rep,ent,eps,lo,lp,lp/lo,target,ent/lp,ent/lo))
d=pd.DataFrame(rows,columns=["separation","weak_sd","rep","entropy","proxy_l1_error","oracle_lmin","proxy_lmin","lmin_inflation","oracle_difficulty","H_over_proxy_lmin","H_over_oracle_lmin"]);d.to_csv(os.path.join(OUT,"phantom_diagnostic_cancellation_raw.csv"),index=False)
y=(d.oracle_difficulty>=d.oracle_difficulty.quantile(.8)).astype(int);res=[]
for col in ["entropy","H_over_proxy_lmin","H_over_oracle_lmin","proxy_l1_error","lmin_inflation"]:res.append((col,spearmanr(d[col],d.oracle_difficulty).statistic,roc_auc_score(y,d[col])))
pd.DataFrame(res,columns=["diagnostic","spearman","top20_auc"]).to_csv(os.path.join(OUT,"phantom_diagnostic_cancellation_summary.csv"),index=False)
