"""Falsification diagnostic: disagreement among causal-geometry constructions."""
import os,numpy as np,pandas as pd
from scipy.special import logsumexp
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
rows=[];mods={"correct":np.array([1.,.3,.08]),"x2":np.array([1.,.3,.16]),"x4":np.array([1.,.3,.32]),"common":np.array([.5,.5,.5])}
for sep in [.5,1.,2.]:
 for rep in range(100):
  N=2400;r=np.random.default_rng(820000000+int(sep*100)*1000+rep);S=r.integers(0,3,N);mu=np.array([-sep,0,sep]);ts=np.array([1.,.3,.08]);Z=r.normal(mu[S],1);T=r.normal(0,ts[S]);H=np.eye(3)[S];oracle=np.min(np.mean(H*T[:,None]**2,0));lz=-.5*(Z[:,None]-mu)**2;gz=np.exp(lz-logsumexp(lz,1)[:,None]);outer=np.linalg.eigvalsh((gz.T*T**2)@gz/N)[0];zdiag=np.min(np.mean(gz*T[:,None]**2,0))
  for name,sig in mods.items():
   lt=-np.log(sig)-.5*(T[:,None]/sig)**2;q=np.exp(lz+lt-logsumexp(lz+lt,1)[:,None]);comp=np.min(np.mean(q*T[:,None]**2,0));v=np.array([outer,zdiag,comp]);rows.append((sep,rep,name,oracle,outer,zdiag,comp,np.log(v.max()/max(v.min(),1e-12)),abs(np.log(comp/oracle))))
d=pd.DataFrame(rows,columns="separation rep model oracle outer zdiag completed spread log_error".split());d.to_csv(os.path.join(OUT,"geometry_disagreement_raw.csv"),index=False);y=(d.log_error>np.log(2)).astype(int);pd.DataFrame([{"spearman":spearmanr(d.spread,d.log_error).statistic,"auc_gt2x":roc_auc_score(y,d.spread),"failure_rate":y.mean()}]).to_csv(os.path.join(OUT,"geometry_disagreement_summary.csv"),index=False)
