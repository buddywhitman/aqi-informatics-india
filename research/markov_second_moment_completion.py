"""Markov validation of posterior second-moment completion.

Two/three-state Markov chain with Gaussian emissions and state-specific treatment
residual variances. Compare:
  oracle J = mean(T^2 diag(H));
  Z-only filtered completion = mean(T^2 diag(P(S_t|Z_1:t)));
  Z,T filtered completion = mean(T^2 diag(P(S_t|Z_1:t,T_1:t))).
Parameters are known to isolate the moment identity from parameter-estimation error.
"""
import os,numpy as np,pandas as pd
from scipy.special import logsumexp
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));OUT=os.path.join(ROOT,"research","results");os.makedirs(OUT,exist_ok=True)
REPS=int(os.getenv("MARKOV_COMPLETION_REPS","150"))
def filt(loglik,P,pi):
 n,k=loglik.shape;la=np.empty((n,k));la[0]=np.log(pi)+loglik[0];la[0]-=logsumexp(la[0])
 for t in range(1,n):
  pred=np.array([logsumexp(la[t-1]+np.log(P[:,j])) for j in range(k)])
  la[t]=pred+loglik[t];la[t]-=logsumexp(la[t])
 return np.exp(la)
rows=[]
for N in [600,1200,2400]:
 for sep in [.5,1.,2.]:
  for persistence in [.7,.9,.97]:
   K=3;off=(1-persistence)/(K-1);P=np.full((K,K),off);np.fill_diagonal(P,persistence);pi=np.ones(K)/K;mu=np.array([-sep,0,sep]);sig=np.array([1.,.3,.08])
   for rep in range(REPS):
    r=np.random.default_rng(610000000+N*10000+int(sep*10)*100+int(persistence*100)+rep);S=np.empty(N,int);S[0]=r.choice(K,p=pi)
    for t in range(1,N):S[t]=r.choice(K,p=P[S[t-1]])
    Z=r.normal(mu[S],1);T=r.normal(0,sig[S]);H=np.eye(K)[S]
    lz=-.5*(Z[:,None]-mu[None,:])**2
    lt=-np.log(sig)[None,:]-.5*(T[:,None]/sig[None,:])**2
    gz=filt(lz,P,pi);gzt=filt(lz+lt,P,pi)
    oracle=np.min(np.mean(H*(T*T)[:,None],axis=0));zonly=np.min(np.mean(gz*(T*T)[:,None],axis=0));complete=np.min(np.mean(gzt*(T*T)[:,None],axis=0))
    rows.append((N,sep,persistence,rep,oracle,zonly,complete,zonly/oracle,complete/oracle))
d=pd.DataFrame(rows,columns=["N","separation","persistence","rep","oracle_lmin","Z_filter_lmin","ZT_filter_completed_lmin","Z_ratio","ZT_ratio"]);d.to_csv(os.path.join(OUT,"markov_second_moment_completion_raw.csv"),index=False)
s=d.groupby(["N","separation","persistence"]).mean(numeric_only=True).reset_index().drop(columns=["rep"]);s.to_csv(os.path.join(OUT,"markov_second_moment_completion_summary.csv"),index=False)
