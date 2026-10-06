"""Adversarial integrated latent DGP with nonlinear treatment response and misspecified linear proxy estimator.

K=3 Markov states; imperfect emission filtering; state-specific treatment scales.
Outcome contains theta_s*T + q_s*T^2 + noise, while oracle/proxy estimators fit only
state-specific linear slopes. Compare proxy-oracle displacement with linear score
operator evaluated at oracle linear projection. Because weighted least squares remains
linear in theta, the identity still holds algebraically conditional on Jp/Sp; this test
instead measures whether the proxy-vs-true structural effect interpretation breaks.
"""
import os,numpy as np,pandas as pd
from scipy.special import logsumexp
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
def filt(z,mu,P):
 n=len(z);k=len(mu);la=np.empty((n,k));la[0]=-np.log(k)-.5*(z[0]-mu)**2;la[0]-=logsumexp(la[0])
 for t in range(1,n):
  pred=np.array([logsumexp(la[t-1]+np.log(P[:,j])) for j in range(k)]);la[t]=pred-.5*(z[t]-mu)**2;la[t]-=logsumexp(la[t])
 return np.exp(la)
r=np.random.default_rng(281000000);rows=[]
for curvature in [0,.2,.5,1.,2.]:
 for sep in [.5,1.,2.]:
  for rep in range(120):
   N=5000;K=3;p=.9;P=np.full((K,K),(1-p)/2);np.fill_diagonal(P,p);S=np.empty(N,int);S[0]=r.integers(K)
   for t in range(1,N):S[t]=r.choice(K,p=P[S[t-1]])
   mu=np.array([-sep,0,sep]);sd=np.array([1.,.3,.08]);theta=np.array([.5,1.,1.5]);q=curvature*np.array([.2,-.1,.3]);Z=r.normal(mu[S],1);T=r.normal(0,sd[S]);Y=theta[S]*T+q[S]*T*T+r.normal(size=N);H=np.eye(K)[S];g=filt(Z,mu,P)
   J=(H.T*(T*T))@H/N;So=np.mean(H*T[:,None]*Y[:,None],0);tho=np.linalg.solve(J,So);Jp=(g.T*(T*T))@g/N;Sp=np.mean(g*T[:,None]*Y[:,None],0);thp=np.linalg.solve(Jp+1e-12*np.eye(K),Sp)
   structural_err=np.linalg.norm(thp-theta);oracle_projection_err=np.linalg.norm(tho-theta);proxy_projection_gap=np.linalg.norm(thp-tho)
   rows.append((curvature,sep,rep,structural_err,oracle_projection_err,proxy_projection_gap,np.linalg.eigvalsh(Jp)[0]/np.linalg.eigvalsh(J)[0]))
pd.DataFrame(rows,columns=["curvature","separation","rep","proxy_structural_error","oracle_linear_projection_error","proxy_oracle_projection_gap","lmin_inflation"]).to_csv(os.path.join(OUT,"integrated_latent_nonlinear_outcome.csv"),index=False)
