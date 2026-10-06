"""Integrated latent-state DGP beyond the linearized score experiment.

K=3 Markov states; Gaussian emissions; state-specific treatment variance/effects.
Infer states from emissions with Gaussian HMM-like filtering using known parameters.
Compare oracle state-specific slope vector with soft proxy weighted slopes. Compute:
- posterior error;
- oracle/proxy information spectra;
- empirical target bias;
- first-order score perturbation b evaluated at oracle theta;
- AJ^-1 b prediction for several targets.
This tests the operator in an actual generated-state nonlinear ratio estimator.
"""
import os,numpy as np,pandas as pd
from scipy.special import logsumexp
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
def filt(z,mu,P):
 n=len(z);k=len(mu);la=np.empty((n,k));la[0]=-np.log(k)-.5*(z[0]-mu)**2;la[0]-=logsumexp(la[0])
 for t in range(1,n):
  pred=np.array([logsumexp(la[t-1]+np.log(P[:,j])) for j in range(k)]);la[t]=pred-.5*(z[t]-mu)**2;la[t]-=logsumexp(la[t])
 return np.exp(la)
r=np.random.default_rng(271000000);rows=[]
for N in [1000,5000,20000]:
 for sep in [.5,1.,2.]:
  for weak in [.08,.2]:
   for rep in range(80):
    K=3;p=.9;P=np.full((K,K),(1-p)/2);np.fill_diagonal(P,p);S=np.empty(N,int);S[0]=r.integers(K)
    for t in range(1,N):S[t]=r.choice(K,p=P[S[t-1]])
    mu=np.array([-sep,0,sep]);sd=np.array([1.,.3,weak]);theta=np.array([.5,1.,1.5]);Z=r.normal(mu[S],1);T=r.normal(0,sd[S]);Y=theta[S]*T+r.normal(size=N);H=np.eye(K)[S];g=filt(Z,mu,P)
    J=(H.T*(T*T))@H/N;Soracle=np.mean(H*T[:,None]*Y[:,None],0);th_or=np.linalg.solve(J,Soracle)
    Jp=(g.T*(T*T))@g/N;Sp=np.mean(g*T[:,None]*Y[:,None],0);th_p=np.linalg.solve(Jp+1e-12*np.eye(K),Sp)
    # proxy score perturbation at oracle target relative to oracle score zero
    b=Sp-Jp@th_or
    for an,a in {"avg":np.ones(K)/K,"state0":np.array([1.,0,0]),"state2":np.array([0,0,1.])}.items():
     actual=a@(th_p-th_or);pred=a@np.linalg.solve(Jp+1e-12*np.eye(K),b)
     rows.append((N,sep,weak,rep,an,actual,pred,abs(actual-pred),np.linalg.eigvalsh(J)[0],np.linalg.eigvalsh(Jp)[0]))
pd.DataFrame(rows,columns=["N","separation","weak_sd","rep","target","actual_proxy_minus_oracle","operator_prediction","abs_prediction_error","oracle_lmin","proxy_lmin"]).to_csv(os.path.join(OUT,"integrated_latent_dgp.csv"),index=False)
