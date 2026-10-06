"""Operational toy: augment proxy moment with calibration/validation moment.

Latent binary S, proxy gamma=S+epsilon*h. Target solves E[S(Y-theta T)]=0.
Naive replaces S by gamma. If a validation/calibration sample identifies
m_h(theta)=E[h(Y-theta T)], augmented score
 E[gamma(Y-theta T)] - epsilon*m_h(theta)
recovers oracle moment exactly in this controlled perturbation family.
Then perturb correction coefficient to test robustness.
"""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
r=np.random.default_rng(251000000);N=1000000;S=r.binomial(1,.5,N).astype(float);T=.5+S+r.normal(size=N);Y=2*T+S+r.normal(size=N);h=np.tanh(T)-np.mean(np.tanh(T))
def ratio(w):return np.mean(w*Y)/np.mean(w*T)
oracle=ratio(S);rows=[]
for eps in [.001,.01,.05,.1,.2]:
 gamma=S+eps*h;naive=ratio(gamma)
 # exact correction to numerator and denominator using known perturbation direction h
 for rel in [0.,.1,.25,.5]:
  ec=eps*(1+rel);corr=(np.mean(gamma*Y)-ec*np.mean(h*Y))/(np.mean(gamma*T)-ec*np.mean(h*T))
  rows.append((eps,rel,abs(naive-oracle),abs(corr-oracle)))
pd.DataFrame(rows,columns=["epsilon","relative_correction_error","naive_abs_error","augmented_abs_error"]).to_csv(os.path.join(OUT,"orthogonal_latent_moment.csv"),index=False)
