"""Toy first-order representation debiasing.

Target theta = E[S Y]/E[S T] in a stylized observed-score model, but S is replaced by
posterior proxy gamma=S+epsilon*h. Naive plug-in has O(epsilon) perturbation.
Assume a validation/calibration moment estimates derivative D at epsilon=0; corrected
target subtracts epsilon*D, leaving O(epsilon^2). This is a proof-of-concept numerical
Taylor cancellation, not yet an operational estimator.
"""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
r=np.random.default_rng(241000000);N=2000000;S=r.binomial(1,.5,N).astype(float);T=.5+S+r.normal(size=N);Y=2*T+S+r.normal(size=N);h=np.tanh(T)-np.mean(np.tanh(T))
def target(g):return np.mean(g*Y)/np.mean(g*T)
base=target(S);# numerical derivative around zero
d=1e-5;D=(target(S+d*h)-target(S-d*h))/(2*d)
rows=[]
for eps in np.logspace(-4,-.5,15):
 naive=target(S+eps*h);corr=naive-eps*D;rows.append((eps,abs(naive-base),abs(corr-base),D))
pd.DataFrame(rows,columns=["epsilon","naive_abs_error","first_order_corrected_abs_error","derivative"]).to_csv(os.path.join(OUT,"representation_orthogonal_toy.csv"),index=False)
