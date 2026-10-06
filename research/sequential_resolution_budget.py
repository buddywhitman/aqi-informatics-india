"""Stylized simulation for sequential causal-resolution budget.

Single target contrast, with effective state sample size N/K_N. Treatment scale N^-alpha.
Errors follow AR(1) with rho_N=1-c*N^-eta. Effect separation N^-beta.
We estimate slope and use the *known* AR(1) long-run variance approximation to form
a z statistic. Goal: organize noncentrality by exponent
1-eta-kappa-2alpha-2beta, not provide a production estimator.
"""
import os,numpy as np,pandas as pd
from scipy.stats import norm
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));OUT=os.path.join(ROOT,"research","results");os.makedirs(OUT,exist_ok=True)
REPS=int(os.getenv("SEQ_RES_REPS","500"));rows=[]
settings=[(.0,.0,.2,.1),(.2,.2,.1,.1),(.25,.25,.15,.1),(.3,.3,.15,.1),(.4,.2,.15,.15)]
for eta,kappa,alpha,beta in settings:
 expo=1-eta-kappa-2*alpha-2*beta
 for N in [800,3200,12800]:
  K=max(1,int(round(N**kappa)));n=max(20,N//K);rho=max(0.,1-N**(-eta)) if eta>0 else 0.;delta=N**(-beta);rej=0;zs=[]
  for rep in range(REPS):
   r=np.random.default_rng(411_000_000+int(eta*100)*1000000+int(kappa*100)*10000+N+rep);t=r.normal(0,N**(-alpha),n)
   e=np.empty(n);e[0]=r.normal()
   for i in range(1,n):e[i]=rho*e[i-1]+np.sqrt(max(1-rho*rho,1e-12))*r.normal()
   y=delta*t+e;den=np.sum(t*t);hat=np.sum(t*y)/den
   # approximate iid slope variance times AR(1) LRV inflation
   se=np.sqrt((1+rho)/(1-rho))/np.sqrt(den);z=hat/se;zs.append(z);rej+=abs(z)>norm.ppf(.975)
  rows.append((eta,kappa,alpha,beta,expo,N,K,n,rho,delta,rej/REPS,np.mean(np.abs(zs))))
d=pd.DataFrame(rows,columns=["eta","kappa","alpha","beta","information_exponent","N","K","n_per_state","rho","delta","power","mean_abs_z"]);d.to_csv(os.path.join(OUT,"sequential_resolution_budget.csv"),index=False);print(d.to_string(index=False))
