"""Factorial validation of representation-to-task reliability.

Independently manipulates latent-state proxy error and post-residualization task
conditioning. Conditioning is changed through state-specific treatment innovation
variance, so it survives correctly specified regime-aware nuisance residualization.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from src.or_dml import PurgedBlockKFold

SIGMA_GRID=[0.08,0.15,0.30,0.50,0.75,1.00,1.50]
EPS_GRID=[0.00,0.01,0.02,0.05,0.10,0.20,0.40]

def generate_factorial_dgp(N=1200,sigma_state1=1.0,seed=700000):
    rng=np.random.RandomState(seed); persistence,p10=0.88,0.15; p01=1.0-persistence
    P=np.array([[persistence,p01],[p10,1.0-p10]])
    stat=np.array([p10/(p01+p10),p01/(p01+p10)])
    S=np.zeros(N,dtype=int); S[0]=0 if rng.rand()<stat[0] else 1
    for t in range(1,N): S[t]=0 if rng.rand()<P[S[t-1],0] else 1
    X=np.zeros((N,3)); inn=rng.randn(N,3); X[0]=inn[0]
    for t in range(1,N): X[t]=0.65*X[t-1]+np.sqrt(1-0.65**2)*inn[t]
    V=rng.randn(N); sigma=np.where(S==0,1.0,sigma_state1)
    T=2.0*(1-S)+5.0*S+0.8*X[:,0]-0.5*X[:,1]+sigma*V
    theta=np.array([0.75,2.50]); U=rng.normal(0,1.2,N)
    Y=theta[S]*T+15.0*(1-S)+50.0*S+1.2*X[:,0]+0.9*X[:,2]+U
    H=np.column_stack([1-S,S]).astype(float)
    return Y,T,X,H,theta

def oracle_crossfit_nuisances(Y,T,X,H):
    N=len(Y); K=H.shape[1]; ty=np.zeros((K,N)); tt=np.zeros((K,N))
    splitter=PurgedBlockKFold(n_splits=4,embargo_tau=12)
    for k in range(K):
        w=np.maximum(H[:,k],1e-4)
        for tr,te in splitter.split(N):
            my=Ridge(alpha=1.0).fit(X[tr],Y[tr],sample_weight=w[tr])
            mt=Ridge(alpha=1.0).fit(X[tr],T[tr],sample_weight=w[tr])
            ty[k,te]=Y[te]-my.predict(X[te]); tt[k,te]=T[te]-mt.predict(X[te])
    return ty,tt

def evaluate_cell(H,theta_true,ty,tt,eps):
    alpha=eps/2.0; gamma=(1.0-alpha)*H+alpha*H[:,::-1]
    J=np.zeros((2,2)); S=np.zeros(2)
    for j in range(2):
        S[j]=np.mean(gamma[:,j]*tt[j]*ty[j])
        for k in range(2): J[j,k]=np.mean(gamma[:,j]*gamma[:,k]*tt[j]*tt[k])
    lmin=float(np.min(np.linalg.eigvalsh(J))); theta=np.linalg.pinv(J)@S
    return lmin,float(np.linalg.norm(theta-theta_true))

def run(n_reps=100,N=1200):
    rows=[]
    for rep in range(n_reps):
        for sigma1 in SIGMA_GRID:
            Y,T,X,H,theta=generate_factorial_dgp(N,sigma1,700000+rep)
            ty,tt=oracle_crossfit_nuisances(Y,T,X,H)
            for eps in EPS_GRID:
                lmin,err=evaluate_cell(H,theta,ty,tt,eps)
                rows.append((rep,sigma1,eps,lmin,err,eps/max(lmin,1e-12)))
    raw=pd.DataFrame(rows,columns=["Replication","Residual_SD_State1","Proxy_Error_Eps","Lambda_Min_J","Causal_Error_L2","Difficulty_Ratio"])
    os.makedirs("reports",exist_ok=True); raw.to_csv("reports/factorial_reliability_raw.csv",index=False)
    summary=raw.groupby(["Residual_SD_State1","Proxy_Error_Eps"],as_index=False).agg(
        Mean_Lambda_Min=("Lambda_Min_J","mean"),Mean_Causal_Error_L2=("Causal_Error_L2","mean"),
        Median_Causal_Error_L2=("Causal_Error_L2","median"),Std_Causal_Error_L2=("Causal_Error_L2","std"),
        Mean_Difficulty_Ratio=("Difficulty_Ratio","mean"))
    summary.to_csv("reports/factorial_reliability_summary.csv",index=False)
    cell=summary.copy(); cell["Inv_Lambda"]=1.0/cell.Mean_Lambda_Min
    corr=pd.DataFrame([{
        "N_Replications":n_reps,"N_Cells":len(summary),"N_Runs":len(raw),
        "Spearman_Run_Difficulty":spearmanr(raw.Causal_Error_L2,raw.Difficulty_Ratio).statistic,
        "Spearman_Run_ProxyError":spearmanr(raw.Causal_Error_L2,raw.Proxy_Error_Eps).statistic,
        "Spearman_Run_InvLambda":spearmanr(raw.Causal_Error_L2,1.0/raw.Lambda_Min_J).statistic,
        "Spearman_Cell_Difficulty":spearmanr(cell.Mean_Causal_Error_L2,cell.Mean_Difficulty_Ratio).statistic,
        "Spearman_Cell_ProxyError":spearmanr(cell.Mean_Causal_Error_L2,cell.Proxy_Error_Eps).statistic,
        "Spearman_Cell_InvLambda":spearmanr(cell.Mean_Causal_Error_L2,cell.Inv_Lambda).statistic}])
    corr.to_csv("reports/factorial_reliability_correlations.csv",index=False)
    risk=[]
    for coverage in [1.0,0.8,0.6,0.4,0.2]:
        q=raw.Difficulty_Ratio.quantile(coverage); sub=raw[raw.Difficulty_Ratio<=q]
        risk.append((coverage,q,sub.Causal_Error_L2.mean(),sub.Causal_Error_L2.median(),sub.Causal_Error_L2.quantile(0.95)))
    pd.DataFrame(risk,columns=["Coverage","Difficulty_Threshold","Mean_Causal_Error","Median_Causal_Error","P95_Causal_Error"]).to_csv("reports/factorial_risk_coverage.csv",index=False)
    print(corr.to_string(index=False))
    return raw,summary,corr

if __name__=="__main__": run()
