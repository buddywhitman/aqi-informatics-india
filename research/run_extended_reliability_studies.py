"""Exploratory follow-up studies for task-relative reliability.

Not part of the AISTATS submission evidence chain. The default replication counts
match the exploratory audit; increase them for publication-grade Monte Carlo precision.
"""
import os, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT=os.path.join(ROOT,"research","results"); os.makedirs(OUT,exist_ok=True)
TH=np.array([0.75,2.50])
PHASE_REPS=int(os.getenv("PHASE_REPS","10"))
REG_REPS=int(os.getenv("REG_REPS","20"))
TRANSITION_REPS=int(os.getenv("TRANSITION_REPS","20"))

def draw_score(N,sigma1,eps,seed):
    r=np.random.default_rng(seed); P=np.array([[.88,.12],[.15,.85]]); pi=np.array([.5556,.4444])
    S=np.empty(N,int); S[0]=r.choice(2,p=pi)
    for t in range(1,N): S[t]=r.choice(2,p=P[S[t-1]])
    V=r.normal(size=N)*np.where(S==0,1.0,sigma1); U=r.normal(0,1.2,N)
    H=np.c_[1-S,S].astype(float); a=eps/2.0; g=(1-a)*H+a*H[:,::-1]
    tt=np.vstack([V,V]); ty=np.vstack([TH[0]*V+U,TH[1]*V+U])
    J=np.zeros((2,2)); q=np.zeros(2)
    for j in range(2):
        q[j]=np.mean(g[:,j]*tt[j]*ty[j])
        for k in range(2): J[j,k]=np.mean(g[:,j]*g[:,k]*tt[j]*tt[k])
    return J,q

def phase_scaling():
    rows=[]
    for N in [300,600,1200,2400,4800]:
      for sig in [.12,.25,.5,1.,1.5]:
       for eps in [0,.01,.03,.07,.15]:
        for rep in range(PHASE_REPS):
         J,q=draw_score(N,sig,eps,10_000_000+N*10000+int(sig*100)*100+int(eps*1000)+rep)
         lm=max(np.linalg.eigvalsh(J)[0],1e-12); est=np.linalg.solve(J,q); err=np.linalg.norm(est-TH)
         rows.append((N,sig,eps,rep,lm,err,eps/lm,np.sqrt(N)*eps/lm))
    d=pd.DataFrame(rows,columns="N sigma1 eps rep lmin err difficulty scaled_difficulty".split())
    d.to_csv(os.path.join(OUT,"phase_scaling_raw.csv"),index=False)
    corr=[]
    for N,g in d.groupby("N"):
        corr.append((N,spearmanr(g.err,g.difficulty).statistic,spearmanr(g.err,g.eps).statistic,spearmanr(g.err,1/g.lmin).statistic))
    pd.DataFrame(corr,columns=["N","rho_diff","rho_eps","rho_invlam"]).to_csv(os.path.join(OUT,"phase_scaling_correlations.csv"),index=False)
    x=d[(d.eps>0)&(d.err>0)]
    m=LinearRegression().fit(np.c_[np.log(x.difficulty),np.log(x.N)],np.log(x.err))
    pd.DataFrame([{"coef_log_difficulty":m.coef_[0],"coef_log_N":m.coef_[1],"R2":m.score(np.c_[np.log(x.difficulty),np.log(x.N)],np.log(x.err))}]).to_csv(os.path.join(OUT,"phase_scaling_loglog.csv"),index=False)

def directional_sensitivity():
    rows=[]
    for cond in [2,5,20,100]:
        lm=1/cond; J=np.diag([lm,1.]); bnorm=.05
        for a in np.linspace(0,np.pi/2,37):
            b=bnorm*np.array([np.cos(a),np.sin(a)]); exact=np.linalg.norm(np.linalg.solve(J,b)); bound=bnorm/lm
            rows.append((cond,a*180/np.pi,exact,bound,exact/bound))
    pd.DataFrame(rows,columns=["condition_number","angle_from_weak_eigenvector_deg","exact_error","scalar_bound","bound_tightness"]).to_csv(os.path.join(OUT,"directional_sensitivity.csv"),index=False)


def k3_orientation_study():
    """Same average proxy error can have very different downstream impact in K=3."""
    theta=np.array([.5,1.5,3.]); sig=np.array([.2,1.,1.]); rows=[]
    for a in [.03,.06,.10,.20,.30]:
      for mode in ["weak","strong","balanced"]:
       for rep in range(100):
        r=np.random.default_rng(50_000_000+rep); N=5000; S=r.integers(0,3,N); V=r.normal(size=N)*sig[S]; U=r.normal(0,1,N); H=np.eye(3)[S]; g=H.copy()
        if mode=="weak":
            m=S==0; g[m,0]-=a; g[m,1]+=a
        elif mode=="strong":
            m=S==1; g[m,1]-=a; g[m,2]+=a
        else:
            aa=a/3
            for k in range(3):
                m=S==k; g[m,k]-=aa; g[m,(k+1)%3]+=aa
        eps=np.abs(g-H).sum(1).mean(); tt=np.tile(V,(3,1)); ty=np.vstack([theta[k]*V+U for k in range(3)]); J=np.zeros((3,3)); q=np.zeros(3)
        for j in range(3):
            q[j]=np.mean(g[:,j]*tt[j]*ty[j])
            for k in range(3): J[j,k]=np.mean(g[:,j]*g[:,k]*tt[j]*tt[k])
        lm=np.linalg.eigvalsh(J)[0]; b=q-J@theta; est=np.linalg.pinv(J)@q
        rows.append((a,mode,rep,eps,lm,np.linalg.norm(est-theta),eps/lm,np.linalg.norm(b),np.linalg.norm(b)/lm,np.linalg.norm(np.linalg.solve(J,b))))
    d=pd.DataFrame(rows,columns=["a","orientation","rep","eps","lmin","error","eps_over_lmin","b_norm","b_over_lmin","directional_error"])
    d.to_csv(os.path.join(OUT,"k3_orientation_raw.csv"),index=False)
    d.groupby(["a","orientation"],as_index=False).mean(numeric_only=True).to_csv(os.path.join(OUT,"k3_orientation_summary.csv"),index=False)

def adaptive_regularization():
    L=np.r_[0,np.logspace(-4,0,13)]; rows=[]
    for N in [600,1200,2400]:
      for sig in [.12,.25,.5,1.]:
       for eps in [.01,.03,.07,.15]:
        mse={l:[] for l in L}; lms=[]
        for rep in range(REG_REPS):
            J,q=draw_score(N,sig,eps,20_000_000+N*10000+int(sig*100)*100+int(eps*1000)+rep); lms.append(np.linalg.eigvalsh(J)[0])
            for l in L: mse[l].append(np.linalg.norm(np.linalg.solve(J+l*np.eye(2),q)-TH)**2)
        mm={l:np.mean(v) for l,v in mse.items()}; best=min(mm,key=mm.get)
        rows.append((N,sig,eps,np.mean(lms),eps/np.mean(lms),best,mm[0],mm[best]))
    d=pd.DataFrame(rows,columns=["N","sigma1","eps","lmin","difficulty","oracle_lambda","mse_unreg","mse_oracle"]); d["gain_pct"]=100*(d.mse_unreg-d.mse_oracle)/d.mse_unreg
    d.to_csv(os.path.join(OUT,"adaptive_regularization_oracle.csv"),index=False)

def synthetic_transition_localization():
    rows=[]
    for sep in [.5,1.,2.,3.]:
      for rep in range(TRANSITION_REPS):
        N=1500; r=np.random.default_rng(30_000_000+int(sep*1000)+rep); P=np.array([[.94,.06],[.08,.92]]); pi=np.array([.5714,.4286]); S=np.empty(N,int); S[0]=r.choice(2,p=pi)
        for t in range(1,N): S[t]=r.choice(2,p=P[S[t-1]])
        mus=np.array([-sep/2,sep/2]); z=r.normal(mus[S],1); a=np.zeros((N,2)); a[0]=pi*np.exp(-.5*(z[0]-mus)**2); a[0]/=a[0].sum()
        for t in range(1,N):
            pred=a[t-1]@P; lik=np.exp(-.5*(z[t]-mus)**2); a[t]=pred*lik; a[t]/=a[t].sum()
        err=np.abs(a-np.c_[1-S,S]).sum(1); ent=-(a*np.log(np.clip(a,1e-12,1))).sum(1); cps=np.flatnonzero(np.r_[False,S[1:]!=S[:-1]]); dist=np.full(N,N); ar=np.arange(N)
        for cp in cps: dist=np.minimum(dist,np.abs(ar-cp))
        for lo,hi,name in [(0,1,"0-1"),(2,3,"2-3"),(4,8,"4-8"),(9,99999,"9+")]:
            m=(dist>=lo)&(dist<=hi); rows.append((sep,rep,name,m.sum(),err[m].mean(),ent[m].mean()))
    pd.DataFrame(rows,columns=["separation","rep","distance_band","n","posterior_l1_error","entropy"]).to_csv(os.path.join(OUT,"transition_localization.csv"),index=False)

def real_transition_diagnostics():
    try:
        sys.path.insert(0,ROOT); from src.or_dml import LatentRegimeHMM
    except Exception as e:
        print("Skipping real transition diagnostics:",e); return
    p=os.path.join(ROOT,"data","processed_clean","combined_hourly_clean.csv")
    if not os.path.exists(p): print("Skipping real transition diagnostics: data absent"); return
    d=pd.read_csv(p); feat=["temperature","wind_speed","humidity","pressure","hour_sin","hour_cos"]; req=feat+["no2","pm25","pm25_lag_1h","no2_lag_1h","pm25_roll_3h","no2_roll_3h"]; rows=[]
    for city in ["Delhi","Mumbai","Bengaluru","Kolkata"]:
        x=d[d.city==city].dropna(subset=req); Z=x[feat].values; Z=(Z-Z.mean(0))/(Z.std(0)+1e-8)
        h=LatentRegimeHMM(2,n_iter=50,random_state=42).fit(Z); gf=h.predict_posteriors(Z,"filter"); gs=h.predict_posteriors(Z,"smooth"); state=gs.argmax(1)
        cps=np.flatnonzero(np.r_[False,state[1:]!=state[:-1]]); dist=np.full(len(x),len(x)); ar=np.arange(len(x))
        for cp in cps: dist=np.minimum(dist,np.abs(ar-cp))
        ef=-(gf*np.log(np.clip(gf,1e-12,1))).sum(1); es=-(gs*np.log(np.clip(gs,1e-12,1))).sum(1); div=np.abs(gf-gs).sum(1)
        for lo,hi,name in [(0,1,"0-1"),(2,3,"2-3"),(4,8,"4-8"),(9,24,"9-24"),(25,99999,"25+")]:
            m=(dist>=lo)&(dist<=hi)
            if m.sum(): rows.append((city,name,m.sum(),ef[m].mean(),es[m].mean(),div[m].mean()))
    pd.DataFrame(rows,columns=["city","distance_band","n","filter_entropy","smooth_entropy","filter_smooth_l1"]).to_csv(os.path.join(OUT,"real_transition_diagnostics.csv"),index=False)

if __name__=="__main__":
    phase_scaling(); directional_sensitivity(); k3_orientation_study(); adaptive_regularization(); synthetic_transition_localization(); real_transition_diagnostics()
