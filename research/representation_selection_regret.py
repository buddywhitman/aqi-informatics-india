"""Search for large causal/task regret from global representation selection.

Generate many synthetic representation error profiles over x in [0,1] and localized
task leverage functions. Compare representation chosen by global mean error with
minimax and per-task oracle. Records regret ratios as leverage localization increases.
"""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
r=np.random.default_rng(141000000);N=200000;x=r.random(N);rows=[]
for L in [5,10,50,100,500,1000]:
 tasks={}
 centers=np.linspace(.1,.9,9)
 for j,c in enumerate(centers):tasks[j]=np.where(np.abs(x-c)<.05,L,1.)
 profiles={}
 profiles["uniform"]=np.full(N,.01)
 for j,c in enumerate(centers):
  profiles[f"spec{j}"]=np.where(np.abs(x-c)<.05,.001,.012)
 # globally attractive but vulnerable around center 0.5
 profiles["global_best"]=np.where(np.abs(x-.5)<.05,.05,.003)
 names=list(profiles);risk=np.array([[np.sum(w*profiles[n])/np.sum(w) for w in tasks.values()] for n in names]);glob=np.array([profiles[n].mean() for n in names])
 ig=glob.argmin();imm=risk.max(1).argmin();oracle=risk.min(0)
 rows.append((L,names[ig],risk[ig].mean(),risk[ig].max(),names[imm],risk[imm].mean(),risk[imm].max(),oracle.mean(),risk[ig].max()/risk[imm].max(),risk[ig].mean()/oracle.mean()))
pd.DataFrame(rows,columns=["leverage_ratio","global_pick","global_avg","global_worst","minimax_pick","minimax_avg","minimax_worst","oracle_avg","worst_regret_ratio","avg_regret_to_oracle"]).to_csv(os.path.join(OUT,"representation_selection_regret.csv"),index=False)
