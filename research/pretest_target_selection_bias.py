"""Selection-induced bias from choosing the target/subspace after seeing noisy effects.

K independent root-N effect estimates under theta=0. Compare:
- predeclared coordinate;
- post-hoc largest absolute coordinate;
- split-sample selection then independent estimation.
Quantifies winner's curse introduced by data-adaptive target selection.
"""
import os,numpy as np,pandas as pd
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),"research","results");os.makedirs(OUT,exist_ok=True)
r=np.random.default_rng(221000000);rows=[]
for K in [2,5,10,20,50,100]:
 for N in [200,1000,5000]:
  R=20000;pre=[];post=[];split=[]
  for _ in range(R):
   z=r.normal(size=K)/np.sqrt(N);z2=r.normal(size=K)/np.sqrt(N);j=np.argmax(np.abs(z));pre.append(z[0]);post.append(abs(z[j]));split.append(abs(z2[j]))
  rows.append((K,N,np.mean(np.abs(pre)),np.mean(post),np.mean(split),np.mean(post)/np.mean(np.abs(pre))))
pd.DataFrame(rows,columns=["K","N","predeclared_abs_error","posthoc_selected_abs_error","split_selected_abs_error","posthoc_inflation"]).to_csv(os.path.join(OUT,"pretest_target_selection_bias.csv"),index=False)
