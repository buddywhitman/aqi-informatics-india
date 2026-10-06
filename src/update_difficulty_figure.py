"""Regenerate the main difficulty-frontier figure from authoritative reports."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    b=pd.read_csv("reports/or_dml_benchmark_summary.csv")
    r=pd.read_csv("reports/factorial_reliability_raw.csv")
    methods=["Oracle DML","Regime FE DML (Hard)","Spectral OR-DML (Ours)","Standard DML"]
    fig,axs=plt.subplots(1,4,figsize=(12.5,2.7))
    for m in methods:
        s=b[b.Method==m].sort_values("Delta_Z")
        if len(s):
            axs[0].plot(s.Delta_Z,s.Abs_Bias,marker="o",label=m.replace(" (Ours)",""))
            axs[1].plot(s.Delta_Z,s.RMSE,marker="o")
    axs[0].set_yscale("log"); axs[1].set_yscale("log")
    axs[0].set_xlabel(r"$\Delta_Z$"); axs[1].set_xlabel(r"$\Delta_Z$")
    axs[0].set_ylabel("Mean abs. bias"); axs[1].set_ylabel("RMSE")
    axs[0].set_title("(a) Observability"); axs[1].set_title("(b) Estimation error")
    for m in ["Oracle DML","Spectral OR-DML (Ours)","Filtered OR-DML (Ours)"]:
        s=b[b.Method==m].sort_values("Delta_Z")
        if len(s): axs[2].plot(s.Delta_Z,s.Coverage_PATE_95_Pct,marker="o",label=m.replace(" (Ours)",""))
    axs[2].axhline(95,ls="--",lw=1); axs[2].set_ylim(-5,105)
    axs[2].set_xlabel(r"$\Delta_Z$"); axs[2].set_ylabel("PATE coverage (%)"); axs[2].set_title("(c) Interval coverage")
    rr=r.sample(min(1800,len(r)),random_state=42)
    axs[3].scatter(rr.Difficulty_Ratio+1e-5,rr.Causal_Error_L2+1e-5,s=5,alpha=.18)
    axs[3].set_xscale("log"); axs[3].set_yscale("log")
    axs[3].set_xlabel(r"$\varepsilon_\gamma/\lambda_{\min}(J)$")
    axs[3].set_ylabel(r"$\|\hat\theta-\theta^*\|_2$"); axs[3].set_title("(d) Corrected factorial")
    axs[0].legend(fontsize=6,frameon=False,loc="best")
    fig.tight_layout(pad=.8)
    os.makedirs("plots",exist_ok=True); os.makedirs("paper/plots",exist_ok=True)
    for p in ["plots/fig2_difficulty_frontier.png","paper/plots/fig2_difficulty_frontier.png"]:
        fig.savefig(p,dpi=300,bbox_inches="tight")
    plt.close(fig)

if __name__=="__main__": main()
