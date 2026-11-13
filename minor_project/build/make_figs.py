import pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, subprocess, re, collections
NAVY,OR,BL,GR="#14213D","#E8651A","#3B6FB6","#8A8F98"
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#555","axes.grid":True,"grid.color":"#e6e6e6","grid.linewidth":.6})
A="minor_project/assets/"
# coverage
fig,ax=plt.subplots(1,2,figsize=(7.2,2.7))
stats=[]
for c,col in zip(["Delhi","Mumbai","Bengaluru","Kolkata"],[NAVY,OR,BL,GR]):
    d=pd.read_csv(f"data/raw_hourly/{c}_pollution_hourly.csv")
    d["t"]=pd.to_datetime(d.timestamp.str.extract(r"utc='([^']+)'")[0],utc=True)
    print(c,d.parameter.unique()[:12],len(d),d.t.min(),d.t.max())
    for p,a in (("no2",ax[0]),("pm25",ax[1])):
        s=d[d.parameter==p].set_index("t").value.resample("MS").count()
        a.plot(s.index.tz_localize(None),s.values,color=col,lw=1.6,label=c)
    stats.append((c,len(d),str(d.t.min().date()),str(d.t.max().date())))
import matplotlib.dates as md
for a in ax: a.xaxis.set_major_locator(md.MonthLocator(interval=3)); a.xaxis.set_major_formatter(md.DateFormatter("%b %y"))
ax[0].set_title("(a) NO$_2$ readings per month");ax[1].set_title("(b) PM$_{2.5}$ readings per month")
ax[0].legend(frameon=False,fontsize=8);
fig.tight_layout();fig.savefig(A+"coverage.png",dpi=300)
pd.DataFrame(stats,columns=["city","rows","start","end"]).to_csv("minor_project/build/coverage_stats.csv",index=False)
# commits
out=subprocess.check_output(["git","log","--format=%ad","--date=short"],text=True).split()
s=pd.Series(1,index=pd.to_datetime(out)).resample("W").sum()
fig,a=plt.subplots(figsize=(7.2,2.3));a.bar(s.index,s.values,width=6,color=NAVY)
a.set_ylabel("commits per week");a.set_title("Development activity across the project")
for lbl,dt in (("applied study","2026-03-23"),("AISTATS pivot","2026-09-28"),("consolidation\n& corrections","2026-10-06")):
    a.axvline(pd.Timestamp(dt),color=OR,ls="--",lw=1);a.text(pd.Timestamp(dt),a.get_ylim()[1]*.92,lbl,color=OR,fontsize=8,ha="left")
fig.tight_layout();fig.savefig(A+"commits.png",dpi=300);print(len(out),s.max())
