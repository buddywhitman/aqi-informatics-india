import numpy as np
# 1) symmetric flip example: exact closed form
for eps in [0.0,0.1,0.3,0.6,0.9]:
    a,b=1-eps/2,eps/2; s2=1.7
    M=np.array([[a,b],[b,a]]); J=s2/2*M@M; th=np.array([0.75,2.5]); S=s2/2*M@th
    tg=np.linalg.solve(J,S); err=np.linalg.norm(tg-th)
    print(eps, err, eps/(1-eps+1e-12)*abs(th[0]-th[1])/np.sqrt(2), np.linalg.eigvalsh(J).min(), s2*(1-eps)**2/2)
# 2) Gram-score orthogonality wrt m: discrete X, soft vs hard gamma, finite difference
rng=np.random.default_rng(0)
nx=4; px=np.ones(nx)/nx
def run(soft):
    # joint over X (4 values), S (2), gamma value, T value 
    # build exact finite distribution: S|X, gamma = f(S,u) with u in {0,1} noise, T = m_S(X)+V, V in {-1,1}
    pS=np.array([[.3,.7],[.5,.5],[.6,.4],[.2,.8]])
    mS=np.array([[0.,1.],[.5,1.5],[0.,2.],[1.,0.]])
    atoms=[]
    for x in range(nx):
      for s in range(2):
        for u in range(2):   # proxy noise: u=1 flips posterior toward wrong
          for v in (-1,1):
            p=px[x]*pS[x,s]*0.5*0.5
            if soft:
                g1 = (0.9 if s==0 else 0.1) if u==0 else (0.6 if s==0 else 0.4)
            else:
                g1 = float(s==0) if u==0 else float(s==1)
            atoms.append((p,x,s,g1,mS[x,s]+v, 1.3*(s==0)+0.7*(s==1)*1 + 0.5*x + (2.0 if s==0 else -1.0)*v))
    return atoms
def Pi(atoms,m,mu,theta):
    out=np.zeros(2)
    for p,x,s,g1,T,Y in atoms:
        g=np.array([g1,1-g1])
        Tt=T-m[:,x]; Yt=Y-mu[:,x]
        v=g*Tt
        out+=p*(g*Tt*Yt - g*Tt*(v@theta)*0 - (g*Tt)*(g@(Tt*theta)))
    return out
for soft in (True,False):
    atoms=run(soft)
    # gamma-weighted nuisances
    m=np.zeros((2,nx)); mu=np.zeros((2,nx))
    for k in range(2):
        for x in range(nx):
            num=den=nu2=0
            for p,xx,s,g1,T,Y in atoms:
                if xx!=x: continue
                g=[g1,1-g1][k]; num+=p*g*T; den+=p*g; nu2+=p*g*Y
            m[k,x]=num/den; mu[k,x]=nu2/den
    th=np.array([0.8,-0.4])
    base=Pi(atoms,m,mu,th)
    d=[]
    for k in range(2):
      for x in range(nx):
        h=1e-6; m2=m.copy(); m2[k,x]+=h
        d.append(np.abs((Pi(atoms,m2,mu,th)-base)/h).max())
    mu_d=[]
    for k in range(2):
      for x in range(nx):
        h=1e-6; mu2=mu.copy(); mu2[k,x]+=h
        mu_d.append(np.abs((Pi(atoms,m,mu2,th)-base)/h).max())
    print('soft' if soft else 'hard','max |dPsi/dm|',max(d),'max |dPsi/dmu|',max(mu_d))
