"""Prototype: calibrated-posterior identification by linear-in-gamma moment regression (no residual-state bias).
Compares OR-DML (posterior-adjusted, biased) with the mixture-moment (MM) estimator on the Theorem-4 DGP."""
import numpy as np
rng = np.random.default_rng(1)

def draw(N, a, b, dth, sig, shape, th0=0.5, mc=0.5):
    g = np.clip(rng.beta(shape, shape, N), 1e-6, 1-1e-6)
    S = (rng.random(N) < g).astype(float)
    X = rng.normal(size=N)
    T = mc*X + a*S + rng.normal(0, sig, N)
    Y = (th0 + dth*S)*T + 0.8*X + b*S + rng.normal(size=N)
    return g, S, X, T, Y

def ols_resid(F, v): return v - F @ np.linalg.lstsq(F, v, rcond=None)[0]

def ordml_pa(g, X, T, Y):
    F = np.c_[np.ones(len(g)), X, g, g*X]
    rt, ry = ols_resid(F, T), ols_resid(F, Y); W = np.c_[1-g, g]
    J = np.einsum('ni,nj,n->ij', W, W, rt**2)/len(g); S_ = (W*(rt*ry)[:, None]).mean(0)
    return np.linalg.solve(J, S_)

def mm(g, X, T, Y):
    """Regress every moment of Z=(1,X,T) and Z*Y on (1-g, g): coefficients are regime-conditional moments."""
    Z = np.c_[np.ones(len(g)), X, T]; W = np.c_[1-g, g]
    ZZ = np.einsum('ni,nj->nij', Z, Z).reshape(len(g), -1); ZY = Z*Y[:, None]
    A = np.linalg.lstsq(W, np.c_[ZZ, ZY], rcond=None)[0]       # 2 x (9+3)
    th = []
    for s in range(2):
        M = A[s, :9].reshape(3, 3); c = A[s, 9:]
        th.append(np.linalg.solve(M, c)[2])
    return np.array(th)

if __name__ == '__main__':
    print("a   shape  true        OR-DML(PA)        MM            sd(MM) over 20 reps")
    for a, shape in [(0.5,.6),(1,.6),(2,.6),(4,.6),(2,2.0),(2,5.0)]:
        true = np.array([.5, 1.5]); r_or, r_mm = [], []
        for _ in range(20):
            g,S,X,T,Y = draw(200000, a, 3.0, 1.0, 1.0, shape)
            r_or.append(ordml_pa(g,X,T,Y)); r_mm.append(mm(g,X,T,Y))
        r_or, r_mm = np.array(r_or), np.array(r_mm)
        print(f"{a:<3} {shape:<5} {true}  {r_or.mean(0).round(3)}  {r_mm.mean(0).round(3)}  {r_mm.std(0).round(3)}  var(g)={shape*shape/((2*shape)**2*(2*shape+1)):.3f}")
