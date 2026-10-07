"""Anchor-recalibrated mixture-moment estimator.
E[T | X, g] = m(X) + a*q(g), q(g)=P(S=1|g) unknown monotone. Fit f(g) with a piecewise-linear spline in g (X linear),
isotonic-project, normalise by anchor quantiles -> q_hat; run MM with q_hat. Needs a != 0 and anchors (q~0 / q~1 at the extremes)."""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from mixture_moment_proto import draw, mm, ordml_pa, rng
sig = lambda x: 1/(1+np.exp(-x)); logit = lambda p: np.log(p/(1-p))

def recal(g, X, T, K=10, lo_q=0.01, hi_q=0.99):
    knots = np.quantile(g, np.linspace(0, 1, K+1)[1:-1])
    B = np.c_[g, *[np.maximum(g-k, 0) for k in knots]]
    F = np.c_[np.ones(len(g)), X, B]
    coef = np.linalg.lstsq(F, T, rcond=None)[0]
    f = B @ coef[2:]
    order = np.argsort(g); iso = IsotonicRegression(increasing='auto').fit(g[order], f[order])
    fi = iso.predict(g)
    flo, fhi = np.quantile(fi, lo_q), np.quantile(fi, hi_q)
    return np.clip((fi-flo)/(fhi-flo), 1e-4, 1-1e-4)

def run(label, make, reps=10):
    e = {'OR-DML': [], 'MM raw': [], 'MM recal': []}; ece = []
    for _ in range(reps):
        g_rep, X, T, Y, q_true = make()
        ece.append(np.abs(g_rep-q_true).mean()); q = recal(g_rep, X, T)
        for k, th in (('OR-DML', ordml_pa(g_rep,X,T,Y)), ('MM raw', mm(g_rep,X,T,Y)), ('MM recal', mm(q,X,T,Y))):
            e[k].append(np.linalg.norm(th-[.5,1.5]))
    print(f"{label:<34} miscal={np.mean(ece):.3f}  " + "  ".join(f"{k}={np.mean(v):.3f}" for k, v in e.items()))

def mk(k=1, c=0, lo=0.0, a=2.0, N=200000):
    def f():
        g, S, X, T, Y = draw(N, a, 3.0, 1.0, 1.0, 0.6)
        if lo > 0: g = lo + (1-2*lo)*g                      # no anchors: true probs confined to [lo, 1-lo]
        S = (rng.random(N) < g).astype(float)
        T = 0.5*X + a*S + rng.normal(size=N); Y = (0.5+S)*T + 0.8*X + 3.0*S + rng.normal(size=N)
        gr = np.clip(sig(k*logit(g)+c), 1e-6, 1-1e-6)
        return gr, X, T, Y, g
    return f

if __name__ == '__main__':
    for k, c in [(1,0),(1.3,0),(0.7,0),(2,0),(1,.5),(1,-.5),(0.5,-.5)]:
        run(f"anchors, k={k}, c={c}", mk(k, c))
    for lo in (0.05, 0.15, 0.3):
        run(f"NO anchors (probs in [{lo},{1-lo}])", mk(1, 0, lo))
    for a in (0.5, 1.0):
        run(f"anchors, k=2, weak treat. shift a={a}", mk(2, 0, a=a))
