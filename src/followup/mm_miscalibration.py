"""MM estimator under a miscalibrated reported posterior: S ~ Bern(g_true), estimator sees g_rep = sigmoid(k*logit(g_true)+c)."""
import numpy as np
from mixture_moment_proto import draw, mm, ordml_pa, rng
import mixture_moment_proto as P
sig = lambda x: 1/(1+np.exp(-x)); logit = lambda p: np.log(p/(1-p))
print("k(sharpen)  c(shift)   mean|ECE|   OR-DML err   MM err   (a=2, true=[.5,1.5])")
for k, c in [(1,0),(1.3,0),(0.7,0),(2,0),(1,0.5),(1,-0.5),(1.5,0.5),(0.5,-0.5)]:
    e_or, e_mm, ece = [], [], []
    for _ in range(10):
        g, S, X, T, Y = draw(200000, 2.0, 3.0, 1.0, 1.0, 0.6)
        gr = np.clip(sig(k*logit(g)+c), 1e-6, 1-1e-6)
        ece.append(np.abs(gr-g).mean())
        e_or.append(np.linalg.norm(ordml_pa(gr,X,T,Y)-[.5,1.5])); e_mm.append(np.linalg.norm(mm(gr,X,T,Y)-[.5,1.5]))
    print(f"{k:<10} {c:<9} {np.mean(ece):.3f}      {np.mean(e_or):.3f}        {np.mean(e_mm):.3f}")
