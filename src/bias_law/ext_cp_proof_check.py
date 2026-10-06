"""
ext_cp_proof_check.py -- numerical verification of the ingredients of the change-point theorem (App. proofs).
Isolated change at 0; walks W_k (k>=1 right, k<=-1 left) have iid N(-d^2/2, d^2) steps, W_0=0, posterior p_k ~ exp(W_k).
Checks, for d in {3,4,5}, q=exp(-d^2/8):
 (i)  m_0 = E[ AB/(A+B)^2 ] equals E_u psi(u), psi(u)=E_W h(e^{W+u}) = e^{u/2-u^2/2d^2} E[h(e^W) e^{uW/d^2}],  u=log((1+eta)/(1+lam))
 (ii) E[sqrt(1+eta)-1] <= q/(1-q),  E[1-(1+lam)^(-1/2)] <= q/(1-q)
 (iii) m_t <= q^(t+1)/(2(1-q)) for t>=1
 (iv) relative remainder (m_0 - m_NN)/m_NN vs q.
    python src/bias_law/ext_cp_proof_check.py -> reports/bias_law/x31_cp_proof_check.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import logsumexp

OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'bias_law')
h = lambda w: 1.0 / (4 * np.cosh(w / 2) ** 2)


def nn(d):
    m = -d * d / 2
    f = lambda w: h(w) * np.exp(-(w - m) ** 2 / (2 * d * d)) / (d * np.sqrt(2 * np.pi))
    return quad(f, -60, 60, limit=400, points=[0])[0]


def psi_exact(u, d):
    m = -d * d / 2
    f = lambda w: h(w + u) * np.exp(-(w - m) ** 2 / (2 * d * d)) / (d * np.sqrt(2 * np.pi))
    return quad(f, -80, 80, limit=400, points=[0, -u])[0]


def run(d, n=3_000_000, J=25, seed=0):
    rng = np.random.default_rng(seed)
    q = np.exp(-d * d / 8)
    xi = rng.normal(-d * d / 2, d, size=(n, 2 * J + 1))
    # right walk W_1..W_J from xi[:, :J]; left walk W_-1..W_-J from xi[:, J:2J]; extra step xi[:, 2J] unused
    Wr = np.cumsum(xi[:, :J], 1); Wl = np.cumsum(xi[:, J:2 * J], 1)
    # t=0: A = 1 + sum_k e^{Wl}, B = sum e^{Wr}
    lamL = np.exp(Wl).sum(1); B = np.exp(Wr).sum(1)
    A = 1 + lamL
    m0 = np.mean(A * B / (A + B) ** 2)
    eta = np.exp(Wr[:, 1:] - Wr[:, [0]]).sum(1)
    lam = lamL
    u = np.log((1 + eta) / (1 + lam))
    # psi(u) by interpolation on a grid of u
    ug = np.linspace(u.min(), u.max(), 400)
    ps = np.array([psi_exact(x, d) for x in ug])
    m0_psi = np.mean(np.interp(u, ug, ps))
    mnn = nn(d)
    a = np.mean(np.sqrt(1 + eta) - 1); b = np.mean(1 - (1 + lam) ** -0.5)
    mt = []
    for t in range(1, 7):
        # A = sum_{k<=t} e^{W_k} = 1 + left + sum_{k=1..t} e^{Wr_k}; B = sum_{l>t}
        At = 1 + lamL + np.exp(Wr[:, :t]).sum(1); Bt = np.exp(Wr[:, t:]).sum(1)
        mt.append(np.mean(At * Bt / (At + Bt) ** 2))
    return dict(d=d, q=q, m0=m0, m0_via_psi=m0_psi, m_nn=mnn, rel_remainder_m0=m0 / mnn - 1, rel_over_q=(m0 / mnn - 1) / q,
                a=a, b=b, bound_ab=q / (1 - q), m1=mt[0], m2=mt[1], m3=mt[2], bound_m1=q ** 2 / (2 * (1 - q)), bound_m2=q ** 3 / (2 * (1 - q)), sum_t_ge1=sum(mt), bound_sum=q ** 2 / (2 * (1 - q) ** 2),
                M_total=2 * (m0 + sum(mt)), M_nn=2 * mnn, M_over_Mnn_minus1=(m0 + sum(mt)) / mnn - 1, M_d_over_q=2 * (m0 + sum(mt)) * d / q, share_t_ge1=sum(mt) / (m0 + sum(mt) - mnn) if m0 + sum(mt) > mnn else float('nan'))


if __name__ == '__main__':
    rows = [run(d, seed=int(d)) for d in (3.0, 4.0, 5.0)]
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, 'x31_cp_proof_check.csv'), index=False)
    print(df.T.round(5).to_string())
