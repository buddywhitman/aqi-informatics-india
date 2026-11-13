# !Annexure G: Assumptions and Proofs

This annexure states the assumptions used in the theory and gives the complete derivations of the results that are proved by direct computation. The remaining proofs (graceful degradation, posterior-adjusted nuisances, asymptotics) are in appendices D, E and H of the manuscript (Annexure F).

## G.1 Assumptions

**(A1) Unconfoundedness and innovation exogeneity.** Conditional on $X_t$, $S_t$ and the proxy history $Z_{1:N}$, potential outcomes are independent of $T_t$, and $\mathbb{E}[U_t \mid X_t, S_t, T_t, Z_{1:N}] = 0$.

**(A2) Overlap, positivity and envelopes.** The stationary law satisfies $\min_k \pi^*_k \ge \eta > 0$, and $P(S_t=k\mid X_t=x) \ge c_0 > 0$ almost everywhere. Variables are bounded, $|T_t|\le B_T$ and $|Y_t|\le B_Y$. Within each regime the treatment variance is bounded away from zero and infinity, $c_1 \le \mathrm{Var}(T_t \mid X_t, S_t=k) \le c_2$.

**(A3) Latent proxy sufficiency.** $P(S_t=k \mid X_t, T_t, Z_{1:N}) = P(S_t=k\mid Z_{1:N}) = \gamma_{tk}$ (or the filtered version). This assumption is used for the structural interpretation of the posterior; the degradation bound of Theorem 3 does not require it, and the simulation designs deliberately violate it.

**(A4) Mixing and nuisance rates.** The process is strictly stationary and geometrically $\alpha$-mixing, $\alpha(m) \le C_\alpha e^{-c_\alpha m}$. Nuisance estimators satisfy $\|\hat\mu_Y-\mu_Y\|_{L_2}\,\|\hat{\boldsymbol\mu}_D-\boldsymbol\mu_D\|_{L_2} = o_P(N^{-1/2})$, with individual rates $o_P(N^{-1/4})$.

## G.2 Proof of Theorem 1 (omitted regime bias)

Let $\pi_k(X)=P(S_t{=}k\mid X_t)$, $m_k(X)=\mathbb{E}[T_t\mid X_t,S_t{=}k]$, $\bar m(X)=\sum_k\pi_k m_k$, $g_k(X)=g(X,k)$ and $\bar g=\sum_k\pi_k g_k$. Residualizing on $X$ alone gives
$$\tilde T_t = T_t - \bar m(X_t) = \sum_k \mathbb{1}(S_t{=}k)\,\Delta m_k(X_t) + V_t, \qquad \Delta m_k = m_k - \bar m .$$
The outcome residual is $\tilde Y_t = \sum_k\theta_k^*\mathbb{1}(S_t{=}k)T_t - \sum_k\theta_k^*\pi_k m_k + g(X_t,S_t)-\bar g(X_t)+U_t$. Writing $T_t=\tilde T_t+\bar m(X_t)$, the treatment term splits into $\sum_k\theta_k^*\mathbb{1}(S_t{=}k)\tilde T_t$ plus a term in $\bar m(X_t)\mathbb{1}(S_t{=}k)-\pi_k m_k$. Now take the inner product with $\tilde T_t$ and the expectation, term by term.

1. $\mathbb{E}[\tilde T_t U_t]=0$ by (A1).
2. $\mathbb{E}\big[\tilde T_t \sum_k\theta_k^*\mathbb{1}(S_t{=}k)\tilde T_t\big] = \mathbb{E}[\tilde T_t^2]\sum_k\omega_k\theta_k^*$ with $\omega_k=\mathbb{E}[\mathbb{1}(S_t{=}k)\tilde T_t^2]/\mathbb{E}[\tilde T_t^2]$ and $\sum_k\omega_k=1$.
3. Because $\mathbb{E}[\tilde T_t\mid X_t]=0$ and $\mathbb{E}[\tilde T_t\mathbb{1}(S_t{=}k)\mid X_t]=\pi_k\Delta m_k$, the mean-adjustment term equals $\sum_k\theta_k^*\mathbb{E}[\bar m\,\Delta m_k\,\pi_k]$, which is $\mathcal{B}_{\rm het}\,\mathbb{E}[\tilde T_t^2]$.
4. The confounding term is $\mathbb{E}[\tilde T_t(g(X_t,S_t)-\bar g(X_t))]=\sum_k\mathbb{E}[\pi_k\Delta m_k\Delta g_k]$, which is $\mathcal{B}_{\rm conf}\,\mathbb{E}[\tilde T_t^2]$.

Dividing by $\mathbb{E}[\tilde T_t^2]$ gives the stated decomposition. $\square$

## G.3 Proof of Proposition 1 and Corollary 1

*Setting.* With oracle regime-wise nuisances, the regime-$k$ residuals evaluated on an observation from regime $s$ are $\tilde T_k = (m_s-m_k)+\sigma_s V$ and $\tilde Y_k = (\mu_s-\mu_k)+\theta_s\sigma_s V + U$, where $V$ has mean zero and unit variance and is independent of $U$ and of $S$. The covariate parts cancel because the nuisances are regime-correct. Since $\gamma_j$ is a function of $s$ alone,
$$J_{jk}=\sum_s\pi_s\gamma_j(s)\gamma_k(s)\big[(m_s-m_j)(m_s-m_k)+\sigma_s^2\big],\qquad S_j=\sum_s\pi_s\gamma_j(s)\big[(m_s-m_j)(\mu_s-\mu_j)+\theta_s\sigma_s^2\big].$$

*Symmetric case.* Put $a=1-\alpha$, $b=\alpha$, $\pi_s=\tfrac12$, $\sigma_s=\sigma$. The diagonal entries are $\tfrac12[a^2\sigma^2+b^2(\Delta^2+\sigma^2)]$. The off-diagonal entry is $ab\sigma^2$, because on either regime the two residuals are $\sigma V$ and $\pm\Delta+\sigma V$, whose product has mean $\sigma^2$. A symmetric $2\times2$ matrix with equal diagonal has eigenvectors $(1,\pm1)$ and eigenvalues "diagonal $\pm$ off-diagonal", which gives
$$\lambda_+=\tfrac12\sigma^2+\tfrac12\alpha^2\Delta^2,\qquad \lambda_-=\tfrac12\sigma^2(1-2\alpha)^2+\tfrac12\alpha^2\Delta^2$$
after using $a+b=1$. For the displacement write $S-J\theta^*=b_\gamma$, project $b_\gamma$ on $(1,\pm1)$ and divide by $2\lambda_\pm$. The $(1,1)$ component collects the leaked outcome contrast $\alpha\Delta G$ and the quadratic leakage $-\tfrac12\alpha^2\Delta^2(\theta_0+\theta_1)$; the $(1,-1)$ component collects the effect heterogeneity. The identity is checked symbolically in `src/synthesis/closed_form_factorial.py`.

*Corollary 1.* (a) With $\alpha=\varepsilon_\gamma/2$, $\lambda_-(\varepsilon_\gamma)-\sigma^2/2 = \tfrac12\sigma^2(\varepsilon_\gamma^2-2\varepsilon_\gamma)+\varepsilon_\gamma^2\Delta^2/8$, which is positive exactly when $\varepsilon_\gamma(\sigma^2+\Delta^2/4) > 2\sigma^2$, that is $\varepsilon_\gamma > 8\sigma^2/(\Delta^2+4\sigma^2)$. (b) Dropping the $\mathcal{O}(\alpha^2)$ numerator term and the heterogeneity component, the error is $\approx\sqrt2\,\alpha|\Delta G|/(\sigma^2+\alpha^2\Delta^2)$. Maximizing $\varepsilon_\gamma/(\sigma^2+\varepsilon_\gamma^2\Delta^2/4)$ gives $\varepsilon^\star = 2\sigma/|\Delta|$ and the stated peak. (c) The leading displacement is proportional to $(1,1)$, the $\lambda_+$ eigenvector, so $\|J^{-1}b_\gamma\| = \|b_\gamma\|/\lambda_+ \le \|b_\gamma\|/\lambda_-$. $\square$

## G.4 Proof of Theorem 4(b): pooled residual-state bias

Take homogeneous effects ($\theta_0=\theta_1=\theta$), treatment $T_t=m(X_t)+aS_t+V_t$ and outcome $Y_t=\theta T_t+g(X_t)+bS_t+U_t$. Because the posterior is calibrated given the features $F_t$, $\mathbb{E}[S_t\mid F_t]=\gamma_t$, so the posterior-adjusted nuisances are $\mathbb{E}[T\mid F]=m(X)+a\gamma$ and $\mathbb{E}[Y\mid F]=\theta(m+a\gamma)+g+b\gamma$. The residuals are
$$\tilde T_t = a(S_t-\gamma_t)+V_t,\qquad \tilde Y_t = \theta\tilde T_t + b(S_t-\gamma_t)+U_t .$$
Hence the pooled residual-on-residual estimator converges to
$$\theta+\frac{\mathbb{E}[\tilde T_t\, b(S_t-\gamma_t)]}{\mathbb{E}[\tilde T_t^2]}=\theta+\frac{ab\,\mathbb{E}[(S_t-\gamma_t)^2]}{a^2\mathbb{E}[(S_t-\gamma_t)^2]+\sigma^2}=\theta+\frac{ab\,v}{a^2v+\sigma^2},$$
where calibration gives $\mathbb{E}[(S_t-\gamma_t)^2]=\mathbb{E}[\gamma_t(1-\gamma_t)]=v$, and $V_t$ is independent of $S_t$ with variance $\sigma^2$. The bias is $b\,v\,h(a)$ with $h(a)=a/(a^2v+\sigma^2)$. Setting $h'(a)=0$ gives $a^2 v=\sigma^2$, so the peak is at $|a|=\sigma/\sqrt v$ with value $h=1/(2\sigma\sqrt v)$ and therefore $\sup|B|=|b|\sqrt v/(2\sigma)$. The bias is zero at $a=0$ (statement (a)). Statement (c) follows because $J=\mathbb{E}[ww^\top(a^2v+\sigma^2)]$ grows with $a^2 v$. $\square$

## G.5 Proof sketch of Lemma 1 and Theorem 2

*Lemma 1* is proved in Section 5.1. *Theorem 2* is proved by the construction in Section 3.3: the two binary-outcome models in Table {{tab:nonid}} have identical laws of $(Y,T,X,Z)_{1:N}$ for every $N$, because $P(Y_t=1\mid T_t=0)=0.40$ and $P(Y_t=1\mid T_t=1)=0.60$ in both, while their regime-effect vectors differ.

@tab nonid | The two models of the non-identification construction
| Model | $\theta_0$ | $\theta_1$ | $g_0$ | $g_1$ | $P(Y{=}1\mid T{=}0)$ | $P(Y{=}1\mid T{=}1)$ |
|---|---|---|---|---|---|---|
| $P_1$ | 0.20 | 0.20 | 0.30 | 0.50 | 0.40 | 0.60 |
| $P_2$ | 0.10 | 0.30 | 0.40 | 0.40 | 0.40 | 0.60 |

Both models have population ATE 0.20, which randomized treatment identifies. The result is non-identification of the *regime-specific vector* within the stated memoryless model class; it does not imply that every persistent latent model with uninformative proxies is unidentified.
