We thank the Program Chairs for the AI review. We checked each item against the source and the data files. Below, "confirmed" means we reproduced the issue, and "to do" means we have not yet re-derived it; we do not claim to have fixed those.

**Confirmed and corrected**
- *Missing sign (App. H, proof of Theorem 5, Step 1).* The held-out score expansion should read $\hat\Psi-\Psi=\gamma\{-\delta_m q-r\delta_\mu+2hr\delta_m+\delta_m\delta_\mu-h\delta_m^2\}$. The "+" before $\delta_m\delta_\mu$ was dropped in typesetting. This does not change the argument: the first three terms are mean zero, and the last two are $O_P(r_N^2)$. Corrected in our source.
- *ROC-AUC for the HMM (0.615 vs. 0.5895).* They come from two different result files (the reliability-AUC file and the calibration-intervention file), produced by different runs. We will report one source and say which run it comes from.
- *Table 3 vs. Table 20.* Mumbai agrees. For screened Kolkata the regime estimates differ ($0.191,\,6.916$ vs. $-0.027,\,7.084$) and for Bengaluru slightly ($0.185,\,-0.156$ vs. $0.182,\,-0.158$). We have not yet traced the screening specification behind the difference; we will report one specification.
- *Table 10 was not referenced in the text.* It is now referenced, and the comparison with the joint MLE and proximal 2SLS has moved into the main text together with a new deconvolution estimator.

**A clarification, not an error: $C_K$.** For $K=2$, $\sup_p 4p(1-p)/H(p)=1/\ln 2$ (nats), so "$C_K\le 1/\ln 2$" is valid and sharp for two regimes, and $C_K=2$ is the general-$K$ constant. We will state the $K$ dependence explicitly. We agree the interior-minimizer condition ($C\varepsilon_\gamma<\lambda_{\min}(J)M_\theta$) should be stated where the minimizer is introduced.

**Accepted, to do**
- *Cross-fitting under $\alpha$-mixing (App. G) and the $N\varepsilon_\tau$ scaling of unnormalized sums.* We agree the proof should import the $\beta$-mixing block-coupling conditions of Assumption 8, or restrict the learner class. We are re-deriving this and will state any change to the statement.
- *Theorem 3 phrasing.* The shared nuisance approximates the realized-state oracle nuisance, and this requires $(T,Y)\perp Z_{1:N}\mid(X,S)$, not only the weaker conditioning we wrote. We will state this. We will also check the $\sqrt K$ factor in the coordinatewise bound.
- *Appendix J and Eq. (41).* The complexity statement and Gram expression describe the earlier posterior-weighted design; we will update them to the posterior-adjusted design.
- *Notation.* We will harmonize regime indices, $v_t$ vs. $\bar v$, $D_t$, and $B_\lambda$ (matrix vs. scalar bound).
- *Interpreting $b^\star$.* Agreed: with heterogeneous effects it bounds the composite shift $b+a(\theta_1-\bar\theta)+\bar m(\theta_1-\theta_0)$, and overlap between controls and proxies ($\delta_X>0$) affects its denominator. We now report budget-based intervals in terms of this composite shift.
- *References.* We will add Ogburn and VanderWeele (2012), Kuroki and Pearl (2014), and Murphy and Topel (1985).

**Scope of the new material.** Following the human reviews, the revision also adds an identification result under an exclusion restriction, a deconvolution estimator, and joint-likelihood baselines; none of the items above depends on them.
