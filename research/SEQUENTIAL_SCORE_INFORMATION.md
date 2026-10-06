# Sequential Score Information

The relevant temporal dependence for causal precision is dependence of the score/influence process, not persistence of the latent state by itself.

In the stylized product score psi_t=T_t U_t with independent AR(1) processes T and U, lag-h score correlation is approximately (rho_T rho_U)^h. Thus long-run variance inflation is
    (1+rho_T rho_U)/(1-rho_T rho_U).
If either residual process is serially uncorrelated, first-order score autocorrelation vanishes even if the other process is highly persistent.

This formalizes the earlier caveat: regime persistence is not itself an effective-sample-size penalty. Temporal diagnostics should estimate score/influence autocorrelation or long-run variance directly.

Reproduction: research/sequential_score_dependence.py
