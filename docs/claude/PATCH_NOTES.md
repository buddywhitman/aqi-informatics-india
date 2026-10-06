# Handoff notes for the agent: bibliography + theorem patch (main.tex, AISTATS 2027)

Files: `bibliography.tex` (drop-in replacement of the whole `thebibliography` block, tex lines ~473-648),
`theorem_repairs.tex` (labelled replacement blocks R1-R6), `verify_patch_math.py` (numerical checks).
Both .tex files were compile-tested with natbib[round] + amsmath/amsthm + the paper's macros (no errors, no undefined refs, 82 unique keys).
`aistats2027.sty` was not available here, so the full manuscript was not compiled: **recompile and re-check the 8-page main-text limit.**

## A. Bibliography
1. Replace lines from `% \bibliographystyle{plainnat}` through `\end{thebibliography}` with `bibliography.tex`.
2. All existing keys are kept, so no `\cite` breaks, EXCEPT `sankararaman2023selective` (unverifiable) -> use `elyaniv2010foundations` and `chow1970optimum` on tex line 77.
3. Keys whose printed year/authors changed (key name is now a misnomer but harmless; rename with sed if you prefer):
   `huang2023sdci` -> Huang et al. 2020 (JMLR 21(89)); `clouth2024latent` -> Clouth, Bijlsma, Pauws, Vermunt 2025 (Sociol. Methods Res.);
   `kennedy2023towards` -> 2022 arXiv 2203.06469; `hendrickx2021machine` -> 2024; `pearl2012bias` -> UAI 2010; `semenova2023generalized` -> Semenova et al. 2023 (Quant. Econ.).
4. Fixed entries (what was wrong): Clouth (fabricated authors/venue; real paper is the parametric g-formula one), Banna (title/authors were a blend; real = Banna-Merlevede-Youssef, RMTA 2016),
   Bica (no "Lambert", no such title; real = Time Series Deconfounder, ICML 2020), Huang (wrong title/year/volume), D'Amour (wrong title; real = "Overlap in observational studies with high-dimensional covariates", J. Econometrics),
   Semenova (non-existent title; replaced by Semenova-Goldman-Chernozhukov-Taddy, QE 2023), Shalit (wrong title and pages), Kennedy (arXiv ID belonged to another paper), Pearl (UAI 2010 not 2012; title corrected),
   Mastouri (ICML 2021 not NeurIPS; title; first name), Hendrickx (year/pages), Tchetgen-Tchetgen (bibitem label listed the wrong authors; the entry body was right).
5. **Verification status.** Confirmed by search this session: Ciganovic (arXiv ID), Clouth (authors/venue/DOI), Banna et al. (title), Bica (title/venue), Huang (title/venue), D'Amour (title/arXiv), Semenova et al. (title/venue),
   Shalit (title/venue), Kennedy (arXiv ID/title), Hendrickx (title/arXiv), Mastouri (title/venue), Pearl (title/arXiv). Page numbers and volumes marked `% verify:` and everything else are from memory of well-known papers:
   **run each through doi.org / arXiv / the publisher page once** (Crossref was rate-limited from this environment, so no bulk automated check was possible). The ones I'm least sure of:
   Hatt & Feuerriegel (exact title/pages), Mastouri (author list), Vu et al. (author list/order), Cui et al. (pages), Kallus et al. and D'Amour 2019 (PMLR page ranges, omitted), Banna et al. (check the mixing condition: alpha vs beta mixing, before citing it for geometric alpha-mixing at tex line 1021).
6. `hendrickx2021machine` was in the old bib but never cited; cite it in the selective-reliability sentence or drop it.

## B. Where to cite the new entries (length-neutral; mostly extra keys inside existing \citep/\citet)
- Related work (tex line 77): (1) DML: + robinson1988root, chernozhukov2022locally, athey2019generalized. (2) dependent/time-series: + bojinov2019time, robins2000marginal, bonhomme2022discretizing, hamilton1990analysis, kim1999state.
  (3) latent/proxy: + bica2020time, hatt2021sequential, wang2019blessings, ogburn2019comment, damour2019multicause, louizos2017causal, kuroki2014measurement, cui2024semiparametric (damour2019multicause and kallus2019interval are AISTATS papers; reviewers like that).
  (4) overlap/selective: + petersen2012diagnosing, khan2010irregular, chow1970optimum, elyaniv2010foundations, bartlett2008classification, cortes2016learning, geifman2019selectivenet, hendrickx2021machine.
- Theorem 1 / OVB (lines ~183, 198): + frisch1933partial, lovell1963seasonal, cinelli2020making, kuroki2014measurement. Consider retitling Theorem 1 "Proposition" (it is the OVB formula).
- Purged cross-fitting / embargo (Sec. 4.3): + lopezdeprado2018advances (origin of purged K-fold + embargo), racine2000consistent, burman1994cross, bergmeir2018note; HAC: + newey1987simple, andrews1991heteroskedasticity.
- HMM/EM + multi-restart (Sec. 4.1, App. E.2): + dempster1977maximum, wu1983convergence, baum1966statistical, rabiner1989tutorial, gassiat2016inference, leroux1992maximum.
- App. C line ~1021: + tropp2012user; ridge: hoerl1970ridge; Weyl: horn2012matrix. App. E proof: ibragimov1962some (cited in the proof text but was missing from the bib), bradley2005basic, bickel1998asymptotic, douc2004asymptotic.
- Empirical section: + grange2019using, vu2019assessing (meteorological normalisation / ML for air-quality interventions).

## C. Theorem repairs (see `theorem_repairs.tex`; each block states what it replaces)
| Block | Problem found | Repair |
|---|---|---|
| R1 | Thm 3 hypotheses `L_g`, `L_m` are never used; the proof uses an unstated envelope `M_Y` and pointwise positivity `c_0` | New Assumption (envelopes + positivity) |
| R2 Thm 3 | `+O_P(N^-1/2)` hides the `1/lambda_min` amplification of sampling noise; Gram matrix is not orthogonal so its error is first order in nuisance error | Explicit constants; finite-sample bound `(2/lambda_min)(...)`, rate `O_P(N^-1/2 + r_N)/lambda_min` |
| R3 Lemma A | "Score is Neyman-orthogonal" is proved only for the cross-moment; the coupled Gram part is non-orthogonal for soft gamma (numerically verified: derivative 0.02 vs ~1e-7 for hard) | Lemma with exact structure: orthogonal for hard gamma, `O(sqrt(eps))` otherwise |
| R4 Thm 4 | Cross term used a bias rate "by Theorem 5" (Thm 5 gives none); "sharpening to N^-3/2" underived; 2nd-moment expansion needs UI; adaptive-lambda cap `N^-1/2` contradicts lambda=0.10-0.20 | Minkowski (no cross term): `E||.||^2 <= (B+sqrt V)^2 <= 2B^2+2V`; UI assumption; oracle inequality lemma; fixed `lambda_bar` |
| R5 Thm 5 | CLT applied to a non-centred score (mean `b_gamma`); "iff" unproved (only "if"); omits HMM-parameter variance (contradicts App. E.2); needs `C>1/c_alpha`, `lambda_min+lambda>=c`, Gram-nuisance rate | Centre at `theta_lambda`; "if"; stacked variance with `kappa A I^-1 A'`; Assumption (H),(G),(M) |
| R6 Prop. flip | (addition) closed-form bias in the symmetric flip construction used in the factorial experiment | `||theta_gamma - theta*|| = eps/(1-eps) |theta_1-theta_2|/sqrt2`, `lambda_min = sigma^2(1-eps)^2/2`; shows `eps/lambda_min` alone is scale-dependent |

Main-text edits needed (length-neutral): (i) Thm 3/4/5 statements per R2/R4/R5; (ii) the sentence "the score is Neyman-orthogonal to (mu_k, m_k)..." -> Lemma A wording;
(iii) "holds iff" -> "holds if"; (iv) wherever Theorem 2 is invoked for Delta_Z <= 1 (Sec. 6 findings, App. B/E text "By Theorem 2, point identification fails"):
say Theorem 2 covers only the exact limit Delta_Z=0 with p=0.5/static, and that Delta_Z<=1 failures are finite-sample ill-conditioning (Theorem 3); (v) code: use tau* = max(Bartlett tau, ceil(C log N)), C > 1/c_alpha, not hard-coded 12/24 rows.

## D. Not repaired (needs your decision / new work, not a rewrite)
- A matching minimax **lower bound** for `eps/lambda_min` (two-point Le Cam) is the biggest rigor upgrade; R6 is an exact-example substitute, not a lower bound.
- Thm 5 is pointwise (fixed `lambda_min`); a local-to-singular analysis (`lambda_min ~ N^-1/2`) is open and is exactly the Mumbai regime.
- Prop. 1 assumes Bayesian calibration of the *estimated* filter; not implied by Thm 5(H).
- Everything numeric (Table 1 single-init, Table 6 vs CSV, Mumbai seeds, Table 10 generators) is from my previous message.

## E. Numerical checks (`verify_patch_math.py`)
1. Flip model: closed form matches to ~1e-11 for eps in {0,.1,.3,.6,.9}; `lambda_min` formula exact.
2. Lemma A: finite-difference derivative of the coupled score wrt `m`: soft gamma 2.0e-2, hard gamma 1.0e-7; wrt `mu`: ~0 in both.
