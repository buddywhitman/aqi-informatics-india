# Exploratory Research: Task-Relative Reliability

This directory is deliberately **outside the AISTATS submission evidence chain**. Nothing here modifies `paper/`, the canonical supplementary archive, or submitted claims. The purpose is adversarial follow-up research: test where the current scalar difficulty idea generalizes, where it is loose, and which extensions are worth developing.

## Studies and current findings

### R1. Finite-sample scaling of the scalar difficulty
We independently vary proxy error, residual treatment information, and sample size. In the exploratory run (10 replications/cell), Spearman correlation between causal error and `eps/lambda_min(J)` increases from 0.783 at N=300 to 0.915 at N=4800, and exceeds either constituent at every N. A log-log regression gives an exploratory slope 0.74 on log difficulty and only -0.081 on log N (R2=0.858). The simple `sqrt(N)*difficulty` scaling does **not** improve rank correlation. This is evidence for the difficulty ordering, but not evidence for a universal root-N phase boundary.

### R2. Direction matters: the scalar bound can be very loose
At the population score level, fix `||b||`, fix the spectrum of J, and rotate the perturbation b relative to the weakest eigenvector. The exact propagation is `||J^{-1} b||`, while `||b||/lambda_min(J)` is the worst-case scalar bound. With condition number 100, the scalar bound is tight when b aligns with the weak direction but can be 100x loose when b is orthogonal. This strongly motivates a sharper **directional task-relative representation error**
`R_task = ||(J+lambda I)^{-1} b_gamma||`.
This is a mathematical stress test, not an independent empirical discovery.

### R3. K>=3 exposes a major limitation of the scalar difficulty ratio
A three-state experiment holds average posterior L1 error and $\lambda_{\min}(J)$ in roughly the same range while changing **which states receive the posterior perturbation**. At perturbation level $a=0.03$, the three designs have $\varepsilon_\gamma\approx0.020$ and $\lambda_{\min}(J)\approx0.0125$--$0.0133$, hence scalar difficulty about 1.5--1.6, but mean causal L2 error ranges from 0.109 (weak-state perturbation) and 0.127 (strong-state perturbation) to **0.667** (balanced cyclic perturbation). Across all 1,500 runs, Spearman correlation of error with $\varepsilon_\gamma/\lambda_{\min}$ is only 0.424. The score-aware worst-case bound $\|b_\gamma\|/\lambda_{\min}$ improves this to 0.712, while the directional propagation $\|J^{-1}b_\gamma\|$ exactly tracks the unregularized population moment error by construction.

This is the most important new finding: the current scalar ratio is a useful worst-case/order diagnostic in the two-state experiments, but it is **not a sufficient task-relative representation metric in higher-dimensional latent spaces**. Error orientation relative to the full spectrum matters.

### R4. Regularization helps, but scalar difficulty does not identify the optimal lambda
Across 48 synthetic settings (20 replications each), oracle choice over a fixed ridge grid reduces MSE substantially: median gains are 81.8%, 87.3%, and 85.5% for N=600,1200,2400. However, Spearman correlation between oracle lambda and scalar difficulty is only 0.393. Thus the current difficulty score is useful for detecting danger but appears insufficient by itself for tuning regularization. This is an important negative result and argues for a richer risk estimator involving perturbation direction, noise level, and target magnitude.

### R5. Posterior error is transition-localized
For a correctly specified two-state Gaussian HMM using causal filtering, posterior L1 error is concentrated near true state transitions. At emission separation 2.0, mean L1 error is 0.577 within 0-1 steps of a transition versus 0.134 at distance >=9; at separation 3.0 it is 0.270 versus 0.056. This suggests that aggregate posterior metrics may hide a small set of temporally localized, high-impact representation failures.

### R6. The same transition phenomenon appears qualitatively in the four-city sensor data
Using a standardized two-state HMM on the six exogenous meteorological features, filter-vs-smoother disagreement is sharply concentrated around inferred transitions. In Delhi, mean L1 disagreement is 0.923 within 0-1 hours of an inferred transition versus 0.009 at >=25 hours. Mumbai shows 0.413 versus 0.017. Bengaluru/Kolkata switch much more frequently, so long interior segments are scarce. Because inferred transitions are defined by the same fitted model, this is a descriptive diagnostic, not ground-truth validation.

### R7. Frozen-sensor sensitivity is a material empirical limitation, especially in Mumbai
Removing complete-analysis rows belonging to NO2 constant-value runs of length >=6 hours leaves Delhi unchanged but removes 885/4,180 Mumbai rows (21.2%), 163/4,093 Bengaluru rows (4.0%), and 110/1,209 Kolkata rows (9.1%). Re-fitting the same smoothed OR-DML specification changes Mumbai's overall estimate from 1.59 to 2.92, regime estimates from (-0.03, 3.30) to (3.86, 2.25), and lambda_min from 0.317 to 0.178. Bengaluru changes little; Kolkata remains extremely ill-conditioned and uncertain. This is not evidence that the filtered estimate is "truer"—removing long constant runs may discard legitimate periods—but it shows that the Mumbai empirical decomposition is materially telemetry-sensitive and should not be treated as a stable physical effect without source-level sensor QA.

### R8. Real-data conclusions are sensitive to the chosen number of latent regimes
Re-fitting the identical observational specification with K=2,3,4 leaves Delhi's overall ATE relatively stable (0.374, 0.428, 0.422) and Bengaluru near zero (-0.026, 0.064, 0.038), but Mumbai changes materially (1.59, 2.83, 5.99). At K=4 Mumbai's lambda_min falls to 0.029 and one regime coefficient reaches 23.0, a clear warning of weak identification rather than credible fine-grained heterogeneity. Kolkata remains near-singular for every K. The fitted HMM BIC decreases monotonically from K=2 through K=5 in all four cities, so K=2 is an interpretability/coarse-graining choice rather than a BIC-selected latent-state count. This is a substantive model-selection limitation and a promising research direction: task-aware state aggregation should trade emission fit against downstream identifiability.

A useful positive control: five different HMM random seeds (1,7,19,42,97) produced identical reported fits for all four cities under the current implementation, so the observed K sensitivity is not merely random initialization noise.

### R9. Emission-optimal latent granularity can be causally suboptimal
A deliberately simple four-microstate experiment separates **representation fit** from **downstream causal utility**. Four observational microstates have well-separated Gaussian emissions (means -3,-1,1,3), but pairs share the same causal effect, yielding two causal macro-regimes. Two microstates also have weak residual treatment variation. Gaussian-mixture BIC prefers K=4 over K=2 in 100% of 100 replications at N=400,800,1600 (mean BIC advantage 283, 587, 1210 respectively). Yet estimating four microstate slopes and then aggregating is far less accurate than causally appropriate two-state pooling: mean macro-effect L2 error is 0.642 vs 0.129 at N=400, 0.438 vs 0.089 at N=800, and 0.321 vs 0.061 at N=1600; the merged representation wins in 96-97% of replications.

This suggests a potentially broad new principle: **latent-state complexity should be selected jointly with downstream identifiability, not solely by emission likelihood/state-prediction fit.** The real-data BIC/K sensitivity in R8 points in the same direction, but does not establish ground truth.

### R10. A simple data-driven task-aware coarsener nearly recovers the oracle macro-representation
The coarsening result is not limited to knowing the true grouping. In a separate experiment, a K=4 Gaussian mixture first recovers the observational microstates (mean ARI about 0.99). State-specific residual slopes and standard errors are then estimated, and the four states are paired into two macro-states by minimizing within-pair Wald discrepancy. This uses downstream effect similarity, not the true macro labels.

At N=400/800/1600, the method identifies the true causal pairing in 89.0%/94.8%/98.8% of 500 replications. Mean causal error is 0.127/0.085/0.063, essentially matching the oracle macro-state estimator (0.128/0.085/0.063), while the fully refined K=4 estimator has error 0.597/0.459/0.327. Task-aware coarsening beats the refined estimator in 95.0%/97.6%/95.6% of replications.

This is only a proof-of-concept algorithm: using the same data to estimate slopes and choose merges can induce selection bias, and a publishable method should cross-fit the merge decision. But it demonstrates that the representation paradox is actionable rather than merely diagnostic.

### R11. State-refinement paradox under local weak overlap: finer can become inconsistent while coarser stays root-N
This is the strongest theoretical/simulation finding so far. Consider four **perfectly observed** and observationally distinguishable microstates. States 0/1 share causal effect $\theta_A$ and states 2/3 share $\theta_B$, but one microstate in each pair has residual treatment standard deviation $\sigma_N=N^{-\alpha}$. If all four microstate slopes are estimated separately and then occupancy-averaged, the weak-state slope has standard deviation of order
$(N\sigma_N^2)^{-1/2}=N^{\alpha-1/2}.$
Thus:
- $\alpha<1/2$: the refined estimator converges, but slower than root-N;
- $\alpha=1/2$: its error remains $O_p(1)$;
- $\alpha>1/2$: its error **diverges with more data**.
By contrast, pooling causally equivalent microstates preserves $O(N)$ treatment information from the strong member of each pair, so the coarsened estimator remains root-N consistent.

The simulation is stark. For $\alpha=0.75$, mean refined error increases from 5.58 at N=400 to 10.64 at N=6400, while task-aware coarsened error falls from 0.126 to 0.0327. At $\alpha=1$, refined error rises 24.9 -> 95.2 while coarsened error again falls 0.126 -> 0.0327. This occurs with **perfect microstate labels**, so it is not a representation-recovery failure. It is a downstream information failure induced by over-refining the representation.

Combined with R9, this yields a genuine representation paradox: a generative model can consistently prefer and perfectly recover a finer latent state space while that finer representation is statistically worse—and under local weak overlap can become asymptotically unusable—for the downstream causal functional. A coarser task-sufficient representation can be dramatically better.

This appears to motivate a new object: **task-sufficient latent abstraction**, where states are distinguished only to the extent that doing so changes the downstream functional enough to justify the information cost.

### R12. Task abstraction has a sharp bias-variance frontier; coarsening is not universally beneficial
To prevent the coarsening story from becoming one-sided, a held-out experiment introduces genuine within-pair causal heterogeneity $\delta$ and evaluates pooled coefficients against all four true microstate effects. With weak-state information varied independently, task-aware coarsening reduces total microstate-effect error by 65.1% at $\delta=0$, 58.7% at $\delta=0.1$, 38.3% at $\delta=0.25$, and only 9.6% at $\delta=0.5$. At $\delta=1$, coarsening is worse by 55.8%; at $\delta=2$, worse by 174.7%.

This is the necessary counterweight to R9-R11: the optimal abstraction is neither maximally fine nor maximally coarse. It lies on a **task-specific bias-information frontier**. Merging is beneficial only while the information gained by pooling outweighs the task heterogeneity erased by the merge. Any principled method therefore needs an explicit approximation-bias term, not merely a conditioning penalty.

### R13. There is no universally optimal abstraction: the same latent process needs different state partitions for different downstream tasks
A clean multi-task experiment uses the **same four latent microstates, same observations, and same fitted K=4 representation**, but defines two downstream causal functionals. Task A has effects $(0.5,0.5,2.5,2.5)$, so its task-sufficient partition is $(0,1)|(2,3)$. Task B has effects $(0.5,2.5,0.5,2.5)$, so its task-sufficient partition is $(0,2)|(1,3)$. A sample-split Wald coarsener recovers the appropriate, mutually incompatible partition for each task in 100% of 100 replications at N=800,1600,3200, with held-out effect-vector error decreasing approximately root-N.

This provides a direct experimental demonstration of the broad principle: **representation optimality is a relation between a representation and a downstream functional, not an intrinsic property of the representation alone.** The exact same high-fidelity latent representation should be abstracted differently depending on the query.

### R14. A train-only estimated risk can adaptively choose whether to coarsen, but the hard heterogeneity boundary remains difficult
A proof-of-concept selector evaluates the fine K=4 partition and all three K=2 pairings using only the training split, with estimated risk = within-group effect heterogeneity + pooled estimation variance. The chosen partition is frozen before held-out effect estimation. For small/moderate heterogeneity it is close to the oracle partition selector: at delta=0 and N=1600, mean held-out error is 0.089 versus oracle 0.085 and refined 0.681; at delta=0.25, 0.247 versus oracle 0.215 and refined 0.675. As heterogeneity becomes large, selection becomes harder and the fine representation catches up: at delta=1-2 the selector no longer reliably beats refinement. This is desirable behavior conceptually but the current criterion is not yet minimax or theoretically calibrated.

The result strengthens the feasibility of **data-adaptive task abstraction** while also showing that partition selection near the bias-information crossover is itself a nontrivial statistical problem.

### R15. Dual-resolution trilemma: fine adjustment and coarse target are different operations
The earlier wording "coarsen the latent representation" was too broad. A fine microstate may remain necessary to remove confounding even when estimating a separate effect for every microstate is statistically harmful. The exact construction in `DUAL_RESOLUTION_PRINCIPLE.md` shows three regimes. With fine-state nuisance adjustment + fine-state target parameters, local weak overlap induces the $N^{\alpha-1/2}$ rate penalty. With coarse nuisance adjustment + coarse target, unresolved microstate treatment/outcome baselines create persistent omitted-microstate bias; in the committed construction the asymptotic slope bias is exactly $4/3$. With **fine adjustment + coarse target**, confounding is removed at full resolution while causally equivalent effect parameters pool residual information and remain root-N.

This is an important correction: the strongest principle is to decouple **identification resolution** from **estimand resolution**, not to indiscriminately discard fine latent states.

### R16. Multitask abstraction incompatibility: per-task compression can be exponentially smaller than any universal sufficient abstraction
The two-task experiment already has crossing optimal partitions: Task A needs $(0,1)|(2,3)$ while Task B needs $(0,2)|(1,3)$. Neither partition is a refinement of the other, so no single two-state hierarchy can represent both as cuts. Generalizing, let the fine state be an $m$-bit vector and let task $j$ depend only on bit $j$. Every task individually has a 2-state sufficient abstraction, but any single deterministic abstraction sufficient for all $m$ tasks must recover every bit and therefore needs all $2^m$ states.

This exact construction implies that a compact universal task-sufficient abstraction need not exist. The natural architecture is a rich shared latent substrate with **task-specific quotient heads**, not a single globally coarsened state space. The set-theoretic result is elementary; its potential significance here is the interaction with weak-overlap causal estimation and the statistical cost of fine target parameterization.

### R17. Information-theoretic causal resolution limit: a perfectly observed state can have an asymptotically undetectable causal effect
The local weak-overlap sequence yields a stronger result than estimator divergence. In a perfectly observed microstate with residual treatment SD $\sigma_N=N^{-\alpha}$, compare two effect models separated by $\delta_N=N^{-\beta}$. Their expected KL divergence is
$E\,KL = C N^{1-2\alpha-2\beta}.$
Hence the exact detection boundary is $\alpha+\beta=1/2$. Below it, causal heterogeneity is asymptotically detectable; on it, testing remains local/nontrivial; above it, KL and total variation vanish, so **no test can reliably distinguish the two causal-effect models**. Most strikingly, for $\alpha>1/2$, even a constant nonzero effect difference ($\beta=0$) becomes asymptotically invisible despite perfect state labels.

At N=25,600 with $\alpha=.6,\beta=0$, the analytic noncentrality is only 0.256 and two-sided 5% Wald power is about 5.76%; at $\alpha=.75$, power is 5.04%, essentially test size. Meanwhile the latent state itself is assumed perfectly observed. This creates a sharp mismatch between **observational state resolution** and **causal resolution**: the data can tell us exactly which state we are in while containing asymptotically zero information about whether that state's causal effect differs.

This is the strongest impossibility result found on the branch so far. It implies that beyond the causal-resolution boundary, demanding a distinct effect parameter for every real latent microstate is not merely inefficient; the distinction is statistically unlearnable from the observational experiment.

### R18. Crossing task abstractions imply an exponential universal-compression gap
The two-task experiment's optimal partitions cross. Generalizing to an $m$-bit latent state and $m$ tasks where task $j$ depends only on bit $j$, each task individually needs only 2 abstract states, while any single deterministic abstraction sufficient for all tasks must preserve all $2^m$ fine states. This exact construction does not by itself establish novelty, but combined with R17 it implies a severe statistical tension: a universal fine target parameterization may preserve every possible task distinction while forcing estimation of distinctions that individual tasks neither need nor can identify under weak overlap. Task-specific target heads avoid that unnecessary resolution cost.

### R19. Real data show a consistent generative-resolution / causal-information divergence as K increases
Combining the previously computed HMM BIC and OR-DML K-sensitivity gives a striking descriptive pattern in all four cities. Moving from K=2 to K=4 improves HMM BIC by 8,401 (Delhi), 8,900 (Mumbai), 10,138 (Bengaluru), and 2,103 (Kolkata), while the smallest downstream score eigenvalue simultaneously collapses by factors of 4.8x, 10.8x, 4.5x, and 3.1x, respectively. Mumbai is the clearest case: BIC strongly favors greater latent resolution while $\lambda_{\min}(J)$ falls from 0.317 to 0.029 and the overall effect estimate shifts from 1.59 to 5.99.

This does **not** prove that K=4 is causally wrong or that K=2 is correct—the true state count and effects are unknown. But it is exactly the empirical signature predicted by the synthetic resolution-mismatch story: additional latent distinctions can improve the observation model while fragmenting downstream treatment information. It motivates reporting a two-axis model-selection diagnostic: generative fit versus causal effective information, rather than selecting K from likelihood alone.

### R20. Resolution thresholding exposes an estimand-versus-identification distinction: unresolved does not mean zero
A spectral projection experiment suppresses directions whose nominal resolution $1/\sqrt{N\lambda_j}$ exceeds a chosen scientific scale. When the true target has no component in the suppressed direction, this can remove enormous variance: at N=400, alpha=.5, the unregularized full-vector MSE is about 2.55 while the projected estimator has MSE 0.048. But if the same weak direction carries a true coefficient 0.8, projection has total MSE 0.688, of which 0.640 is pure approximation error from changing the target. Oracle ridge is better (0.298) but still cannot create information that is absent.

This is an important negative result: a causal-resolution spectrum tells us what the data can resolve, **not what the unresolved effect equals**. Hard spectral truncation is legitimate only if the estimand is explicitly redefined/projected or external structure justifies the restriction. Otherwise unsupported contrasts should be reported as weakly identified/partially learned, not silently set to zero.

The result sharpens the framework into three distinct questions:
1. What fine latent information is required for identification?
2. Which causal contrasts are resolvable from the data?
3. Which unresolved contrasts may be constrained/pooled based on scientifically defensible structure?

### R21. Resolution divergence: more data can strengthen evidence that states are different while erasing evidence that their causal effects differ
A joint local-sequence construction fixes Gaussian emission separation while shrinking within-state residual treatment SD as $N^{-\alpha}$. The generative/state evidence accumulates at order $N$ (BIC penalty only $O(\log N)$), while information for a fixed causal-effect contrast is order $N^{1-2\alpha}$. For $\alpha>1/2$, these move in opposite directions: state-model evidence diverges while causal-effect KL goes to zero.

An exploratory 50-replication simulation makes the contrast concrete. At $\alpha=.75$, mean BIC advantage for the two-state observation model grows from 32.6 at N=400 to 764.0 at N=6400, with state classification error about 6.7% throughout, while causal-contrast test power remains around nominal size (2-8%). At $\alpha=.25$, by contrast, causal power rises from 56% to 100% over the same N range. Thus more data can make us increasingly certain that two latent states are genuinely distinct while providing asymptotically less ability to learn whether their causal effects differ.

This is a stronger statement than generative/causal model-selection disagreement at fixed N: **observational resolution and causal resolution can diverge in opposite asymptotic directions on the same sequence of data-generating processes.**

### R22. Honest inference can get wider with more data when causal information per observation collapses
For a weak causal direction with information eigenvalue $\lambda_N=N^{-2\alpha}$, the honest 95% interval width scales as $N^{\alpha-1/2}$. At $\alpha=.75$, its expected Gaussian width increases from about 17.5 at N=400 to 49.6 at N=25,600. A naive root-N interval simultaneously shrinks from 0.196 to 0.0245, but its approximate coverage collapses from 1.75% to 0.077%. This is a classical weak-identification phenomenon in spirit, not a novelty claim; here it is a diagnostic consequence of latent-state causal resolution. The lesson is operational: **apparent precision can increase exactly while actual causal information decreases.**

### R23. Granularity-induced weak identification: causal resolution can collapse even with healthy overlap inside every state
Let the learned/fine state count grow as $K_N=N^\kappa$, with equal occupancy and residual treatment variance bounded away from zero in every state. Each state then receives only $N/K_N=N^{1-\kappa}$ observations, so a state-specific effect has standard error $O(N^{(\kappa-1)/2})$. At $\kappa=1$, fine-state causal error no longer vanishes despite perfect state labels and healthy within-state treatment variation; a pooled task effect still uses $O(N)$ information and remains root-N.

Combining growing granularity $K_N=N^\kappa$, treatment SD $N^{-\alpha}$, and local effect separation $N^{-\beta}$ yields the unified per-state causal-information law
$KL_{\mathrm{state}}\asymp N^{1-\kappa-2\alpha-2\beta}.$
Thus the causal-resolution boundary is
$\boxed{\kappa+2\alpha+2\beta=1.}$
This gives a simple **causal resolution budget**: representation granularity, overlap deterioration, and increasingly subtle heterogeneity all consume the same information exponent. The earlier $\alpha+\beta=1/2$ boundary is the special case $\kappa=0$.

This is especially relevant to learned latent representations: model capacity/data can increase the number of statistically real states even when every state has good local overlap, while the downstream demand for one effect per state outruns the available sample information.

### R24. Sequential resolution budget: persistence matters only through dependence of the causal score
The growing-state/weak-overlap boundary extends to dependent sequences when the **relevant score/influence process** loses effective sample size. If score dependence has long-run variance inflation of order $N^\eta$, then effective information is $N^{1-\eta}$ and the unified boundary becomes
$\boxed{\eta+\kappa+2\alpha+2\beta=1.}$
Here $\eta$ is a dependence tax, $\kappa$ a granularity tax, $2\alpha$ an overlap/treatment-information tax, and $2\beta$ an effect-resolution tax.

A crucial negative clarification emerged while designing the simulation: **persistent latent states alone do not automatically impose the $\eta$ tax.** If treatment residuals are serially independent and mean-zero, multiplying them by a persistent outcome error can destroy score autocovariance. The dependence term belongs to the actual orthogonal score/influence sequence, not to state persistence as a descriptive property. The committed simulation therefore correlates both treatment residuals and outcome innovations so that the score itself has persistent covariance. This prevents overclaiming that Markov persistence by itself reduces causal effective sample size.

The budget is thus a diagnostic decomposition of *causal score information*, not a formula obtained from HMM transition persistence alone.

### R25. Phantom causal resolution: representation error can make weak causal geometry look healthy
A new K=3 experiment reveals a failure mode of proxy-weighted conditioning diagnostics. True residual treatment SDs are $(1,.3,.08)$, so the oracle causal information has condition number about 153-158. When state emissions overlap, inferred states mix high-information observations into the weak state. At N=1200 and emission separation 0.5, the true $\lambda_{\min}$ is about 0.00215, but soft posteriors report 0.0246 (**11.4x inflation**) and hard assignments 0.0457 (**21.2x inflation**). The apparent condition number collapses from 153 to 5.9/4.0. Even at separation 3, soft/hard $\lambda_{\min}$ remain inflated by about 1.54x/1.90x.

Thus representation uncertainty need not merely reduce information; it can **manufacture apparent overlap / causal resolution**. A proxy-weighted $\lambda_{\min}$ is not automatically conservative for oracle conditioning.

### R26. Correcting phantom resolution creates a second inverse problem
With symmetric hard misclassification rate $p$, observed state-information moments equal a confusion-matrix mixture of oracle moments. Known-confusion deconvolution uses $C^{-1}$, whose operator norm is $1/(1-2p)$. As classification approaches chance, correction becomes singular. In a two-state experiment with a 100x weak-information state, naive inferred $\lambda_{\min}$ is inflated about 49.4x at $p=.49$. Deconfusion removes the mixing bias in expectation but is extremely noisy: even at N=20,000, the corrected weak-information moment is negative in about 47% of runs and has RMSE 0.215 around a true moment near 0.005.

This yields a **double ill-posedness** picture:
$\text{proxy moments}\xrightarrow{\;C^{-1}\;}\text{latent-state geometry}\xrightarrow{\;J^{-1}\;}\text{causal effects}.$
First-order errors can be amplified by both the representation-channel inverse and the causal inverse. This provides a sharper mechanism for the original representation-by-task interaction and suggests joint, rather than stagewise, regularization.

### R27. Self-canceling reliability diagnostics: proxy conditioning can erase the uncertainty warning it is meant to complement
In a 3,500-run K=3 experiment, the mechanistic oracle difficulty is $\varepsilon_\gamma/\lambda_{\min}(J_{true})$. Entropy alone has Spearman 0.329 and top-20%-failure AUC 0.640; posterior L1 error has AUC 0.671. But the seemingly richer operational score $H/\lambda_{\min}(J_{proxy})$ collapses to Spearman **0.093** and AUC **0.470**, slightly worse than chance. Replacing the contaminated denominator with oracle $\lambda_{\min}(J_{true})$ yields AUC **0.994**.

The mechanism is self-cancellation: worse state separation raises entropy but also inflates proxy $\lambda_{\min}$ through state mixing, so dividing by the proxy geometry suppresses the warning. Eigenvalue inflation itself has AUC 0.944 for the hardest oracle-difficulty cases. This offers a concrete explanation for why coupled entropy/conditioning scores can underperform entropy alone even when the underlying representation-by-task principle is correct.

### R28. Phantom resolution becomes more statistically convincing with more data
At fixed poor state separation, the oracle $\lambda_{\min}$ stays near 0.00215 while soft-proxy $\lambda_{\min}$ converges near 0.0242, roughly 11x too large. Across-run proxy SD shrinks from 0.0130 at N=300 to 0.00360 at N=4800, so the oracle-proxy discrepancy grows from 1.74 to **6.11 proxy standard deviations**. Thus ordinary resampling/stability checks can become increasingly confident in the wrong optimistic geometry: they estimate sampling uncertainty around a proxy pseudo-parameter, not error relative to the unobserved oracle geometry.

### R29. Posterior second-moment completion can recover oracle causal geometry without confusion inversion in a correctly specified model
The phantom-resolution analysis revealed that posterior means are the wrong object for latent-state second moments. For one-hot state $H$ and information set $\mathcal I$ with posterior $q=P(S|\mathcal I)$,
$E[HH^\top|\mathcal I]=\operatorname{diag}(q),$
not $qq^\top$. Therefore for any weight $W$ measurable in $\mathcal I$,
$E[W HH^\top]=E[W\operatorname{diag}(q)].$

In a K=3 model with state-specific treatment variances, a posterior using emissions alone, $P(S|Z)$, is insufficient for a $T^2$ information moment because $T$ itself contains state information; diagonal completion using that posterior inflates weak information by 5x-47x. But updating the diagnostic posterior to $q=P(S|Z,T)$ and using $E[T^2\operatorname{diag}(q)]$ recovers oracle $\lambda_{\min}$ essentially exactly. Across N=600-4800 and emission separations .5-2, the completed/oracle ratio stays about 0.993-1.005.

This gives a constructive alternative to unstable confusion-matrix inversion: estimate **posterior latent second moments** under an information set containing the variables in the diagnostic moment. It also exposes two distinct proxy-geometry errors: posterior-mean substitution and an insufficient posterior information set. Using treatment-updated state posteriors inside the causal estimator itself is not automatically valid; the current result is a diagnostic moment identity and needs causal-theory work before estimator use.

### R30. Sequential/Markov dependence does not break posterior second-moment completion under a correct filtered joint model
The completion identity extends directly to causal filtering: with $q_t=P(S_t|\mathcal I_t)$, $E[H_tH_t^\top|\mathcal I_t]=\mathrm{diag}(q_t)$. A committed Markov experiment varies persistence, emission separation, and sample size while comparing oracle treatment-information geometry against filtered $P(S_t|Z_{1:t},T_{1:t})$ completion. This isolates temporal dependence from parameter-estimation error; under known correct dynamics, the identity remains valid. The real difficulty is therefore not Markov dependence per se but whether the diagnostic posterior conditions on the right information and is calibrated.

### R31. Moment completion is highly model-sensitive: correcting phantom resolution can recreate it under treatment-model misspecification
The exact identity does **not** make the method automatically robust. True treatment SDs are $(1,.3,.08)$. At N=2400 and emission separation 1, correct joint completion gives completed/oracle weak information about 1.00. If the diagnostic model incorrectly uses a common treatment variance across states, the ratio is about **26.2x**; assuming the weak-state SD is only 2x too large already yields **2.05x**, and 4x too large yields **5.96x**. At separation .5 the corresponding distortions are 46.1x, 2.37x, and 7.84x.

Thus posterior moment completion trades structural plug-in bias for joint-model specification risk. A practical causal-resolution diagnostic should report a **sensitivity envelope** over plausible state-treatment models rather than a naked completed eigenvalue. Large disagreement among Z-only, hard/soft plug-in, and joint-completed geometries is itself evidence that causal resolution is not robustly identified.

## Reproduction

Run:
```bash
python research/run_extended_reliability_studies.py
python research/frozen_sensor_sensitivity.py
python research/latent_state_count_sensitivity.py
python research/task_aware_state_coarsening.py
python research/data_driven_task_coarsening.py
python research/crossfit_task_coarsening.py
python research/weak_overlap_refinement_paradox.py
python research/abstraction_tradeoff.py
python research/multitask_abstraction.py
python research/risk_selected_abstraction.py
```

The script writes raw/summary CSVs under `research/results/`. Increase `PHASE_REPS`, `REG_REPS`, and `TRANSITION_REPS` for publication-grade Monte Carlo precision.

## Interpretation discipline

Promising:
- task-relative **directional** sensitivity is a stronger theoretical object than the scalar worst-case ratio;
- transition-localized uncertainty is a plausible sequential extension;
- scalar difficulty robustly ranks error across sample sizes.

Negative/non-promising as currently formulated:
- `eps/lambda_min` alone does not reliably determine the MSE-optimal regularization parameter;
- no evidence from these runs supports a simple universal `sqrt(N) eps/lambda_min` phase boundary;
- real-data transition diagnostics do not establish true latent transitions.

## Next rigorous tests

1. Derive/estimate `b_gamma` and compare `||(J+lambda I)^-1 b_gamma||` against the scalar bound on held-out synthetic worlds.
2. Hold `||b_gamma||` fixed while rotating its direction in K>=3 latent-state DGPs, not only score-level algebra.
3. Test whether transition-weighted proxy error predicts downstream causal error better than global ECE/NLL/entropy.
4. Learn lambda from training worlds using directional risk features and evaluate on held-out worlds; do not tune on target causal error.
5. Stress wrong-K, semi-Markov durations, non-Gaussian emissions, and time-varying transition matrices.


### R32. Robust causal geometry: disagreement is a falsification signal
A new experiment compares posterior-outer, representation-only diagonal completion, and joint state-treatment completion. The hypothesis is deliberately one-sided: large disagreement among these spectra should flag that a nominal completed lambda_min is model-dependent. See research/geometry_disagreement_diagnostic.py.

### R33. Task-weighted calibration
Global ECE/Brier can miss errors concentrated on observations carrying most causal information. A new experiment weights posterior error by residual-treatment leverage and compares it with ordinary calibration as a predictor of oracle/proxy spectral distortion. See research/task_weighted_calibration.py and research/ROBUST_CAUSAL_GEOMETRY.md.

These studies are prospective until their full Monte Carlo outputs are committed; no favorable numeric claim is made yet.


### R34. Matched global calibration can hide a ~3000x task-weighted error gap
The automated Monte Carlo completed successfully. Two representations were constructed with exactly the same global posterior squared error (0.016) but with mistakes concentrated on either the bottom or top 20% of treatment leverage. Across N=1,000, 5,000, 20,000, the high-leverage representation has task-weighted calibration error about 0.069, while the low-leverage representation is about 2.3e-5: roughly a **3,000x gap despite identical global calibration error**. The corresponding causal-geometry log error is about 0.46 versus 0.0093-0.0096, roughly a 48-50x difference.

This is a clean impossibility result for task-agnostic calibration: marginal/global calibration magnitude alone cannot determine downstream causal reliability when error location relative to score leverage is unconstrained.

### R35. Generic geometry disagreement is NOT a reliable misspecification detector
The proposed spread among posterior-outer, Z-only diagonal completion, and joint-completed spectra failed as a generic falsification diagnostic. Spearman correlation with completed-geometry error is only 0.128 and AUC for detecting >2x completion error is 0.486, essentially chance, despite a 69.7% failure prevalence in the designed sweep. This negative result is important: disagreement among several biased geometry constructions does not automatically produce a conservative diagnostic.

### R36. Task-weighted calibration helps modestly in the broad temperature sweep, not dramatically
Across the automated temperature/separation/weak-information sweep, ordinary Brier error has Spearman 0.470 with causal-geometry distortion; treatment-leverage-weighted Brier improves this only to 0.505. Thus task weighting is theoretically necessary in adversarial constructions but is not by itself a universally strong predictor in broad smooth distortions. The useful contribution is the impossibility/counterexample and the need for task-aware calibration, not a claim that one simple weighted Brier score solves reliability.


### R37. Multitask calibration ranking reversal is exact and ~100x
The automated experiment confirms that two representations with indistinguishable global posterior error (~0.016) can reverse reliability ranking across tasks. At N=50,000, left-concentrated error has task-A/task-B weighted errors 0.07693/0.000770, while right-concentrated error has 0.000768/0.07692. Each representation is roughly **100x worse for one task and 100x better for the other**, with the same global calibration. The reversal is stable from N=2,000 to 50,000.

This strengthens the impossibility result: without restrictions on downstream leverage functions, there is no task-independent scalar ordering of representations by reliability. Task dependence governs both state abstraction and the measure under which representation error matters.

### R38. Representation evaluation should be a task-conditioned Pareto problem, not a universal leaderboard
A new experiment constructs multiple error profiles and downstream leverage tasks, then evaluates the full risk matrix and Pareto frontier. The goal is to test whether globally best calibration can be dominated for relevant task families and whether specialist representations occupy different non-dominated regions. This is being run automatically; no numeric claim is made until outputs are committed.


### R39. Global representation leaderboards can be entirely Pareto-incomplete
The automated task-family experiment yields five representations and four downstream leverage tasks; **all five are Pareto non-dominated**. The globally best-calibrated representation has error 0.00835 overall and performs well on left/right tasks (~0.00326), but its center-task error is 0.02895. A center specialist with worse global error (0.01162) achieves 0.00247 on that task, an **11.7x improvement**. Thus even within a fixed finite task family, global calibration can select a representation far from optimal for a relevant downstream functional.

The appropriate comparison object is a task-conditioned risk vector/Pareto frontier. A scalar ranking requires an explicit task distribution, minimax criterion, or other declared utility.

### R40. Minimax task-family selection
A follow-up experiment compares the globally best representation, average-task selection, minimax worst-task selection, and per-task oracle as the declared task family expands. This operationalizes the decision-theoretic consequence of R39. Results are generated automatically; no numerical claim is recorded until committed.
