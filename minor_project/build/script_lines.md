
## 5. Line-by-line scripts (read aloud, adapt to your own voice)

Speakers: **P** = Pulkit Kumar, **U** = Utsav Verma, **N** = Nishant Kumar Upadhyay. Slide numbers match the deck. Times total about 15 minutes. `[click]` means advance the slide. Words in *italics* are stage directions.

### Slide 1. Cover (N, 20 s)

**N:** Good morning, ma'am, and good morning everyone. We are Pulkit Kumar, Utsav Verma and Nishant Upadhyay from Electrical Engineering, and our guide is Dr. Savitha G.
**N:** Our minor project is titled "Reliable Causal Estimation under Latent Markov Confounding".
**N:** The research output is a paper that we have submitted to AISTATS 2027. [click]

### Slide 2. The project in one slide (P, 60 s)

**P:** Here is the whole project in one slide.
**P:** Many methods first estimate a hidden state from data, and then adjust for it. A common rule says that if the state estimate looks confident and well conditioned, the causal answer can be trusted.
**P:** We show that this rule can fail, and we show exactly how. We also build an estimator that is more reliable.
**P:** Three numbers summarise it. The exact law we derive matches 4,900 simulated runs, with an observed-to-predicted ratio of 0.999.
**P:** In 82 percent of runs where the hidden state is learned, the learned state looks better conditioned than the true one. That is the trap.
**P:** And we tested on real sensor data from four Indian cities. [click]

### Slide 3. The question (N, 45 s)

**N:** Let me explain the concrete question.
**N:** Traffic-related nitrogen dioxide and fine particulate matter, PM2.5, both rise together when the air is stagnant.
**N:** A forecast only tells us they move together. A policy maker wants to know something different: if we cut NO2, what happens to PM2.5?
**N:** The difficulty is that stagnation is a hidden state. We never measure it directly. We only see proxies such as temperature, humidity and wind.
**N:** This is a classic confounding problem, and it persists for hours to days, so the data are not independent either. [click]

### Slide 4. Objectives (N, 30 s)

**N:** We set six objectives, O1 to O6.
**N:** In short: formalise the problem, show where standard methods fail, derive exact laws, build an estimator, test on synthetic and real data, and make everything reproducible.
**N:** All six were met, and the conclusion chapter maps each one back to a chapter of the report.
**N:** Chapter 2 of the report surveys the related work, with 147 references that we checked against bibliographic databases. It shows that no existing line covers dependent data, a latent regime and the plug-in posterior together. Now Utsav will describe the data. [click]

### Slide 5. Data (U, 60 s)

**U:** Thank you, Nishant. Our data are hourly pollutant and weather records for four cities: Delhi, Mumbai, Bengaluru and Kolkata.
**U:** The record is about sixteen months for most cities. Kolkata is shorter, about four months, and ends in June 2025. Delhi has the densest record.
**U:** The coverage plot on the slide shows where data exist and where they have gaps.
**U:** An audit of the time grid found a defect: half-hour timestamps had been rounded in a way that collapsed the series. Fixing it changed Delhi's single-index estimate from 0.58 to 0.37.
**U:** So the first lesson is that data quality alone can move a result by a large amount. Now Pulkit will explain why standard methods struggle. [click]

### Slide 6. Why standard methods fail (P, 60 s)

**P:** Thank you, Utsav. Standard double machine learning assumes we observe all confounders.
**P:** When a hidden regime is omitted, the estimate is biased. Adding more controls can even make it worse. This effect is called bias amplification.
**P:** The intuition: the denominator of the estimate shrinks as controls explain more of the treatment, but the confounding part in the numerator does not move.
**P:** In our simulation, standard DML returns about 8.4 when the true effect is 1.53. That is not a small error. [click]

### Slide 7. Approach (P, 45 s)

**P:** Our approach has three ideas.
**P:** First, model the hidden regime with a hidden Markov model.
**P:** Second, adjust for the regime through its posterior probability.
**P:** Third, cross-fit with purged blocks: we split the series into contiguous blocks and leave a time gap around each, because neighbouring hours leak information. We call the estimator OR-DML. [click]

### Slide 8. Key discovery (P, 60 s)

**P:** The most important finding came from testing the trust rule itself.
**P:** The rule says: well conditioned means reliable. We found the opposite can happen.
**P:** On the left, the error is a hump as the proxy error grows. In the middle, the smallest eigenvalue of the Gram matrix, our conditioning measure, grows as the state estimate gets worse.
**P:** The right panel shows 82 percent of learned runs above the identity line. The state looks more informative than the truth, when it is actually less accurate. We call this phantom resolution. [click]

### Slide 9. The rules: why it fails (P, 75 s)

**P:** This behaviour is not an accident. It has exact laws.
**P:** The first law decomposes the bias of the standard estimator into a heterogeneity term and a confounding term. The weights are the variance shares of each regime.
**P:** The second law gives the smallest eigenvalue of the generated-state matrix. Here sigma squared is the treatment noise, Delta is the gap between regime means, and epsilon is the posterior error.
**P:** The key point is that this smallest eigenvalue can increase when epsilon increases, which is phantom resolution.
**P:** These formulas were checked against 4,900 simulated runs and they match. Full proofs are in Annexure G. [click]

### Slide 10. The rules: what we build (P, 75 s)

**P:** From these laws we derive three design rules.
**P:** One: orthogonality. With nuisances conditioned on the posterior, first-order errors cancel for every posterior.
**P:** Two: an error bound. The estimation error is controlled by the posterior error divided by the smallest eigenvalue, so the two errors add and neither hides the other.
**P:** Three: a residual-state bias. If the posterior leaves some state variance v unexplained, a bias remains. It peaks when the regime shift in treatment, a, equals sigma over the square root of v.
**P:** So the right report is not only the estimate, but also the residual variance and a robustness value. [click]

### Slide 11. The estimator (P, 60 s)

**P:** Here is the estimator in code.
**P:** Step one: fit the hidden Markov model and compute the regime posterior for each hour.
**P:** Step two: estimate the treatment and outcome nuisances while conditioning on that posterior, using purged block cross-fitting.
**P:** Step three: solve the weighted target equation with a spectral regularisation, and compute HAC standard errors that allow for serial dependence.
**P:** The report documents the code and its tests in Chapter 4. [click]

### Slide 12. Results (P, 75 s)

**P:** Does it work? On synthetic data it is the best regime-aware estimator in 7 of 7 cells of our main benchmark.
**P:** When the proxies separate the regimes well, it ties with the simplest baseline. For example 0.129 against 0.141 in error.
**P:** And we must be honest about a limit. When the proxies barely separate the regimes, there is no information to recover the state, and every method fails, including ours.
**P:** On a benchmark calibrated to real city data, our advantage is for regime-specific effects. For the pooled average effect, a simpler method is best in Mumbai and Bengaluru. Utsav will now show real sensors. [click]

### Slide 13. Real sensors, honest verdicts (U, 75 s)

**U:** Thank you, Pulkit. With real data, there is no known true effect, so the results are exposure associations, not policy effects.
**U:** We ran aligned placebo tests: if a future change in NO2 predicts today's PM2.5, something other than causation is present.
**U:** In Delhi, the association is positive in both regimes and survives our robustness check, but the lead placebo rejects. So we describe it as an association.
**U:** Mumbai, Bengaluru and Kolkata are inconclusive, because of data quality, failed placebos or wide intervals.
**U:** We think refusing to over-claim is part of the contribution. [click]

### Slide 14. Reproducibility (U, 45 s)

**U:** Everything is reproducible. One generator script writes every number in the paper, 247 values, into a claims file.
**U:** A verifier script, verify_submission.py, regenerates them and fails if any output is different.
**U:** The repository also has regression tests for the inference code. [click]

### Slide 15. The journey (U, 45 s)

**U:** This slide shows how the project developed over three phases.
**U:** Phase one was an applied study. An audit found a fabricated scalar and a sign error in an absolute value, so we retired those claims.
**U:** Phase two was the pivot to regime-aware DML. Phase three consolidated and corrected the work.
**U:** Each retired claim is documented in Chapter 10. Now Nishant will say where this is submitted. [click]

### Slide 16. Where it is submitted (N, 60 s)

**N:** Thank you, Utsav. We submitted the paper to AISTATS, the International Conference on Artificial Intelligence and Statistics.
**N:** In artificial intelligence, the top conferences play the role that journals play in many other fields. The AISTATS proceedings appear in the PMLR series.
**N:** It is rated CORE A, and the 2025 edition accepted 583 of 1,861 submissions, about 31 percent.
**N:** To be clear, our paper is submitted and under review. It is not yet accepted.
**N:** We have not quoted an h5-index in the slides, because the official Google Scholar page could not be checked automatically, as it blocks automated access. Third-party sites list a value near 100. [click]

### Slide 17. Conclusions (P, 45 s)

**P:** To conclude, two rules survive.
**P:** A design rule: use an orthogonal, posterior-adjusted estimator with purged blocks.
**P:** And a reporting rule: do not trust conditioning alone. Report the residual state variance and a robustness value.
**P:** Together these meet the six objectives. [click]

### Slide 18. Limits and next steps (P, 45 s)

**P:** We state the limits before we are asked.
**P:** First, when regimes are barely separable, no method recovers them.
**P:** Second, the four-city analysis is observational, so it shows associations, not policy effects.
**P:** Future work includes longer data, a design that uses a real policy shock, and a software package. Those are hypotheses, not findings. [click]

### Slide 19. Thank you (N, 15 s)

**N:** Thank you, ma'am and everyone. We are happy to take questions.
**N:** Pulkit will answer technical questions and Utsav will answer data questions.

### Per-person reading order

- **Pulkit:** slides 2, 6, 7, 8, 9, 10, 11, 12, 17, 18 (about 10 minutes).
- **Utsav:** slides 5, 13, 14, 15 (about 3.75 minutes).
- **Nishant:** slides 1, 3, 4, 16, 19 (about 2.75 minutes).
- Hand-offs happen on the last line before each `[click]` where the next speaker is named.
