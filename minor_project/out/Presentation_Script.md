# Presentation Script and Run Sheet

Reliable Causal Estimation under Latent Markov Confounding. 19 slides, about 15 minutes plus questions.
Presenters: Pulkit Kumar, Utsav Verma, Nishant Kumar Upadhyay. Guide: Dr. Savitha G.

How to use: do not memorise sentences. Memorise the one-line **point** of each slide and the **anchor** (a number or word to say). Say the rest in your own words. The slides carry the detail; speaker notes in the PPTX repeat the facts.

## 1. Ownership and contribution (proposed, edit to match what each of you actually did)

The repository history is single-author. Before the viva, each person must be able to explain and defend every claim under their name. Change any line that is not true.

| Member | Claimed ownership | Report chapters to know well |
|---|---|---|
| Pulkit Kumar | Research design, theory (laws, orthogonality, error bound), OR-DML estimator, experiments, AISTATS manuscript, verification pipeline | 3, 4, 5, 7, 11, Annexure G |
| Utsav Verma | Data assembly and audit, four-city analysis, placebo and robustness checks, reproducibility package, figures and tables | 6, 8, 9, 10 |
| Nishant Kumar Upadhyay | Problem framing and literature, objectives, project management, report editing, conference and submission details | 1, 2, 10.6, Annexure A |

Division of speaking: Pulkit about 9 minutes, Utsav about 4, Nishant about 2.

## 2. Flow and slide-by-slide script

Each block: **Speaker. Point. Anchor. Say (loosely). Bridge. Report section.**

**1. Cover (Nishant, 20 s).** Point: who we are and what this is. Anchor: "submitted to AISTATS 2027". Say: names, guide, one sentence on the topic. Bridge: "To see why it matters, start with a simple question." Report: front matter.

**2. The project in one slide (Pulkit, 60 s).** Point: a trust rule people use fails; we show how and fix the estimator. Anchors: 4,900 runs, 82%, four cities. Say: "The usual rule says a confident hidden-state estimate means a reliable causal answer. We show it can mislead, exactly, and we build a better estimator." Bridge: "Here is the concrete question." Report: abstract, 1.1.

**3. The question (Nishant, 45 s).** Point: forecasts do not tell you what cutting NO₂ does. Anchor: "hidden state, only proxies seen". Say: stagnant air raises both pollutants; we never observe that state, only weather. Bridge: "So what did we set out to do?" Report: 1.2, 3.1.

**4. Objectives (Nishant, 30 s).** Point: six objectives O1 to O6. Say: read the first words of each, do not recite. Bridge: "Start with the data." Report: 1.3, 11.1 maps back to O1 to O6.

**5. Data (Utsav, 60 s).** Point: hourly multi-city data, audited. Anchor: the coverage plot and the number of months per city (about 16; Kolkata about 4). Say: where it came from, what the audit fixed (time grid, gaps). Bridge: "With data in hand, why do standard methods fail?" Report: Chapter 6.

**6. Why standard methods fail (Pulkit, 60 s).** Point: a hidden regime that is omitted biases the estimate; more controls can worsen it. Anchor: "8.4 against a true 1.5". Say: standard DML lands at about 8.4 when the truth is 1.53. Bridge: "How do we attack it?" Report: 3.2, 7.3.

**7. Approach (Pulkit, 45 s).** Point: model the regime, adjust on its posterior, cross-fit with a time embargo. Anchor: "purged block cross-fitting". Say: why embargo (time series leak across folds). Bridge: "The key finding came from testing the trust rule." Report: 4.1 to 4.3.

**8. Key discovery (Pulkit, 60 s).** Point: worse state estimates can look better conditioned. Anchor: "phantom resolution, 70 times". Say: the apparent information grows as the state estimate degrades, so a conditioning check is fooled. Bridge: "This has an exact law." Report: 5.3, 7.1.

**9. The rules: why it fails (Pulkit, 75 s).** Point: two laws. Anchors: the bias decomposition; λ₋ formula; "matches 4,900 runs". Say: explain symbols in words (σ² noise, Δ regime gap, ε state error). Do not derive. Bridge: "From those we get three design rules." Report: 3.2, 5.3, Annexure G.2, G.3.

**10. The rules: what we build (Pulkit, 75 s).** Point: orthogonality, error bound, residual-state bias. Anchors: ∂E ψ = 0; ε/λ_min; a·b·v over a²v+σ². Say: first-order errors cancel; errors add, neither hides the other; the bias peaks at |a| = σ/√v. Bridge: "Putting it together." Report: 5.1, 5.2, 5.4, G.4.

**11. The estimator (Pulkit, 60 s).** Point: OR-DML in code. Anchor: the snippet. Say: walk the three steps (posterior, nuisances conditioned on it, weighted target). Bridge: "Does it work?" Report: Chapter 4.

**12. Results (Pulkit, 75 s).** Point: best regime-aware estimator, ties at high separation, fails at weak separation. Anchors: 7 of 7 cells; 0.129 against 0.141. Say the limit yourself. Bridge: "Now real sensors." Report: 7.3, 7.4, 7.8.

**13. Real sensors, honest verdicts (Utsav, 75 s).** Point: four cities, aligned placebos, robustness values. Anchor: Delhi's lead placebo rejects. Say: an association, not a policy effect. Bridge: "Everything is regenerable." Report: Chapter 8.

**14. Reproducibility (Utsav, 45 s).** Point: one script writes every number; a verifier fails on mismatch. Anchor: 247 macros, `verify_submission.py`. Bridge: "How we got here." Report: Chapter 9.

**15. The journey (Utsav, 45 s).** Point: audits corrected earlier mistakes. Anchor: the timeline. Say: one example of a claim we withdrew. Bridge: "Where this is submitted." Report: Chapter 10.

**16. Where it is submitted (Nishant, 60 s).** Point: AISTATS is a top-tier peer-reviewed venue. Anchors: CORE A, 31% accepted in 2025 (583 of 1,861). Say: in AI the top conferences play the role journals play elsewhere; submitted, under review, not accepted yet. Do not quote an h5-index unless you checked Google Scholar Metrics. Bridge: "To conclude." Report: 10.6.

**17. Conclusions (Pulkit, 45 s).** Point: two rules survive (design, reporting). Say: map back to the objectives. Report: 11.1.

**18. Limits and next steps (Pulkit, 45 s).** Point: say the limits before they are asked. Anchors: weak separation fails; four-city data is observational. Report: 11.2 to 11.4.

**19. Thank you (Nishant, 15 s).** Invite questions; Pulkit leads technical answers, Utsav data answers.

## 3. Likely questions (answer in one or two sentences)

- **Why a minor-project of this level?** It is the required peer-reviewed output; the report documents the full work.
- **Is the paper accepted?** No. It is submitted and under review.
- **What is a hidden regime in plain terms?** An unobserved condition, such as stagnation, that shifts both pollutants.
- **Does cutting NO₂ cut PM₂.₅?** Not shown. The analysis is observational; Delhi's placebo rejects. Policy shocks are future work.
- **Why does a better state estimate not always help?** Error and apparent information grow together; see slide 8 and Theorem 3.
- **How do you know the numbers are right?** One generator script and a verifier that fails on mismatch (Chapter 9).
- **What would you do with more time?** Longer data, policy-shock design, software package (11.4).
- If you do not know: "That is outside what we tested; it is in our limits section." Do not guess.

## 4. Running the presentation

- Files: `Minor_Project_Presentation.pdf` (safest), `.pptx` (slides are images; speaker notes included).
- Open the PDF full screen (F11 in a PDF viewer). Use the arrow keys. Test the projector 10 minutes before.
- Keep the report open on a laptop and know the chapter for each slide (column above).
- Rehearse twice: once by anchors only, once timed. Hand-offs happen on the bridge sentence; agree it aloud.
- If time is short, drop slides 4, 11 and 15 aloud, keep everything else.
