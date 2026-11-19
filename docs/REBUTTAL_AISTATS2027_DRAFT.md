# AISTATS 2027 rebuttal, Submission 1823 (v3)

Status: ready to post after your read-through. Nothing has been posted.

## 1. What to post where (deadline: November 22, 2026, AoE)

| Button | Reply to | File | Limit | Size |
|---|---|---|---|---|
| Rebuttal | Reviewer Qg1H (2, conf. 5) | `docs/rebuttal/A_Qg1H.md` | 5,000 | about 3,900 |
| Rebuttal | Reviewer BBdA (2, conf. 3) | `docs/rebuttal/B_BBdA.md` | 5,000 | about 4,100 |
| Rebuttal | Reviewer mC7v (3) | `docs/rebuttal/C_mC7v.md` | 5,000 | about 1,850 |
| Rebuttal | Reviewer wRgp (4) | `docs/rebuttal/D_wRgp.md` | 5,000 | about 3,300 |
| Rebuttal / Official Comment on the meta-review | Area chair (preliminary meta-review) | `docs/rebuttal/E_AC_PMR.md` | **2,500** | about 1,130 |

The rules come from the AISTATS 2027 rebuttal guidelines:
- One response per review, at most 5,000 characters each. The answer to the preliminary meta-review gets 2,500.
- Responses become visible to reviewers at the end of November 23 (AoE).
- Author-reviewer discussion runs November 23 to December 3. Use **Official Comment** then, to answer reviewer follow-ups.
- **Flag Review** is only for inappropriate content (or to mark an excellent review). Flags go to the Program Chairs only, and flagging a constructive negative review "is not acceptable". None of these reviews qualifies. Do not flag.
- No revised PDF before acceptance (the 2025 FAQ says text-only; the 2027 pages do not say otherwise).
- Camera-ready allows 9 pages, against 8 at submission; the appendix is unlimited.
- Titles: only minor changes are allowed. That is why the replies propose only dropping "Reliable" and ask the chairs, rather than a full retitle.

## 2. New results behind the replies (all reproducible, each script under 2 minutes)

| Script | Output | Result |
|---|---|---|
| `src/rebuttal/deconv_estimator.py` | `reports/rebuttal/deconv_{raw,summary}.csv` | Frontier, 100 datasets per cell. At Delta_Z = 1, DC-DML bias +0.01 vs. +2.13 (OR-DML) and coverage 95% vs. 0%. At Delta_Z = 0.5, RMSE 16.7 vs. 5.9, with coverage 97% (wide intervals). At Delta_Z = 2 and 4 the two are equal. Joint coverage of both regime effects is 81% at Delta_Z = 1. |
| `src/rebuttal/deconv_semisynthetic.py` | `reports/rebuttal/deconv_semisynthetic_*.csv` | City benchmark, 30 per city. DC-DML mean abs ATE bias 0.054 / 0.200 / 0.234 / 1.849 (Kolkata median 0.616), vs. OR-DML 0.42 / 1.65 / 1.69 / 2.05. Coverage 90 / 97 / 100 / 100%, vs. 0%. The variant with additive Z terms is unstable in Kolkata (max bias 1,328), so it is not used. |
| `src/rebuttal/deconv_table10.py` | `reports/rebuttal/deconv_table10_*.csv` | Table 10 scenarios. Joint MLE about 0.01 or less; DC-DML 0.03 to 0.28; posterior-DML 0.18 to 0.66. **The joint MLE wins every scenario.** |
| `src/rebuttal/joint_semisynthetic.py` | `reports/rebuttal/joint_semisynthetic_*.csv` | City benchmark. Regime-specific joint MLE 0.025 / 0.097 / 0.438 / 0.082 (beats DC-DML in Delhi, Mumbai and Kolkata; loses in Bengaluru). Common-theta joint MLE 0.19 to 0.30. |
| `src/rebuttal/bias_aware_ci.py` | `reports/rebuttal/bias_aware_ci_*.csv` | Budget-based intervals. At Delta_Z = 1: 0% with no budget, 5% at half the true shift, 94% at the true shift, 99% at twice the true shift. |
| `src/rebuttal/hetero_law_check.py` | `reports/rebuttal/hetero_law_check.csv` | Heteroscedastic law: error 0.07% to 1.7%; the naive average-variance law is off by 42% to 55%. |
| (inline check) | `reports/rebuttal/lambda_rule_check.csv` | The fixed lambda rule gives about 7e-4 to 9e-4, with sqrt(N) lambda about 0.03 (not o(1)); shrinkage bias about 0.004. |

## 3. The honest story the replies now tell

1. The paper's diagnostic finding (phantom resolution) and bias law stand.
2. **Identification is proved**, under an exclusion restriction plus informative proxies (`docs/REVISED_TEXT_AISTATS2027.md` section 6). Theorem 2 is the singular case.
3. Two-stage plug-in designs, including OR-DML, are biased under imperfect proxies. That is exactly what Theorem 4 says.
4. DC-DML, which follows from the identification result, removes that bias at a variance cost and gives near-nominal coverage at moderate separation.
5. **A joint MLE with regime-specific effects is better still on most benchmarks**, including the city benchmark, where its emission model is misspecified. The replies say so. Hiding it would be indefensible: Qg1H already asked about Table 10.

## 4. Do not claim

- That OR-DML is the best estimator. It is only the best two-stage design.
- That DC-DML beats the joint MLE. It does not, in most cells.
- That the DC-DML intervals include HMM-estimation uncertainty. The posterior is held fixed in the bootstrap.
- That the heteroscedastic law covers the coupled K-regime estimator. It is not proved.
- That the identification proposition has been checked by anyone but me. **Check the proof** (`docs/REVISED_TEXT_AISTATS2027.md` section 6) before posting.

## 5. Still open

- The `[CITE]` in the revised introduction: real citations are needed for "these diagnostics are routinely reported".
- Data inconsistencies, not traced: the ROC-AUC 0.615 vs. 0.5895, and Tables 3 vs. 20 for Kolkata and Bengaluru (`docs/REVISED_TEXT_AISTATS2027.md` section 9).
- If the paper is accepted: implement DC-DML with machine-learning nuisances and a full-pipeline bootstrap (HMM refit) for the camera-ready version. Promise only what is in the replies.
