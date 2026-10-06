# Representation Zoo Reanalysis Under the New Framework

This pass uses only committed aggregate zoo outputs; no per-observation posterior/leverage arrays were found in the report tables inspected here, so directional/task-weighted calibration cannot be reconstructed honestly from these CSVs alone.

## What the stored results actually support

Across 15 worlds, HMM has the strongest reliability ROC-AUC in the aggregate table (0.6153), followed by SSM (0.5492), GRU (0.5228), Transformer (0.5150). Entropy baseline AUC is nearly identical to D for every architecture; reported D-over-H advantages are -0.0055, +0.0011, -0.0023, -0.0029. Thus the original coupled operational score does not add robust discrimination over entropy.

Temperature scaling sharply improves neural calibration but not reliability discrimination:
- GRU ECE falls 47.9%, ROC-AUC changes -0.0035;
- Transformer ECE falls 60.3%, ROC-AUC changes -0.0075;
- SSM ECE falls 15.8%, ROC-AUC changes -0.0030.
This is direct empirical evidence that global probability calibration and downstream reliability are distinct.

However, representation quality is not irrelevant. Within-seed correlations with downstream reliability are strong for F1 (rho .80), ECE (-.76), NLL (-.84). LOWO prediction improves from Spearman .374 using F1 alone to .527 using F1+NLL. Hierarchical regression gives within-R2 .431 for F1 alone and .524 for F1+NLL; NLL coefficient -0.0653 has p=.0442 while F1 becomes nonsignificant conditional on NLL.

Therefore the honest synthesis is:
1. global representation metrics contain predictive signal;
2. no single metric is sufficient;
3. calibration interventions that improve global ECE/NLL do not causally improve the existing reliability score;
4. the coupled D score adds essentially nothing over entropy in this zoo;
5. aggregate CSVs cannot test the new directional/task-weighted hypothesis because they lack observation-level error x causal-leverage information.

## New interpretation

The calibration intervention is particularly consistent with the task-weighted calibration principle: temperature scaling changes global probability geometry while leaving the location/orientation of errors relative to causal leverage largely unchanged. This is a mechanism hypothesis, not established by the stored aggregate outputs.

## Required next data product

To test it, persist per observation/world/model:
- posterior vector;
- true synthetic state;
- treatment residual / score leverage;
- oracle and proxy score contribution;
- scientific target/contrast;
- failure label/downstream error.

Then compute ordinary versus task-weighted/directional calibration and ask whether the latter explains the residual reliability variation after F1/NLL.

## Negative-result preservation

Do not claim that deep models have high state F1 yet universally fail causal reliability: HMM itself dominates both state recovery and reliability here, while neural architectures vary. The stronger supported statement is that representation metrics correlate with reliability but calibration improvement alone is not sufficient, and the proposed D score does not outperform entropy.
