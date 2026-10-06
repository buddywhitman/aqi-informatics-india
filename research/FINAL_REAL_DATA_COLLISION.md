# Final Real-Data Collision Test

This audit uses committed empirical outputs only. It does not infer oracle latent states or structural causal truth from observational sensor data.

## 1. Delhi: statistically precise but causal falsification is problematic

Delhi has very healthy reported proxy geometry (lambda_min 111.8, kappa 1.58) and small entropy (.0158). The contemporaneous regime effects are precise (0.704 +/- .051 and .128 +/- .074).

However the placebo table contains strongly significant *pre-treatment* associations:
- horizon -6: ATE .566 +/- .126, p<.001;
- horizon -3: .0687 +/- .0253, p=.0067;
- horizon -1: -.0147 +/- .0036, p<.001.

Therefore the real Delhi result cannot be treated as clean causal validation merely because representation uncertainty and conditioning look favorable. This directly supports the proxy/oracle/scientific-target separation: healthy representation geometry does not establish causal identification.

## 2. Mumbai: data-quality intervention changes the scientific picture materially

Original Mumbai:
- ATE 1.59 +/- 4.36;
- lambda_min .3165;
- entropy .1157.

Removing frozen NO2 runs >=6h changes:
- N 4180 -> 3295;
- ATE 1.59 -> 2.92;
- SE 4.36 -> 2.82;
- regime effects (-.03, 3.30) -> (3.86, 2.25);
- lambda_min .3165 -> .1781;
- kappa 1.09 -> 3.20;
- entropy .1157 -> .0596.

This is unusually informative: posterior entropy improves while conditioning worsens and the regime-effect pattern changes qualitatively. It is a real-data example of why entropy/state certainty and downstream causal geometry are distinct axes. It is NOT proof of phantom resolution because oracle states/information are unknown.

## 3. Bengaluru: apparently significant heterogeneity is not robustly a strong causal claim

Regime 2 contemporaneous estimate is -.153 +/- .069 (p=.0266), but the previously measured HAC sensitivity weakens significance at longer lags. Frozen-run removal changes little, which is reassuring for that particular data-quality concern. Pre-treatment placebo at -6 is borderline (p=.064), while -3/-1 are null.

The correct interpretation is observational heterogeneity with sensitivity to dependence assumptions, not decisive state-specific causality.

## 4. Kolkata: unresolved

Kolkata has N=1209 and lambda_min .0025 with SEs 1.40 and 2.67 for regime effects. Frozen-run removal barely changes lambda_min (.00248 -> .00245) and increases ATE SE. The empirical data support the label "unresolved/weak information"; they do not support substantive regime-effect interpretation.

## 5. Transition-localized uncertainty is real and highly city dependent

Filtering/smoothing disagreement and entropy are concentrated near inferred transitions:
- Delhi distance 0-1: filter/smooth L1 .923 versus .009 at distance 25+.
- Mumbai: .413 near transition versus .017 at 25+.
- Bengaluru/Kolkata have much larger fractions of observations near transitions.

This suggests a concrete real-data task-aware validation strategy: prioritize transition neighborhoods for state adjudication, but weight that priority further by causal score leverage. Transition uncertainty alone is not sufficient.

## 6. Imputation sensitivity is secondary relative to identification/data-quality issues

Strict versus full-sample ATE changes are modest relative to SE for Delhi, Mumbai, Bengaluru and Kolkata. This does not eliminate missingness concerns, but the larger threats in committed diagnostics are pre-treatment falsification, frozen-sensor sensitivity, weak information and dependence.

## 7. What the real data do NOT establish

They do not establish:
- phantom causal resolution (no oracle geometry);
- causal correctness of regime labels/effects;
- that lower entropy means better causal reliability;
- that spectral regularization repairs identification;
- that the new task-weighted calibration metric works operationally.

## 8. What they DO complement

The observational outputs independently motivate the multi-axis framework:
- representation uncertainty: entropy/filter-smoother disagreement;
- data quality: frozen sensor runs;
- temporal validity: HAC/placebo behavior;
- causal information: lambda spectrum/SE;
- scientific validity: requires falsification beyond all of the above.

The strongest real-data lesson is therefore diagnostic rather than causal: apparently favorable state certainty and conditioning can coexist with failed causal falsification, while data cleaning can improve state certainty but worsen conditioning.
