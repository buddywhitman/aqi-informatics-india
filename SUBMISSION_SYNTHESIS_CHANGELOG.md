# Submission Synthesis Changelog

This branch promotes only audit-surviving changes into the AISTATS submission package.

## Frozen metadata

The registered title and abstract are preserved. AISTATS 2027 permits only minor title/abstract corrections after the abstract deadline; this branch therefore does not use the later research program as a pretext for a major metadata rewrite.

## Main manuscript changes

1. Narrowed the scalar difficulty claim to its verified role:
   - strong controlled evidence in the two-state factorial design;
   - worst-direction/conservative diagnostic interpretation;
   - explicit warning that K>=3 error orientation relative to the information spectrum matters.

2. Rewrote Mumbai observational interpretation:
   - retained the committed nominal estimate;
   - added frozen-sensor sensitivity;
   - removed stable physical-causal interpretation.

3. Rewrote Bengaluru observational interpretation:
   - retained the nominal regime estimate;
   - removed unsupported mechanistic attribution;
   - disclosed HAC-lag sensitivity.

4. Strengthened Discussion/Limitations:
   - generated-state geometry can be anti-conservative;
   - Delhi placebo failure;
   - Mumbai telemetry sensitivity;
   - Bengaluru dependence sensitivity;
   - Kolkata weak information;
   - D does not uniformly dominate entropy;
   - financial transfer remains synthetic.

5. Expanded the mandatory AI Use Statement to disclose the actual hybrid workflow: conceptual/proof assistance, experiment design, falsification, code, analysis/interpretation, literature work, and manuscript editing, with explicit author verification/responsibility.

6. Added Appendix J, Post-Audit Stress Tests:
   - phantom causal resolution;
   - posterior second-moment completion identity and misspecification caveat;
   - higher-dimensional directional limitation of the scalar ratio;
   - dual adjustment/target resolution;
   - observational collision tests.

## Supplement changes

The canonical supplementary builder now packages selected post-audit evidence and scripts for:
- phantom causal resolution;
- posterior second-moment completion;
- completion misspecification;
- directional reliability;
- dual resolution;
- final real-data collision audit;
- representation-zoo reanalysis;
- novelty/claim audit.

## Research results intentionally NOT promoted as headline submission claims

- generic task-aware calibration as a novel concept;
- generic weak identification or strongly identified functionals;
- generated-covariate asymptotics;
- latent-class classification correction;
- Pareto representation selection;
- Neyman orthogonality itself;
- representation orthogonality as a completed operational method;
- observational sensor effects as causal ground truth.

## Verification

The final CI workflow:
1. runs scientific invariance checks;
2. compiles the manuscript with the AISTATS source;
3. runs artifact synchronization checks;
4. rebuilds the canonical supplementary ZIP;
5. checks AI Use Statement ordering and anonymity;
6. uploads the compiled submission package as a workflow artifact.

No success claim should be made until that workflow completes successfully.


## Final packaging guardrails

The supplementary builder now packages an anonymous `SUPPLEMENT_README.md` instead of the repository README. The final workflow scans the ZIP for identifying repository strings before committing the compiled PDF/archive upstream.
