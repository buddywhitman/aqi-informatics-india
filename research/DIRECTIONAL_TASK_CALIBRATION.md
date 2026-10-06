# Directional Task Calibration

Status: exploratory; not submission material.

Task weighting is still too coarse when the downstream target is vector-valued. Different causal contrasts induce different score directions and therefore different leverage measures.

For effect vector theta and contrast v, define a direction-specific representation error under the causal-score measure

    C(v) = E[w_v(X,T,S) e_gamma^2] / E[w_v(X,T,S)],

where w_v is the leverage/information contribution for contrast v.

The collection {C(v): ||v||=1} is a **directional calibration spectrum**. A single task-weighted scalar averages over directions and can hide a representation that is reliable for one causal contrast but poor for another.

This mirrors the causal-resolution spectrum:
- resolution spectrum: which effect directions the data can identify;
- calibration spectrum: in which effect directions representation errors contaminate the score.

A natural joint object is direction-wise risk proportional to representation perturbation divided by directional information, rather than a global entropy divided by lambda_min.

This may be the correct generalization of the original scalar difficulty idea:
    D(v) = B_gamma(v) / sqrt(v'Jv)
or an appropriate squared/information-scaled analogue.

The worst-case scalar is sup_v D(v), while a scientific target uses D(v_target). Averaging or using lambda_min can be unnecessarily pessimistic in irrelevant weak directions and optimistic when proxy geometry is distorted.

Reproduction:
research/directional_calibration_spectrum.py
