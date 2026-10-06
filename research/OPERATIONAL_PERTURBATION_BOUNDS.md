# Operational Bounds When Representation Perturbation Is Not Identified

The exact reliability operator needs b_gamma, but latent-state representation error is generally not observed. A realistic method may therefore need uncertainty sets rather than a point estimate.

For target row
    c = a' J^{-1},
scientific representation bias is
    |c b|.

If b lies in an L2 ball ||b||<=rho:
    |c b| <= rho ||c||.

If coordinatewise bounds |b_j|<=rho_j are available:
    |c b| <= sum_j |c_j| rho_j.

If b lies in ellipsoid b' W^{-1} b <= 1:
    |c b| <= sqrt(c W c').

These are support functions of the perturbation uncertainty set. They connect representation validation directly to target-specific causal sensitivity.

## Interpretation

The geometry of uncertainty about b matters as much as its magnitude. An isotropic norm ball can be extremely conservative when representation uncertainty is known to avoid weak target-relevant directions. Directional/ellipsoidal validation information can sharply tighten the bound.

Potential sources for rho/W:
- labeled validation states in a small subset;
- repeated proxy measurements;
- posterior predictive calibration;
- sensitivity assumptions on misclassification;
- simulation-calibrated perturbation classes.

This is a possible operational route when exact representation orthogonality is unavailable: report a target-specific worst-case bias bound over a scientifically defensible perturbation set.

The next empirical question is how loose L2, coordinate-box, and ellipsoidal bounds are under anisotropic uncertainty.
