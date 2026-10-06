# Robust Target Families

Status: exploratory methodology; not submission material.

Pure target-aware inference assumes the scientific target is fixed. Pure global conditioning protects every possible contrast and can be arbitrarily pessimistic. A middle ground is to declare a family of plausible scientific targets A in F.

Define family sensitivity
    R(F,J) = sup_{A in F} ||A J^-1||.

This interpolates:
- singleton F: fixed-target sensitivity;
- all unit contrasts: global 1/lambda_min;
- restricted cone/subspace: robust target-family sensitivity.

Regularization can then minimize a family-aware bias/stability objective rather than either isotropic worst-case stabilization or single-target specialization.

This is especially appropriate when:
- primary/secondary estimands are predeclared;
- future exploratory targets are expected to lie in a known subspace/cone;
- a representation will be reused for a bounded family of downstream causal queries.

The script robust_target_family_regularization.py studies a two-dimensional cone of possible future target rotations and chooses selective regularization under a stability penalty. As the target cone widens, the optimal procedure should protect more of the weak direction, interpolating between aggressive specialization and global robustness.

The conceptual hierarchy is:
    fixed target < target family < unrestricted target space.

This gives an explicit knob for scientific reusability rather than treating task specificity as binary.
