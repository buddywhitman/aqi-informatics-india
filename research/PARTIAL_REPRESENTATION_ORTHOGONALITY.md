# Partial Representation Orthogonality and Correction Priority

Status: exploratory design principle; not submission material.

The partial-orthogonality experiment gives a severe warning. Correcting representation error in strong/medium information directions can have almost no effect if the uncorrected remainder lies in a weak causal direction.

With J=diag(1,.1,.01) and an equal-weight target, the strong-direction bias contribution is about .00577 while the weak-direction contribution is about .57735, 100x larger. Correcting the strong direction removes its own error exactly but leaves 100% of weak-direction bias. For a mixed perturbation, correcting the first two directions still leaves about 90% of target-amplified bias because the weak direction dominates.

Therefore limited correction effort should not be allocated by raw representation error magnitude alone.

## Priority principle

In a diagonal/eigenbasis approximation, direction j contributes approximately |a_j b_j|/lambda_j to target bias. If only d perturbation directions can be modeled/corrected, the oracle first-order priority is to correct directions with the largest target-amplified contribution, not necessarily the largest raw posterior error, the smallest eigenvalue alone, or the most common latent state.

The script research/orthogonality_priority.py compares these rules under deliberately conflicting constructions.

This links representation orthogonality directly to the directional reliability operator: correction resources should target the directions that dominate A J^-1 b.
