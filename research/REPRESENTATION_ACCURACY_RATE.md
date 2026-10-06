# Representation Accuracy Rate Requirement

Status: stylized local-rate calculation; not submission material.

The reliability budget implies that "consistent representation learning" is not enough for valid causal inference. Representation error must vanish fast enough relative to both sample size and downstream information.

Suppose along a scientific direction:
    target information lambda_N = N^{-s},
    target-projected score perturbation b_N = N^{-r}.

Causal displacement from representation error scales as
    b_N / lambda_N = N^{s-r}.

Under a simple score model with variance proportional to lambda_N, sampling standard error scales as
    1/sqrt(N lambda_N) = N^{(s-1)/2}.

Therefore representation bias relative to sampling error scales as
    N^{(s+1)/2-r}.

The boundary is
    r = (1+s)/2.

For asymptotically negligible representation bias:
    r > (1+s)/2.

Special cases:
- strong information s=0: need r>1/2;
- information decays as N^-1/2 (s=.5): need r>.75;
- information decays as N^-1 (s=1): need r>1.

Thus weak causal information raises the accuracy rate demanded of the latent representation. A representation can be statistically consistent (r>0) yet still invalidate first-order causal inference.

## Interpretation

This couples representation learning rates and causal resolution in a single inferential requirement. Better downstream information relaxes the representation-accuracy requirement; weaker information makes nuisance/representation quality increasingly stringent.

The exact exponent depends on the score covariance/information relation and should not be universalized beyond the stylized local experiment without theorem-level assumptions.

Reproduction:
research/representation_accuracy_rate_requirement.py
