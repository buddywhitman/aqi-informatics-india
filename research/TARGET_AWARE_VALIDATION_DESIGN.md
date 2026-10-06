# Target-Aware Validation Design

Status: exploratory operational methodology; not submission material.

If a small labeled validation set can observe true latent states or high-quality adjudicated proxies, it can estimate representation perturbation moments and turn the abstract uncertainty set for b into data-driven bounds.

But validation examples are not equally valuable for a causal target.

Let c=a'J^-1 be target sensitivity and let coordinate j's perturbation observation have variance v_j. If m_j validation observations inform coordinate j independently, propagated variance is approximately

    sum_j c_j^2 v_j / m_j.

Under a fixed validation budget M, the variance-minimizing allocation is the Neyman-type rule

    m_j proportional |c_j| sqrt(v_j).

Thus validation effort should be concentrated where representation uncertainty and causal target sensitivity jointly matter.

This is the same directional principle appearing again:
- correction priority: |a_j b_j|/lambda_j;
- validation priority: |c_j| sqrt(v_j);
- reliability: c b.

## Implication

Uniform labeling, labeling proportional to state frequency, or labeling only where the representation is most uncertain can all be inefficient for causal reliability. Rare weak-information states can deserve disproportionate validation effort if the scientific target is sensitive to them.

This suggests a practical active-validation loop:
1. estimate provisional target sensitivity;
2. identify high-impact representation directions/states;
3. acquire/adjudicate labels preferentially there;
4. update perturbation bounds;
5. stop when target-specific bias bound falls below a scientific/inferential tolerance.

The current experiment compares equal, frequency, raw-uncertainty, and target-sensitive allocations.
