# Proxy, Oracle, and Scientific Target Are Three Different Benchmarks

Status: adversarial conceptual check; not submission material.

Total scientific error decomposes exactly as

    theta_proxy - theta_star
      = (theta_proxy - theta_oracle)
      + (theta_oracle - theta_star).

The first term measures generated-representation degradation relative to an oracle estimator. The second measures whether that oracle estimator targets the desired scientific/structural estimand.

These terms can reinforce or cancel.

## Consequence: accidental improvement

A worse proxy can be closer to structural truth than the oracle estimator if proxy error happens to oppose oracle-model bias. This does not make the proxy scientifically superior in a stable sense; it is cancellation between two errors.

Conversely, improving state recovery can move the proxy estimator closer to a misspecified oracle and therefore farther from structural truth.

Hence monotonicity of proxy-to-oracle fidelity does not imply monotonicity of scientific validity.

## Evaluation hierarchy

Every experiment should distinguish:
1. representation fidelity: proxy state versus latent state;
2. proxy-estimator fidelity: theta_proxy versus theta_oracle;
3. oracle identification: theta_oracle versus theta_star;
4. final scientific error: theta_proxy versus theta_star.

Synthetic/semi-synthetic experiments with known theta_star can measure all four. Observational real-data experiments generally cannot measure (3) or (4) without additional identification assumptions.

This prevents a common circularity: calling an estimator reliable because it approaches an oracle estimator whose scientific validity was never established.

## Reproduction
research/oracle_validity_decomposition.py
