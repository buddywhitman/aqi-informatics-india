"""Enumerate the binary models in the current non-identification proof.

Checks the complete joint law and the proxy-sufficiency posterior, rather
than just comparing conditional means. Independence over time then yields
observational equivalence for every sequence length.
"""
import numpy as np


def check():
    models = [([0.2, 0.2], [0.3, 0.5]), ([0.1, 0.3], [0.4, 0.4])]
    tables = []
    for theta, baseline in models:
        table = np.zeros((2, 2, 2))  # state, treatment, outcome
        for state in range(2):
            for treatment in range(2):
                p = baseline[state] + theta[state] * treatment
                assert 0 <= p <= 1
                table[state, treatment] = 0.25 * np.array([1 - p, p])
        np.testing.assert_allclose(table.sum(), 1)
        state_treatment = table.sum(axis=2)
        np.testing.assert_allclose(state_treatment / state_treatment.sum(axis=0), 0.5)
        np.testing.assert_allclose(state_treatment.sum(axis=1), 0.5)
        np.testing.assert_allclose(table.sum(axis=0)[:, 1] / 0.5, [0.4, 0.6])
        tables.append(table.sum(axis=0))
    np.testing.assert_allclose(tables[0], tables[1], atol=1e-15)
    assert sorted(models[0][0]) != sorted(models[1][0])
    np.testing.assert_allclose([np.mean(m[0]) for m in models], 0.2)
    print('NON-IDENTIFICATION CHECK PASSED: full observable law equal; '
          'proxy sufficiency holds; effect vectors differ; ATE is shared.')


if __name__ == '__main__':
    check()
