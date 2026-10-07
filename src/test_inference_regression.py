"""Numerical regressions for estimating-equation and variance consistency.

Run: python -m unittest discover -s src -p 'test_inference_regression.py' -v
"""
import unittest
import numpy as np
from sklearn.linear_model import Ridge
from or_dml import OverlapAwareRegimeDML
from synthesis.estimators import hac


def centered_hac(values, lag):
    values = values - values.mean(axis=0)
    out = values.T @ values / len(values)
    for h in range(1, lag + 1):
        cross = values[h:].T @ values[:-h] / len(values)
        out += (1 - h / (lag + 1)) * (cross + cross.T)
    return out


class InferenceRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(192)
        n = 180
        state = rng.integers(0, 2, n)
        cls.X = rng.normal(size=(n, 2))
        cls.Z = (state + rng.normal(size=n))[:, None]
        cls.T = 2 * state + cls.X[:, 0] + rng.normal(size=n)
        cls.Y = (0.4 + state) * cls.T + 2 * state + cls.X[:, 1] + rng.normal(size=n)

    def fitted(self, weighting='coupled', solve='coupled'):
        return OverlapAwareRegimeDML(
            n_regimes=2, n_splits=3, embargo_tau=3, n_inits=1,
            nuisance_mode='posterior', nuisance_model=Ridge(),
            weighting_mode=weighting, solve_mode=solve,
            reg_lambda=0.08, hac_lag=3, random_state=7,
        ).fit(self.Y, self.T, self.X, self.Z)

    def moments(self, model):
        w, rt, ry = model.weights_, model.tilde_T_.T, model.tilde_Y_.T
        design = w * rt
        matrices = design[:, :, None] * design[:, None, :]
        if model.weighting_mode == 'matched':
            for k in range(2):
                matrices[:, k, k] = w[:, k] * rt[:, k] ** 2
        if model.solve_mode == 'decoupled':
            matrices[:, 0, 1] = matrices[:, 1, 0] = 0
        theta = np.array(list(model.theta_regimes_.values()))
        score = w * rt * ry - np.einsum('nij,j->ni', matrices, theta)
        bread = np.linalg.inv(matrices.mean(0) + model.effective_lambda_ * np.eye(2))
        return theta, score, bread

    def test_sandwich_matches_derivative_of_solved_equation(self):
        for weighting in ['coupled', 'matched']:
            for solve in ['coupled', 'decoupled']:
                with self.subTest(weighting=weighting, solve=solve):
                    model = self.fitted(weighting, solve)
                    _, score, bread = self.moments(model)
                    np.testing.assert_allclose(score.mean(0), model.effective_lambda_ *
                                               np.array(list(model.theta_regimes_.values())), atol=1e-12)
                    expected = bread @ centered_hac(score, 3) @ bread.T / len(self.T)
                    np.testing.assert_allclose(model.Sigma_, expected, rtol=1e-10, atol=1e-12)

    def test_occupation_weighted_se_includes_cross_covariance(self):
        model = self.fitted()
        theta, score, bread = self.moments(model)
        influence = score @ bread.T @ model.regime_weights_
        influence += (model.weights_ - model.regime_weights_) @ theta
        expected = np.sqrt(centered_hac(influence[:, None], 3)[0, 0] / len(self.T))
        self.assertAlmostEqual(model.ate_se_, expected, places=11)

    def test_reference_hac_is_invariant_to_nonzero_score_mean(self):
        rng = np.random.default_rng(13)
        score = rng.normal(size=(80, 2))
        np.testing.assert_allclose(hac(score + [3, -8], 4), hac(score, 4), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
