"""
adaptive_spectral_optimizer.py
==============================
Implements the exact data-driven spectral regularization minimizer derived from Theorem 4:

Risk Bound Objective:
R(lambda) = [ (C * eps_gamma + lambda * M_theta)^2 + (1/N) * Tr(Omega) ] / (lambda_min(J) + lambda)^2

Given:
- J_hat: Empirical coupled Gram matrix (K, K)
- S_hat: Empirical coupled score vector (K,)
- Omega_hat: Long-run score covariance matrix (K, K) from HAC sandwich
- eps_gamma_hat: Estimated proxy uncertainty surrogate via Proposition 3:
  eps_gamma_hat <= sqrt(2 * ln(2) * H_bar(gamma))
- M_theta_hat: Norm bound on initial effect estimate ||theta_init||_2
- C_hat: Empirical Lipschitz constant of score Jacobian

Solves:
lambda_star = argmin_{lambda >= 0} R(lambda)
using 1D bounded scalar minimization (scipy.optimize.minimize_scalar).

Guarantees:
- If lambda_min(J) is large (well-conditioned), lambda_star -> 0 (minimizing shrinkage bias).
- If lambda_min(J) is small (ill-conditioned), lambda_star > 0 (suppressing variance explosion).
"""

import numpy as np
from scipy.optimize import minimize_scalar
from typing import Tuple, Dict, Optional


class AdaptiveSpectralOptimizer:
    """
    Automated data-driven spectral regularization selector for OR-DML.
    Minimizes the leading-order finite-sample risk upper bound (Theorem 4).
    """
    def __init__(self,
                 lambda_max: float = 1.0,
                 c_lipschitz: float = 1.0,
                 use_entropy_surrogate: bool = True):
        self.lambda_max = lambda_max
        self.c_lipschitz = c_lipschitz
        self.use_entropy_surrogate = use_entropy_surrogate
        self.optimal_lambda_ = 0.0
        self.risk_at_optimal_ = 0.0
        self.risk_at_zero_ = 0.0
        self.risk_reduction_ratio_ = 1.0

    def compute_optimal_lambda(self,
                               J: np.ndarray,
                               Omega: np.ndarray,
                               N: int,
                               gamma: np.ndarray,
                               theta_init: Optional[np.ndarray] = None) -> float:
        """
        Compute plug-in optimal spectral penalty lambda* >= 0.
        """
        K = J.shape[0]
        eigvals = np.linalg.eigvalsh(J)
        lambda_min = float(max(eigvals[0], 1e-6))
        tr_omega = float(max(np.trace(Omega), 1e-6))
        
        # 1. State proxy error surrogate via Proposition 3 (Entropy bridge)
        if self.use_entropy_surrogate:
            # H_bar = -1/N sum_t sum_k gamma_tk log(gamma_tk)
            eps = 1e-12
            ent_t = -np.sum(gamma * np.log(np.maximum(gamma, eps)), axis=1)
            mean_ent = float(np.mean(ent_t))
            # Pinsker / Gini bound: eps_gamma <= sqrt(2 * ln(2) * H_bar)
            eps_gamma = float(np.sqrt(2.0 * np.log(2.0) * mean_ent))
        else:
            eps_gamma = 0.05
            
        # 2. Parameter norm bound
        if theta_init is not None:
            m_theta = float(max(np.linalg.norm(theta_init), 1.0))
        else:
            m_theta = 2.0
            
        c_lip = self.c_lipschitz
        
        # 3. Define the Theorem 4 risk objective function
        def risk_obj(lam: float) -> float:
            numerator = (c_lip * eps_gamma + lam * m_theta) ** 2 + (tr_omega / N)
            denominator = (lambda_min + lam) ** 2
            return float(numerator / denominator)
            
        r_zero = risk_obj(0.0)
        self.risk_at_zero_ = r_zero
        
        # 4. Minimize over [0, lambda_max]
        res = minimize_scalar(risk_obj, bounds=(0.0, self.lambda_max), method='bounded')
        lam_opt = float(res.x)
        r_opt = float(res.fun)
        
        self.optimal_lambda_ = lam_opt
        self.risk_at_optimal_ = r_opt
        self.risk_reduction_ratio_ = float(r_zero / max(r_opt, 1e-12))
        
        return lam_opt
