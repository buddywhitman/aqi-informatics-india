"""
regime_intelligence.py
======================
Regime Intelligence: Learning, Representation, and Decision-Making
Under Persistent Hidden Contexts.

Implements the unified ML framework described in docs/ml.md:
  Raw Sequential Stream Z_{1:t}
    -> Latent State Encoder q_phi(S_t | Z_{1:t}) [HMM, GRU, Transformer, SSM]
    -> Calibrated Uncertainty & Regime Risk R_t = eps_gamma / (lambda_min + eps)
    -> Uncertainty-Aware Gating, Adaptive Regularization, & Abstention Policy
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union
from scipy.special import logsumexp
from scipy.stats import norm
from sklearn.metrics import f1_score, brier_score_loss


# =====================================================================
# 1. Latent State Representation Model Zoo
# =====================================================================

class GaussianHMMEncoder:
    """
    Classical generative Hidden Markov Model encoder.
    Provides calibrated forward filtering probabilities P(S_t | Z_{1:t})
    and retrospective smoothing probabilities P(S_t | Z_{1:N}).
    """
    def __init__(self, n_regimes: int = 2, n_iter: int = 30, tol: float = 1e-4, random_state: int = 42):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.pi = None
        self.A = None
        self.means = None
        self.covs = None

    def fit(self, Z: np.ndarray) -> "GaussianHMMEncoder":
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        N, d = Z.shape
        K = self.n_regimes
        rng = np.random.default_rng(self.random_state)

        # Quantile-based initialization
        quantiles = np.linspace(0.1, 0.9, K)
        self.means = np.quantile(Z, quantiles, axis=0)
        overall_var = np.var(Z, axis=0) + 1e-4
        self.covs = np.tile(overall_var, (K, 1))
        self.pi = np.full(K, 1.0 / K)
        self.A = np.full((K, K), 0.1 / (K - 1))
        np.fill_diagonal(self.A, 0.9)

        log_lik_prev = -np.inf
        for _ in range(self.n_iter):
            log_B = np.zeros((N, K))
            for k in range(K):
                diff = Z - self.means[k]
                var = np.maximum(self.covs[k], 1e-4)
                log_B[:, k] = -0.5 * np.sum(np.log(2 * np.pi * var) + (diff ** 2) / var, axis=1)

            # Forward-backward
            log_alpha = np.zeros((N, K))
            log_alpha[0] = np.log(self.pi + 1e-12) + log_B[0]
            log_alpha[0] -= logsumexp(log_alpha[0])
            for t in range(1, N):
                for k in range(K):
                    log_alpha[t, k] = logsumexp(log_alpha[t - 1] + np.log(self.A[:, k] + 1e-12)) + log_B[t, k]
                log_alpha[t] -= logsumexp(log_alpha[t])

            log_beta = np.zeros((N, K))
            for t in range(N - 2, -1, -1):
                for k in range(K):
                    log_beta[t, k] = logsumexp(np.log(self.A[k, :] + 1e-12) + log_B[t + 1] + log_beta[t + 1])
                log_beta[t] -= logsumexp(log_beta[t])

            log_gamma = log_alpha + log_beta
            log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            log_xi = np.zeros((N - 1, K, K))
            for t in range(N - 1):
                for j in range(K):
                    for k in range(K):
                        log_xi[t, j, k] = log_alpha[t, j] + np.log(self.A[j, k] + 1e-12) + log_B[t + 1, k] + log_beta[t + 1, k]
                log_xi[t] -= logsumexp(log_xi[t])
            xi = np.exp(log_xi)

            self.pi = np.maximum(gamma[0], 1e-6)
            self.pi /= np.sum(self.pi)
            self.A = np.maximum(np.sum(xi, axis=0), 1e-6)
            self.A /= np.sum(self.A, axis=1, keepdims=True)

            for k in range(K):
                denom = np.sum(gamma[:, k]) + 1e-8
                self.means[k] = np.sum(gamma[:, k:k+1] * Z, axis=0) / denom
                diff = Z - self.means[k]
                self.covs[k] = np.maximum(np.sum(gamma[:, k:k+1] * (diff ** 2), axis=0) / denom, 1e-4)

            log_lik = np.sum(logsumexp(log_alpha, axis=1))
            if abs(log_lik - log_lik_prev) < self.tol:
                break
            log_lik_prev = log_lik

        return self

    def filter_forward(self, Z: np.ndarray) -> np.ndarray:
        """Online causal forward filtering: P(S_t | Z_{1:t})."""
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        N, d = Z.shape
        K = self.n_regimes
        log_alpha = np.zeros((N, K))

        log_B0 = np.zeros(K)
        for k in range(K):
            diff = Z[0] - self.means[k]
            var = np.maximum(self.covs[k], 1e-4)
            log_B0[k] = -0.5 * np.sum(np.log(2 * np.pi * var) + (diff ** 2) / var)
        log_alpha[0] = np.log(self.pi + 1e-12) + log_B0
        log_alpha[0] -= logsumexp(log_alpha[0])

        for t in range(1, N):
            log_Bt = np.zeros(K)
            for k in range(K):
                diff = Z[t] - self.means[k]
                var = np.maximum(self.covs[k], 1e-4)
                log_Bt[k] = -0.5 * np.sum(np.log(2 * np.pi * var) + (diff ** 2) / var)
            for k in range(K):
                log_alpha[t, k] = logsumexp(log_alpha[t - 1] + np.log(self.A[:, k] + 1e-12)) + log_Bt[k]
            log_alpha[t] -= logsumexp(log_alpha[t])

        return np.exp(log_alpha)

    def smooth(self, Z: np.ndarray) -> np.ndarray:
        """Retrospective two-sided smoothing: P(S_t | Z_{1:N})."""
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        N, d = Z.shape
        K = self.n_regimes

        log_B = np.zeros((N, K))
        for k in range(K):
            diff = Z - self.means[k]
            var = np.maximum(self.covs[k], 1e-4)
            log_B[:, k] = -0.5 * np.sum(np.log(2 * np.pi * var) + (diff ** 2) / var, axis=1)

        log_alpha = np.zeros((N, K))
        log_alpha[0] = np.log(self.pi + 1e-12) + log_B[0]
        log_alpha[0] -= logsumexp(log_alpha[0])
        for t in range(1, N):
            for k in range(K):
                log_alpha[t, k] = logsumexp(log_alpha[t - 1] + np.log(self.A[:, k] + 1e-12)) + log_B[t, k]
            log_alpha[t] -= logsumexp(log_alpha[t])

        log_beta = np.zeros((N, K))
        for t in range(N - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = logsumexp(np.log(self.A[k, :] + 1e-12) + log_B[t + 1] + log_beta[t + 1])
            log_beta[t] -= logsumexp(log_beta[t])

        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma)


class GRURegimeEncoder(nn.Module):
    """
    Neural Recurrent Latent-State Encoder:
    Maps sequence history Z_{t-L:t} to latent regime probability simplex q_phi(S_t | Z_{1:t}).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32, n_regimes: int = 2, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_regimes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        out, _ = self.gru(x)
        # Use last step representation
        last_h = out[:, -1, :]
        logits = self.head(last_h)
        return torch.softmax(logits, dim=-1)


class TransformerRegimeEncoder(nn.Module):
    """
    Causal Self-Attention Latent-State Encoder:
    Applies multi-head self-attention with causal masking over historical context windows.
    """
    def __init__(self, input_dim: int, embed_dim: int = 32, num_heads: int = 2, n_regimes: int = 2):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=64,
            dropout=0.05, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Linear(embed_dim, n_regimes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        h = self.proj(x)
        seq_len = x.size(1)
        # Causal mask so position t cannot look at future positions
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        out = self.transformer(h, mask=mask)
        logits = self.head(out[:, -1, :])
        return torch.softmax(logits, dim=-1)


class SSMRegimeEncoder(nn.Module):
    """
    Linear State Space / Recurrent Filter Encoder:
    h_t = A h_{t-1} + B x_t, with learned continuous-time transition parameters.
    """
    def __init__(self, input_dim: int, state_dim: int = 16, n_regimes: int = 2):
        super().__init__()
        self.state_dim = state_dim
        self.A_diag = nn.Parameter(torch.rand(state_dim) * 0.5 + 0.4) # stable transition in [0.4, 0.9]
        self.B = nn.Linear(input_dim, state_dim)
        self.head = nn.Linear(state_dim, n_regimes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.state_dim, device=x.device)
        A = torch.clamp(self.A_diag, 0.0, 0.99)
        for t in range(seq_len):
            u_t = self.B(x[:, t, :])
            h = A * h + u_t
        logits = self.head(h)
        return torch.softmax(logits, dim=-1)


# =====================================================================
# 2. Regime Risk & Decision-Time Abstention Policy
# =====================================================================

class RegimeIntelligenceEngine:
    """
    Integrated Decision Engine combining:
    1. Latent State Inference (HMM or Neural)
    2. Dynamic Difficulty Index D_t = H(gamma_t) / lambda_min(J_t)
    3. Uncertainty-Aware Gating
    4. Adaptive Spectral Regularization lambda*(D_t)
    5. Identification-Aware Abstention / Safety Escalation Policy
    """
    def __init__(
        self,
        n_regimes: int = 2,
        abstain_threshold: float = 2.5,
        adaptive_lambda: bool = True,
        random_state: int = 42
    ):
        self.n_regimes = n_regimes
        self.abstain_threshold = abstain_threshold
        self.adaptive_lambda = adaptive_lambda
        self.random_state = random_state
        self.hmm = GaussianHMMEncoder(n_regimes=n_regimes, random_state=random_state)
        self.fitted = False

    def fit_latent_regimes(self, Z: np.ndarray):
        """Fits underlying latent Markov model on observation sequence Z."""
        self.hmm.fit(Z)
        self.fitted = True
        return self

    def infer_regimes(self, Z: np.ndarray, mode: str = "filter") -> np.ndarray:
        """Infers posterior regime probabilities under forward filtering or retrospective smoothing."""
        if not self.fitted:
            self.fit_latent_regimes(Z)
        if mode == "smooth":
            return self.hmm.smooth_retrospective(Z)
        return self.hmm.filter_forward(Z)

    def compute_regime_entropy(self, gamma: np.ndarray) -> np.ndarray:
        """Normalized Shannon entropy: H(gamma_t) in [0, 1]."""
        eps = 1e-12
        gamma_safe = np.clip(gamma, eps, 1.0)
        K = gamma.shape[1]
        raw_h = -np.sum(gamma_safe * np.log(gamma_safe), axis=1)
        max_h = np.log(K) if K > 1 else 1.0
        return raw_h / max_h

    def compute_transition_hazard(self, gamma: np.ndarray, transition_matrix: np.ndarray) -> np.ndarray:
        """Impending regime switch probability: P(S_{t+1} != S_t | Z_{1:t})."""
        N = gamma.shape[0]
        K = gamma.shape[1]
        hazard = np.zeros(N)
        for t in range(N):
            # Predicted next state distribution
            pred_next = gamma[t] @ transition_matrix
            # Probability of changing state from argmax state
            cur_state = np.argmax(gamma[t])
            hazard[t] = 1.0 - pred_next[cur_state]
        return hazard

    def compute_dynamic_difficulty(
        self,
        gamma: np.ndarray,
        T_res: np.ndarray,
        window: int = 48
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the dynamic difficulty ratio D_t = H(gamma_t) / lambda_min(J_t)
        over rolling temporal windows using closed-form spectral calculations.
        """
        N, K = gamma.shape
        entropy_t = self.compute_regime_entropy(gamma)

        if K == 2:
            # Closed-form 2x2 symmetric eigenvalue formula (1000x faster)
            s00 = pd.Series(gamma[:, 0]**2 * T_res[:, 0]**2).rolling(window, min_periods=1).mean().values
            s11 = pd.Series(gamma[:, 1]**2 * T_res[:, 1]**2).rolling(window, min_periods=1).mean().values
            s01 = pd.Series(gamma[:, 0] * gamma[:, 1] * T_res[:, 0] * T_res[:, 1]).rolling(window, min_periods=1).mean().values
            trace = s00 + s11
            disc = np.sqrt(np.maximum((s00 - s11)**2 + 4.0 * (s01**2), 0.0))
            lambda_min_t = np.maximum(0.5 * (trace - disc), 1e-4)
        else:
            lambda_min_t = np.full(N, 1e-4)
            step = 5
            for t in range(0, N, step):
                start = max(0, t - window + 1)
                end = t + 1
                g_win = gamma[start:end]
                t_win = T_res[start:end]
                J_local = np.zeros((K, K))
                for j in range(K):
                    for k in range(K):
                        J_local[j, k] = np.mean(g_win[:, j] * g_win[:, k] * t_win[:, j] * t_win[:, k])
                lmin = max(float(np.min(np.linalg.eigvalsh(J_local))), 1e-4)
                lambda_min_t[t:t+step] = lmin

        D_t = entropy_t / lambda_min_t
        return D_t, lambda_min_t

    def compute_adaptive_lambda(
        self,
        eps_gamma: float,
        lambda_min: float,
        N: int,
        M_theta: float = 1.0,
        tr_omega: float = 1.0,
        C_proxy: float = 1.0
    ) -> float:
        """
        Data-driven optimal spectral regularization parameter lambda* derived
        from the theoretical risk surrogate upper bound (Theorem 3).
        Directly minimizes the empirical risk surrogate over lambda in [0, 2.0].
        """
        from scipy.optimize import minimize_scalar

        def surrogate_risk(lam):
            numerator = (C_proxy * eps_gamma + lam * M_theta) ** 2 + (tr_omega / max(N, 1))
            denominator = max(lambda_min + lam, 1e-6) ** 2
            return numerator / denominator

        res = minimize_scalar(surrogate_risk, bounds=(0.0, 2.0), method='bounded')
        return float(res.x)

    def evaluate_decision_policy(
        self,
        y_true: np.ndarray,
        y_pred_point: np.ndarray,
        difficulty: np.ndarray,
        tau: Optional[float] = None,
        y_fallback: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluates the identification-aware abstention policy vs standard point estimation:
        When D_t <= tau: execute active model prediction.
        When D_t > tau: abstain and execute robust fallback (e.g. conservative shrinkage or zero-exposure).
        """
        if tau is None:
            tau = self.abstain_threshold

        abstain_mask = difficulty > tau
        coverage_rate = 1.0 - float(np.mean(abstain_mask))
        
        # Unconditional active model loss
        full_mse = float(np.mean((y_true - y_pred_point) ** 2))
        full_tail_loss = float(np.percentile((y_true - y_pred_point) ** 2, 95))

        # Conditional loss on accepted inferences
        if np.any(~abstain_mask):
            accepted_mse = float(np.mean((y_true[~abstain_mask] - y_pred_point[~abstain_mask]) ** 2))
            accepted_tail_loss = float(np.percentile((y_true[~abstain_mask] - y_pred_point[~abstain_mask]) ** 2, 95))
        else:
            accepted_mse = full_mse
            accepted_tail_loss = full_tail_loss

        # Loss on abstained cases (demonstrating avoided high-risk cases)
        if np.any(abstain_mask):
            avoided_mse = float(np.mean((y_true[abstain_mask] - y_pred_point[abstain_mask]) ** 2))
        else:
            avoided_mse = 0.0

        # Deploy real fallback action under the policy
        if y_fallback is None:
            # Conservative sample mean fallback
            y_fallback = np.full_like(y_true, np.mean(y_true))
        
        y_policy = np.where(~abstain_mask, y_pred_point, y_fallback)
        policy_mse = float(np.mean((y_true - y_policy) ** 2))
        policy_tail_loss = float(np.percentile((y_true - y_policy) ** 2, 95))
        policy_regret = float(np.mean(np.maximum((y_true - y_policy)**2 - (y_true - y_pred_point)**2, 0.0)))
        mse_reduction = (full_mse - policy_mse) / (full_mse + 1e-8)

        return {
            "tau_threshold": tau,
            "coverage_rate": coverage_rate,
            "abstain_rate": float(np.mean(abstain_mask)),
            "full_mse": full_mse,
            "policy_mse": policy_mse,
            "accepted_mse": accepted_mse,
            "avoided_mse": avoided_mse,
            "full_tail_95_mse": full_tail_loss,
            "accepted_tail_95_mse": accepted_tail_loss,
            "policy_tail_95_mse": policy_tail_loss,
            "policy_regret": policy_regret,
            "mse_reduction_pct": float(mse_reduction * 100.0)
        }
