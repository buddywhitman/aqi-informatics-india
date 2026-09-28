"""
train_real_representation_zoo.py
================================
Trains REAL neural sequence encoders (GRU, Causal Transformer, Linear SSM)
and Gaussian HMM on sequence regimes under temporal distribution shifts.
NO mock perturbations. Real weights, real gradients, real metrics.
Automatically utilizes CUDA GPU if available.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, brier_score_loss

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.regime_intelligence import (
    GaussianHMMEncoder,
    GRURegimeEncoder,
    TransformerRegimeEncoder,
    SSMRegimeEncoder
)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    K = probs.shape[1]
    ece = 0.0
    for k in range(K):
        p = probs[:, k]
        y = (labels == k).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        for i in range(n_bins):
            m = (p >= bins[i]) & (p < bins[i + 1])
            if m.sum() > 0:
                ece += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return float(ece / K)


def generate_regime_data(N: int, K: int = 2, rho: float = 0.85, Delta_Z: float = 2.0, shift: bool = False, seed: int = 42):
    rng = np.random.default_rng(seed)
    if not shift:
        off = max(1e-4, (1.0 - rho) / (K - 1))
        A = np.full((K, K), off)
        np.fill_diagonal(A, rho)
        mu_Z = np.array([[-Delta_Z, 0.0], [Delta_Z, 0.0]])
    else:
        # Distribution shift: transition probabilities scramble + emission centers shift
        A = np.array([[0.60, 0.40], [0.40, 0.60]])
        mu_Z = np.array([[-0.8 * Delta_Z, 0.5], [0.8 * Delta_Z, -0.5]])

    S = np.zeros(N, dtype=int)
    S[0] = rng.integers(K)
    for t in range(1, N):
        S[t] = rng.choice(K, p=A[S[t - 1]])

    Z = mu_Z[S] + rng.normal(0, 1.0, (N, 2))
    return Z, S


def run_real_representation_zoo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"TRAINING REAL LATENT REPRESENTATION ZOO (Device: {device})")
    print("=" * 70)

    seq_len = 16
    N_train, N_test, N_ood = 1200, 800, 1500

    Z_train, S_train = generate_regime_data(N_train, K=2, rho=0.85, Delta_Z=2.0, shift=False, seed=42)
    Z_test, S_test = generate_regime_data(N_test, K=2, rho=0.85, Delta_Z=2.0, shift=False, seed=1042)
    Z_ood, S_ood = generate_regime_data(N_ood, K=2, rho=0.85, Delta_Z=2.0, shift=True, seed=2042)

    def make_windows(Z, S):
        X = np.array([Z[t - seq_len:t] for t in range(seq_len, len(Z))])
        y = S[seq_len - 1 : len(Z) - 1]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    X_tr, y_tr = make_windows(Z_train, S_train)
    X_te, y_te = make_windows(Z_test, S_test)
    X_ood, y_ood = make_windows(Z_ood, S_ood)

    X_tr_dev, y_tr_dev = X_tr.to(device), y_tr.to(device)

    # 1. Gaussian HMM
    print("\n[1/4] Fitting Gaussian HMM (Causal Filter)...")
    t0 = time.time()
    hmm = GaussianHMMEncoder(n_regimes=2, random_state=42)
    hmm.fit(Z_train)
    p_hmm_te = hmm.filter_forward(Z_test)[seq_len - 1 : len(Z_test) - 1]
    p_hmm_ood = hmm.filter_forward(Z_ood)[seq_len - 1 : len(Z_ood) - 1]
    hmm_time = time.time() - t0
    print(f"      HMM fitted in {hmm_time:.2f}s")

    # Helper for neural training
    def train_neural(model, name, lr=0.01, epochs=15):
        print(f"\nTraining {name} on {device} ({epochs} epochs)...")
        t0 = time.time()
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.NLLLoss()

        model.train()
        num_samples = len(X_tr_dev)
        for epoch in range(epochs):
            perm = torch.randperm(num_samples, device=device)
            for i in range(0, num_samples, 64):
                idx = perm[i : i + 64]
                optimizer.zero_grad()
                out = model(X_tr_dev[idx])
                loss = criterion(torch.log(out + 1e-12), y_tr_dev[idx])
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            p_te = model(X_te.to(device)).cpu().numpy()
            p_ood = model(X_ood.to(device)).cpu().numpy()

        train_time = time.time() - t0
        print(f"      {name} trained in {train_time:.2f}s")
        return p_te, p_ood

    # 2. Neural GRU
    gru = GRURegimeEncoder(input_dim=2, hidden_dim=32, n_regimes=2)
    p_gru_te, p_gru_ood = train_neural(gru, "[2/4] Neural GRU Encoder", lr=0.01, epochs=15)

    # 3. Causal Transformer
    tf = TransformerRegimeEncoder(input_dim=2, embed_dim=32, num_heads=2, n_regimes=2)
    p_tf_te, p_tf_ood = train_neural(tf, "[3/4] Causal Transformer Encoder", lr=0.008, epochs=15)

    # 4. Linear SSM
    ssm = SSMRegimeEncoder(input_dim=2, state_dim=16, n_regimes=2)
    p_ssm_te, p_ssm_ood = train_neural(ssm, "[4/4] Linear State Space Model (SSM)", lr=0.01, epochs=15)

    # Evaluate all models with genuine metrics
    y_te_np = y_te.numpy()
    y_ood_np = y_ood.numpy()

    models = [
        ("Gaussian HMM (Causal Filter)", p_hmm_te, p_hmm_ood),
        ("Neural GRU Encoder", p_gru_te, p_gru_ood),
        ("Causal Transformer Encoder", p_tf_te, p_tf_ood),
        ("Linear State Space Model (SSM)", p_ssm_te, p_ssm_ood),
    ]

    crit = nn.NLLLoss()
    zoo_rows = []
    print("\n" + "=" * 70)
    print("GENUINE REPRESENTATION ZOO BENCHMARK RESULTS")
    print("=" * 70)

    for name, pt, po in models:
        f1i = float(f1_score(y_te_np, np.argmax(pt, axis=1), average="macro"))
        f1o = float(f1_score(y_ood_np, np.argmax(po, axis=1), average="macro"))
        bi = float(np.mean([brier_score_loss(y_te_np == k, pt[:, k]) for k in range(2)]))
        bo = float(np.mean([brier_score_loss(y_ood_np == k, po[:, k]) for k in range(2)]))
        ei = compute_ece(pt, y_te_np)
        eo = compute_ece(po, y_ood_np)
        nlli = float(crit(torch.log(torch.tensor(pt) + 1e-12), torch.tensor(y_te_np, dtype=torch.long)))
        nllo = float(crit(torch.log(torch.tensor(po) + 1e-12), torch.tensor(y_ood_np, dtype=torch.long)))
        gap = round(f1i - f1o, 4)

        zoo_rows.append({
            "Architecture": name,
            "InDist_Macro_F1": round(f1i, 4),
            "OOD_Shift_F1": round(f1o, 4),
            "F1_Generalization_Gap": gap,
            "InDist_Brier": round(bi, 4),
            "OOD_Shift_Brier": round(bo, 4),
            "InDist_ECE": round(ei, 4),
            "OOD_Shift_ECE": round(eo, 4),
            "InDist_NLL": round(nlli, 4),
            "OOD_Shift_NLL": round(nllo, 4)
        })
        print(f"{name:<34} | In F1: {f1i:.4f} | OOD F1: {f1o:.4f} | Gap: {gap:+.4f} | In ECE: {ei:.4f} | OOD ECE: {eo:.4f}")

    df_models = pd.DataFrame(zoo_rows)
    os.makedirs("reports", exist_ok=True)
    df_models.to_csv("reports/latent_regime_bench_models.csv", index=False)
    print("\nSaved genuine trained metrics to reports/latent_regime_bench_models.csv")
    return df_models


if __name__ == "__main__":
    run_real_representation_zoo()
