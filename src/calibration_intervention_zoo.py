"""
calibration_intervention_zoo.py
================================
Evaluates Post-Hoc Calibration Interventions (Platt / Temperature Scaling)
on Deep Neural Sequence Representations in LatentRegimeBench.

Question:
  Does post-hoc temperature calibration of deep neural sequence representations
  restore downstream task reliability (ROC-AUC D_t) under compound distribution shift?

Methodology:
  1. On each random seed, train Gaussian HMM, Neural GRU, Causal Transformer,
     and Linear SSM on in-distribution stream (N=1,200).
  2. On held-out compound shift (Shift C), split into calibration partition (N=400)
     and test evaluation partition (N=800).
  3. Optimize temperature parameter T* > 0 on calibration logits via NLL minimization.
  4. Evaluate raw vs temperature-calibrated representations on out-of-sample test split:
     - Expected Calibration Error (ECE)
     - Negative Log-Likelihood (NLL)
     - Downstream Task Difficulty ROC-AUC (D_t = H_t / lambda_min(J_t))
     - Downstream Task Difficulty PR-AUC (D_t)
     - Posterior Entropy Failure ROC-AUC (H_t)
  5. Save results to reports/representation_zoo_calibration_intervention.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, roc_auc_score, auc, precision_recall_curve
from scipy.optimize import minimize_scalar

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.regime_intelligence import (
    GaussianHMMEncoder,
    GRURegimeEncoder,
    TransformerRegimeEncoder,
    SSMRegimeEncoder
)
from src.train_real_representation_zoo import (
    generate_regime_stream,
    compute_ece,
    compute_nll
)


def temperature_scale(probs: np.ndarray, temp: float, eps: float = 1e-12) -> np.ndarray:
    """Apply temperature scaling to posterior probabilities."""
    log_p = np.log(np.clip(probs, eps, 1.0))
    scaled_log_p = log_p / max(temp, 1e-4)
    scaled_p = np.exp(scaled_log_p - np.max(scaled_log_p, axis=1, keepdims=True))
    scaled_p = scaled_p / np.sum(scaled_p, axis=1, keepdims=True)
    return scaled_p


def fit_temperature(probs_cal: np.ndarray, labels_cal: np.ndarray) -> float:
    """Find temperature T* > 0 minimizing NLL on calibration set."""
    def nll_obj(t):
        p_scaled = temperature_scale(probs_cal, t)
        return compute_nll(p_scaled, labels_cal)

    res = minimize_scalar(nll_obj, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)


def evaluate_calibration_intervention(n_seeds: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Calibration Intervention Evaluation across {n_seeds} seeds on {device}...")
    seq_len = 16
    N_tr = 1200
    N_eval = 1200
    N_cal = 400

    results_per_seed = []

    for seed in range(42, 42 + n_seeds):
        print(f"  Running Seed {seed}...")
        torch.manual_seed(seed)
        np.random.seed(seed)

        Z_tr, S_tr, X_tr, T_tr, Y_tr = generate_regime_stream(N_tr, shift_mode="none", seed=seed)
        Z_sc, S_sc, X_sc, T_sc, Y_sc = generate_regime_stream(N_eval, shift_mode="shift_c", seed=seed + 100)

        def make_windows(Z, S):
            X_w = np.array([Z[t - seq_len:t] for t in range(seq_len, len(Z))])
            y_w = S[seq_len - 1 : len(Z) - 1]
            return torch.tensor(X_w, dtype=torch.float32), torch.tensor(y_w, dtype=torch.long)

        X_tr_w, y_tr_w = make_windows(Z_tr, S_tr)
        X_sc_w, y_sc_w = make_windows(Z_sc, S_sc)
        X_tr_dev, y_tr_dev = X_tr_w.to(device), y_tr_w.to(device)

        # 1. HMM
        hmm = GaussianHMMEncoder(n_regimes=2, random_state=seed)
        hmm.fit(Z_tr)
        p_hmm_tr = hmm.filter_forward(Z_tr)[seq_len - 1 : len(Z_tr) - 1]
        p_hmm_sc = hmm.filter_forward(Z_sc)[seq_len - 1 : len(Z_sc) - 1]

        # Helper for neural training
        def train_neural(model, lr=0.01, epochs=12):
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = nn.NLLLoss()
            model.train()
            num_samples = len(X_tr_dev)
            for _ in range(epochs):
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
                p_tr = model(X_tr_dev).cpu().numpy()
                p_sc = model(X_sc_w.to(device)).cpu().numpy()
            return p_tr, p_sc

        # 2. GRU
        gru = GRURegimeEncoder(input_dim=2, hidden_dim=32, n_regimes=2)
        p_gru_tr, p_gru_sc = train_neural(gru, lr=0.01)

        # 3. Transformer
        tf = TransformerRegimeEncoder(input_dim=2, embed_dim=32, num_heads=2, n_regimes=2)
        p_tf_tr, p_tf_sc = train_neural(tf, lr=0.008)

        # 4. SSM
        ssm = SSMRegimeEncoder(input_dim=2, state_dim=16, n_regimes=2)
        p_ssm_tr, p_ssm_sc = train_neural(ssm, lr=0.01)

        models_dict = {
            "Gaussian HMM (Causal Filter)": (p_hmm_tr, p_hmm_sc),
            "Neural GRU Encoder": (p_gru_tr, p_gru_sc),
            "Causal Transformer Encoder": (p_tf_tr, p_tf_sc),
            "Linear State Space Model (SSM)": (p_ssm_tr, p_ssm_sc)
        }

        sl_tr = slice(seq_len - 1, N_tr - 1)
        sl_sc = slice(seq_len - 1, N_eval - 1)
        y_sc_all = y_sc_w.numpy()

        # Calibration vs Test split on Shift C
        # Note: seq_len-1 offset already present
        cal_slice = slice(0, N_cal)
        test_slice = slice(N_cal, len(y_sc_all))

        y_cal = y_sc_all[cal_slice]
        y_test = y_sc_all[test_slice]

        X_tr_s = X_tr[sl_tr]
        T_tr_s = T_tr[sl_tr]
        Y_tr_s = Y_tr[sl_tr]

        X_sc_s = X_sc[sl_sc]
        T_sc_s = T_sc[sl_sc]
        Y_sc_s = Y_sc[sl_sc]

        X_test_s = X_sc_s[test_slice]
        T_test_s = T_sc_s[test_slice]
        Y_test_s = Y_sc_s[test_slice]

        for m_name, (p_tr_m, p_sc_m) in models_dict.items():
            p_cal = p_sc_m[cal_slice]
            p_test = p_sc_m[test_slice]

            # Fit temperature scaling on calibration set
            if "HMM" in m_name:
                t_star = 1.0  # HMM is already calibrated probabilistic generative model
            else:
                t_star = fit_temperature(p_cal, y_cal)

            p_test_calibrated = temperature_scale(p_test, t_star)

            # Fit downstream Ridge model on training split using raw posteriors
            Phi_tr = np.column_stack([
                p_tr_m[:, 0], p_tr_m[:, 1],
                T_tr_s,
                p_tr_m[:, 0] * T_tr_s, p_tr_m[:, 1] * T_tr_s,
                X_tr_s[:, 0], X_tr_s[:, 1]
            ])
            downstream = Ridge(alpha=1.0).fit(Phi_tr, Y_tr_s)

            # Evaluate downstream loss on test split
            Phi_test = np.column_stack([
                p_test[:, 0], p_test[:, 1],
                T_test_s,
                p_test[:, 0] * T_test_s, p_test[:, 1] * T_test_s,
                X_test_s[:, 0], X_test_s[:, 1]
            ])
            Y_hat = downstream.predict(Phi_test)
            loss_test = (Y_test_s - Y_hat) ** 2

            # Evaluate failure threshold on calibration loss
            Phi_cal = np.column_stack([
                p_cal[:, 0], p_cal[:, 1],
                T_sc_s[cal_slice],
                p_cal[:, 0] * T_sc_s[cal_slice], p_cal[:, 1] * T_sc_s[cal_slice],
                X_sc_s[cal_slice, 0], X_sc_s[cal_slice, 1]
            ])
            loss_cal = (Y_sc_s[cal_slice] - downstream.predict(Phi_cal)) ** 2
            fail_thresh = float(np.percentile(loss_cal, 90))
            y_fail = (loss_test[1:] > fail_thresh).astype(int)

            # Nuisances on training split
            reg_m0 = Ridge(alpha=1.0).fit(X_tr_s, T_tr_s, sample_weight=np.clip(p_tr_m[:, 0], 1e-4, 1.0))
            reg_m1 = Ridge(alpha=1.0).fit(X_tr_s, T_tr_s, sample_weight=np.clip(p_tr_m[:, 1], 1e-4, 1.0))

            T_res0 = T_test_s - reg_m0.predict(X_test_s)
            T_res1 = T_test_s - reg_m1.predict(X_test_s)

            # Function to compute ROC-AUC and PR-AUC given probabilities
            def eval_difficulty(p_eval):
                w = 48
                s00 = pd.Series(p_eval[:, 0]**2 * T_res0**2).rolling(w, min_periods=1).mean().values
                s11 = pd.Series(p_eval[:, 1]**2 * T_res1**2).rolling(w, min_periods=1).mean().values
                s01 = pd.Series(p_eval[:, 0] * p_eval[:, 1] * T_res0 * T_res1).rolling(w, min_periods=1).mean().values
                trace = s00 + s11
                disc = np.sqrt(np.maximum((s00 - s11)**2 + 4.0 * (s01**2), 0.0))
                lmin_t = np.maximum(0.5 * (trace - disc), 1e-4)

                gamma_safe = np.clip(p_eval, 1e-12, 1.0)
                H_t = -np.sum(gamma_safe * np.log(gamma_safe), axis=1) / np.log(2.0)
                D_t = H_t / lmin_t

                D_lead = D_t[:-1]
                H_lead = H_t[:-1]

                roc_D = float(roc_auc_score(y_fail, D_lead)) if len(np.unique(y_fail)) > 1 else 0.5
                roc_H = float(roc_auc_score(y_fail, H_lead)) if len(np.unique(y_fail)) > 1 else 0.5
                prec, rec, _ = precision_recall_curve(y_fail, D_lead)
                pr_D = float(auc(rec, prec)) if len(np.unique(y_fail)) > 1 else float(np.mean(y_fail))
                return roc_D, pr_D, roc_H

            raw_roc_D, raw_pr_D, raw_roc_H = eval_difficulty(p_test)
            cal_roc_D, cal_pr_D, cal_roc_H = eval_difficulty(p_test_calibrated)

            raw_ece = compute_ece(p_test, y_test)
            cal_ece = compute_ece(p_test_calibrated, y_test)
            raw_nll = compute_nll(p_test, y_test)
            cal_nll = compute_nll(p_test_calibrated, y_test)

            results_per_seed.append({
                "Seed": seed,
                "Architecture": m_name,
                "Optimal_Temperature": round(t_star, 4),
                "Raw_ECE": raw_ece,
                "Calibrated_ECE": cal_ece,
                "ECE_Reduction_Pct": (raw_ece - cal_ece) / max(raw_ece, 1e-12) * 100.0,
                "Raw_NLL": raw_nll,
                "Calibrated_NLL": cal_nll,
                "Raw_ROC_AUC_D": raw_roc_D,
                "Calibrated_ROC_AUC_D": cal_roc_D,
                "Delta_ROC_AUC": cal_roc_D - raw_roc_D,
                "Raw_PR_AUC_D": raw_pr_D,
                "Calibrated_PR_AUC_D": cal_pr_D,
                "Raw_ROC_AUC_H": raw_roc_H,
                "Calibrated_ROC_AUC_H": cal_roc_H
            })

    df_all = pd.DataFrame(results_per_seed)
    
    # Summary across seeds
    summary = []
    for arch in df_all["Architecture"].unique():
        sub = df_all[df_all["Architecture"] == arch]
        summary.append({
            "Architecture": arch,
            "Optimal_Temperature": f"{sub['Optimal_Temperature'].mean():.2f} +/- {sub['Optimal_Temperature'].std():.2f}",
            "Raw_ECE": f"{sub['Raw_ECE'].mean():.4f} +/- {sub['Raw_ECE'].std():.4f}",
            "Calibrated_ECE": f"{sub['Calibrated_ECE'].mean():.4f} +/- {sub['Calibrated_ECE'].std():.4f}",
            "ECE_Reduction_Pct": f"{sub['ECE_Reduction_Pct'].mean():.1f}%",
            "Raw_NLL": f"{sub['Raw_NLL'].mean():.4f}",
            "Calibrated_NLL": f"{sub['Calibrated_NLL'].mean():.4f}",
            "Raw_ROC_AUC_D": f"{sub['Raw_ROC_AUC_D'].mean():.4f} +/- {sub['Raw_ROC_AUC_D'].std():.4f}",
            "Calibrated_ROC_AUC_D": f"{sub['Calibrated_ROC_AUC_D'].mean():.4f} +/- {sub['Calibrated_ROC_AUC_D'].std():.4f}",
            "Delta_ROC_AUC": f"{sub['Delta_ROC_AUC'].mean():+.4f}",
            "Raw_PR_AUC_D": f"{sub['Raw_PR_AUC_D'].mean():.4f}",
            "Calibrated_PR_AUC_D": f"{sub['Calibrated_PR_AUC_D'].mean():.4f}"
        })

    df_sum = pd.DataFrame(summary)
    os.makedirs("reports", exist_ok=True)
    out_csv = "reports/representation_zoo_calibration_intervention.csv"
    df_sum.to_csv(out_csv, index=False)
    print("\n" + "=" * 90)
    print("CALIBRATION INTERVENTION EVALUATION SUMMARY")
    print("=" * 90)
    print(df_sum.to_string(index=False))
    return df_sum


if __name__ == "__main__":
    evaluate_calibration_intervention(n_seeds=5)
