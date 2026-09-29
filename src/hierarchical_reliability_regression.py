"""
hierarchical_reliability_regression.py
======================================
Tests whether calibration (ECE, NLL) predicts downstream task reliability
after controlling for latent-state classification accuracy (F1) across
D independent benchmark worlds.

Model:
  ReliabilityAUC_{d,m} = \alpha_d + \beta_1 * F1_{d,m} + \beta_2 * Calibration_{d,m} + \epsilon_{d,m}
where \alpha_d represents world-level fixed effects and standard errors are
clustered at the independent world level (d = 1, ..., D).
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
from scipy import stats

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


def evaluate_world(world_id: int, device: torch.device):
    seq_len = 16
    N_tr = 1200
    N_eval = 1000

    seed_tr = world_id * 100 + 1
    seed_eval = world_id * 100 + 2

    Z_tr, S_tr, X_tr, T_tr, Y_tr = generate_regime_stream(N_tr, shift_mode="none", seed=seed_tr)
    Z_sc, S_sc, X_sc, T_sc, Y_sc = generate_regime_stream(N_eval, shift_mode="shift_c", seed=seed_eval)

    def make_windows(Z, S):
        X_w = np.array([Z[t - seq_len:t] for t in range(seq_len, len(Z))])
        y_w = S[seq_len - 1 : len(Z) - 1]
        return torch.tensor(X_w, dtype=torch.float32), torch.tensor(y_w, dtype=torch.long)

    X_tr_w, y_tr_w = make_windows(Z_tr, S_tr)
    X_sc_w, y_sc_w = make_windows(Z_sc, S_sc)
    X_tr_dev, y_tr_dev = X_tr_w.to(device), y_tr_w.to(device)

    # 1. HMM
    hmm = GaussianHMMEncoder(n_regimes=2, random_state=seed_tr)
    hmm.fit(Z_tr)
    p_hmm_tr = hmm.filter_forward(Z_tr)[seq_len - 1 : len(Z_tr) - 1]
    p_hmm_sc = hmm.filter_forward(Z_sc)[seq_len - 1 : len(Z_sc) - 1]

    # Helper for fast training
    def train_neural(model, lr=0.01, epochs=10):
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
    p_gru_tr, p_gru_sc = train_neural(gru, lr=0.01, epochs=10)

    # 3. Transformer
    tf = TransformerRegimeEncoder(input_dim=2, embed_dim=32, num_heads=2, n_regimes=2)
    p_tf_tr, p_tf_sc = train_neural(tf, lr=0.008, epochs=10)

    # 4. SSM
    ssm = SSMRegimeEncoder(input_dim=2, state_dim=16, n_regimes=2)
    p_ssm_tr, p_ssm_sc = train_neural(ssm, lr=0.01, epochs=10)

    models_dict = {
        "HMM": (p_hmm_tr, p_hmm_sc),
        "GRU": (p_gru_tr, p_gru_sc),
        "Transformer": (p_tf_tr, p_tf_sc),
        "SSM": (p_ssm_tr, p_ssm_sc)
    }

    sl_tr = slice(seq_len - 1, N_tr - 1)
    sl_sc = slice(seq_len - 1, N_eval - 1)
    y_test_true = y_sc_w.numpy()

    world_records = []
    for m_name, (p_tr, p_sc) in models_dict.items():
        f1 = float(f1_score(y_test_true, np.argmax(p_sc, 1), average="macro"))
        ece = float(compute_ece(p_sc, y_test_true))
        nll = float(compute_nll(p_sc, y_test_true))

        # Downstream model
        X_tr_s = X_tr[sl_tr]
        T_tr_s = T_tr[sl_tr]
        Y_tr_s = Y_tr[sl_tr]

        Phi_tr = np.column_stack([
            p_tr[:, 0], p_tr[:, 1],
            T_tr_s,
            p_tr[:, 0] * T_tr_s, p_tr[:, 1] * T_tr_s,
            X_tr_s[:, 0], X_tr_s[:, 1]
        ])
        downstream = Ridge(alpha=1.0).fit(Phi_tr, Y_tr_s)

        X_sc_s = X_sc[sl_sc]
        T_sc_s = T_sc[sl_sc]
        Y_sc_s = Y_sc[sl_sc]

        Phi_sc = np.column_stack([
            p_sc[:, 0], p_sc[:, 1],
            T_sc_s,
            p_sc[:, 0] * T_sc_s, p_sc[:, 1] * T_sc_s,
            X_sc_s[:, 0], X_sc_s[:, 1]
        ])
        Y_hat = downstream.predict(Phi_sc)
        downstream_loss = (Y_sc_s - Y_hat) ** 2

        # Coupled Gram
        reg_m0 = Ridge(alpha=1.0).fit(X_tr_s, T_tr_s, sample_weight=np.clip(p_tr[:, 0], 1e-4, 1.0))
        reg_m1 = Ridge(alpha=1.0).fit(X_tr_s, T_tr_s, sample_weight=np.clip(p_tr[:, 1], 1e-4, 1.0))

        T_res0 = T_sc_s - reg_m0.predict(X_sc_s)
        T_res1 = T_sc_s - reg_m1.predict(X_sc_s)

        w = 48
        s00 = pd.Series(p_sc[:, 0]**2 * T_res0**2).rolling(w, min_periods=1).mean().values
        s11 = pd.Series(p_sc[:, 1]**2 * T_res1**2).rolling(w, min_periods=1).mean().values
        s01 = pd.Series(p_sc[:, 0] * p_sc[:, 1] * T_res0 * T_res1).rolling(w, min_periods=1).mean().values

        trace = s00 + s11
        disc = np.sqrt(np.maximum((s00 - s11)**2 + 4.0 * (s01**2), 0.0))
        lmin_t = np.maximum(0.5 * (trace - disc), 1e-4)

        gamma_safe = np.clip(p_sc, 1e-12, 1.0)
        H_t = -np.sum(gamma_safe * np.log(gamma_safe), axis=1) / np.log(2.0)
        D_t = H_t / lmin_t

        # Calibration-frozen thresholding
        cal_len = 350
        fail_thresh = float(np.percentile(downstream_loss[:cal_len], 90))

        test_lead_loss = downstream_loss[cal_len:][1:]
        fail_bin = (test_lead_loss > fail_thresh).astype(int)
        D_lead = D_t[cal_len:-1]
        H_lead = H_t[cal_len:-1]

        auc_D = float(roc_auc_score(fail_bin, D_lead)) if len(np.unique(fail_bin)) > 1 else 0.5
        auc_H = float(roc_auc_score(fail_bin, H_lead)) if len(np.unique(fail_bin)) > 1 else 0.5

        world_records.append({
            "World_ID": world_id,
            "Architecture": m_name,
            "F1": f1,
            "ECE": ece,
            "NLL": nll,
            "Reliability_AUC_D": auc_D,
            "Reliability_AUC_H": auc_H
        })

    return world_records


def run_hierarchical_analysis(n_worlds: int = 15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Hierarchical Regression across {n_worlds} Independent Benchmark Worlds on {device}...")

    all_data = []
    for w in range(1, n_worlds + 1):
        print(f"  Evaluating World {w}/{n_worlds}...")
        w_recs = evaluate_world(w, device)
        all_data.extend(w_recs)

    df = pd.DataFrame(all_data)
    os.makedirs("reports", exist_ok=True)
    df.to_csv("reports/representation_zoo_world_evaluations.csv", index=False)

    # Fit clustered regression with World Fixed Effects (within-world demeaned OLS)
    # y_demeaned = y - y_bar_world
    results = []

    # Model 1: Reliability_AUC on F1 alone
    # Model 2: Reliability_AUC on F1 + ECE
    # Model 3: Reliability_AUC on F1 + NLL

    worlds = df["World_ID"].unique()
    n_obs = len(df)

    def fit_fe_ols(y_col, x_cols):
        # Demean by world
        y = df[y_col].values
        X = df[x_cols].values

        y_dm = np.zeros_like(y)
        X_dm = np.zeros_like(X)

        for w in worlds:
            idx = (df["World_ID"] == w).values
            y_dm[idx] = y[idx] - np.mean(y[idx])
            X_dm[idx] = X[idx] - np.mean(X[idx], axis=0)

        # OLS on demeaned data
        beta, residuals, rank, s = np.linalg.lstsq(X_dm, y_dm, rcond=None)
        res = y_dm - X_dm @ beta
        dof = n_obs - len(worlds) - len(x_cols)

        # Clustered variance-covariance matrix (by world)
        bread = np.linalg.inv(X_dm.T @ X_dm)
        meat = np.zeros((len(x_cols), len(x_cols)))
        for w in worlds:
            idx = (df["World_ID"] == w).values
            score_w = X_dm[idx].T @ res[idx]
            meat += np.outer(score_w, score_w)

        # Finite-sample cluster correction
        c_fac = (len(worlds) / (len(worlds) - 1)) * ((n_obs - 1) / max(1, dof))
        V_cluster = c_fac * (bread @ meat @ bread)
        se = np.sqrt(np.diag(V_cluster))

        t_stats = beta / np.maximum(se, 1e-12)
        p_vals = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stats), df=max(1, len(worlds) - 1)))

        # Total variance of demeaned y
        r2 = 1.0 - np.sum(res**2) / np.maximum(np.sum(y_dm**2), 1e-12)
        return beta, se, t_stats, p_vals, r2

    # Fit Model 1: F1 alone
    b_f1, se_f1, t_f1, p_f1, r2_1 = fit_fe_ols("Reliability_AUC_D", ["F1"])
    results.append({
        "Model": "Model 1: F1 Alone",
        "Predictor": "Classification F1",
        "Coefficient": round(float(b_f1[0]), 4),
        "Clustered_SE": round(float(se_f1[0]), 4),
        "t_stat": round(float(t_f1[0]), 4),
        "p_value": round(float(p_f1[0]), 4),
        "Within_R2": round(float(r2_1), 4),
        "N_Worlds": len(worlds),
        "N_Obs": n_obs
    })

    # Fit Model 2: F1 + ECE
    b_fe, se_fe, t_fe, p_fe, r2_2 = fit_fe_ols("Reliability_AUC_D", ["F1", "ECE"])
    results.append({
        "Model": "Model 2: F1 + ECE",
        "Predictor": "Classification F1",
        "Coefficient": round(float(b_fe[0]), 4),
        "Clustered_SE": round(float(se_fe[0]), 4),
        "t_stat": round(float(t_fe[0]), 4),
        "p_value": round(float(p_fe[0]), 4),
        "Within_R2": round(float(r2_2), 4),
        "N_Worlds": len(worlds),
        "N_Obs": n_obs
    })
    results.append({
        "Model": "Model 2: F1 + ECE",
        "Predictor": "Calibration ECE",
        "Coefficient": round(float(b_fe[1]), 4),
        "Clustered_SE": round(float(se_fe[1]), 4),
        "t_stat": round(float(t_fe[1]), 4),
        "p_value": round(float(p_fe[1]), 4),
        "Within_R2": round(float(r2_2), 4),
        "N_Worlds": len(worlds),
        "N_Obs": n_obs
    })

    # Fit Model 3: F1 + NLL
    b_fn, se_fn, t_fn, p_fn, r2_3 = fit_fe_ols("Reliability_AUC_D", ["F1", "NLL"])
    results.append({
        "Model": "Model 3: F1 + NLL",
        "Predictor": "Classification F1",
        "Coefficient": round(float(b_fn[0]), 4),
        "Clustered_SE": round(float(se_fn[0]), 4),
        "t_stat": round(float(t_fn[0]), 4),
        "p_value": round(float(p_fn[0]), 4),
        "Within_R2": round(float(r2_3), 4),
        "N_Worlds": len(worlds),
        "N_Obs": n_obs
    })
    results.append({
        "Model": "Model 3: F1 + NLL",
        "Predictor": "Log-Loss NLL",
        "Coefficient": round(float(b_fn[1]), 4),
        "Clustered_SE": round(float(se_fn[1]), 4),
        "t_stat": round(float(t_fn[1]), 4),
        "p_value": round(float(p_fn[1]), 4),
        "Within_R2": round(float(r2_3), 4),
        "N_Worlds": len(worlds),
        "N_Obs": n_obs
    })

    df_res = pd.DataFrame(results)
    df_res.to_csv("reports/representation_zoo_hierarchical_regression.csv", index=False)
    print("\n" + "=" * 80)
    print("HIERARCHICAL RELIABILITY REGRESSION RESULTS")
    print("=" * 80)
    print(df_res.to_string(index=False))
    return df_res


if __name__ == "__main__":
    run_hierarchical_analysis(n_worlds=15)
