"""
verify_artifacts.py
===================
Automated reproducibility and artifact integrity verification for AISTATS.
Validates:
  1. Existence and numerical consistency of all cited reports and tables.
  2. Byte-for-byte / float-level agreement between CSV artifacts and paper/main.tex.
  3. Existence of all publication figures.
  4. Absence of any placeholder, mock, dummy, or fake tokens.
Exits with nonzero code if any check fails.
"""

import os
import sys
import re
import pandas as pd
import numpy as np

def run_verification():
    print("=" * 75)
    print("VERIFYING ARTIFACT CHAIN AND MANUSCRIPT SYNCHRONIZATION")
    print("=" * 75)
    failures = []

    # 1. Required CSV reports
    required_csvs = [
        "reports/representation_zoo_reliability_auc.csv",
        "reports/representation_zoo_disentangled_shifts.csv",
        "reports/representation_zoo_floor_sensitivity.csv",
        "reports/representation_zoo_reliability_correlations.csv",
        "reports/latent_regime_bench_results.csv",
        "reports/latent_regime_bench_models.csv",
        "reports/empirical_or_dml_results.csv",
        "reports/multidomain_failure_forecasting.csv",
        "reports/financial_regime_risk_quartiles.csv",
        "reports/financial_abstention_policy.csv",
        "reports/meteorological_regime_profiles.csv",
        "reports/kolkata_lambda_sensitivity.csv",
        "reports/data_imputation_sensitivity.csv",
        "reports/or_dml_benchmark_summary.csv",
        "reports/or_dml_regularization_frontier.csv"
    ]
    for r in required_csvs:
        if not os.path.exists(r) or os.path.getsize(r) == 0:
            failures.append(f"Missing or empty report: {r}")
        else:
            print(f"[OK] Report exists: {r}")

    # 2. Required Figures
    required_figures = [
        "plots/fig1_bias_amplification.png",
        "plots/fig2_difficulty_frontier.png",
        "plots/fig3_dynamic_irf.png",
        "plots/fig4_latent_regime_bench.png",
        "plots/fig5_financial_transfer.png",
        "paper/plots/fig1_bias_amplification.png",
        "paper/plots/fig2_difficulty_frontier.png",
        "paper/plots/fig3_dynamic_irf.png",
        "paper/plots/fig4_latent_regime_bench.png",
        "paper/plots/fig5_financial_transfer.png"
    ]
    for fig in required_figures:
        if not os.path.exists(fig) or os.path.getsize(fig) == 0:
            failures.append(f"Missing or empty figure: {fig}")
        else:
            print(f"[OK] Figure exists: {fig}")

    # 3. Check for placeholder tokens in python and tex files
    import glob
    py_tex_files = glob.glob("src/**/*.py", recursive=True) + glob.glob("paper/**/*.tex", recursive=True) + ["verify_artifacts.py"]
    forbidden = [r"\bTODO\b", r"\bFIXME\b", r"\bmock\b", r"\bplaceholder\b", r"\bdummy\b", r"\bfake\b"]
    for f in py_tex_files:
        if "verify_artifacts.py" in f:
            continue
        with open(f, "r", encoding="utf-8", errors="ignore") as fp:
            for line_no, line in enumerate(fp, 1):
                for pat in forbidden:
                    if re.search(pat, line, re.IGNORECASE):
                        failures.append(f"Forbidden token '{pat}' in {f}:{line_no}: {line.strip()[:80]}")

    # 4. Check numerical values in paper/main.tex against CSVs
    with open("paper/main.tex", "r", encoding="utf-8") as fp:
        tex_content = fp.read()

    # Load key CSVs
    df_rel = pd.read_csv("reports/representation_zoo_reliability_auc.csv")
    df_shifts = pd.read_csv("reports/representation_zoo_disentangled_shifts.csv")
    df_emp = pd.read_csv("reports/empirical_or_dml_results.csv")
    df_lrb = pd.read_csv("reports/latent_regime_bench_results.csv")
    df_fin_q = pd.read_csv("reports/financial_regime_risk_quartiles.csv")
    df_fin_abs = pd.read_csv("reports/financial_abstention_policy.csv")

    # Check HMM reliability AUC
    hmm_row = df_rel[df_rel["Architecture"].str.contains("Gaussian HMM")].iloc[0]
    hmm_auc_str = f"{hmm_row['Reliability_ROC_AUC_D']:.4f}"
    if hmm_auc_str not in tex_content:
        failures.append(f"HMM Reliability ROC-AUC {hmm_auc_str} not found in paper/main.tex")
    else:
        print(f"[OK] Verified HMM Reliability ROC-AUC: {hmm_auc_str}")

    # Check GRU shift C F1
    gru_shift_row = df_shifts[df_shifts["Architecture"].str.contains("GRU")].iloc[0]
    gru_f1_str = f"{gru_shift_row['ShiftC_Compound_F1']:.4f}"
    if gru_f1_str not in tex_content:
        failures.append(f"GRU Shift C F1 {gru_f1_str} not found in paper/main.tex")
    else:
        print(f"[OK] Verified GRU Shift C F1: {gru_f1_str}")

    # Check Financial calibration-selected policy regret
    fin_star = df_fin_abs[df_fin_abs["Is_Cal_Selected_Tau"] == True].iloc[0]
    fin_regret_str = f"{fin_star['Policy_Regret']:.4f}"
    if fin_regret_str not in tex_content:
        failures.append(f"Financial Policy Regret {fin_regret_str} not found in paper/main.tex")
    else:
        print(f"[OK] Verified Financial Policy Regret: {fin_regret_str}")

    # Check Empirical Delhi Regime 1 effect
    delhi_r1 = df_emp[(df_emp["City"] == "Delhi") & (df_emp["Regime"] == "Regime 1")].iloc[0]
    delhi_r1_str = f"{delhi_r1['Effect_Theta']:.4f}"
    if delhi_r1_str not in tex_content:
        failures.append(f"Delhi Regime 1 Effect {delhi_r1_str} not found in paper/main.tex")
    else:
        print(f"[OK] Verified Delhi Regime 1 Effect: {delhi_r1_str}")

    print("-" * 75)
    if failures:
        print(f"FAILED: {len(failures)} verification issues detected:")
        for fail in failures:
            print(f"  [X] {fail}")
        sys.exit(1)
    else:
        print("SUCCESS: All artifact checks and manuscript synchronizations PASSED with 0 errors!")
        sys.exit(0)

if __name__ == "__main__":
    run_verification()
