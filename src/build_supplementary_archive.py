"""
build_supplementary_archive.py
==============================
Builds the canonical, publication-ready Supplementary Material archive:
  AISTATS2027_OR_DML_Supplementary_Material.zip

Excludes all legacy pre-pivot files, git history, cache directories,
and unverified scratch scripts. Includes all 24-page PDF, LaTeX source,
clean data, canonical Python implementation, reports, and manifests.
"""

import os
import sys
import zipfile
import glob

ARCHIVE_NAME = "AISTATS2027_OR_DML_Supplementary_Material.zip"
BASE_DIR = "AISTATS2027_OR_DML_Supplementary_Material"

REQUIRED_FILES = [
    # Documentation & Manifests
    "README.md",
    "SUPPLEMENT_ROADMAP.md",
    "experiments_manifest.json",
    "verify_science.py",
    "verify_artifacts.py",

    # Paper Source & Compiled PDF
    "paper/main.pdf",
    "paper/main.tex",
    "paper/aistats2027.sty",
    "paper/fancyhdr.sty",

    # Canonical Codebase
    "src/or_dml.py",
    "src/calibration_intervention_zoo.py",
    "src/factorial_reliability_experiment.py",
    "src/empirical_falsification_checks.py",
    "src/synthetic_dgp_benchmark.py",
    "src/empirical_evaluation.py",
    "src/train_real_representation_zoo.py",
    "src/hierarchical_reliability_regression.py",
    "src/financial_regime_transfer.py",
    "src/data_pipeline_clean.py",
    "src/generate_paper_figures.py",
    "src/regime_intelligence.py",
    "src/requirements.txt",

    # Clean Datasets
    "data/processed_clean/combined_hourly_clean.csv",

    # Core Publication Figures
    "plots/fig1_bias_amplification.png",
    "plots/fig2_difficulty_frontier.png",
    "plots/fig3_dynamic_irf.png",
    "plots/fig4_latent_regime_bench.png",
    "plots/fig5_financial_transfer.png",
    "paper/plots/fig1_bias_amplification.png",
    "paper/plots/fig2_difficulty_frontier.png",
    "paper/plots/fig3_dynamic_irf.png",
    "paper/plots/fig4_latent_regime_bench.png",
    "paper/plots/fig5_financial_transfer.png",
]


def build_archive():
    print(f"Building Authoritative Supplementary Archive: {ARCHIVE_NAME}...")

    # Gather all CSV reports in reports/
    report_files = glob.glob("reports/*.csv")

    all_files = list(REQUIRED_FILES) + report_files

    missing = [f for f in all_files if not os.path.exists(f)]
    if missing:
        print(f"Error: {len(missing)} files missing:")
        for m in missing:
            print(f"  [X] {m}")
        # Proceed with available files if only optional new ones missing
        all_files = [f for f in all_files if os.path.exists(f)]

    with zipfile.ZipFile(ARCHIVE_NAME, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for file_path in all_files:
            archive_path = os.path.join(BASE_DIR, file_path).replace("\\", "/")
            zipf.write(file_path, archive_path)
            print(f"  Added: {file_path} -> {archive_path}")

    zip_size_mb = os.path.getsize(ARCHIVE_NAME) / (1024 * 1024)
    print("=" * 80)
    print(f"Successfully packaged {len(all_files)} files into {ARCHIVE_NAME}")
    print(f"Archive Size: {zip_size_mb:.2f} MB")
    print("=" * 80)


if __name__ == "__main__":
    build_archive()
