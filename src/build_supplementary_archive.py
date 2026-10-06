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
    "SUPPLEMENT_README.md",
    "SUPPLEMENT_ROADMAP.md",
    "SUBMISSION_COMPLIANCE.md",
    "experiments_manifest.json",
    "verify_science.py",
    "verify_artifacts.py",
    "verify_submission_synthesis.py",

    # Paper Source & Compiled PDF
    "paper/main.pdf",
    "paper/main.tex",
    "paper/aistats2027.sty",
    "paper/fancyhdr.sty",

    # Canonical Codebase
    "src/or_dml.py",
    "src/calibration_intervention_zoo.py",
    "src/factorial_reliability_experiment.py",
    "src/update_difficulty_figure.py",
    "src/empirical_falsification_checks.py",
    "src/synthetic_dgp_benchmark.py",
    "src/empirical_evaluation.py",
    "src/train_real_representation_zoo.py",
    "src/hierarchical_reliability_regression.py",
    "src/financial_regime_transfer.py",
    "src/data_pipeline_clean.py",
    "src/regime_intelligence.py",
    # Audited residual-confounding sensitivity lineage
    "src/bias_law/__init__.py",
    "src/bias_law/bias_law.py",
    "src/bias_law/sim.py",
    "src/bias_law/run_experiments.py",
    "src/bias_law/real_cities_sensitivity.py",
    "src/bias_law/hac_sensitivity.py",
    "src/bias_law/make_figures_tables.py",
    "src/bias_law/ext_misspec_crossfit.py",
    "src/bias_law/ext_cp_proof_check.py",
    "src/bias_law/ext_h2_lemma_check.py",
    "src/bias_law/ext_real_hourly_fix.py",
    "src/bias_law/ext_cp_is.py",
    "src/bias_law/ext_hmm_rate.py",
    "src/bias_law/ext_hmm_window.py",
    "src/bias_law/ext_nn_quad.py",
    "docs/V2_CHANGELOG.md",
    # Phase-2 frontier provenance retained after audit
    "docs/RESEARCH_INVESTIGATION_METHODOLOGY_AND_FINDINGS.md",
    "docs/FUTURE_WORK_AND_FRONTIERS.md",
    "src/deep_empirical_evaluations.py",
    "src/adaptive_spectral_optimizer.py",
    "src/real_megacity_semisynthetic_benchmark.py",

    "src/requirements.txt",

    # Final post-audit evidence promoted to supplementary material
    "research/PHANTOM_CAUSAL_RESOLUTION.md",
    "research/POSTERIOR_SECOND_MOMENT_COMPLETION.md",
    "research/COMPLETION_ROBUSTNESS_LIMITS.md",
    "research/DIRECTIONAL_RELIABILITY_OPERATOR.md",
    "research/DUAL_RESOLUTION_PRINCIPLE.md",
    "research/FINAL_REAL_DATA_COLLISION.md",
    "research/REPRESENTATION_ZOO_REANALYSIS.md",
    "research/NOVELTY_COLLISION_AUDIT.md",
    "research/FINAL_AUDIT_CHECKLIST.md",
    "research/FINAL_CANDIDATE_SYNTHESIS.md",
    "research/MASTER_ARTIFACT_INDEX.md",
    "research/UNIFIED_THEOREM_SKETCH.md",
    "research/results/frozen_sensor_sensitivity.csv",
    "research/results/real_transition_diagnostics.csv",
    "research/generated_state_resolution_inflation.py",
    "research/posterior_second_moment_completion.py",
    "research/completion_misspecification.py",
    "research/real_data_collision_panel.py",
    "research/zoo_task_relative_reanalysis.py",

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
    report_files = glob.glob("reports/*.csv") + glob.glob("reports/bias_law/*")

    all_files = sorted(dict.fromkeys(list(REQUIRED_FILES) + report_files))

    missing_required = [p for p in REQUIRED_FILES if not os.path.isfile(p) or os.path.getsize(p) == 0]
    if missing_required:
        raise FileNotFoundError("Required supplementary evidence missing or empty:\n  " + "\n  ".join(missing_required))

    # Deterministic archive: stable ordering, fixed timestamps/permissions, explicit bytes.
    # This makes repeated builds from the same source tree byte-reproducible.
    with zipfile.ZipFile(ARCHIVE_NAME, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for file_path in all_files:
            if not os.path.isfile(file_path):
                continue
            archive_path = os.path.join(BASE_DIR, file_path).replace("\\", "/")
            info = zipfile.ZipInfo(archive_path, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with open(file_path, "rb") as fh:
                payload = fh.read()
            zipf.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
            print(f"  Added: {file_path} -> {archive_path}")

    zip_size_mb = os.path.getsize(ARCHIVE_NAME) / (1024 * 1024)
    print("=" * 80)
    print(f"Successfully packaged {len(all_files)} files into {ARCHIVE_NAME}")
    print(f"Archive Size: {zip_size_mb:.2f} MB")
    print("=" * 80)


if __name__ == "__main__":
    build_archive()
