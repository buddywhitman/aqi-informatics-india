"""
build_overleaf_pack.py
======================
Builds the standalone, self-contained Overleaf submission package:
  overleaf_submission_pack.zip
and populates the unzipped distribution directory:
  overleaf_submission_pack/

Includes:
  - Canonical LaTeX source: main.tex (strictly 26-page budget, 8-page main text)
  - Conference style file: aistats2027.sty
  - Header formatting package: fancyhdr.sty
  - All 5 canonical publication figures in plots/
"""

import os
import sys
import shutil
import zipfile
import subprocess
import argparse

PACK_DIR = "overleaf_submission_pack"
PACK_ZIP = "overleaf_submission_pack.zip"

FILES_TO_PACK = [
    ("paper/main.tex", "main.tex"),
    ("paper/aistats2027.sty", "aistats2027.sty"),
    ("paper/fancyhdr.sty", "fancyhdr.sty"),
    ("paper/plots/fig1_bias_amplification.png", "plots/fig1_bias_amplification.png"),
    ("paper/plots/fig2_difficulty_frontier.png", "plots/fig2_difficulty_frontier.png"),
    ("paper/plots/fig3_dynamic_irf.png", "plots/fig3_dynamic_irf.png"),
    ("paper/plots/fig4_latent_regime_bench.png", "plots/fig4_latent_regime_bench.png"),
    ("paper/plots/fig5_financial_transfer.png", "plots/fig5_financial_transfer.png"),
]


def build_pack(verify_compile=False):
    print("=" * 70)
    print("BUILDING OVERLEAF SUBMISSION PACK")
    print("=" * 70)

    # 1. Verify source files exist
    for src_path, _ in FILES_TO_PACK:
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Missing required source file: {src_path}")
        if os.path.getsize(src_path) == 0:
            raise ValueError(f"Source file is empty: {src_path}")

    # 2. Populate unzipped directory
    if os.path.exists(PACK_DIR):
        shutil.rmtree(PACK_DIR)
    os.makedirs(os.path.join(PACK_DIR, "plots"), exist_ok=True)

    for src_path, dest_rel in FILES_TO_PACK:
        dest_path = os.path.join(PACK_DIR, dest_rel)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(src_path, dest_path)
        print(f"  [Folder] Added {dest_rel} ({os.path.getsize(dest_path):,} bytes)")

    # 3. Create zip archive
    if os.path.exists(PACK_ZIP):
        os.remove(PACK_ZIP)

    with zipfile.ZipFile(PACK_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src_path, arcname in FILES_TO_PACK:
            zf.write(src_path, arcname=arcname)
            print(f"  [Zip]    Archived {arcname}")

    zip_size = os.path.getsize(PACK_ZIP)
    print("-" * 70)
    print(f"SUCCESS: Packaged {len(FILES_TO_PACK)} files into {PACK_ZIP} ({zip_size / (1024*1024):.2f} MB)")
    print(f"Directory mirror: {PACK_DIR}/")
    print("=" * 70)

    # 4. Optional isolated compilation verification
    if verify_compile:
        print("\nVerifying isolated Overleaf compilation with pdflatex...")
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            for src_path, dest_rel in FILES_TO_PACK:
                target = os.path.join(tmpdir, dest_rel)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copy2(src_path, target)

            cmd = ["pdflatex", "-interaction=nonstopmode", "main.tex"]
            p1 = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True)
            p2 = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True)

            pdf_out = os.path.join(tmpdir, "main.pdf")
            if os.path.exists(pdf_out):
                import pypdf
                reader = pypdf.PdfReader(pdf_out)
                num_pages = len(reader.pages)
                print(f"[OK] Isolated compilation PASSED: {num_pages} pages generated.")
                if num_pages != 26:
                    print(f"[WARNING] Expected 26 pages, but got {num_pages} pages!")
            else:
                print("[ERROR] Isolated compilation FAILED!")
                print(p2.stdout[-1000:])
                sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Overleaf submission package.")
    parser.add_argument("--verify", action="store_true", help="Verify isolated compilation")
    args = parser.parse_args()
    build_pack(verify_compile=args.verify)
