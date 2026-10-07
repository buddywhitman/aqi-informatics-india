#!/usr/bin/env python3
"""
build_supplementary_archive.py -- builds the anonymized supplementary archive
AISTATS2027_Supplementary_Material.zip from the repository working tree.

Contents: manuscript sources and PDF, every generated table/macro, all source code, all result files
(reports/*.csv|json and reports/bias_law, reports/synthesis), the raw hourly sensor data needed to rerun the
real-data analyses, earlier manuscript lineages and the research ledger. Third-party reference PDFs, git
metadata, caches and legacy binaries are excluded. Timestamps are fixed so the archive is reproducible.

    python src/build_supplementary_archive.py
"""
import glob
import os
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, 'AISTATS2027_Supplementary_Material.zip')
PREFIX = 'AISTATS2027_Supplementary_Material/'
FIXED_TIME = (2026, 10, 7, 0, 0, 0)

INCLUDE = [
    'README_SUPPLEMENT.md', 'reproduce.sh', 'verify_submission.py', 'verify_science.py', 'requirements.txt',
    'paper/main.tex', 'paper/appendix_src.tex', 'paper/appendix.tex', 'paper/build_appendix.py', 'paper/prune_bib.py',
    'paper/checklist.tex', 'paper/references_all.tex', 'paper/references.tex', 'paper/LOCKED_ABSTRACT.txt',
    'paper/aistats2027.sty', 'paper/fancyhdr.sty', 'paper/main.pdf',
    'paper/generated/*.tex', 'paper/generated/*.csv', 'paper/generated/biaslaw/*.tex',
    'paper/plots/fig1_phantom_resolution.*', 'paper/plots/fig2_bias_law.*', 'paper/plots/fig3_estimators.*',
    'src/*.py', 'src/bias_law/*.py', 'src/synthesis/*.py',
    'reports/*.csv', 'reports/bias_law/*', 'reports/synthesis/*',
    'data/raw_hourly/*.csv', 'data/processed_clean/*.csv',
    'archive/manuscript_lineages/*.tex', 'docs/V2_CHANGELOG.md',
    'research/*.md', 'research/*.py', 'research/*.csv', 'research/*.json',
]


def main():
    files = []
    for pat in INCLUDE:
        hits = sorted(glob.glob(os.path.join(ROOT, pat)))
        if not hits:
            if pat.startswith(('paper/', 'src/', 'README', 'reproduce', 'verify')):
                raise SystemExit(f'required pattern matched nothing: {pat}')
            print(f'note: optional pattern matched nothing: {pat}')
            continue
        files += [h for h in hits if os.path.isfile(h) and '__pycache__' not in h]
    files = sorted(set(files))
    if os.path.exists(OUT):
        os.remove(OUT)
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            rel = os.path.relpath(f, ROOT)
            info = zipfile.ZipInfo(PREFIX + rel, date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if rel.endswith('.sh') else 0o644) << 16
            with open(f, 'rb') as fh:
                z.writestr(info, fh.read())
    print(f'wrote {os.path.relpath(OUT, ROOT)}: {len(files)} files, {os.path.getsize(OUT) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
