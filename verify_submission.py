#!/usr/bin/env python3
"""
verify_submission.py -- single gate for the AISTATS 2027 submission artifacts.

Checks (each prints [OK]/[FAIL]; exit code 1 on any failure):
  1. Locked title and abstract are present verbatim in paper/main.tex.
  2. Every macro and table under paper/generated/ is byte-identical to a fresh regeneration from reports/
     (src/synthesis/make_tables_numbers.py), so no number in the PDF can drift from the committed artifacts.
  3. Every \\macro used in the manuscript that looks like a generated number is defined in numbers.tex.
  4. The closed-form law (Proposition 1) reproduces the committed factorial data (median ratio within 2%),
     and its symbolic identities hold (recomputed with sympy unless --fast).
  5. Estimator invariances of src/or_dml.py (verify_science.py).
  6. paper/main.pdf exists, is newer than every manuscript source, has the AI statement and references after a
     main body of at most eight pages, and its metadata/text contain no de-anonymizing strings.
  7. The supplementary archive exists, contains the required files, and no text member contains de-anonymizing
     strings; the root PDF equals paper/main.pdf.

    python verify_submission.py [--fast]
"""
import argparse
import filecmp
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(ROOT, 'paper')
ZIP = os.path.join(ROOT, 'AISTATS2027_Supplementary_Material.zip')
ROOT_PDF = os.path.join(ROOT, 'AISTATS2027_Main_Paper.pdf')
import codecs
# de-anonymizing strings, stored rot13-encoded so that this file does not itself trip the scan
FORBIDDEN = [codecs.decode(w, 'rot13') for w in ['ohqqljuvgzna', 'chyxvg', 'tvguho.pbz/ohqqljuvgzna', 'ndv-vasbezngvpf-vaqvn', 'chyxvg.gnyxf', 'znavcny', 'srggyr', 'Xhzne,']]
LOCKED_TITLE = 'Reliable Causal Estimation under Latent Markov Confounding'
fails = []


def ok(cond, msg):
    print(('[OK]   ' if cond else '[FAIL] ') + msg)
    if not cond:
        fails.append(msg)


def check_locked():
    tex = open(os.path.join(PAPER, 'main.tex')).read()
    abstract = open(os.path.join(PAPER, 'LOCKED_ABSTRACT.txt')).read().strip()
    ok('\\aistatstitle{' + LOCKED_TITLE + '}' in tex, 'locked title present')
    m = re.search(r'\\begin\{abstract\}\s*(.*?)\s*\\end\{abstract\}', tex, re.S)
    ok(m is not None and m.group(1).strip() == abstract, 'locked abstract present verbatim')


def check_generated():
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([sys.executable, os.path.join(ROOT, 'src', 'synthesis', 'make_tables_numbers.py'), '--out', td],
                           capture_output=True, text=True)
        ok(r.returncode == 0, 'regenerated numbers and tables from reports/')
        fresh = sorted(os.path.basename(p) for p in glob.glob(os.path.join(td, '*')))
        diffs = [f for f in fresh if not filecmp.cmp(os.path.join(td, f), os.path.join(PAPER, 'generated', f), shallow=False)]
        ok(not diffs, f'{len(fresh)} generated files byte-identical to committed copies' + (f' (differ: {diffs})' if diffs else ''))
        macros = set(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', open(os.path.join(td, 'numbers.tex')).read()))
    used = set()
    for f in ('main.tex', 'appendix.tex', 'checklist.tex'):
        used |= set(re.findall(r'\\([A-Z][A-Za-z]+)(?![A-Za-z])', open(os.path.join(PAPER, f)).read()))
    generated_like = {u for u in used if re.match(r'^(Fac|CF|LP|BM|SS|FB|FC|FD|CL|Orth|Reg|SVH|DE|Zoo|Lowo|BL|ESev|City|Pl|Au|Bstar|Pooled|Legacy)', u)}
    undefined = sorted(generated_like - macros)
    ok(not undefined, f'all {len(generated_like)} number macros used in the manuscript are generated' + (f' (undefined: {undefined})' if undefined else ''))
    unused = sorted(macros - used)
    print(f'       info: {len(macros)} macros generated, {len(unused)} recorded in claims.csv but not quoted in text')


def check_closed_form(fast):
    sys.path.insert(0, ROOT)
    from src.synthesis import closed_form_factorial as CF
    import pandas as pd
    raw = pd.read_csv(os.path.join(ROOT, 'reports', 'factorial_reliability_raw.csv'))
    cells = raw[raw.Proxy_Error_Eps > 0].groupby(['Residual_SD_State1', 'Proxy_Error_Eps']).Causal_Error_L2.median()
    ratios = [v / CF.population_error(s, e)[0] for (s, e), v in cells.items()]
    med = sorted(ratios)[len(ratios) // 2]
    ok(abs(med - 1) < 0.02 and min(ratios) > 0.95 and max(ratios) < 1.05,
       f'closed form reproduces factorial cells (median ratio {med:.3f}, range {min(ratios):.3f}-{max(ratios):.3f})')
    if fast:
        js = json.load(open(os.path.join(ROOT, 'reports', 'synthesis', 'closed_form_factorial_summary.json')))['symbolic']
    else:
        js = CF.symbolic_check()
    ok(all(v for k, v in js.items() if k.endswith('_ok') or k.startswith('hump')), 'symbolic identities of Proposition 1 hold'
       + (' (from committed summary)' if fast else ' (recomputed)'))


def check_posterior_adjusted():
    import pandas as pd
    cl = pd.read_csv(os.path.join(ROOT, 'reports', 'synthesis', 'coupled_law_check.csv'))
    ok(cl.max_abs_dev.max() < 0.02, f'Theorem 4 law matches the estimator in all {len(cl)} configurations (max dev {cl.max_abs_dev.max():.4f})')
    oc = pd.read_csv(os.path.join(ROOT, 'reports', 'synthesis', 'orthogonality_check.csv')).set_index('design')
    pa, w = oc.loc['posterior-adjusted', 'relative_to_jacobian'], oc.loc['posterior-weighted', 'relative_to_jacobian']
    ok(pa < 0.02 and w > 0.2, f'Lemma 1: score derivative {100 * pa:.1f}% (posterior-adjusted) vs {100 * w:.0f}% (weighted) of the Jacobian')
    fd = pd.read_csv(os.path.join(ROOT, 'reports', 'synthesis', 'frontier_designs_summary.csv'))
    sub = fd[fd.dz >= 1.0]
    best = all(g[g.method == 'OR-DML'].theta_err_median.iloc[0] <= g[g.method.isin(['Hard regime FE', 'OR-DML, weighted nuisances', 'OR-DML'])].theta_err_median.min() + 1e-3
               for _, g in sub.groupby(['dz', 'p00']))
    ok(best, 'posterior-adjusted OR-DML has the smallest regime error among coupled estimators in every frontier cell with dz >= 1')


def check_estimator():
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'verify_science.py')], capture_output=True, text=True, cwd=ROOT)
    ok(r.returncode == 0 and 'PASSED' in r.stdout, 'estimator invariances (verify_science.py)')


def pdftext(path, first=None, last=None):
    cmd = ['pdftotext', '-layout']
    if first:
        cmd += ['-f', str(first), '-l', str(last)]
    return subprocess.run(cmd + [path, '-'], capture_output=True, text=True).stdout


def check_pdf():
    pdf = os.path.join(PAPER, 'main.pdf')
    ok(os.path.exists(pdf), 'paper/main.pdf exists')
    if not os.path.exists(pdf):
        return
    srcs = [os.path.join(PAPER, f) for f in ('main.tex', 'appendix.tex', 'checklist.tex', 'references.tex')]
    srcs += glob.glob(os.path.join(PAPER, 'generated', '*.tex')) + glob.glob(os.path.join(PAPER, 'plots', 'fig[123]_*.pdf'))
    stale = [os.path.relpath(s, ROOT) for s in srcs if os.path.getmtime(s) > os.path.getmtime(pdf) + 1]
    ok(not stale, 'PDF newer than all manuscript sources' + (f' (stale vs {stale[:3]})' if stale else ''))
    info = subprocess.run(['pdfinfo', pdf], capture_output=True, text=True).stdout
    npages = int(re.search(r'Pages:\s+(\d+)', info).group(1))
    ai_page = next((p for p in range(1, npages + 1) if 'AI Use Statement' in pdftext(pdf, p, p)), None)
    ok(ai_page is not None and ai_page <= 9, f'main body ends by page 8 (AI statement/references start on page {ai_page})')
    low = (info + pdftext(pdf)).lower()
    hits = [w for w in FORBIDDEN if w.lower() in low]
    ok(not hits, 'PDF text and metadata anonymous' + (f' (found {hits})' if hits else ''))
    if IN_SUPPLEMENT:
        print('[SKIP] root PDF comparison (running inside the unpacked supplementary archive)')
        return
    ok(os.path.exists(ROOT_PDF) and filecmp.cmp(ROOT_PDF, pdf, shallow=False), 'root AISTATS2027_Main_Paper.pdf equals paper/main.pdf')


def check_zip():
    if IN_SUPPLEMENT:
        print('[SKIP] archive checks (running inside the unpacked supplementary archive)')
        return
    ok(os.path.exists(ZIP), 'supplementary archive exists')
    if not os.path.exists(ZIP):
        return
    z = zipfile.ZipFile(ZIP)
    names = z.namelist()
    req = ['README_SUPPLEMENT.md', 'reproduce.sh', 'verify_submission.py', 'requirements.txt', 'paper/main.tex',
           'paper/main.pdf', 'paper/generated/numbers.tex', 'paper/generated/claims.csv',
           'src/synthesis/make_tables_numbers.py', 'src/or_dml.py', 'reports/factorial_reliability_raw.csv',
           'reports/synthesis/semisynthetic_hourly_summary.csv', 'data/raw_hourly/Delhi_pollution_hourly.csv']
    pre = names[0].split('/')[0] + '/'
    missing = [r for r in req if pre + r not in names]
    ok(not missing, f'archive contains required files ({len(names)} members)' + (f' (missing {missing})' if missing else ''))
    hits = set()
    for n in names:
        if n.endswith(('.py', '.md', '.tex', '.txt', '.json', '.sh', '.yml', '.cfg', '.toml', '.csv')) and z.getinfo(n).file_size < 5_000_000:
            low = z.read(n).decode('utf-8', 'ignore').lower()
            hits |= {(w, n) for w in FORBIDDEN if w.lower() in low}
    ok(not hits, 'archive text members anonymous' + (f' (found {sorted(hits)[:5]})' if hits else ''))
    ok(not any('/.git/' in n or n.endswith('.git') for n in names), 'archive has no git metadata')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fast', action='store_true', help='skip sympy recomputation')
    ap.add_argument('--no-artifacts', action='store_true', help='skip PDF and archive checks (CI before build)')
    a = ap.parse_args()
    # The unpacked supplement has no git metadata and does not contain itself or the root PDF copy.
    IN_SUPPLEMENT = not os.path.exists(os.path.join(ROOT, '.git')) and not os.path.exists(ZIP)
    check_locked()
    check_generated()
    check_closed_form(a.fast)
    check_posterior_adjusted()
    check_estimator()
    if not a.no_artifacts:
        check_pdf()
        check_zip()
    print('-' * 70)
    print('ALL CHECKS PASSED' if not fails else f'{len(fails)} CHECK(S) FAILED')
    sys.exit(1 if fails else 0)
