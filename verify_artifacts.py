"""
verify_artifacts.py -- consistency checks between code outputs, generated tables and the v2 manuscript.

    python verify_artifacts.py
Checks: required files exist; every \\input/\\includegraphics in paper/main.tex resolves; result CSVs are non-empty and
finite; headline numbers in reports/bias_law/summary_numbers.json meet the thresholds the paper's text claims; compiled
PDF is not older than its sources; no known credential string is present in the working tree.
"""
import glob
import json
import os
import re
import sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.dirname(__file__))
os.chdir(ROOT)
FAILS = []


def check(name, cond, detail=''):
    print(f"[{'OK' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(name)


# 1. required files
required = ['paper/main.tex', 'paper/main.pdf', 'paper/main_v1.tex', 'paper/aistats2027.sty', 'src/or_dml.py',
            'src/bias_law/bias_law.py', 'src/bias_law/sim.py', 'src/bias_law/run_experiments.py',
            'src/bias_law/real_cities_sensitivity.py', 'src/bias_law/hac_sensitivity.py',
            'src/bias_law/make_figures_tables.py', 'data/processed_clean/combined_hourly_clean.csv',
            'reports/bias_law/summary_numbers.json']
check('required files present', all(os.path.exists(f) for f in required),
      str([f for f in required if not os.path.exists(f)]))

# 2. every include in main.tex resolves
tex = open('paper/main.tex', encoding='utf-8').read()
refs = re.findall(r'\\input\{([^}]+)\}', tex) + re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', tex)
missing = [r for r in refs if not os.path.exists(os.path.join('paper', r if os.path.splitext(r)[1] else r + '.tex'))]
check('all \\input / \\includegraphics targets exist', not missing, str(missing))

# 3. CSVs
bad = []
for f in glob.glob('reports/bias_law/*.csv'):
    d = pd.read_csv(f)
    num = d.select_dtypes('number')
    if d.empty or (not f.endswith('_raw.csv') and not np.isfinite(num.values).all()):  # raw file has intentional NaNs (skipped cells)
        bad.append(f)
check('reports/bias_law/*.csv non-empty and finite', not bad, str(bad))

# 4. headline numbers vs claims in the text
S = json.load(open('reports/bias_law/summary_numbers.json'))
check('E1: corr(law, empirical bias) >= 0.99', S['e1_corr_law_hat'] >= 0.99, f"({S['e1_corr_law_hat']:.4f})")
check('E1: RMSE(law_hat) <= 0.02', S['e1_rmse_law_hat'] <= 0.02, f"({S['e1_rmse_law_hat']:.4f})")
check('E2 (K=3): RMSE <= 0.02', S['e2_rmse_law_hat'] <= 0.02, f"({S['e2_rmse_law_hat']:.4f})")
check('E3 (Table-1 DGP): max |err| <= 0.05', S['e3_max_abs_err_law_true'] <= 0.05, f"({S['e3_max_abs_err_law_true']:.4f})")
check('E4: calibration bound holds in every cell', S['e4_bound_holds_min'] >= 1.0)
check('adjusted-benchmark MAE minimised at bias factor 1', min((k for k in S if k.startswith('adj_mae_bfac_')), key=S.get).endswith('1.0'))

# 5. freshness + hygiene
check('main.pdf not older than main.tex', os.path.getmtime('paper/main.pdf') >= os.path.getmtime('paper/main.tex') - 1)
leak = []
for f in glob.glob('**/*', recursive=True):
    if os.path.isfile(f) and f != 'verify_artifacts.py' and f.split('.')[-1] in ('py', 'md', 'tex', 'json', 'txt', 'yml', 'csv', 'sh') and not f.startswith(('data/', 'archive/')):
        try:
            if ('8535' + '98fc') in open(f, encoding='utf-8', errors='ignore').read():
                leak.append(f)
        except OSError:
            pass
check('no known API-key prefix in working tree', not leak, str(leak))

print(f"\n{'ALL CHECKS PASSED' if not FAILS else str(len(FAILS)) + ' FAILED'}")
sys.exit(1 if FAILS else 0)
