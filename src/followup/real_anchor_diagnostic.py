"""Anchor / linearity diagnostics and MM estimates on real hourly city posteriors (no ground-truth state exists).
Reports: posterior mass near 0/1, Var(gamma), curvature of the gamma-response of NO2 (spline vs linear, adj. for controls),
and OR-DML vs raw-MM vs recalibrated-MM regime effects.  Output: reports/followup/real_anchor_diagnostic.csv"""
import os, sys, warnings
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore')
from src.synthesis.city_stress_hourly import model, REQ, CITIES            # noqa: E402
from src.bias_law import real_cities_sensitivity as R                       # noqa: E402
from src.bias_law.ext_real_hourly_fix import build                          # noqa: E402
from mm_recal import recal                                                  # noqa: E402
from mixture_moment_proto import mm                                         # noqa: E402

def curvature(g, X, T, K=8):
    """Relative drop in RSS when replacing linear-in-gamma by a K-knot spline (controls linear)."""
    base = np.c_[np.ones(len(g)), X]
    def rss(F): r = T - F @ np.linalg.lstsq(F, T, rcond=None)[0]; return float(r @ r)
    knots = np.quantile(g, np.linspace(0, 1, K + 1)[1:-1])
    Fl = np.c_[base, g]; Fs = np.c_[Fl, *[np.maximum(g - k, 0) for k in knots]]
    return (rss(Fl) - rss(Fs)) / rss(Fl), (rss(base) - rss(Fl)) / rss(base)

rows = []
for city in CITIES:
    d = build(city).dropna(subset=REQ)
    Y, T = d.pm25.values.astype(float), d.no2.values.astype(float)
    X, Z = d[R.CONTROLS].values.astype(float), d[R.REGIME_FEATURES].values.astype(float)
    m = model().fit(Y, T, X, Z)
    g = np.asarray(m.weights_)[:, 1]
    # orient so regime 2 has the larger share-weighted mean NO2 (cosmetic)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9); Ts = (T - T.mean()) / T.std(); Ys = (Y - Y.mean()) / Y.std()
    curv, expl = curvature(g, Xs, Ts)
    q = recal(g, Xs, Ts)
    tr = m.theta_regimes_; tr = [tr[k] for k in sorted(tr)] if isinstance(tr, dict) else list(tr)
    th_or = np.asarray(tr, float) * T.std() / Y.std()                # standardised units
    th_mm, th_rc = mm(g, Xs, Ts, Ys), mm(q, Xs, Ts, Ys)
    rows.append(dict(city=city, N=len(d), var_gamma=g.var(), frac_lt05=(g < .05).mean(), frac_gt95=(g > .95).mean(),
                     frac_lt01=(g < .01).mean(), frac_gt99=(g > .99).mean(), curvature=curv, gamma_R2_T=expl,
                     or_dml_1=th_or[0], or_dml_2=th_or[1], mm_raw_1=th_mm[0], mm_raw_2=th_mm[1],
                     mm_recal_1=th_rc[0], mm_recal_2=th_rc[1]))
    print(rows[-1], flush=True)
os.makedirs(os.path.join(ROOT, 'reports', 'followup'), exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(ROOT, 'reports', 'followup', 'real_anchor_diagnostic.csv'), index=False)
print(pd.DataFrame(rows).round(3).T.to_string())
