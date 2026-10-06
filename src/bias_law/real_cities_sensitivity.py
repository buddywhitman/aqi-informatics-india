"""
real_cities_sensitivity.py -- residual-state sensitivity analysis on the four Indian cities.

For each city (and with/without frozen-sensor screening) report
    theta_hat  (pooled, soft conditioning on the HMM posterior, purged cross-fitting, Ridge nuisances),
    v_hat, a_hat (treatment shift), Var(T_tilde),
    the robustness value  b* = |theta_hat| Var(T_tilde) / (|a_hat| v_hat)   (outcome shift that would explain the estimate away)
    in outcome-residual SD units, with block-bootstrap intervals.

    python src/bias_law/real_cities_sensitivity.py
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.or_dml import OverlapAwareRegimeDML, PurgedBlockKFold    # noqa: E402
from src.bias_law import bias_law as BL                           # noqa: E402

warnings.filterwarnings('ignore')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'reports', 'bias_law')
os.makedirs(OUT, exist_ok=True)

REGIME_FEATURES = ['temperature', 'wind_speed', 'humidity', 'pressure', 'hour_sin', 'hour_cos']
CONTROLS = ['pm25_lag_1h', 'no2_lag_1h', 'pm25_roll_3h', 'no2_roll_3h', 'temperature', 'humidity', 'wind_speed', 'pressure']


def frozen_mask(x, min_run=6):
    """True for observations belonging to a run of >= min_run identical consecutive values."""
    x = np.asarray(x)
    keep_drop = np.zeros(len(x), bool)
    start = 0
    for i in range(1, len(x) + 1):
        if i == len(x) or x[i] != x[start]:
            if i - start >= min_run:
                keep_drop[start:i] = True
            start = i
    return keep_drop


def analyse(Y, T, X, Z, n_boot=300, block=72, seed=0, gamma=None):
    N = len(Y)
    if gamma is None:
        m = OverlapAwareRegimeDML(n_regimes=2, n_splits=5, embargo_tau=24, reg_alpha=0.05, posterior_mode='smooth',
                                  nuisance_model=Ridge(alpha=1.0), hac_lag=12, random_state=42).fit(Y, T, X, Z)
        g = m.gamma_
    else:
        g = gamma                                           # externally supplied posterior (e.g. best-of-restarts EM)
    F = np.column_stack([X, g[:, 1]])
    tT = np.zeros(N)
    tY = np.zeros(N)
    for tr, va in PurgedBlockKFold(5, 24).split(N):
        tT[va] = T[va] - Ridge(alpha=1.0).fit(F[tr], T[tr]).predict(F[va])
        tY[va] = Y[va] - Ridge(alpha=1.0).fit(F[tr], Y[tr]).predict(F[va])

    def stats(idx):
        gg, TT, YY, XX, TTr = g[idx], tT[idx], tY[idx], X[idx], T[idx]
        th = float((TT @ YY) / (TT @ TT))
        v = float(np.mean(gg[:, 1] * (1 - gg[:, 1])))
        a = float(BL.treatment_shift(TTr, XX, gg)[0])
        vT = float(np.mean(TT ** 2))
        sdY = float(np.std(YY))
        b_star = BL.robustness_value(th, a, np.array([[v]]), vT)
        return th, v, a, vT, sdY, b_star

    th, v, a, vT, sdY, b_star = stats(np.arange(N))
    # benchmark: observed regime shift in Y (coefficient of gamma_hat in Y ~ 1 + X + T + gamma_hat); not identified, only a yardstick
    cY = float(np.linalg.lstsq(np.column_stack([np.ones(N), X, T, g[:, 1]]), Y, rcond=None)[0][-1])
    rng = np.random.default_rng(seed)
    boots = []
    nb = int(np.ceil(N / block))
    for _ in range(n_boot):
        starts = rng.integers(0, N - block, size=nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:N]
        boots.append(stats(idx))
    B = np.array(boots)
    q = lambda col, p: float(np.nanpercentile(B[:, col], p))
    return dict(N=N, theta=th, theta_lo=q(0, 5), theta_hi=q(0, 95), v_hat=v, a_hat=a, var_Ttilde=vT, sd_Ytilde=sdY,
                b_star=b_star, b_star_sd=b_star / sdY, b_star_sd_lo=q(5, 5) / sdY, b_star_sd_hi=q(5, 95) / sdY,
                dg_obs_sd=cY / sdY, b_star_over_obs=float(b_star / max(abs(cY), 1e-12)), bias_at_1sd=float(a * v * sdY / vT), share_at_1sd=float(abs(a * v * sdY / vT) / max(abs(th), 1e-9)))


def main():
    df = pd.read_csv(os.path.join(ROOT, 'data', 'processed_clean', 'combined_hourly_clean.csv'))
    rows = []
    curves = []
    for screen in (False, True):
        for city in ['Delhi', 'Mumbai', 'Bengaluru', 'Kolkata']:
            d = df[df.city == city].dropna(subset=['no2', 'pm25'] + REGIME_FEATURES + CONTROLS).reset_index(drop=True)
            n_raw = len(d)
            if screen:
                d = d[~frozen_mask(d.no2.values, 6)].reset_index(drop=True)
            Y, T = d.pm25.values.astype(float), d.no2.values.astype(float)
            X, Z = d[CONTROLS].values.astype(float), d[REGIME_FEATURES].values.astype(float)
            r = analyse(Y, T, X, Z)
            r.update(city=city, frozen_screen=screen, n_raw=n_raw, n_removed=n_raw - len(d), unique_no2=int(d.no2.nunique()))
            rows.append(r)
            print(f"{city:9s} screen={screen!s:5s} N={r['N']:5d} (removed {r['n_removed']:4d}) theta={r['theta']:8.4f} "
                  f"[{r['theta_lo']:.3f},{r['theta_hi']:.3f}] v={r['v_hat']:.4f} a={r['a_hat']:8.3f} varT~={r['var_Ttilde']:8.3f} "
                  f"b*={r['b_star_sd']:8.2f} SD [{r['b_star_sd_lo']:.2f},{r['b_star_sd_hi']:.2f}]", flush=True)
            for k in np.linspace(0, 3, 31):
                curves.append(dict(city=city, frozen_screen=screen, dg_sd=k, bias=r['a_hat'] * r['v_hat'] * k * r['sd_Ytilde'] / r['var_Ttilde']))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'e5_real_city_sensitivity.csv'), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(OUT, 'e5_real_city_curves.csv'), index=False)


if __name__ == '__main__':
    main()
