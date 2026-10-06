"""ext_real_hourly_fix.py -- re-run of the four-city analysis on a TRUE hourly grid.

Finding: data_pipeline_clean.py rounds OpenAQ hh:30Z timestamps with .dt.round('h'); pandas rounds ties to even, so every hh:30 maps to an
EVEN hour, each bin averages two readings, and the inner join with hourly weather keeps only even hours.  The processed file is therefore a
2-hourly series (lag_1h = previous row = 2 h earlier; roll_3h = 6 h), although it is labelled hourly.
Here: floor pollutant timestamps to the hour (pairs each reading with the weather hour that started 30 min earlier: causal), reindex to a regular
hourly grid, forward-fill <= 2 grid hours, build lags on the grid (true 1 h / 3 h), and rerun real_cities_sensitivity.analyse.
-> reports/bias_law/x23_real_hourly_fix.csv      (does not modify the original pipeline outputs)"""
import os, sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data_pipeline_clean import POLLUTANT_BOUNDS, WEATHER_BOUNDS, parse_openaq_utc, RAW_PATH   # noqa: E402
from src.bias_law import real_cities_sensitivity as R                                                # noqa: E402
warnings.filterwarnings('ignore')
OUT = R.OUT


def build(city):
    pdf = pd.read_csv(os.path.join(RAW_PATH, f'{city}_pollution_hourly.csv'))
    wdf = pd.read_csv(os.path.join(RAW_PATH, f'{city}_weather_hourly.csv'))
    pdf['timestamp'] = pd.to_datetime(pdf['timestamp'].apply(parse_openaq_utc), utc=True).dt.floor('h')
    pdf = pdf[pdf.parameter.isin(POLLUTANT_BOUNDS)]
    keep = [pdf[(pdf.parameter == k) & (pdf.value >= lo) & (pdf.value <= hi)] for k, (lo, hi) in POLLUTANT_BOUNDS.items()]
    pdf = pd.concat(keep)
    piv = pdf.pivot_table(index='timestamp', columns='parameter', values='value', aggfunc='mean')
    wdf['timestamp'] = pd.to_datetime(wdf['timestamp'], utc=True).dt.floor('h')
    wdf = wdf.set_index('timestamp')
    for c, (lo, hi) in WEATHER_BOUNDS.items():
        if c in wdf.columns: wdf[c] = wdf[c].clip(lo, hi)
    m = piv.join(wdf, how='inner').sort_index()
    m = m.reindex(pd.date_range(m.index.min(), m.index.max(), freq='h'))
    m[['pm25', 'no2']] = m[['pm25', 'no2']].ffill(limit=2)
    for c in ('pm25', 'no2'):
        m[f'{c}_lag_1h'] = m[c].shift(1)
        m[f'{c}_roll_3h'] = m[c].shift(1).rolling(3, min_periods=1).mean()
    m['hour_sin'] = np.sin(2 * np.pi * m.index.hour / 24.0); m['hour_cos'] = np.cos(2 * np.pi * m.index.hour / 24.0)
    m['city'] = city
    return m


if __name__ == '__main__':
    rows = []
    for screen in (False, True):
        for city in ['Delhi', 'Mumbai', 'Bengaluru', 'Kolkata']:
            d = build(city).dropna(subset=['no2', 'pm25'] + R.REGIME_FEATURES + R.CONTROLS).reset_index()
            n_raw = len(d)
            if screen: d = d[~R.frozen_mask(d.no2.values, 6)].reset_index(drop=True)
            gap = d['index'].diff().dt.total_seconds().div(3600) if 'index' in d else None
            r = R.analyse(d.pm25.values.astype(float), d.no2.values.astype(float), d[R.CONTROLS].values.astype(float), d[R.REGIME_FEATURES].values.astype(float))
            r.update(city=city, frozen_screen=screen, n_raw=n_raw, n_removed=n_raw - len(d), unique_no2=int(d.no2.nunique()),
                     first=str(d['index'].min()) if 'index' in d else '', last=str(d['index'].max()) if 'index' in d else '',
                     frac_hourly_steps=float((gap == 1).mean()) if gap is not None else np.nan)
            rows.append(r); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in ('city', 'frozen_screen', 'N', 'theta', 'theta_lo', 'theta_hi', 'v_hat', 'b_star_sd', 'b_star_sd_lo', 'b_star_sd_hi', 'frac_hourly_steps', 'last')}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'x23_real_hourly_fix.csv'), index=False)
