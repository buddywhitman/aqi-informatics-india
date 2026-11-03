"""Data-driven screening of unusual pollution spikes at one Delhi CPCB station.

Run: python pollution_spike_investigation.py
Optional offline run with already-downloaded, real source data:
  python pollution_spike_investigation.py --air-csv path/to/air.csv \
      --weather-json path/to/weather.json

The default download uses the public 2024 demo endpoint of the India Air
Quality Database (XKDR Forum; CPCB source network) and Open-Meteo ERA5
historical weather. The score is an exploratory screening rule, not an
official Indian AQI, calibrated detector, or proof that a reading is false.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


# The selected period is intentional: the public demo key is restricted to
# 2024 and 10,000 rows per request. One station and one month stay well below
# that limit while retaining a meaningful hourly sequence.
STATION_ID = "site_5024"  # Alipur, Delhi - DPCC
STATION_NAME = "Alipur, Delhi - DPCC"
CITY = "Delhi"
LATITUDE = 28.815329
LONGITUDE = 77.153010
START_DATE = "2024-11-01"
END_DATE = "2024-11-30"
DEMO_KEY = "aqi_demo_wbf92Qx21zX-Wa_Tg8Dx1nXe"
AIR_API = "https://airquality.xkdr.org/v1/measurements"
WEATHER_API = "https://archive-api.open-meteo.com/v1/archive"


def fetch_bytes(url: str, headers: dict[str, str] | None = None) -> bytes:
    """Download bytes with a clear error message and a small retry policy."""
    request_headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AQI-Spike-Screening/1.0)",
        "Accept": "text/csv,application/json,*/*",
    }
    request_headers.update(headers or {})
    request = Request(url, headers=request_headers, method="GET")
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=60) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if isinstance(exc, HTTPError) and exc.code < 500 and exc.code != 429:
                break
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(
        "Could not download a real source dataset. Check your internet "
        f"connection and try again. Last error: {last_error}"
    ) from last_error


def obtain_inputs(air_csv: Path | None, weather_json: Path | None,
                  raw_dir: Path) -> tuple[pd.DataFrame, dict]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if air_csv is None:
        params = {
            "station": STATION_ID,
            "parameter": ["PM2.5", "NO2"],
            "start": START_DATE,
            "end": END_DATE,
            "agg": "hourly",
            "format": "csv",
            "limit": 10000,
        }
        url = AIR_API + "?" + urlencode(params, doseq=True)
        content = fetch_bytes(url, {"Authorization": f"Bearer {DEMO_KEY}"})
        saved_air = raw_dir / "delhi_cpcb_hourly_nov2024.csv"
        saved_air.write_bytes(content)
        air = pd.read_csv(io.BytesIO(content))
    else:
        air = pd.read_csv(air_csv)

    if weather_json is None:
        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
            "timezone": "Asia/Kolkata",
            "wind_speed_unit": "kmh",
        }
        url = WEATHER_API + "?" + urlencode(params)
        content = fetch_bytes(url)
        saved_weather = raw_dir / "delhi_weather_nov2024.json"
        saved_weather.write_bytes(content)
        weather = json.loads(content.decode("utf-8"))
    else:
        weather = json.loads(Path(weather_json).read_text(encoding="utf-8"))
    return air, weather


def prepare_hourly_data(air: pd.DataFrame, weather_json: dict) -> pd.DataFrame:
    required = {"parameter_name", "period_start", "mean"}
    missing = required - set(air.columns)
    if missing:
        raise ValueError(f"Air CSV is missing required columns: {sorted(missing)}")

    air = air.copy()
    air["parameter_name"] = air["parameter_name"].astype(str).str.upper()
    air["period_start"] = pd.to_datetime(air["period_start"], errors="coerce")
    air["mean"] = pd.to_numeric(air["mean"], errors="coerce")
    air = air[air["parameter_name"].isin(["PM2.5", "NO2"])]
    air["pollutant"] = air["parameter_name"].replace({"PM2.5": "pm25", "NO2": "no2"})
    wide = (air.pivot_table(index="period_start", columns="pollutant", values="mean",
                            aggfunc="mean").reset_index()
            .rename(columns={"period_start": "timestamp"}))
    if not {"pm25", "no2"}.issubset(wide.columns):
        raise ValueError("The selected source slice must include both PM2.5 and NO2.")

    hourly = weather_json.get("hourly", {})
    if "time" not in hourly:
        raise ValueError("Weather response has no hourly time series.")
    weather = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"], errors="coerce"),
        "temperature": hourly.get("temperature_2m"),
        "humidity": hourly.get("relative_humidity_2m"),
        "pressure": hourly.get("surface_pressure"),
        "wind_speed": hourly.get("wind_speed_10m"),
    })
    for col in ["temperature", "humidity", "pressure", "wind_speed"]:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")

    df = wide.merge(weather, on="timestamp", how="left", validate="one_to_one")
    df["city"] = CITY
    df["station"] = STATION_NAME
    df = df.sort_values("timestamp").dropna(subset=["timestamp", "pm25", "no2"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No complete PM2.5 and NO2 hourly observations were returned.")
    return df


def score_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Implement the assignment's interpretable city-relative score."""
    df = df.sort_values(["city", "timestamp"]).copy()
    for col in ["pm25", "no2"]:
        city_median = df.groupby("city")[col].transform("median")
        city_iqr = df.groupby("city")[col].transform(
            lambda s: s.quantile(0.75) - s.quantile(0.25)
        ).replace(0, np.nan)
        df[f"{col}_severity"] = (df[col] - city_median) / city_iqr

    df["pollution_score"] = df[["pm25_severity", "no2_severity"]].mean(axis=1)
    df["pollution_index"] = (100 + 25 * df["pollution_score"]).clip(0, 500)

    groups = df.groupby("city", group_keys=False)["pollution_index"]
    df["rolling_median_6h"] = groups.transform(
        lambda s: s.shift(1).rolling(6, min_periods=3).median()
    )
    df["rolling_mad_6h"] = groups.transform(
        lambda s: s.shift(1).rolling(6, min_periods=3).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True
        )
    )
    delta = (df["pollution_index"] - df["rolling_median_6h"]).abs()
    df["jump_score"] = delta / (df["rolling_mad_6h"] + 1)
    df["relative_jump"] = delta / (df["rolling_median_6h"] + 1)

    city_median_wind = df.groupby("city")["wind_speed"].transform("median")
    df["low_wind_flag"] = np.where(
        df["wind_speed"].notna() & city_median_wind.notna(),
        (df["wind_speed"] < 0.6 * city_median_wind).astype(int),
        np.nan,
    )
    df["unsupported_spike"] = (
        (df["relative_jump"] > 1.0) & (df["low_wind_flag"] == 0)
    ).astype(int)

    city_q95 = df.groupby("city")["pollution_index"].transform(lambda s: s.quantile(0.95))
    df["extreme_flag"] = (df["pollution_index"] > city_q95).astype(int)
    df["large_jump_flag"] = (df["relative_jump"] > 1.0).astype(int)
    df["pollutant_disagreement"] = (df["pm25_severity"] - df["no2_severity"]).abs()
    df["disagreement_flag"] = (df["pollutant_disagreement"] > 2).astype(int)
    df["suspicion_score"] = (
        35 * df["extreme_flag"]
        + 35 * df["large_jump_flag"]
        + 20 * df["unsupported_spike"]
        + 10 * df["disagreement_flag"]
    )
    df["status"] = pd.cut(
        df["suspicion_score"], bins=[-1, 30, 65, np.inf],
        labels=["Normal", "Watch", "Suspicious"],
    ).astype("string")
    # The first few rows lack a three-hour history; the assignment's score
    # leaves their jump flags off, so they remain eligible and are documented.
    return df


def save_charts(df: pd.DataFrame, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    try:
        import plotly.express as px
        import plotly.graph_objects as go

        timeline = px.line(
            df, x="timestamp", y="pollution_index",
            title=f"{CITY} Alipur: Pollution Severity Timeline",
            labels={"timestamp": "Time (IST)", "pollution_index": "Pollution Severity Index (proxy)"},
        )
        watch = df[df["status"] == "Watch"]
        if not watch.empty:
            timeline.add_trace(go.Scatter(
                x=watch["timestamp"], y=watch["pollution_index"],
                mode="markers", name="Watch hours",
                marker={"color": "#e6a700", "size": 8, "symbol": "circle-open", "line": {"width": 2}},
                text=watch["suspicion_score"],
                hovertemplate="%{x}<br>Index %{y:.1f}<br>Score %{text}<extra></extra>",
            ))
        suspicious = df[df["status"] == "Suspicious"]
        if not suspicious.empty:
            timeline.add_trace(go.Scatter(
                x=suspicious["timestamp"], y=suspicious["pollution_index"],
                mode="markers", name="Suspicious hours",
                marker={"color": "#c0392b", "size": 10, "symbol": "circle-open", "line": {"width": 2}},
                text=suspicious["suspicion_score"],
                hovertemplate="%{x}<br>Index %{y:.1f}<br>Score %{text}<extra></extra>",
            ))
        timeline.update_layout(template="plotly_white", height=520)
        file = output_dir / "pollution_timeline.html"
        timeline.write_html(file, include_plotlyjs=True)
        saved.append(file.name)

        df_scatter = df.copy()
        df_scatter["jump_size"] = df_scatter["relative_jump"].fillna(0).clip(lower=0.05)
        scatter = px.scatter(
            df_scatter, x="pollution_index", y="suspicion_score", color="status",
            size="jump_size", size_max=18, hover_data=["timestamp", "pm25", "no2", "wind_speed", "relative_jump"],
            color_discrete_map={"Normal": "#2ca02c", "Watch": "#e6a700", "Suspicious": "#c0392b"},
            title=f"{CITY}: Pollution Severity and Suspicion Score",
            labels={"pollution_index": "Pollution Severity Index (proxy)", "suspicion_score": "Suspicion Score"},
        )
        scatter.update_layout(template="plotly_white", height=520)
        file = output_dir / "severity_vs_suspicion.html"
        scatter.write_html(file, include_plotlyjs=True)
        saved.append(file.name)

        weather = go.Figure()
        weather.add_trace(go.Scatter(x=df["timestamp"], y=df["pollution_index"], name="Pollution severity", yaxis="y"))
        weather.add_trace(go.Scatter(x=df["timestamp"], y=df["wind_speed"], name="Wind speed (km/h)", yaxis="y2"))
        weather.update_layout(
            title=f"{CITY}: Pollution and Historical Wind Context", template="plotly_white", height=520,
            yaxis={"title": "Pollution Severity Index (proxy)"},
            yaxis2={"title": "Wind speed (km/h)", "overlaying": "y", "side": "right"},
            hovermode="x unified",
        )
        file = output_dir / "pollution_weather_context.html"
        weather.write_html(file, include_plotlyjs=True)
        saved.append(file.name)
    except ImportError:
        # Keep the script runnable in a minimal Python setup. These PNGs are
        # static; install Plotly for the interactive HTML versions above.
        import matplotlib.pyplot as plt

        colors = df["status"].map({"Normal": "#2ca02c", "Watch": "#e6a700", "Suspicious": "#c0392b"})
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df["timestamp"], df["pollution_index"], color="#274c77", linewidth=1.3)
        mask = df["status"].eq("Watch")
        ax.scatter(df.loc[mask, "timestamp"], df.loc[mask, "pollution_index"], color="#e6a700", label="Watch", zorder=3)
        mask = df["status"].eq("Suspicious")
        ax.scatter(df.loc[mask, "timestamp"], df.loc[mask, "pollution_index"], color="#c0392b", label="Suspicious", zorder=4)
        ax.set(title=f"{CITY} Alipur: Pollution Severity Timeline", xlabel="Time (IST)", ylabel="Pollution Severity Index (proxy)")
        ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
        file = output_dir / "pollution_timeline.png"; fig.savefig(file, dpi=160); plt.close(fig); saved.append(file.name)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df["pollution_index"], df["suspicion_score"], c=colors, alpha=.7)
        ax.set(title="Pollution Severity and Suspicion Score", xlabel="Pollution Severity Index (proxy)", ylabel="Suspicion Score")
        fig.tight_layout(); file = output_dir / "severity_vs_suspicion.png"; fig.savefig(file, dpi=160); plt.close(fig); saved.append(file.name)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df["timestamp"], df["pollution_index"], color="#a23b3b", label="Pollution severity")
        ax.set_ylabel("Pollution Severity Index (proxy)")
        ax2 = ax.twinx(); ax2.plot(df["timestamp"], df["wind_speed"], color="#386fa4", alpha=.75, label="Wind speed (km/h)")
        ax2.set_ylabel("Wind speed (km/h)"); ax.set_title("Pollution and Historical Wind Context")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)
        fig.autofmt_xdate(); fig.tight_layout()
        file = output_dir / "pollution_weather_context.png"; fig.savefig(file, dpi=160); plt.close(fig); saved.append(file.name)
        print("Plotly is not installed; static PNG charts were saved. For interactive HTML, run: pip install plotly")
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description="Investigate unusual hourly Delhi pollution readings using real data.")
    parser.add_argument("--air-csv", type=Path, help="Optional already-downloaded XKDR hourly CSV.")
    parser.add_argument("--weather-json", type=Path, help="Optional already-downloaded Open-Meteo weather JSON.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Folder for results and graphs (default: outputs).")
    args = parser.parse_args()
    raw_dir = args.output_dir / "raw_data"

    try:
        air_raw, weather_raw = obtain_inputs(args.air_csv, args.weather_json, raw_dir)
        df = score_observations(prepare_hourly_data(air_raw, weather_raw))
    except Exception as exc:
        print(f"Project stopped: {exc}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "hourly_scored_observations.csv"
    cases_path = args.output_dir / "suspicious_cases.csv"
    df.to_csv(detail_path, index=False)
    columns = ["timestamp", "pollution_index", "pm25", "no2", "wind_speed", "relative_jump", "suspicion_score", "status"]
    df[df["status"] == "Suspicious"].sort_values("suspicion_score", ascending=False)[columns].head(10).to_csv(cases_path, index=False)
    charts = save_charts(df, args.output_dir)

    counts = df["status"].value_counts().reindex(["Normal", "Watch", "Suspicious"], fill_value=0)
    suspicious = df[df["status"] == "Suspicious"].sort_values("suspicion_score", ascending=False)
    summary = {
        "city": CITY,
        "station": STATION_NAME,
        "period_start": str(df["timestamp"].min()),
        "period_end": str(df["timestamp"].max()),
        "observations": int(len(df)),
        "pm25_observations": int(df["pm25"].notna().sum()),
        "no2_observations": int(df["no2"].notna().sum()),
        "weather_matched_observations": int(df["wind_speed"].notna().sum()),
        "status_counts": {str(k): int(v) for k, v in counts.items()},
        "median_pollution_index_proxy": round(float(df["pollution_index"].median()), 2),
        "max_pollution_index_proxy": round(float(df["pollution_index"].max()), 2),
        "highest_scored_case": None if suspicious.empty else {
            "timestamp": str(suspicious.iloc[0]["timestamp"]),
            "score": int(suspicious.iloc[0]["suspicion_score"]),
            "pollution_index_proxy": round(float(suspicious.iloc[0]["pollution_index"]), 2),
            "pm25": round(float(suspicious.iloc[0]["pm25"]), 2),
            "no2": round(float(suspicious.iloc[0]["no2"]), 2),
            "wind_speed_kmh": None if pd.isna(suspicious.iloc[0]["wind_speed"]) else round(float(suspicious.iloc[0]["wind_speed"]), 2),
        },
        "chart_files": charts,
        "index_note": "Exploratory city-relative proxy; not India's regulatory AQI.",
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDATA-DRIVEN POLLUTION SPIKE SCREENING")
    print(f"Station: {STATION_NAME} | Period: {summary['period_start']} to {summary['period_end']} IST")
    print(f"Usable hourly observations: {len(df):,}; weather matched: {summary['weather_matched_observations']:,}")
    print("\nStatus counts:")
    print(counts.to_string())
    print(f"\nMedian / maximum Pollution Severity Index (proxy): {summary['median_pollution_index_proxy']} / {summary['max_pollution_index_proxy']}")
    if summary["highest_scored_case"]:
        case = summary["highest_scored_case"]
        print("\nHighest-scoring case:")
        print(f"{case['timestamp']} | score {case['score']} | proxy {case['pollution_index_proxy']} | PM2.5 {case['pm25']} | NO2 {case['no2']} | wind {case['wind_speed_kmh']} km/h")
    else:
        print("\nNo observations reached the Suspicious category in this period.")
    print("\nSaved outputs:")
    for name in [detail_path.name, cases_path.name, "run_summary.json", *charts]:
        print(f"  {args.output_dir / name}")
    print("\nInterpretation: a flag means investigate the measurement context; it does not establish that the reading is wrong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
