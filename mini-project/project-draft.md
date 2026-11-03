Yes — make it **a real mini data-science investigation**, not just a dashboard.

# Project idea: **“When Should We Distrust an AQI Spike?”**

### Narrative

> “AQI suddenly becomes dangerous. Is this a genuine pollution event, or is the number suspicious because the recent trend, weather, and neighboring observations do not support it?”

You will use:

- `pandas` for cleaning and rolling features;
- `NumPy` for anomaly scores and normalization;
- `Plotly` for interactive visualizations;
- optionally `scikit-learn` for a simple clustering or classification extension.

Your final output is an **AQI Suspicion Score** rather than a prediction model.

---

# The technical question

For every city-hour, calculate:

1. **How unusual is the current AQI compared with recent hours?**
2. **Does the weather support the pollution spike?**
3. **Are pollutant readings behaving consistently?**
4. **Is the reading unusually extreme compared with that city’s history?**

Then classify each hour as:

```text
Normal
Watch
Suspicious
```

This is technically respectable but still easy to explain.

Your project’s conclusion could be:

> “The highest AQI value is not always the most interesting observation. The most interesting observation is the one that is both extreme and poorly supported by the rest of the data.”

---

# Data available in your repository

Your existing pipeline already uses variables such as:

```text
city
pm25
no2
temperature
wind_speed
humidity
pressure
pm25_lag_1h
no2_lag_1h
pm25_roll_3h
no2_roll_3h
hour_sin
hour_cos
```

The repository also uses regular hourly grids, lag features, rolling features, and NumPy time-of-day encodings. The code-search results may be incomplete, but the repository’s existing preprocessing provides a good starting point.

---

# Notebook structure

## 1. Load and inspect the data

```python name=01_load_data.py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DATA_PATH = "data/processed_clean/combined_hourly_clean.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

print(df.shape)
print(df.columns.tolist())
print(df[["city", "timestamp", "pm25", "no2"]].head())
```

If the timestamp column is already the index or has a different name, adjust that one line.

---

## 2. Create an AQI-like pollution index

If the repository does not contain an official AQI column, create a transparent proxy.

Do **not** present this as the legally calculated Indian AQI. Call it a:

> **Pollution Severity Index**

```python name=02_pollution_index.py
pollutants = ["pm25", "no2"]

for col in pollutants:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Simple standardized pollutant scores
for col in pollutants:
    city_median = df.groupby("city")[col].transform("median")
    city_iqr = (
        df.groupby("city")[col]
        .transform(lambda x: x.quantile(0.75) - x.quantile(0.25))
    )

    city_iqr = city_iqr.replace(0, np.nan)

    df[f"{col}_severity"] = (df[col] - city_median) / city_iqr

# Combine the pollutants
df["pollution_score"] = df[
    ["pm25_severity", "no2_severity"]
].mean(axis=1)

# Rescale only for presentation
df["pollution_index"] = (
    100 + 25 * df["pollution_score"]
).clip(lower=0, upper=500)

df[["city", "timestamp", "pollution_index"]].head()
```

This is a good opportunity to explain standardization:

> “We compare each pollutant with what is normal for that city instead of assuming the same concentration means exactly the same thing everywhere.”

---

# 3. Add temporal anomaly features

Use rolling medians rather than only rolling means because medians are less affected by extreme spikes.

```python name=03_temporal_features.py
df = df.sort_values(["city", "timestamp"]).copy()

grouped = df.groupby("city", group_keys=False)

df["rolling_median_6h"] = grouped["pollution_index"].transform(
    lambda s: s.shift(1).rolling(6, min_periods=3).median()
)

df["rolling_mad_6h"] = grouped["pollution_index"].transform(
    lambda s: (
        s.shift(1)
        .rolling(6, min_periods=3)
        .apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    )
)

df["jump_score"] = (
    (df["pollution_index"] - df["rolling_median_6h"]).abs()
    / (df["rolling_mad_6h"] + 1)
)

df["relative_jump"] = (
    (df["pollution_index"] - df["rolling_median_6h"]).abs()
    / (df["rolling_median_6h"] + 1)
)
```

Explain it simply:

> “A value is suspicious when it is much larger than the recent local pattern, not merely because it is numerically large.”

---

# 4. Add weather-support features

A pollution spike is more believable when the weather gives some reason for accumulation.

For example, low wind often makes pollutant accumulation more plausible. This is a simplified exploratory assumption, not a complete atmospheric model.

```python name=04_weather_features.py
weather_cols = ["wind_speed", "humidity", "temperature", "pressure"]

for col in weather_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# City-specific wind reference
df["city_median_wind"] = df.groupby("city")["wind_speed"].transform("median")

# Low wind means weaker atmospheric dispersion
df["low_wind_flag"] = (
    df["wind_speed"] < 0.6 * df["city_median_wind"]
).astype(int)

# A sudden pollution spike during normal/high wind is more suspicious
df["unsupported_spike"] = (
    (df["relative_jump"] > 1.0)
    & (df["low_wind_flag"] == 0)
).astype(int)
```

The class does not need to understand atmospheric physics. Your explanation can be:

> “We are not proving that weather caused the pollution. We are checking whether the weather makes the observed spike more or less plausible.”

---

# 5. Build the AQI Suspicion Score

Use an interpretable formula.

```python name=05_suspicion_score.py
df["extreme_flag"] = (
    df["pollution_index"]
    > df.groupby("city")["pollution_index"].transform(
        lambda x: x.quantile(0.95)
    )
).astype(int)

df["large_jump_flag"] = (df["relative_jump"] > 1.0).astype(int)

df["pollutant_disagreement"] = (
    df["pm25_severity"] - df["no2_severity"]
).abs()

df["suspicion_score"] = (
    35 * df["extreme_flag"]
    + 35 * df["large_jump_flag"]
    + 20 * df["unsupported_spike"]
    + 10 * (df["pollutant_disagreement"] > 2).astype(int)
)

df["status"] = pd.cut(
    df["suspicion_score"],
    bins=[-1, 30, 65, np.inf],
    labels=["Normal", "Watch", "Suspicious"]
)

df["status"].value_counts()
```

This is the technical centerpiece. It uses:

- quantiles;
- Boolean feature engineering;
- grouped statistics;
- a transparent scoring model.

You can say:

> “We intentionally use a transparent scoring system instead of a black-box model, because the purpose is to explain why a reading was flagged.”

---

# 6. Plot the city’s pollution timeline

```python name=06_plot_timeline.py
city = "Delhi"

plot_df = (
    df[df["city"] == city]
    .dropna(subset=["timestamp", "pollution_index", "suspicion_score"])
    .sort_values("timestamp")
)

fig = px.line(
    plot_df,
    x="timestamp",
    y="pollution_index",
    title=f"{city}: Pollution Severity Over Time",
    labels={
        "timestamp": "Time",
        "pollution_index": "Pollution Severity Index"
    }
)

suspicious = plot_df[plot_df["status"] == "Suspicious"]

fig.add_trace(
    go.Scatter(
        x=suspicious["timestamp"],
        y=suspicious["pollution_index"],
        mode="markers",
        name="Suspicious hours",
        marker=dict(
            color="red",
            size=9,
            symbol="circle-open",
            line=dict(width=2)
        ),
        text=suspicious["suspicion_score"].round(0),
        hovertemplate=(
            "Time: %{x}<br>"
            "Pollution index: %{y:.1f}<br>"
            "Suspicion score: %{text}"
            "<extra></extra>"
        )
    )
)

fig.update_layout(template="plotly_white", height=500)
fig.show()
```

This gives you the main presentation plot: a normal timeline with red warning points.

---

# 7. Plot the “why was this flagged?” relationship

A very understandable technical plot is:

- x-axis: pollution severity;
- y-axis: suspicion score;
- color: status;
- size: wind speed or jump magnitude.

```python name=07_scatter_plot.py
fig = px.scatter(
    plot_df,
    x="pollution_index",
    y="suspicion_score",
    color="status",
    size="relative_jump",
    hover_data=[
        "timestamp",
        "pm25",
        "no2",
        "wind_speed",
        "relative_jump"
    ],
    color_discrete_map={
        "Normal": "#2ca02c",
        "Watch": "#ffbf00",
        "Suspicious": "#d62728"
    },
    title=f"{city}: Extreme Pollution vs Suspicion",
    labels={
        "pollution_index": "Pollution Severity Index",
        "suspicion_score": "Suspicion Score"
    }
)

fig.update_layout(template="plotly_white")
fig.show()
```

Your storytelling point:

> “The most polluted observations are not automatically the most suspicious. Suspicion depends on how inconsistent the observation is with the surrounding evidence.”

---

# 8. Show weather context interactively

```python name=08_weather_context.py
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=plot_df["timestamp"],
        y=plot_df["pollution_index"],
        name="Pollution index",
        line=dict(color="firebrick", width=2),
        yaxis="y"
    )
)

fig.add_trace(
    go.Scatter(
        x=plot_df["timestamp"],
        y=plot_df["wind_speed"],
        name="Wind speed",
        line=dict(color="royalblue", width=1.5),
        yaxis="y2"
    )
)

fig.update_layout(
    title=f"{city}: Pollution Spikes and Weather Context",
    template="plotly_white",
    yaxis=dict(title="Pollution Severity Index"),
    yaxis2=dict(
        title="Wind speed",
        overlaying="y",
        side="right"
    ),
    hovermode="x unified",
    height=500
)

fig.show()
```

This plot can support a simple comparison:

- high pollution + low wind = more plausible accumulation;
- high pollution + normal wind + sudden isolated jump = more suspicious.

---

# 9. Make a small result table

```python name=09_investigation_table.py
investigation = (
    plot_df[plot_df["status"] == "Suspicious"]
    .sort_values("suspicion_score", ascending=False)
    [[
        "timestamp",
        "pollution_index",
        "pm25",
        "no2",
        "wind_speed",
        "relative_jump",
        "suspicion_score",
        "status"
    ]]
    .head(10)
)

investigation
```

Present the top result as a “case file”:

```text
Case #1
Time: 2024-...
Pollution index: 387
Suspicion score: 90
Reason:
- top 5% city pollution
- more than 100% jump from recent baseline
- wind did not fall
- PM2.5 and NO₂ disagreed strongly
```

---

# Stronger extension: compare cities

This gives you one impressive summary plot.

```python name=10_city_summary.py
city_summary = (
    df.groupby("city")
    .agg(
        median_pollution=("pollution_index", "median"),
        max_pollution=("pollution_index", "max"),
        suspicious_hours=(
            "status",
            lambda x: (x == "Suspicious").sum()
        ),
        total_hours=("status", "size")
    )
    .reset_index()
)

city_summary["suspicious_percent"] = (
    100 * city_summary["suspicious_hours"]
    / city_summary["total_hours"]
)

fig = px.bar(
    city_summary.sort_values("suspicious_percent", ascending=False),
    x="city",
    y="suspicious_percent",
    color="suspicious_percent",
    color_continuous_scale="Reds",
    title="Percentage of Hours Flagged as Suspicious",
    labels={
        "suspicious_percent": "Suspicious hours (%)",
        "city": "City"
    },
    text_auto=".1f"
)

fig.update_layout(template="plotly_white")
fig.show()
```

Be careful with the interpretation:

> “This does not mean one city has fake data. It means our simple screening rules flag more observations there.”

---

# Presentation storyline

## Opening

Show this:

```text
AQI: 385

Would you trust this number?
```

Ask the class to vote.

## Then reveal

> “An AQI value is not just a number. It is the output of a measurement system.”

## Technical section

Explain the pipeline visually:

```text
Raw hourly data
      ↓
Clean and align timestamps
      ↓
Create rolling baselines
      ↓
Measure unusual jumps
      ↓
Compare with weather
      ↓
Calculate suspicion score
      ↓
Interactive Plotly investigation
```

## Main visual sequence

1. Pollution timeline.
2. Red suspicious points.
3. Zoom into the largest spike.
4. Show wind and pollution together.
5. Show the suspicion-score formula.
6. Compare cities.

## Final conclusion

> “Our project does not decide whether an AQI value is true or false. It identifies observations that deserve a second look.”

That is a very defensible conclusion.

---

# How to connect it to your thesis without exposing the difficult mathematics

You can mention:

> “The larger research project studies how hidden environmental regimes and imperfect observations affect causal conclusions. For this mini-project, we simplified that idea into a practical data-quality question: when should an observed pollution pattern be treated cautiously?”

That makes the mini-project feel connected to serious work without requiring the class to understand your full theory.

---

# Suggested title

## **AQI Under Investigation**
### *Detecting pollution spikes that deserve a second look*

Alternative title:

## **When Numbers Lie**
### *An explainable data-science investigation of suspicious air-quality readings*

The second title is more entertaining; the first sounds more academic.

One caution: your repository search only returned a limited subset of matching files, so the exact processed-data column names should be checked before running the notebook.