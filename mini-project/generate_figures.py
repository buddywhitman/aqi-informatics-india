import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

output_dir = Path("mini-project/figures")
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("mini-project/outputs/hourly_scored_observations.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Set publication style with large, crisp, readable fonts
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 16,
    "lines.linewidth": 2.0,
    "figure.dpi": 300
})

# --- FIGURE 1: Timeline (Wide, Big Fonts) ---
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.plot(df["timestamp"], df["pollution_index"], color="#1f4e79", linewidth=1.8, label="Pollution Severity Index (proxy)")
watch = df[df["status"] == "Watch"]
ax.scatter(watch["timestamp"], watch["pollution_index"], color="#d97706", s=55, zorder=4, label="Watch Class (Score 35-45)", edgecolors="#78350f", linewidth=1.2)
q95 = df["pollution_index"].quantile(0.95)
ax.axhline(q95, color="#dc2626", linestyle="--", linewidth=1.6, label=f"95th Percentile Threshold ({q95:.1f})")
ax.set_title("Temporal Progression of Pollution Severity Index with Anomaly Screening Flags", pad=12, fontweight="bold")
ax.set_xlabel("Timestamp (IST, November 2024)", labelpad=8)
ax.set_ylabel("Severity Index Proxy (0-500)", labelpad=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
ax.grid(True, linestyle=":", alpha=0.7)
ax.legend(loc="upper right", framealpha=0.95, facecolor="white")
fig.tight_layout()
fig.savefig(output_dir / "fig1_timeline.png", dpi=300)
plt.close(fig)

# --- FIGURE 2: Severity vs Suspicion Scatter ---
fig, ax = plt.subplots(figsize=(8.5, 5.2))
palette = {"Normal": "#10b981", "Watch": "#d97706", "Suspicious": "#dc2626"}
for status in ["Normal", "Watch"]:
    sub = df[df["status"] == status]
    ax.scatter(sub["pollution_index"], sub["suspicion_score"], c=palette[status], label=f"{status} (n={len(sub)})", alpha=0.8, edgecolors="black", linewidth=0.5, s=48)
ax.axhline(30, color="#4b5563", linestyle=":", linewidth=1.5, label="Watch Threshold (Score > 30)")
ax.axhline(65, color="#dc2626", linestyle="--", linewidth=1.8, label="Suspicious Threshold (Score > 65)")
ax.set_title("Empirical Decoupling of Pollution Severity and Suspicion Score", pad=12, fontweight="bold")
ax.set_xlabel("Pollution Severity Index Proxy", labelpad=8)
ax.set_ylabel("AQI Suspicion Score (0-100)", labelpad=8)
ax.grid(True, linestyle=":", alpha=0.7)
ax.legend(loc="upper left", framealpha=0.95, facecolor="white")
fig.tight_layout()
fig.savefig(output_dir / "fig2_scatter.png", dpi=300)
plt.close(fig)

# --- FIGURE 3: Dual-Axis Weather Context ---
fig, ax1 = plt.subplots(figsize=(11, 4.8))
color_poll = "#b91c1c"
color_wind = "#0284c7"

ax1.set_xlabel("Timestamp (IST, November 2024)", labelpad=8)
ax1.set_ylabel("Pollution Severity Index (proxy)", color=color_poll, labelpad=8, fontweight="bold")
line1 = ax1.plot(df["timestamp"], df["pollution_index"], color=color_poll, linewidth=1.8, label="Pollution Index")
ax1.tick_params(axis="y", labelcolor=color_poll)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=4))

ax2 = ax1.twinx()
ax2.set_ylabel("ERA5 Wind Speed (km/h)", color=color_wind, labelpad=8, fontweight="bold")
line2 = ax2.plot(df["timestamp"], df["wind_speed"], color=color_wind, linewidth=1.5, alpha=0.85, linestyle="-", label="Wind Speed (10m)")
ax2.tick_params(axis="y", labelcolor=color_wind)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper left", framealpha=0.95, facecolor="white")
ax1.set_title("Inverse Meteorological Coupling: Surface Wind Stagnation vs. Pollution Spikes", pad=12, fontweight="bold")
ax1.grid(True, linestyle=":", alpha=0.6)
fig.tight_layout()
fig.savefig(output_dir / "fig3_weather.png", dpi=300)
plt.close(fig)

# --- FIGURE 4: Distributions (Skewness & Outliers) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
sns.boxplot(data=df[["pm25", "no2"]], ax=ax1, palette=["#ef4444", "#3b82f6"], width=0.45, fliersize=4)
ax1.set_title("(a) Raw Pollutant Box-and-Whisker", fontsize=13, fontweight="bold")
ax1.set_ylabel("Concentration (ug/m3)", labelpad=6)
ax1.grid(True, linestyle=":", alpha=0.6)

sns.histplot(df["jump_score"].dropna(), bins=28, kde=True, ax=ax2, color="#4f46e5")
q95_jump = df["jump_score"].quantile(0.95)
ax2.axvline(q95_jump, color="#dc2626", linestyle="--", linewidth=1.8, label=f"95th pct ({q95_jump:.1f})")
ax2.set_title("(b) Rolling Robust Jump Score [Delta / (MAD+1)]", fontsize=13, fontweight="bold")
ax2.set_xlabel("Robust MAD Jump Score", labelpad=6)
ax2.set_ylabel("Hourly Frequency", labelpad=6)
ax2.legend(loc="upper right", facecolor="white")
ax2.grid(True, linestyle=":", alpha=0.6)

fig.suptitle("Non-Parametric Feature Distributions Validating Robust Scale Selection", fontsize=15, fontweight="bold")
fig.tight_layout()
fig.savefig(output_dir / "fig4_distributions.png", dpi=300)
plt.close(fig)

# --- FIGURE 5: Nov 12-14 Case Study Zoom ---
case_df = df[(df["timestamp"] >= "2024-11-12 12:00:00") & (df["timestamp"] <= "2024-11-14 12:00:00")].copy()
fig, ax1 = plt.subplots(figsize=(10, 4.6))
ax1.plot(case_df["timestamp"], case_df["pm25"], color="#dc2626", marker="o", markersize=5.5, label="PM2.5 (ug/m3)", linewidth=2.0)
ax1.plot(case_df["timestamp"], case_df["no2"], color="#2563eb", marker="s", markersize=4.5, label="NO2 (ug/m3)", linewidth=1.8)
ax1.set_ylabel("Pollutant Concentration (ug/m3)", labelpad=8, fontweight="bold")
ax1.set_xlabel("Time (IST)", labelpad=8)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
plt.xticks(rotation=25)

ax2 = ax1.twinx()
ax2.plot(case_df["timestamp"], case_df["wind_speed"], color="#059669", linestyle="--", linewidth=1.8, label="Wind Speed (km/h)")
ax2.set_ylabel("Wind Speed (km/h)", color="#059669", labelpad=8, fontweight="bold")
ax2.tick_params(axis="y", labelcolor="#059669")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.95, facecolor="white")
ax1.set_title("Forensic Case Inspection of Peak Pollution Surge (November 12-14, 2024)", pad=12, fontweight="bold")
ax1.grid(True, linestyle=":", alpha=0.6)
fig.tight_layout()
fig.savefig(output_dir / "fig5_case_study.png", dpi=300)
plt.close(fig)

print("Regenerated all 5 figures with large, bold, highly legible fonts and high-res layout!")
