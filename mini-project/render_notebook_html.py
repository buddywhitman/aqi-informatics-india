import json
import pandas as pd
import numpy as np
from pathlib import Path

# Load dataset
data_path = Path("mini-project/outputs/hourly_scored_observations.csv")
df = pd.read_csv(data_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

GIST_ID = "57e1f6b40953a2eec1cb1fa176c93fee"
COLAB_URL = f"https://colab.research.google.com/gist/buddywhitman/{GIST_ID}/investigation_notebook.ipynb"
RAW_DATA_URL = f"https://gist.githubusercontent.com/buddywhitman/{GIST_ID}/raw/hourly_scored_observations.csv"

# Cell 1 output: head table
head_table = df[['timestamp', 'pm25', 'no2', 'wind_speed', 'pollution_index', 'status']].head().to_html(
    classes="dataframe", index=True, border=0
)

# Cell 2 output: summary stats table
summary_stats = pd.DataFrame({
    'Metric': ['Median', 'IQR (Q3 - Q1)', 'Skewness', 'Minimum', 'Maximum', '95th Percentile'],
    'PM2.5 (ug/m3)': [
        df['pm25'].median(),
        df['pm25'].quantile(0.75) - df['pm25'].quantile(0.25),
        df['pm25'].skew(),
        df['pm25'].min(),
        df['pm25'].max(),
        df['pm25'].quantile(0.95)
    ],
    'NO2 (ug/m3)': [
        df['no2'].median(),
        df['no2'].quantile(0.75) - df['no2'].quantile(0.25),
        df['no2'].skew(),
        df['no2'].min(),
        df['no2'].max(),
        df['no2'].quantile(0.95)
    ],
    'Wind Speed (km/h)': [
        df['wind_speed'].median(),
        df['wind_speed'].quantile(0.75) - df['wind_speed'].quantile(0.25),
        df['wind_speed'].skew(),
        df['wind_speed'].min(),
        df['wind_speed'].max(),
        df['wind_speed'].quantile(0.95)
    ]
}).round(2)
stats_table = summary_stats.to_html(classes="dataframe", index=False, border=0)

# Cell 3 output: rolling and jump metrics
df['calc_roll_med'] = df['pollution_index'].shift(1).rolling(6, min_periods=3).median()
delta = (df['pollution_index'] - df['calc_roll_med']).abs()
df['calc_rel_jump'] = delta / (df['calc_roll_med'] + 1)
q_rel_jump = df['calc_rel_jump'].quantile([0.5, 0.75, 0.9, 0.95, 0.99, 1.0]).round(3)
cell3_text = f"Max relative jump observed on rescaled index: {round(df['calc_rel_jump'].max(), 3)}\nQuantiles of relative jump:\n{q_rel_jump.to_string()}"

# Cell 4 output: counts
status_counts = df['status'].value_counts()
watch_df = df[df['status'] == 'Watch']
watch_scores = watch_df['suspicion_score'].value_counts()
cell4_text = f"Classification Breakdown (701 observation hours):\n{status_counts.to_string()}\n\nWatch rows score value counts:\n{watch_scores.to_string()}"

# Cell 7 output: experiment
def run_threshold_experiment(jump_threshold=1.0, extreme_pct=0.95):
    test_df = df.copy()
    q_val = test_df['pollution_index'].quantile(extreme_pct)
    ext_flag = (test_df['pollution_index'] > q_val).astype(int)
    jump_flag = (test_df['relative_jump'] > jump_threshold).astype(int)
    low_wind = (test_df['wind_speed'] < 0.6 * test_df['wind_speed'].median()).astype(int)
    unsupp = (jump_flag & (1 - low_wind)).astype(int)
    disag = ((test_df['pm25_severity'] - test_df['no2_severity']).abs() > 2.0).astype(int)
    score = 35 * ext_flag + 35 * jump_flag + 20 * unsupp + 10 * disag
    status = pd.cut(score, bins=[-1, 30, 65, np.inf], labels=['Normal', 'Watch', 'Suspicious'])
    counts = status.value_counts()
    return f"--- Experiment: Jump Threshold = {jump_threshold}, Extreme Percentile = {int(extreme_pct*100)}th ---\n{counts.to_string()}"

cell7_text = run_threshold_experiment(1.0, 0.95) + "\n\n" + run_threshold_experiment(0.35, 0.95)

# Cell 8 output: case study
case = df[(df['timestamp'] >= '2024-11-12 18:00:00') & (df['timestamp'] <= '2024-11-13 12:00:00')].copy()
case_table = case[['timestamp', 'pm25', 'no2', 'wind_speed', 'pollution_index', 'relative_jump', 'suspicion_score', 'status']].head(10).to_html(
    classes="dataframe", index=False, border=0
)

# HTML template for Google Colab / JupyterLab interactive experience
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>investigation_notebook.ipynb - Google Colab / JupyterLab</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Roboto+Mono:wght@400;500&display=swap');
  body {{
    font-family: 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #f8f9fa;
    color: #202124;
  }}
  code, pre, .font-mono {{ font-family: 'Roboto Mono', monospace; }}
  .cell {{
    background: #ffffff;
    border: 1px solid #dadce0;
    border-radius: 8px;
    margin-bottom: 16px;
    transition: box-shadow 0.15s ease;
  }}
  .cell:hover {{
    box-shadow: 0 1px 3px rgba(60,64,67,0.15);
  }}
  .cell-input {{
    background: #f8f9fa;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    border-bottom: 1px solid #edf2f7;
    padding: 12px 16px;
  }}
  .cell-output {{
    padding: 14px 16px;
    overflow-x: auto;
  }}
  table.dataframe {{
    border-collapse: collapse;
    width: 100%;
    font-size: 12px;
    font-family: 'Roboto Mono', monospace;
  }}
  table.dataframe th {{
    background: #f1f3f4;
    color: #202124;
    padding: 6px 12px;
    border: 1px solid #dadce0;
    text-align: right;
    font-weight: 600;
  }}
  table.dataframe td {{
    padding: 6px 12px;
    border: 1px solid #dadce0;
    text-align: right;
  }}
  table.dataframe tr:nth-child(even) {{
    background-color: #f8f9fa;
  }}
  iframe {{ width: 100%; border: none; min-height: 500px; }}
</style>
</head>
<body class="min-h-screen flex flex-col">

  <!-- Top Colab / Jupyter Header -->
  <header class="bg-white border-b border-gray-200 sticky top-0 z-30 shadow-sm">
    <div class="max-w-6xl mx-auto px-4 sm:px-6 py-2.5 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <!-- Colab Logo -->
        <a href="{COLAB_URL}" target="_blank" title="Open directly in Google Colab">
          <svg class="w-8 h-8 flex-shrink-0 cursor-pointer hover:opacity-85 transition" viewBox="0 0 40 40" fill="none">
            <circle cx="20" cy="20" r="18" fill="#F9AB00" fill-opacity="0.2"/>
            <path d="M12 20C12 15.5817 15.5817 12 20 12C24.4183 12 28 15.5817 28 20C28 24.4183 24.4183 28 20 28" stroke="#E37400" stroke-width="3" stroke-linecap="round"/>
            <path d="M28 20C28 24.4183 24.4183 28 20 28C15.5817 28 12 24.4183 12 20" stroke="#F9AB00" stroke-width="3" stroke-linecap="round"/>
          </svg>
        </a>
        <div>
          <div class="flex items-center space-x-2">
            <span class="font-semibold text-gray-900 text-sm">investigation_notebook.ipynb</span>
            <span class="text-[10px] px-1.5 py-0.5 bg-green-50 text-green-700 border border-green-200 rounded font-medium">Ready</span>
            <span class="text-[10px] px-1.5 py-0.5 bg-gray-100 text-gray-600 rounded">Python 3 (ipykernel)</span>
          </div>
          <div class="flex space-x-4 text-xs text-gray-500 mt-0.5">
            <a href="{COLAB_URL}" target="_blank" class="hover:text-amber-600 font-semibold text-amber-700 flex items-center space-x-1">
              <span>Open in Google Colab &nearr;</span>
            </a>
            <span class="text-gray-300">|</span>
            <span class="hover:text-gray-900 cursor-pointer">File</span>
            <span class="hover:text-gray-900 cursor-pointer">Edit</span>
            <span class="hover:text-gray-900 cursor-pointer">View</span>
            <span class="hover:text-gray-900 cursor-pointer">Runtime</span>
          </div>
        </div>
      </div>
      <div class="flex items-center space-x-2">
        <a href="{COLAB_URL}" target="_blank" class="inline-flex items-center px-3 py-1.5 text-xs font-semibold bg-amber-600 text-white rounded hover:bg-amber-700 shadow-sm transition">
          <svg class="w-3.5 h-3.5 mr-1" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
          Run in Google Colab
        </a>
        <a href="interactive_investigation_dashboard.html" class="px-3 py-1.5 text-xs font-medium bg-slate-800 text-white rounded hover:bg-slate-700 transition">
          Dashboard
        </a>
        <a href="investigation_notebook.ipynb" download class="px-2.5 py-1.5 text-xs font-medium border border-gray-300 text-gray-700 rounded hover:bg-gray-50 transition" title="Download .ipynb">
          .ipynb
        </a>
      </div>
    </div>
  </header>

  <!-- Notebook Body -->
  <main class="flex-1 max-w-5xl w-full mx-auto px-4 sm:px-6 py-6 space-y-4">

    <!-- Cell 0: Header Markdown -->
    <div class="bg-white p-6 rounded-lg border border-gray-200 shadow-sm space-y-3">
      <div class="flex items-center justify-between">
        <h1 class="text-2xl font-bold text-gray-900">Data-Driven Detection of Anomalous Pollution Spikes Using Python</h1>
        <a href="{COLAB_URL}" target="_blank">
          <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
        </a>
      </div>
      <p class="text-sm font-medium text-gray-600">Department of Electrical Engineering &bull; Course: ELE 4449 Introduction to Data Science</p>
      <div class="text-xs text-gray-500 flex flex-wrap gap-x-4 gap-y-1 pt-1 border-t border-gray-100">
        <span><strong>Authors:</strong> Pulkit Kumar, Vijayalakshmi AL, Anant Prasad, Rama Sridhrut</span>
        <span><strong>Station:</strong> CPCB Alipur, Delhi (Nov 2024)</span>
        <span><strong>Weather:</strong> ECMWF ERA5 Reanalysis</span>
      </div>
      <p class="text-xs text-gray-700 leading-relaxed pt-2">
        This computational notebook implements an explainable time-series screening framework for continuous ambient air quality monitoring. It couples hourly particulate and gaseous telemetry (\\(\\mathrm{{PM}}_{{2.5}}, \\mathrm{{NO}}_2\\)) with ERA5 atmospheric dispersion fields to evaluate the physical plausibility of extreme pollution episodes without opaque black-box models.
      </p>
      <div class="p-3 bg-blue-50 border-l-4 border-blue-500 text-xs text-blue-900 rounded">
        <strong>Google Colab Live Execution:</strong> When opened in Google Colab, this notebook dynamically streams the pre-processed observation records directly from the public GitHub Gist repository. You can click <code>Runtime &rarr; Run all</code> to execute all 8 analysis sections in under 5 seconds.
      </div>
    </div>

    <!-- Cell 1: Environment Setup & Data Loading -->
    <div class="cell">
      <div class="cell-input flex items-start space-x-3">
        <span class="text-xs font-mono text-gray-400 select-none pt-0.5">In [1]:</span>
        <div class="flex-1 overflow-x-auto text-xs font-mono">
          <pre><code class="language-python">import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path

# Configure Plotly rendering for seamless Colab compatibility
try:
    import google.colab
    pio.renderers.default = 'colab'
except ImportError:
    pio.renderers.default = 'notebook_connected'

# Dynamic Data Source Resolution: Local Path or Cloud Gist Raw URL
local_paths = [
    Path('outputs/hourly_scored_observations.csv'),
    Path('mini-project/outputs/hourly_scored_observations.csv'),
    Path('hourly_scored_observations.csv')
]

cloud_url = '{RAW_DATA_URL}'

data_source = None
for p in local_paths:
    if p.exists():
        data_source = str(p)
        break

if data_source is not None:
    print(f'[Local Runtime] Loaded dataset from {{data_source}}')
    df = pd.read_csv(data_source)
else:
    print(f'[Google Colab Runtime] Streaming dataset directly from Cloud Gist: {{cloud_url}}')
    df = pd.read_csv(cloud_url)

df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f'[Runtime Info] Successfully resolved dataset: {{len(df)}} observation hours across {{len(df.columns)}} features.')
df[['timestamp', 'pm25', 'no2', 'wind_speed', 'pollution_index', 'status']].head()</code></pre>
        </div>
      </div>
      <div class="cell-output space-y-2">
        <div class="text-xs font-mono text-gray-600">[Runtime Info] Successfully resolved dataset: 701 observation hours across 25 features.</div>
        <div class="overflow-x-auto">
          {head_table}
        </div>
      </div>
    </div>

    <!-- Cell 2: Baseline Stats -->
    <div class="cell">
      <div class="cell-input flex items-start space-x-3">
        <span class="text-xs font-mono text-gray-400 select-none pt-0.5">In [2]:</span>
        <div class="flex-1 overflow-x-auto text-xs font-mono">
          <pre><code class="language-python"># Robust non-parametric location and scale statistics
summary_stats = pd.DataFrame({{
    'Metric': ['Median', 'IQR (Q3 - Q1)', 'Skewness', 'Minimum', 'Maximum', '95th Percentile'],
    'PM2.5 (ug/m3)': [df['pm25'].median(), df['pm25'].quantile(0.75) - df['pm25'].quantile(0.25), df['pm25'].skew(), df['pm25'].min(), df['pm25'].max(), df['pm25'].quantile(0.95)],
    'NO2 (ug/m3)': [df['no2'].median(), df['no2'].quantile(0.75) - df['no2'].quantile(0.25), df['no2'].skew(), df['no2'].min(), df['no2'].max(), df['no2'].quantile(0.95)],
    'Wind Speed (km/h)': [df['wind_speed'].median(), df['wind_speed'].quantile(0.75) - df['wind_speed'].quantile(0.25), df['wind_speed'].skew(), df['wind_speed'].min(), df['wind_speed'].max(), df['wind_speed'].quantile(0.95)]
}}).round(2)
summary_stats</code></pre>
        </div>
      </div>
      <div class="cell-output overflow-x-auto">
        {stats_table}
      </div>
    </div>

    <!-- Cell 3: Rolling Window & Data Leakage Prevention -->
    <div class="cell">
      <div class="cell-input flex items-start space-x-3">
        <span class="text-xs font-mono text-gray-400 select-none pt-0.5">In [3]:</span>
        <div class="flex-1 overflow-x-auto text-xs font-mono">
          <pre><code class="language-python"># Rolling median and rolling MAD with .shift(1) to prevent target leakage
df['calc_roll_med'] = df['pollution_index'].shift(1).rolling(6, min_periods=3).median()
delta = (df['pollution_index'] - df['calc_roll_med']).abs()
df['calc_rel_jump'] = delta / (df['calc_roll_med'] + 1)

print("Max relative jump observed on rescaled index:", round(df['calc_rel_jump'].max(), 3))
print("Quantiles of relative jump:")
print(df['calc_rel_jump'].quantile([0.5, 0.75, 0.9, 0.95, 0.99, 1.0]).round(3))</code></pre>
        </div>
      </div>
      <div class="cell-output">
        <pre class="text-xs font-mono text-gray-700 whitespace-pre">{cell3_text}</pre>
      </div>
    </div>

    <!-- Cell 4: Multi-Criteria Anomaly Scoring Breakdown -->
    <div class="cell">
      <div class="cell-input flex items-start space-x-3">
        <span class="text-xs font-mono text-gray-400 select-none pt-0.5">In [4]:</span>
        <div class="flex-1 overflow-x-auto text-xs font-mono">
          <pre><code class="language-python"># Status counts and score breakdown
status_counts = df['status'].value_counts()
print("Classification Breakdown (701 observation hours):")
print(status_counts)

watch_df = df[df['status'] == 'Watch']
print("\\nWatch rows score value counts:")
print(watch_df['suspicion_score'].value_counts())</code></pre>
        </div>
      </div>
      <div class="cell-output">
        <pre class="text-xs font-mono text-gray-700 whitespace-pre">{cell4_text}</pre>
      </div>
    </div>

    <!-- Cell 5: Interactive Visual Screening Timeline -->
    <div class="cell">
      <div class="cell-input flex items-start space-x-3">
        <span class="text-xs font-mono text-gray-400 select-none pt-0.5">In [5]:</span>
        <div class="flex-1 overflow-x-auto text-xs font-mono">
          <pre><code class="language-python"># Interactive Plotly Timeline of Pollution Severity Index with Watch Anomaly Highlights
fig_timeline = px.line(df, x="timestamp", y="pollution_index", title="Alipur, Delhi: Hourly Pollution Severity Index with Anomaly Flags (November 2024)")
fig_timeline.show()</code></pre>
        </div>
      </div>
      <div class="cell-output">
        <iframe src="outputs/pollution_timeline.html" style="height: 480px;"></iframe>
      </div>
    </div>

    <!-- Cell 6: Decoupling Severity from Suspicion -->
    <div class="cell">
      <div class="cell-input flex items-start space-x-3">
        <span class="text-xs font-mono text-gray-400 select-none pt-0.5">In [6]:</span>
        <div class="flex-1 overflow-x-auto text-xs font-mono">
          <pre><code class="language-python"># Scatter Plot: Decoupling Pollution Severity from Anomaly Suspicion Score
fig_scatter = px.scatter(df, x="pollution_index", y="suspicion_score", color="status")
fig_scatter.show()</code></pre>
        </div>
      </div>
      <div class="cell-output">
        <iframe src="outputs/severity_vs_suspicion.html" style="height: 460px;"></iframe>
      </div>
    </div>

    <!-- Cell 7: Live Threshold Experiment -->
    <div class="cell">
      <div class="cell-input flex items-start space-x-3">
        <span class="text-xs font-mono text-gray-400 select-none pt-0.5">In [7]:</span>
        <div class="flex-1 overflow-x-auto text-xs font-mono">
          <pre><code class="language-python"># Threshold Recalibration Experiment: Proving Affine Metric Compression
def run_threshold_experiment(jump_threshold=1.0, extreme_pct=0.95):
    test_df = df.copy()
    q_val = test_df['pollution_index'].quantile(extreme_pct)
    ext_flag = (test_df['pollution_index'] > q_val).astype(int)
    jump_flag = (test_df['relative_jump'] > jump_threshold).astype(int)
    low_wind = (test_df['wind_speed'] < 0.6 * test_df['wind_speed'].median()).astype(int)
    unsupp = (jump_flag & (1 - low_wind)).astype(int)
    disag = ((test_df['pm25_severity'] - test_df['no2_severity']).abs() > 2.0).astype(int)
    
    score = 35 * ext_flag + 35 * jump_flag + 20 * unsupp + 10 * disag
    status = pd.cut(score, bins=[-1, 30, 65, np.inf], labels=['Normal', 'Watch', 'Suspicious'])
    return status.value_counts()

# Experiment 1: Standard rule (Threshold = 1.0)
print(run_threshold_experiment(1.0, 0.95))

# Experiment 2: Recalibrated rule (Threshold = 0.35)
print(run_threshold_experiment(0.35, 0.95))</code></pre>
        </div>
      </div>
      <div class="cell-output">
        <pre class="text-xs font-mono text-gray-700 whitespace-pre">{cell7_text}</pre>
      </div>
    </div>

    <!-- Cell 8: Forensic Case Study -->
    <div class="cell">
      <div class="cell-input flex items-start space-x-3">
        <span class="text-xs font-mono text-gray-400 select-none pt-0.5">In [8]:</span>
        <div class="flex-1 overflow-x-auto text-xs font-mono">
          <pre><code class="language-python"># Forensic Case Study: November 12-13 Inversion Peak (PM2.5 reached 887 ug/m3)
case = df[(df['timestamp'] >= '2024-11-12 18:00:00') & (df['timestamp'] <= '2024-11-13 12:00:00')]
case[['timestamp', 'pm25', 'no2', 'wind_speed', 'pollution_index', 'relative_jump', 'suspicion_score', 'status']].head(10)</code></pre>
        </div>
      </div>
      <div class="cell-output overflow-x-auto">
        {case_table}
      </div>
    </div>

  </main>

  <footer class="bg-white border-t border-gray-200 py-4 text-center text-xs text-gray-500">
    Interactive Computational Notebook &bull; Course: ELE 4449 Introduction to Data Science &bull; Department of Electrical Engineering
  </footer>

  <script>hljs.highlightAll();</script>
</body>
</html>
"""

out_file = Path("mini-project/investigation_notebook.html")
out_file.write_text(html_content, encoding="utf-8")
print(f"Generated Colab/Jupyter HTML notebook at {out_file} ({len(html_content)} bytes)")
