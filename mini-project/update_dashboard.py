import json
import pandas as pd
from pathlib import Path

df = pd.read_csv("mini-project/outputs/hourly_scored_observations.csv")

records = df[[
    "timestamp", "pm25", "no2", "wind_speed", "pollution_index", 
    "pm25_severity", "no2_severity", "rolling_median_6h", "relative_jump", 
    "suspicion_score", "status"
]].copy()

records["time_str"] = records["timestamp"].astype(str)
records_json = records.to_json(orient="records")

html_template = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Air Quality Anomaly Screening - Interactive Investigation Dashboard</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #fafafa;
    color: #1e293b;
  }
  code, pre, .font-mono { font-family: 'JetBrains Mono', monospace; }
  .tab-btn.active {
    border-bottom-color: #0f172a;
    color: #0f172a;
    font-weight: 600;
  }
  .tab-pane { display: none; }
  .tab-pane.active { display: block; }
  iframe { width: 100%; border: none; min-height: 520px; }
</style>
</head>
<body class="min-h-screen flex flex-col">

  <main class="flex-1 max-w-6xl w-full mx-auto px-4 sm:px-6 py-8 space-y-6">
    
    <!-- Title & Metadata -->
    <div class="border-b border-slate-200 pb-5">
      <div class="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-3">
        <div>
          <div class="flex items-center space-x-2">
            <h1 class="text-xl sm:text-2xl font-bold tracking-tight text-slate-900">Data-Driven Detection of Anomalous Pollution Spikes</h1>
            <span id="kpi-mode-badge" class="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-100 text-slate-600 border border-slate-200">Baseline</span>
          </div>
          <p class="text-sm text-slate-500 mt-1">Exploratory time-series screening &bull; CPCB CAAQM Station: Alipur, Delhi (November 2024)</p>
        </div>
        <div class="flex flex-wrap items-center gap-2">
          <a href="Data_Driven_Detection_of_Anomalous_Pollution_Spikes_Report.pdf" target="_blank" class="inline-flex items-center text-xs font-medium text-slate-700 bg-white border border-slate-300 hover:bg-slate-50 px-3 py-1.5 rounded shadow-sm transition">
            PDF Report
          </a>
          <a href="https://colab.research.google.com/gist/buddywhitman/57e1f6b40953a2eec1cb1fa176c93fee/investigation_notebook.ipynb" target="_blank" class="inline-flex items-center text-xs font-semibold text-white bg-amber-600 hover:bg-amber-700 px-3 py-1.5 rounded shadow-sm transition">
            <svg class="w-3.5 h-3.5 mr-1.5 text-white" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
            Open in Google Colab
          </a>
          <a href="investigation_notebook.html" target="_blank" class="inline-flex items-center text-xs font-medium text-slate-700 bg-white border border-slate-300 hover:bg-slate-50 px-3 py-1.5 rounded shadow-sm transition">
            Notebook (HTML View)
          </a>
          <a href="investigation_notebook.ipynb" download class="inline-flex items-center text-xs font-medium text-slate-700 bg-white border border-slate-300 hover:bg-slate-50 px-2 py-1.5 rounded shadow-sm transition" title="Download raw .ipynb for local JupyterLab">
            .ipynb
          </a>
        </div>
      </div>
    </div>

    <!-- Minimal KPI Grid -->
    <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
      <div class="bg-white p-3.5 rounded-lg border border-slate-200 shadow-sm">
        <span class="text-xs text-slate-500 block font-medium">Observations</span>
        <span class="text-lg font-semibold text-slate-900 mt-0.5 block font-mono">701 <span class="text-xs font-normal text-slate-400">/ 744</span></span>
        <span class="text-[11px] text-slate-400">94.2% coverage</span>
      </div>
      <div class="bg-white p-3.5 rounded-lg border border-slate-200 shadow-sm transition-all" id="card-normal">
        <span class="text-xs text-slate-500 block font-medium">Normal Class</span>
        <span class="text-lg font-semibold text-emerald-700 mt-0.5 block font-mono" id="stat-normal">666</span>
        <span class="text-[11px] text-slate-400" id="pct-normal">95.0%</span>
      </div>
      <div class="bg-white p-3.5 rounded-lg border border-slate-200 shadow-sm transition-all" id="card-watch">
        <span class="text-xs text-slate-500 block font-medium">Watch Class</span>
        <span class="text-lg font-semibold text-amber-700 mt-0.5 block font-mono" id="stat-watch">35</span>
        <span class="text-[11px] text-slate-400" id="pct-watch">5.0%</span>
      </div>
      <div class="bg-white p-3.5 rounded-lg border border-slate-200 shadow-sm transition-all" id="card-suspicious">
        <span class="text-xs text-slate-500 block font-medium">Suspicious Class</span>
        <span class="text-lg font-semibold text-rose-700 mt-0.5 block font-mono" id="stat-suspicious">0</span>
        <span class="text-[11px] text-slate-400" id="pct-suspicious">0.0%</span>
      </div>
      <div class="bg-white p-3.5 rounded-lg border border-slate-200 shadow-sm">
        <span class="text-xs text-slate-500 block font-medium">Median Index</span>
        <span class="text-lg font-semibold text-slate-900 mt-0.5 block font-mono">102.6</span>
        <span class="text-[11px] text-slate-400">Baseline proxy</span>
      </div>
      <div class="bg-white p-3.5 rounded-lg border border-slate-200 shadow-sm">
        <span class="text-xs text-slate-500 block font-medium">Peak Score</span>
        <span class="text-lg font-semibold text-slate-900 mt-0.5 block font-mono" id="stat-peak">45 <span class="text-xs font-normal text-slate-400">/ 100</span></span>
        <span class="text-[11px] text-slate-400" id="stat-peak-sub">Extreme + Disag</span>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="border-b border-slate-200 flex space-x-6 text-sm overflow-x-auto">
      <button onclick="setTab('visuals')" id="tab-visuals" class="tab-btn active py-2.5 border-b-2 border-slate-900 text-slate-900 whitespace-nowrap">
        Visual Screening
      </button>
      <button onclick="setTab('sandbox')" id="tab-sandbox" class="tab-btn py-2.5 border-b-2 border-transparent text-slate-500 hover:text-slate-800 whitespace-nowrap">
        Live Threshold Sandbox
      </button>
      <button onclick="setTab('cases')" id="tab-cases" class="tab-btn py-2.5 border-b-2 border-transparent text-slate-500 hover:text-slate-800 whitespace-nowrap">
        Case Review Table <span id="tab-case-count" class="ml-1 px-1.5 py-0.2 bg-slate-100 text-slate-700 rounded-full text-[10px] font-mono">35</span>
      </button>
      <button onclick="setTab('diagnostics')" id="tab-diagnostics" class="tab-btn py-2.5 border-b-2 border-transparent text-slate-500 hover:text-slate-800 whitespace-nowrap">
        Mathematical Diagnostic
      </button>
    </div>

    <!-- TAB 1: VISUAL SCREENING -->
    <div id="pane-visuals" class="tab-pane active space-y-6">
      <div class="bg-white p-5 rounded-lg border border-slate-200 shadow-sm space-y-3">
        <div class="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-1">
          <h2 class="text-sm font-semibold text-slate-900">1. Temporal Pollution Timeline with Anomaly Highlights</h2>
          <span class="text-xs text-slate-500">Interactive: Box-zoom into mid-November to inspect individual hour spikes</span>
        </div>
        <div class="overflow-hidden border border-slate-100 rounded">
          <iframe src="outputs/pollution_timeline.html"></iframe>
        </div>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div class="bg-white p-5 rounded-lg border border-slate-200 shadow-sm space-y-3">
          <h2 class="text-sm font-semibold text-slate-900">2. Pollution Severity vs. Suspicion Score</h2>
          <div class="overflow-hidden border border-slate-100 rounded">
            <iframe src="outputs/severity_vs_suspicion.html" style="min-height: 460px;"></iframe>
          </div>
        </div>

        <div class="bg-white p-5 rounded-lg border border-slate-200 shadow-sm space-y-3">
          <h2 class="text-sm font-semibold text-slate-900">3. Meteorological Alignment (Wind Stagnation)</h2>
          <div class="overflow-hidden border border-slate-100 rounded">
            <iframe src="outputs/pollution_weather_context.html" style="min-height: 460px;"></iframe>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: LIVE THRESHOLD SANDBOX (DEMO) -->
    <div id="pane-sandbox" class="tab-pane space-y-6">
      <div class="bg-white p-6 rounded-lg border border-slate-200 shadow-sm space-y-5">
        <div class="border-b border-slate-100 pb-3 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <div>
            <h2 class="text-base font-semibold text-slate-900">Interactive Threshold Calibration Sandbox</h2>
            <p class="text-xs text-slate-500 mt-0.5">
              Drag the sliders below to re-calibrate thresholds in real-time across all 701 observation hours.
            </p>
          </div>
          <button onclick="resetSliders()" class="self-start sm:self-auto px-3 py-1 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded border border-slate-300 text-xs font-medium transition">
            Reset to Baseline
          </button>
        </div>

        <!-- Sliders -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-5">
          <div class="space-y-1.5 bg-slate-50 p-4 rounded border border-slate-200">
            <div class="flex justify-between items-center text-xs">
              <label class="font-medium text-slate-700">Relative Jump Threshold (&tau;<sub>jump</sub>)</label>
              <span id="lbl-jump" class="font-mono text-slate-900 font-bold bg-white px-2 py-0.5 rounded border border-slate-200">1.00</span>
            </div>
            <input type="range" id="slider-jump" min="0.10" max="1.50" step="0.05" value="1.00" class="w-full h-2 bg-slate-200 rounded cursor-pointer accent-slate-900" oninput="recompute()">
            <div class="flex justify-between text-[10px] text-slate-400">
              <span>0.10 (Sensitive)</span>
              <span class="text-amber-700 font-medium">Default: 1.00 (100% surge)</span>
              <span>1.50 (Strict)</span>
            </div>
            <span class="text-[11px] text-slate-500 block pt-1">Lowering to ~0.35 triggers the Large Jump rule, showing why affine compression masked spikes.</span>
          </div>

          <div class="space-y-1.5 bg-slate-50 p-4 rounded border border-slate-200">
            <div class="flex justify-between items-center text-xs">
              <label class="font-medium text-slate-700">Extreme Percentile Cutoff (Q<sub>ext</sub>)</label>
              <span id="lbl-q" class="font-mono text-slate-900 font-bold bg-white px-2 py-0.5 rounded border border-slate-200">95th</span>
            </div>
            <input type="range" id="slider-q" min="80" max="99" step="1" value="95" class="w-full h-2 bg-slate-200 rounded cursor-pointer accent-slate-900" oninput="recompute()">
            <div class="flex justify-between text-[10px] text-slate-400">
              <span>80th percentile</span>
              <span class="text-slate-600 font-medium">Default: 95th (~135.7 proxy)</span>
              <span>99th percentile</span>
            </div>
            <span class="text-[11px] text-slate-500 block pt-1">Awards 35 points for observations falling in the top monthly extreme tail.</span>
          </div>

          <div class="space-y-1.5 bg-slate-50 p-4 rounded border border-slate-200">
            <div class="flex justify-between items-center text-xs">
              <label class="font-medium text-slate-700">Low Wind Multiplier (&alpha;<sub>wind</sub>)</label>
              <span id="lbl-wind" class="font-mono text-slate-900 font-bold bg-white px-2 py-0.5 rounded border border-slate-200">0.60 &times; median</span>
            </div>
            <input type="range" id="slider-wind" min="0.20" max="1.00" step="0.05" value="0.60" class="w-full h-2 bg-slate-200 rounded cursor-pointer accent-slate-900" oninput="recompute()">
            <div class="flex justify-between text-[10px] text-slate-400">
              <span>0.20 (1.16 km/h)</span>
              <span class="text-slate-600 font-medium">Default: 0.60 (3.48 km/h)</span>
              <span>1.00 (5.80 km/h)</span>
            </div>
            <span class="text-[11px] text-slate-500 block pt-1">Wind below cutoff permits natural stagnation; surges under high wind receive +20 pts.</span>
          </div>

          <div class="space-y-1.5 bg-slate-50 p-4 rounded border border-slate-200">
            <div class="flex justify-between items-center text-xs">
              <label class="font-medium text-slate-700">Pollutant Disagreement Cutoff (|&Delta;Sev|)</label>
              <span id="lbl-disag" class="font-mono text-slate-900 font-bold bg-white px-2 py-0.5 rounded border border-slate-200">2.00</span>
            </div>
            <input type="range" id="slider-disag" min="0.50" max="3.50" step="0.25" value="2.00" class="w-full h-2 bg-slate-200 rounded cursor-pointer accent-slate-900" oninput="recompute()">
            <div class="flex justify-between text-[10px] text-slate-400">
              <span>0.50 (Frequent)</span>
              <span class="text-slate-600 font-medium">Default: 2.0 IQR units</span>
              <span>3.50 (Rare)</span>
            </div>
            <span class="text-[11px] text-slate-500 block pt-1">Flags severe divergence between particulate and gaseous combustion severities (+10 pts).</span>
          </div>
        </div>

        <!-- Dynamic Visual Class Distribution Bar -->
        <div class="space-y-2 pt-2">
          <div class="flex justify-between items-center text-xs">
            <span class="font-medium text-slate-700">Live Dataset Classification Split (701 Hours):</span>
            <span class="text-slate-400 font-mono text-[11px]" id="sim-bar-label">666 Normal / 35 Watch / 0 Suspicious</span>
          </div>
          <div class="w-full bg-slate-100 h-3 rounded-full overflow-hidden flex shadow-inner">
            <div id="bar-normal" class="bg-emerald-500 h-full transition-all duration-200" style="width: 95.0%;"></div>
            <div id="bar-watch" class="bg-amber-500 h-full transition-all duration-200" style="width: 5.0%;"></div>
            <div id="bar-suspicious" class="bg-rose-500 h-full transition-all duration-200" style="width: 0.0%;"></div>
          </div>
        </div>

        <!-- Sandbox Output Status Banner -->
        <div class="p-4 bg-slate-900 text-white rounded-lg flex flex-col sm:flex-row items-center justify-between gap-4 font-mono text-xs">
          <div>
            <span class="text-slate-400 block uppercase text-[10px] tracking-wider font-semibold">Simulated Classification Totals:</span>
            <div class="flex flex-wrap gap-x-6 gap-y-1 mt-1 text-xs">
              <span>Normal: <b class="text-emerald-400 font-bold" id="sim-normal">666</b> (<span id="sim-pct-normal">95.0%</span>)</span>
              <span>Watch: <b class="text-amber-400 font-bold" id="sim-watch">35</b> (<span id="sim-pct-watch">5.0%</span>)</span>
              <span>Suspicious: <b class="text-rose-400 font-bold" id="sim-suspicious">0</b> (<span id="sim-pct-suspicious">0.0%</span>)</span>
            </div>
          </div>
          <button onclick="setTab('cases')" class="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-amber-300 rounded border border-slate-700 text-xs font-sans transition flex items-center space-x-1">
            <span>Review Rows in Table</span>
            <span>&rarr;</span>
          </button>
        </div>

        <div id="sim-alert" class="p-3 text-xs bg-amber-50 text-amber-900 border border-amber-200 rounded hidden leading-relaxed">
          <!-- Populated dynamically -->
        </div>
      </div>
    </div>

    <!-- TAB 3: CASE REVIEW TABLE -->
    <div id="pane-cases" class="tab-pane space-y-4">
      <div class="bg-white p-5 rounded-lg border border-slate-200 shadow-sm space-y-3">
        <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-slate-100 pb-3">
          <div>
            <div class="flex items-center space-x-2">
              <h2 class="text-sm font-semibold text-slate-900">Observation Records Review Table</h2>
              <span id="table-filter-badge" class="text-[10px] px-2 py-0.5 rounded font-mono bg-amber-50 text-amber-800 border border-amber-200">Watch & Suspicious Only</span>
            </div>
            <p class="text-xs text-slate-500 mt-0.5">Dynamically reflects the active calibration thresholds set in the Live Sandbox.</p>
          </div>
          <div class="flex items-center space-x-2">
            <select id="status-filter" onchange="filterTable()" class="px-2 py-1 text-xs border border-slate-300 rounded focus:outline-none focus:border-slate-500 bg-white">
              <option value="all-flagged">Flagged Only (Watch + Suspicious)</option>
              <option value="Suspicious">Suspicious Only</option>
              <option value="Watch">Watch Only</option>
              <option value="all">All 701 Records</option>
            </select>
            <input type="text" id="table-search" placeholder="Search timestamp..." oninput="filterTable()" class="px-3 py-1 text-xs border border-slate-300 rounded w-40 focus:outline-none focus:border-slate-500">
          </div>
        </div>

        <div class="overflow-x-auto max-h-[600px] overflow-y-auto border border-slate-100 rounded">
          <table class="w-full text-left text-xs text-slate-600">
            <thead class="bg-slate-50 text-slate-800 uppercase text-[10px] tracking-wider border-b border-slate-200 font-semibold sticky top-0 z-10 shadow-sm">
              <tr>
                <th class="px-3 py-2.5">Timestamp (IST)</th>
                <th class="px-3 py-2.5 text-right">PM2.5 (&mu;g/m&sup3;)</th>
                <th class="px-3 py-2.5 text-right">NO2 (&mu;g/m&sup3;)</th>
                <th class="px-3 py-2.5 text-right">Wind (km/h)</th>
                <th class="px-3 py-2.5 text-right">Index Proxy</th>
                <th class="px-3 py-2.5 text-right">Rel Jump</th>
                <th class="px-3 py-2.5 text-center">Score</th>
                <th class="px-3 py-2.5 text-center">Status</th>
              </tr>
            </thead>
            <tbody id="case-rows" class="divide-y divide-slate-100 font-mono">
              <!-- Populated by JavaScript -->
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- TAB 4: MATHEMATICAL DIAGNOSTIC -->
    <div id="pane-diagnostics" class="tab-pane space-y-6">
      <div class="bg-white p-6 rounded-lg border border-slate-200 shadow-sm space-y-4 text-xs text-slate-700 leading-relaxed">
        <h2 class="text-sm font-bold text-slate-900 border-b border-slate-200 pb-2">Diagnostic Proof: Scale Distortion Under Affine Transformations</h2>
        
        <p>
          A central analytical finding of this project centers on why zero observations crossed the Suspicious threshold (&gt; 65). On raw PM2.5, concentrations more than doubled (&gt; 100% surge) across <strong>42 separate hours</strong>, with surges exceeding 435%.
        </p>

        <p>
          However, the project rule evaluated relative jump on the rescaled <code>pollution_index</code>:
        </p>
        <div class="bg-slate-50 p-3 rounded font-mono border border-slate-200 text-slate-900">
          Index = 100 + 25 &times; Score<br>
          Relative Jump = |Index_t - Median_6h| / (Median_6h + 1)
        </div>

        <p>
          Let y = ax + b represent an affine transformation with an additive offset b = 100. The relative change of y compared to x is:
        </p>
        <div class="bg-slate-50 p-3 rounded font-mono border border-slate-200 text-slate-900">
          &Delta;y / y = a &Delta;x / (ax + b) = &Delta;x / (x + b/a) &lt; &Delta;x / x
        </div>

        <p>
          Because b = 100, the denominator is anchored around 100 to 140. For a relative jump to exceed 1.0, the index would have had to jump by over +100 points in a single hour. Since the entire monthly range of the index was bounded between 66.4 and 163.8, an instantaneous +100 jump was mathematically impossible. The maximum observed index jump was 0.629. Consequently, the Large Jump flag remained 0 across all 701 hours.
        </p>

        <div class="p-3 bg-slate-50 border-l-2 border-slate-800 text-slate-900">
          <strong>Key Methodological Conclusion:</strong> Affine transformations preserve rank order and linear correlation, but fundamentally compress ratio metrics. Scale-dependent heuristic thresholds cannot be blindly transferred across transformed coordinate spaces without re-calibration.
        </div>
      </div>
    </div>

  </main>

  <script>
    function setTab(name) {
      document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
      
      const pane = document.getElementById('pane-' + name);
      const btn = document.getElementById('tab-' + name);
      if (pane && btn) {
        pane.classList.add('active');
        btn.classList.add('active');
      }
    }

    const rawDataset = %%RECORDS_JSON%%;
    const medWind = 5.8;
    
    // Dataset copy with dynamic simulated fields
    let dataset = rawDataset.map(r => ({
      ...r,
      sim_score: r.suspicion_score,
      sim_status: r.status
    }));

    function renderTable() {
      const tbody = document.getElementById('case-rows');
      const search = (document.getElementById('table-search')?.value || '').trim().toLowerCase();
      const filter = document.getElementById('status-filter')?.value || 'all-flagged';
      
      tbody.innerHTML = '';
      
      let filtered = dataset.filter(r => {
        const matchesSearch = !search || r.time_str.toLowerCase().includes(search);
        if (!matchesSearch) return false;
        
        if (filter === 'all-flagged') return r.sim_status === 'Watch' || r.sim_status === 'Suspicious';
        if (filter === 'Suspicious') return r.sim_status === 'Suspicious';
        if (filter === 'Watch') return r.sim_status === 'Watch';
        return true; // 'all'
      });

      filtered.sort((a, b) => b.sim_score - a.sim_score);

      // Display up to 50 rows
      const displayed = filtered.slice(0, 60);

      if (displayed.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" class="text-center py-6 text-slate-400 font-sans">No matching observation records found for current criteria.</td></tr>';
        return;
      }

      displayed.forEach(r => {
        const tr = document.createElement('tr');
        const isSusp = r.sim_status === 'Suspicious';
        const isWatch = r.sim_status === 'Watch';
        tr.className = isSusp ? 'bg-rose-50/70 hover:bg-rose-100/70' : (isWatch ? 'hover:bg-amber-50/50' : 'hover:bg-slate-50');
        
        const statusBadge = isSusp 
          ? '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-600 text-white shadow-sm">Suspicious</span>'
          : (isWatch ? '<span class="px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-100 text-amber-800 border border-amber-200">Watch</span>'
                     : '<span class="px-2 py-0.5 rounded text-[10px] font-medium bg-emerald-50 text-emerald-700">Normal</span>');

        tr.innerHTML = `
          <td class="px-3 py-2 text-slate-900 font-medium whitespace-nowrap">${r.timestamp}</td>
          <td class="px-3 py-2 text-right text-rose-700">${Number(r.pm25).toFixed(1)}</td>
          <td class="px-3 py-2 text-right text-slate-700">${Number(r.no2).toFixed(1)}</td>
          <td class="px-3 py-2 text-right text-slate-700">${Number(r.wind_speed).toFixed(1)}</td>
          <td class="px-3 py-2 text-right font-medium">${Number(r.pollution_index).toFixed(1)}</td>
          <td class="px-3 py-2 text-right text-amber-700 font-semibold">${Number(r.relative_jump).toFixed(3)}</td>
          <td class="px-3 py-2 text-center font-bold ${isSusp ? 'text-rose-700 text-sm' : 'text-slate-900'}">${r.sim_score}</td>
          <td class="px-3 py-2 text-center whitespace-nowrap">${statusBadge}</td>
        `;
        tbody.appendChild(tr);
      });

      // Update case count badge
      const flaggedCount = dataset.filter(r => r.sim_status === 'Watch' || r.sim_status === 'Suspicious').length;
      document.getElementById('tab-case-count').innerText = flaggedCount;
    }

    function filterTable() {
      renderTable();
    }

    function recompute() {
      const jumpThresh = parseFloat(document.getElementById('slider-jump').value);
      const qPct = parseInt(document.getElementById('slider-q').value) / 100.0;
      const windMult = parseFloat(document.getElementById('slider-wind').value);
      const disagCut = parseFloat(document.getElementById('slider-disag').value);

      document.getElementById('lbl-jump').innerText = jumpThresh.toFixed(2);
      document.getElementById('lbl-q').innerText = (qPct * 100).toFixed(0) + 'th';
      document.getElementById('lbl-wind').innerText = windMult.toFixed(2) + ' × median';
      document.getElementById('lbl-disag').innerText = disagCut.toFixed(2);

      const indices = rawDataset.map(d => d.pollution_index).sort((a,b) => a - b);
      const qIndex = Math.min(Math.floor(indices.length * qPct), indices.length - 1);
      const qVal = indices[qIndex];

      let nNormal = 0, nWatch = 0, nSuspicious = 0;
      let peakScore = 0;

      dataset.forEach(r => {
        const ext = r.pollution_index > qVal ? 1 : 0;
        const jump = r.relative_jump > jumpThresh ? 1 : 0;
        const lowWind = r.wind_speed < (windMult * medWind) ? 1 : 0;
        const unsupp = (jump === 1 && lowWind === 0) ? 1 : 0;
        const disag = Math.abs(r.pm25_severity - r.no2_severity) > disagCut ? 1 : 0;

        const score = 35 * ext + 35 * jump + 20 * unsupp + 10 * disag;
        r.sim_score = score;
        if (score > peakScore) peakScore = score;

        if (score <= 30) {
          r.sim_status = 'Normal';
          nNormal++;
        } else if (score <= 65) {
          r.sim_status = 'Watch';
          nWatch++;
        } else {
          r.sim_status = 'Suspicious';
          nSuspicious++;
        }
      });

      const total = dataset.length;
      const pctNormal = ((nNormal/total)*100).toFixed(1);
      const pctWatch = ((nWatch/total)*100).toFixed(1);
      const pctSuspicious = ((nSuspicious/total)*100).toFixed(1);

      // Update Sandbox values
      document.getElementById('sim-normal').innerText = nNormal;
      document.getElementById('sim-watch').innerText = nWatch;
      document.getElementById('sim-suspicious').innerText = nSuspicious;

      document.getElementById('sim-pct-normal').innerText = pctNormal + '%';
      document.getElementById('sim-pct-watch').innerText = pctWatch + '%';
      document.getElementById('sim-pct-suspicious').innerText = pctSuspicious + '%';

      // Update Top KPI Grid cards!
      document.getElementById('stat-normal').innerText = nNormal;
      document.getElementById('pct-normal').innerText = pctNormal + '%';
      document.getElementById('stat-watch').innerText = nWatch;
      document.getElementById('pct-watch').innerText = pctWatch + '%';
      document.getElementById('stat-suspicious').innerText = nSuspicious;
      document.getElementById('pct-suspicious').innerText = pctSuspicious + '%';
      document.getElementById('stat-peak').innerHTML = `${peakScore} <span class="text-xs font-normal text-slate-400">/ 100</span>`;

      // Update distribution bar
      document.getElementById('bar-normal').style.width = pctNormal + '%';
      document.getElementById('bar-watch').style.width = pctWatch + '%';
      document.getElementById('bar-suspicious').style.width = pctSuspicious + '%';
      document.getElementById('sim-bar-label').innerText = `${nNormal} Normal / ${nWatch} Watch / ${nSuspicious} Suspicious`;

      // Check if altered from defaults
      const isDefault = (jumpThresh === 1.00 && Math.abs(qPct - 0.95) < 0.001 && Math.abs(windMult - 0.60) < 0.001 && Math.abs(disagCut - 2.00) < 0.001);
      const modeBadge = document.getElementById('kpi-mode-badge');
      if (isDefault) {
        modeBadge.className = 'text-[10px] font-mono px-2 py-0.5 rounded bg-slate-100 text-slate-600 border border-slate-200';
        modeBadge.innerText = 'Baseline';
      } else {
        modeBadge.className = 'text-[10px] font-mono px-2 py-0.5 rounded bg-amber-100 text-amber-800 border border-amber-300 font-semibold';
        modeBadge.innerText = 'Live Recalibration Active';
      }

      // Dynamic Alert Box
      const alertBox = document.getElementById('sim-alert');
      if (jumpThresh <= 0.45 && nSuspicious > 0) {
        alertBox.classList.remove('hidden');
        alertBox.innerHTML = `<strong>Live Discovery:</strong> Lowering &tau;<sub>jump</sub> to <strong>${jumpThresh.toFixed(2)}</strong> activates the Large Jump flag, classifying <strong>${nSuspicious} hour(s) as Suspicious</strong>. This proves that the absence of suspicious flags under the default &tau;<sub>jump</sub> = 1.00 is solely an artifact of affine metric compression (\(y = 100 + 25z\)), confirming our theoretical diagnostic.`;
      } else if (!isDefault) {
        alertBox.classList.remove('hidden');
        alertBox.innerHTML = `<strong>Custom Calibration:</strong> Yields ${nNormal} Normal (${pctNormal}%), ${nWatch} Watch (${pctWatch}%), and ${nSuspicious} Suspicious (${pctSuspicious}%). Check the Case Review Table to inspect the updated classifications.`;
      } else {
        alertBox.classList.add('hidden');
      }

      // Re-render table with newly classified rows
      renderTable();
    }

    function resetSliders() {
      document.getElementById('slider-jump').value = 1.00;
      document.getElementById('slider-q').value = 95;
      document.getElementById('slider-wind').value = 0.60;
      document.getElementById('slider-disag').value = 2.00;
      recompute();
    }

    // Initial render
    recompute();
  </script>
</body>
</html>
"""

final_html = html_template.replace("%%RECORDS_JSON%%", records_json)
dash_file = Path("mini-project/interactive_investigation_dashboard.html")
dash_file.write_text(final_html, encoding="utf-8")
print(f"Minimalist academic dashboard successfully updated at {dash_file} ({len(final_html)} bytes)")

# Also update build_interactive_dashboard.py so it matches
Path("mini-project/build_interactive_dashboard.py").write_text(
    f'import json\nimport pandas as pd\nfrom pathlib import Path\n\ndf = pd.read_csv("mini-project/outputs/hourly_scored_observations.csv")\nrecords = df[["timestamp", "pm25", "no2", "wind_speed", "pollution_index", "pm25_severity", "no2_severity", "rolling_median_6h", "relative_jump", "suspicion_score", "status"]].copy()\nrecords["time_str"] = records["timestamp"].astype(str)\nrecords_json = records.to_json(orient="records")\nhtml_template = """{html_template}"""\nfinal_html = html_template.replace("%%RECORDS_JSON%%", records_json)\ndash_file = Path("mini-project/interactive_investigation_dashboard.html")\ndash_file.write_text(final_html, encoding="utf-8")\nprint("Generated", dash_file)\n',
    encoding="utf-8"
)
print("Updated build_interactive_dashboard.py")
