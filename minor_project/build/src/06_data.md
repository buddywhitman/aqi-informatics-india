# Data and Experimental Design

## 6.1 Data sources

The empirical work uses two open sources.

* **OpenAQ / CPCB.** Continuous ambient air-quality records from monitoring stations of India's Central Pollution Control Board, accessed through the OpenAQ platform: PM₂.₅, PM₁₀, NO, NO₂, NOₓ, CO, SO₂, O₃ and station-reported meteorology.
* **Open-Meteo.** Hourly ECMWF ERA5 reanalysis: temperature, humidity, precipitation, wind speed and direction, pressure.

The raw hourly files are stored in `data/raw_hourly/`. Pollution files exist for four cities, and weather files for those four plus three further cities that the earlier applied study used (Chennai, Hyderabad, Ahmedabad).

@tab cov | Raw pollution records per city (all parameters, half-hourly UTC stamps)
| City | Rows | First reading (UTC) | Last reading (UTC) |
|---|---|---|---|
| Delhi | 123,323 | 2025-02-18 | 2026-06-16 |
| Mumbai | 61,628 | 2025-02-18 | 2026-06-16 |
| Bengaluru | 94,713 | 2025-02-18 | 2026-06-16 |
| Kolkata | 22,847 | 2025-02-18 | 2025-06-23 |

@fig cover | coverage.png | **Data coverage.** Monthly count of NO₂ and PM₂.₅ readings by city. Kolkata's record ends in June 2025; sparse coverage (and, in Mumbai, long constant runs, Section 8.2) are why the four cities behave so differently in Chapter 8. | w

## 6.2 Variable construction

The treatment $T$ is NO₂ (µg/m³), the outcome $Y$ is PM₂.₅ (µg/m³), the regime proxies $Z$ are meteorological series (temperature, humidity, wind speed and direction, pressure), and the controls $X$ are lagged pollutants and lagged weather. NO₂ is a combustion marker whose emission is dominated by traffic and industry, which is why it serves as an emission-intensity proxy for treatment; the associated limitation, that NO₂ is itself chemically related to PM₂.₅ formation, is stated in Section 8.5.

## 6.3 The time-grid audit

A data-quality audit found that the legacy panel had rounded half-hour timestamps with ties-to-even, collapsing the series onto a **two-hour grid**. What had been labelled "hourly" lags were therefore two hours, and every horizon, lag length and embargo built on them was mis-scaled. The effect was not cosmetic: the Delhi single-index posterior-feature estimate moved from 0.58 to 0.37 once the grid was corrected. The corrected pipeline (`src/data_pipeline_clean.py`) aggregates properly to an hourly grid, and all real-data numbers in the report are recomputed on it.

> The lesson recorded in the project: *a correct estimator on a mis-specified grid produces confident wrong numbers.* The audit is the reason the manuscript contains a provenance appendix.

## 6.4 Experimental designs

The evidence is deliberately layered, from fully known truth to fully observational.

@tab design | Experimental designs and what each can establish
| Design | Truth known? | What it tests | Size |
|---|---|---|---|
| Factorial ($7\times7$) | yes | closed-form law; rank correlation of failure scores | 4,900 runs, 42 cells |
| Learned-posterior grid | yes | removes the circularity of synthetic proxies; EM posteriors | 720 runs, 36 cells, $N=1{,}200$ |
| Synthetic frontier | yes | observability ($\Delta_Z$) $\times$ persistence | 900 datasets |
| Residual-state law checks | yes | Theorem 4 against estimator, $N=10^6$ | 9 configurations; 72 learned-posterior cells |
| Real-data-calibrated semi-synthetic | effects known; everything else real | four cities, real gaps, meteorology, NO₂ series | 30 datasets per city |
| Representation zoo | yes | do better encoders give better causal reliability? | 15 shifted worlds, 4 encoders |
| Observational stress test | **no** | audited associations, placebos, robustness values | four cities |

**Factorial design.** A $7\times 7$ grid over proxy error $\varepsilon_\gamma$ and treatment-innovation scale $\sigma_1$ with 100 runs per cell controls the posterior error directly, so the closed form of Proposition 1 can be checked cell by cell.

**Learned posteriors.** To remove the circularity of synthetic proxies, posteriors are learned with a Gaussian HMM (EM, three restarts, smoothing) from noisy proxies across proxy separation $\Delta_Z$, innovation scale $\sigma_1$ and treatment shift $\Delta_T$. All diagnostics use the *realized* generated-state Gram matrix.

**Real-data-calibrated semi-synthetic benchmark.** The benchmark keeps everything real except the causal effects: the meteorological proxies and controls, the observed NO₂ series, the hourly missingness pattern, and regime persistence and overlap. True state paths are drawn by forward-filtering backward-sampling from an HMM calibrated per city (stay probabilities 0.90–0.96), so truth is known while the structure is realistic. It replaces an earlier benchmark in which the true state was a deterministic function of the proxies.

## 6.5 Estimators compared and metrics

Estimators: oracle-state DML (infeasible lower bound), standard DML, block DML, DML with $Z$ controls, posterior-feature DML, hard regime fixed effects, OR-DML with weighted nuisances (ablation) and posterior-adjusted OR-DML. Metrics are the mean absolute ATE bias, the median regime-effect error $\|\hat\theta - \theta^*\|_2$, RMSE, 95% interval coverage, posterior error $\varepsilon_\gamma$, and the raw and standardized $\lambda_{\min}$.
