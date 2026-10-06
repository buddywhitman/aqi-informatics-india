"""Final submission-synthesis consistency checks.

Checks only promoted post-audit claims against committed evidence.
"""
from pathlib import Path
import pandas as pd
import numpy as np

tex=Path("paper/main.tex").read_text()

# Frozen-sensor sensitivity promoted into main text.
fr=pd.read_csv("research/results/frozen_sensor_sensitivity.csv")
m0=fr[(fr.city=="Mumbai")&(fr["sample"]=="original")].iloc[0]
m1=fr[(fr.city=="Mumbai")&(fr["sample"]=="remove_frozen_ge6")].iloc[0]
assert int(m0.N)==4180 and int(m1.N)==3295
assert abs(m0.ate-1.591357)<1e-5 and abs(m1.ate-2.916814)<1e-5
assert abs(m0.lmin-0.316537)<1e-5 and abs(m1.lmin-0.178138)<1e-5
for token in ["4{,}180","3{,}295","0.317","0.178"]:
    assert token in tex, token

# Delhi placebo qualification.
pl=pd.read_csv("reports/empirical_placebo_falsification.csv")
d=pl[(pl.City=="Delhi")&(pl.Type=="Pre-Treatment Placebo")]
assert len(d)==3 and (d.p_value<.01).all()
assert "pre-treatment placebos" in tex

# Calibration intervention: large ECE gains, no AUC improvement.
cal=pd.read_csv("reports/representation_zoo_calibration_intervention.csv")
for arch in ["Neural GRU Encoder","Causal Transformer Encoder"]:
    r=cal[cal.Architecture==arch].iloc[0]
    assert r.ECE_Reduction_Pct in ["47.9%","60.3%"]
    assert float(r.Delta_ROC_AUC)<0
assert "without improving downstream ROC-AUC" in tex

# Phantom-resolution appendix must be present and numerically scoped.
assert "Phantom causal resolution" in tex
assert "11.4x/21.2x inflation" in tex
assert "not automatically a conservative certificate" in tex

# Scope/AI disclosure.
assert "worst-direction diagnostic" in tex
assert "Generative AI tools were used as research assistants" in tex

print("Final submission synthesis checks passed.")


# Cross-branch bias-law integration gates.
from src.bias_law.bias_law import law_bias, peak_dT, sup_bias
assert abs(law_bias(2.0, 3.0, 0.25, 1.0) - 1.0) < 1e-12
assert abs(peak_dT(0.25, 1.0) - 2.0) < 1e-12
assert abs(sup_bias(3.0, 0.25, 1.0) - 0.75) < 1e-12
assert "residual-confounding identity" in tex
assert "not monotone" in tex
assert "lower-bound/optimistic diagnostic" in tex
print("Cross-branch residual-confounding integration checks passed.")


# Central-thesis and provenance consistency gates.
assert "phantom causal resolution" in tex.lower()
assert "11.4\\times/21.2\\times" in tex
assert "effectively two-hour grid" in tex
assert "x23\\_real\\_hourly\\_fix.csv" in tex
assert tex.index("AI Use Statement") < tex.index("\\begin{thebibliography}")
assert "zero efficiency loss" not in tex
assert "verified multi-season dataset" not in tex
print("Central thesis, provenance, and AI-statement ordering checks passed.")


# Corrected-hourly audit must agree with the promoted appendix values.
hourly=pd.read_csv("reports/bias_law/x23_real_hourly_fix.csv")
for city,n,theta in [("Delhi",9024,0.3703095053),("Mumbai",8113,2.2349491649),("Bengaluru",7775,0.0132185163),("Kolkata",2322,1.7901017124)]:
    r=hourly[(hourly.city==city)&(hourly.frozen_screen==False)].iloc[0]
    assert int(r.N)==n and abs(float(r.theta)-theta)<1e-8
for token in ["0.370","2.235","0.013","1.790","16 distinct"]:
    assert token in tex, token
print("Corrected-hourly manuscript values match the committed audit table.")
