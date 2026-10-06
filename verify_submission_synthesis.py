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
