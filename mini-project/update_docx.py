import docx

doc = docx.Document("mini-project/Data_Driven_Detection_of_Anomalous_Pollution_Spikes_Report.docx")

# Update Table 0 (Metadata)
table0 = doc.tables[0]
for row in table0.rows:
    for cell in row.cells:
        if "[Add student names and roll numbers]" in cell.text:
            cell.text = (
                "1. Pulkit Kumar (Lead Architect & Diagnostics)\n"
                "2. Vijayalakshmi AL (Statistical Signal Processing)\n"
                "3. Anant Prasad (Data Systems & Pipeline ETL)\n"
                "4. Rama Sridhrut (EDA & Empirical Validation)"
            )
        elif "[Professor / department]" in cell.text:
            cell.text = "Prof. Bharathi R. B.\nSchool of Electrical Engineering"
        elif "Data Science Using Python" in cell.text:
            cell.text = "ELE 4449: Introduction to Data Science (Semester VII)"

# Update Table 7 (Contributions)
table7 = doc.tables[7]
contributions = [
    ("Pulkit Kumar\n(Lead Architect & Diagnostics)", 
     "Scoring Architecture & Mathematical Diagnostics",
     "Formulated the 4-component weighted scoring equation (Section 5). Diagnosed the affine metric compression effect (Section 7), proving mathematically why 100% relative jump thresholds compress under an additive +100 offset. Verified O(1) streaming computational complexity."),
    ("Vijayalakshmi AL\n(Statistical Signal Processing)",
     "Non-Parametric Feature Engineering & Signal Processing",
     "Engineered the Median and IQR robust scaling formulation (Section 4) to eliminate outlier contamination from heavy-tailed pollution distributions. Designed the 6-hour backward-shifted rolling median and rolling MAD baselines, implementing the strict .shift(1) data leakage safeguard."),
    ("Anant Prasad\n(Data Systems & Pipeline ETL)",
     "API Engineering, Temporal Alignment & Data Hygiene",
     "Constructed the automated REST API ingestion pipelines for CPCB CAAQM (XKDR Forum) and ECMWF ERA5 (Open-Meteo). Enforced IST temporal synchronization (701 / 744 hours = 94.2% completeness). Established and defended the Zero-Imputation policy and built offline caching."),
    ("Rama Sridhrut\n(EDA & Empirical Validation)",
     "Exploratory Data Analysis, Visualization & Forensic Case Studies",
     "Directed exploratory data analysis and visual storytelling (Section 7). Built the dual-axis meteorological visualization and severity-versus-suspicion diagnostics. Executed the forensic case study of the November 12-13 peak episode (887 ug/m3), demonstrating the physical distinction between atmospheric stagnation and hardware glitches.")
]

for idx, (member, area, deliverable) in enumerate(contributions):
    row_idx = idx + 1
    table7.rows[row_idx].cells[0].text = member
    table7.rows[row_idx].cells[1].text = area
    table7.rows[row_idx].cells[2].text = deliverable

doc.save("mini-project/Data_Driven_Detection_of_Anomalous_Pollution_Spikes_Report.docx")
print("Successfully updated Data_Driven_Detection_of_Anomalous_Pollution_Spikes_Report.docx with team members, roles, and course metadata!")
