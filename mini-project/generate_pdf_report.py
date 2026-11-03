import base64
from pathlib import Path
from playwright.sync_api import sync_playwright

def img_to_b64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("utf-8")

fig_dir = Path("mini-project/figures")
b64_fig1 = img_to_b64(fig_dir / "fig1_timeline.png")
b64_fig2 = img_to_b64(fig_dir / "fig2_scatter.png")
b64_fig3 = img_to_b64(fig_dir / "fig3_weather.png")
b64_fig4 = img_to_b64(fig_dir / "fig4_distributions.png")
b64_fig5 = img_to_b64(fig_dir / "fig5_case_study.png")

template_path = Path("mini-project/report_template.html")
html_text = template_path.read_text(encoding="utf-8")

html_text = html_text.replace("%%B64_FIG1%%", b64_fig1)
html_text = html_text.replace("%%B64_FIG2%%", b64_fig2)
html_text = html_text.replace("%%B64_FIG3%%", b64_fig3)
html_text = html_text.replace("%%B64_FIG4%%", b64_fig4)
html_text = html_text.replace("%%B64_FIG5%%", b64_fig5)

out_html = Path("mini-project/report_publication.html")
out_html.write_text(html_text, encoding="utf-8")
print(f"Generated {out_html}")

pdf_path = Path("mini-project/Data_Driven_Detection_of_Anomalous_Pollution_Spikes_Report.pdf")
print("Compiling camera-ready PDF via Playwright...")
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(f"file:///{out_html.resolve().as_posix()}", wait_until="networkidle")
    page.pdf(
        path=pdf_path.resolve().as_posix(),
        format="A4",
        print_background=True,
        margin={"top": "15mm", "bottom": "18mm", "left": "14mm", "right": "14mm"},
        display_header_footer=True,
        header_template='<div style="font-size: 8pt; font-family: serif; width: 100%; text-align: center; color: #666;">ELE 4449: Introduction to Data Science &bull; Capstone Mini-Project Report &bull; Prof. Bharathi R. B.</div>',
        footer_template='<div style="font-size: 8pt; font-family: serif; width: 100%; display: flex; justify-content: space-between; padding: 0 20mm; color: #666;"><span>School of Electrical Engineering</span><span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span><span>Alipur, Delhi Monitoring</span></div>'
    )
    browser.close()

print(f"Camera-ready publication PDF successfully compiled: {pdf_path} (size: {pdf_path.stat().st_size} bytes)")
