import os
import markdown2
from xhtml2pdf import pisa
import re

def convert_latex_to_unicode(text):
    text = text.replace('µg/m³', 'ug/m3')
    text = text.replace('µg/m3', 'ug/m3')
    text = text.replace('R²', 'R2')
    text = text.replace('Δ', 'Delta ')
    text = text.replace('β', 'Beta')
    text = text.replace('Σ', 'Sum')
    text = text.replace('π', 'Pi')
    text = text.replace('μ', 'u')
    text = text.replace('σ', 'Sigma')
    text = text.replace('℃', 'C')
    text = text.replace('°C', 'C')
    text = re.sub(r'\$\$(.*?)\$\$', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\$(.*?)\$', r'\1', text)
    return text

def fix_image_paths(html_content, base_dir):
    """Ensure image paths are absolute for xhtml2pdf."""
    abs_base = os.path.abspath(base_dir)
    # The markdown uses ![alt](../plots/...png)
    html_content = html_content.replace('src="../plots', f'src="{os.path.join(abs_base, "plots")}')
    return html_content

def generate_si_pdf():
    md_path = "manuscript/Supplementary_Information.md"
    if not os.path.exists(md_path):
        print("SI Markdown not found.")
        return
        
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    md_content = convert_latex_to_unicode(md_content)

    html_content = markdown2.markdown(md_content, extras=['tables', 'fenced-code-blocks'])
    html_content = fix_image_paths(html_content, os.getcwd())

    styled_html = f"""
    <html>
    <head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; font-size: 11pt; }}
        h1 {{ color: #2c3e50; font-size: 18pt; text-align: center; margin-bottom: 20px; }}
        h2 {{ color: #2c3e50; font-size: 14pt; margin-top: 20px; border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
        h3 {{ color: #34495e; font-size: 12pt; margin-top: 15px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 9pt; }}
        th, td {{ border: 1px solid #ddd; padding: 6px; text-align: left; }}
        th {{ background-color: #f8f9fa; font-weight: bold; }}
        img {{ max-width: 100%; height: auto; margin: 15px 0; text-align: center; }}
        code {{ background-color: #f4f4f4; padding: 2px 4px; font-family: monospace; font-size: 10pt; }}
        .page-break {{ page-break-before: always; }}
    </style>
    </head>
    <body>
    {html_content}
    </body>
    </html>
    """

    output_filename = "Supplementary_Information_Q1.pdf"
    
    with open(output_filename, "w+b") as result_file:
        pisa_status = pisa.CreatePDF(styled_html, dest=result_file)

    if not pisa_status.err:
        print(f"Supplementary PDF generated successfully: {output_filename}")
    else:
        print("Error generating PDF")

if __name__ == "__main__":
    generate_si_pdf()
