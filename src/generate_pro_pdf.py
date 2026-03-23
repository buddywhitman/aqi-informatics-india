import os
import markdown2
from xhtml2pdf import pisa
import re

def convert_latex_to_unicode(text):
    """Replace common LaTeX sequences with Unicode, carefully avoiding Markdown syntax."""
    # 1. Temporarily hide images/paths to avoid mangling them
    placeholders = []
    def hide_paths(match):
        placeholders.append(match.group(0))
        return f"__PATH_PLACEHOLDER_{len(placeholders)-1}__"
    
    # Matches ![alt](path)
    text = re.sub(r'!\[.*?\]\(.*?\)', hide_paths, text)

    replacements = {
        r'\Delta': 'Δ',
        r'\phi': 'φ',
        r'\beta': 'β',
        r'\tau': 'τ',
        r'\mu': 'μ',
        r'\sum': 'Σ',
        r'\subseteq': '⊆',
        r'\setminus': '∖',
        r'\cup': '∪',
        r'\times': '×',
        r'\cdot': '·',
        r'\infty': '∞',
        r'^{sin}': 'ᔆᴵᴺ',
        r'^{cos}': 'ᶜᴼˢ',
        r'_{t-w...t}': 'ₜ₋𝓌...ₜ',
        r'_i': 'ᵢ',
        r'_t': 'ₜ',
        r'_0': '₀',
        r'_1': '₁',
        r'^2': '²',
        r'^3': '³',
        r'µg/m³': 'μg/m³',
        r'ug/m3': 'μg/m³'
    }
    for latex, unicode_val in replacements.items():
        text = text.replace(latex, unicode_val)
    
    # Special specific math terms
    text = text.replace('Delta M', 'ΔM').replace('Delta X', 'ΔX')
    
    # Remove $ signs
    text = text.replace('$', '')

    # 2. Restore images/paths
    for i, placeholder in enumerate(placeholders):
        text = text.replace(f"__PATH_PLACEHOLDER_{i}__", placeholder)
        
    return text

def generate_pro_pdf():
    # 1. Load the manuscript
    md_path = "Final_Nature_Manuscript_Extended.md"
    if not os.path.exists(md_path):
        # Fallback to the other extended name if needed
        md_path = "manuscript/Final_Nature_Manuscript_Extended.md"
        
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # 2. Pre-process text (Math cleanup)
    md_text = convert_latex_to_unicode(md_text)

    # 3. Convert Markdown to HTML
    # Extras: tables, fenced-code-blocks, toc, etc.
    html_content = markdown2.markdown(md_text, extras=["tables", "fenced-code-blocks", "cuddled-lists"])

    # 4. Add CSS for Journal Style
    css = """
    @page {
        size: a4 portrait;
        margin: 2cm;
        @frame footer_frame {
            -pdf-frame-content: footer_content;
            left: 2cm; width: 17cm; bottom: 1cm; height: 1cm;
        }
    }
    body {
        font-family: "Times New Roman", serif;
        font-size: 11pt;
        line-height: 1.5;
        color: black;
        text-align: justify;
    }
    h1 {
        font-size: 18pt;
        text-align: center;
        margin-bottom: 20pt;
        font-weight: bold;
    }
    h2 {
        font-size: 14pt;
        border-bottom: 1px solid #ccc;
        margin-top: 20pt;
        margin-bottom: 10pt;
        font-weight: bold;
    }
    h3 {
        font-size: 12pt;
        margin-top: 15pt;
        font-weight: bold;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 15pt;
    }
    th, td {
        border: 1px solid black;
        padding: 5pt;
        font-size: 10pt;
    }
    th {
        background-color: #f2f2f2;
    }
    img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 10pt auto;
    }
    .footer {
        text-align: center;
        font-size: 9pt;
        color: #666;
    }
    """

    # 5. Build full HTML
    full_html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>{css}</style>
    </head>
    <body>
        <div id="content">
            {html_content}
        </div>
        <div id="footer_content" class="footer">
            Page <pdf:pagenumber>
        </div>
    </body>
    </html>
    """

    # 6. Convert to PDF
    output_filename = "Final_Q1_Manuscript_30Pages.pdf"
    with open(output_filename, "wb") as result_file:
        pisa_status = pisa.CreatePDF(full_html, dest=result_file)

    if not pisa_status.err:
        print(f"Professional PDF generated successfully: {output_filename}")
    else:
        print("Error generating PDF")

if __name__ == "__main__":
    generate_pro_pdf()
