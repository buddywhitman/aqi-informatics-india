import markdown2
from weasyprint import HTML, CSS
import os

def generate_pdf():
    # Read the markdown
    with open("Final_Nature_Manuscript_Extended.md", "r", encoding="utf-8") as f:
        md_text = f.read()

    # Convert to HTML with tables support
    html_content = markdown2.markdown(md_text, extras=["tables", "fenced-code-blocks", "cuddled-lists"])

    # Wrap in basic HTML structure
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Manuscript</title>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Define CSS for a Nature/Lancet style manuscript
    css_string = """
    @page {
        size: A4;
        margin: 2.5cm;
        @bottom-center {
            content: counter(page);
            font-family: "Times New Roman", Times, serif;
            font-size: 10pt;
        }
    }
    body {
        font-family: "Times New Roman", Times, serif;
        font-size: 12pt;
        line-height: 1.6;
        color: #000;
        text-align: justify;
    }
    h1 {
        font-family: Arial, Helvetica, sans-serif;
        font-size: 18pt;
        font-weight: bold;
        text-align: center;
        margin-bottom: 24pt;
        page-break-after: avoid;
    }
    h2 {
        font-family: Arial, Helvetica, sans-serif;
        font-size: 14pt;
        font-weight: bold;
        margin-top: 24pt;
        margin-bottom: 12pt;
        page-break-after: avoid;
    }
    h3, h4 {
        font-family: Arial, Helvetica, sans-serif;
        font-size: 12pt;
        font-weight: bold;
        margin-top: 18pt;
        margin-bottom: 6pt;
        page-break-after: avoid;
    }
    p {
        margin-bottom: 12pt;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 12pt;
        margin-bottom: 24pt;
        page-break-inside: avoid;
    }
    th, td {
        border-top: 1px solid #000;
        border-bottom: 1px solid #000;
        padding: 8pt;
        text-align: left;
    }
    th {
        font-weight: bold;
        background-color: #f9f9f9;
    }
    img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 20px auto;
        page-break-inside: avoid;
    }
    """

    # Generate PDF
    HTML(string=full_html, base_url=os.getcwd()).write_pdf(
        "Final_Q1_Manuscript_30Pages.pdf",
        stylesheets=[CSS(string=css_string)]
    )
    print("Successfully generated Final_Q1_Manuscript_30Pages.pdf")

if __name__ == "__main__":
    generate_pdf()
