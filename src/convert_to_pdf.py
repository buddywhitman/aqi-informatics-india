import os
import markdown2
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def markdown_to_pdf(md_file, pdf_file):
    if not os.path.exists(md_file):
        print(f"File {md_file} not found.")
        return

    with open(md_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Create PDF
    doc = SimpleDocTemplate(pdf_file, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = styles['Title']
    h1_style = styles['Heading1']
    h2_style = styles['Heading2']
    normal_style = styles['Normal']
    
    elements = []
    
    # Split content into sections (very basic parser)
    lines = content.split('\n')
    for line in lines:
        if line.startswith('# '):
            elements.append(Paragraph(line[2:], title_style))
            elements.append(Spacer(1, 12))
        elif line.startswith('## '):
            elements.append(Paragraph(line[3:], h1_style))
            elements.append(Spacer(1, 10))
        elif line.startswith('### '):
            elements.append(Paragraph(line[4:], h2_style))
            elements.append(Spacer(1, 8))
        elif line.startswith('!['):
            # Placeholder for image handling
            # Extract path: ![alt](path)
            start = line.find('(') + 1
            end = line.find(')')
            img_path = line[start:end]
            if os.path.exists(img_path):
                img = Image(img_path, width=400, height=250)
                elements.append(img)
                elements.append(Spacer(1, 12))
        elif line.startswith('|'):
            # Table handling (very basic)
            continue # Skip tables for now or handle later
        else:
            if line.strip():
                elements.append(Paragraph(line, normal_style))
                elements.append(Spacer(1, 6))

    doc.build(elements)
    print(f"PDF generated: {pdf_file}")

if __name__ == "__main__":
    markdown_to_pdf("Final_Manuscript.md", "Research_Manuscript.pdf")
