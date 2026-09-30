from fpdf import FPDF
import os

class ManuscriptPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font('helvetica', 'I', 8)
            self.cell(0, 10, 'Deep Spatiotemporal Informatics - High-Resolution Pan-India Study', 0, 0, 'R')
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf():
    pdf = ManuscriptPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title Page
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 16)
    title = 'Deep Spatiotemporal Informatics, Deep Hybrid Architectures, and Causal Evaluation of Urban Air Quality: A Pan-India High-Resolution Study (2021-2026)'
    pdf.multi_cell(0, 10, title, 0, 'C')
    pdf.ln(20)
    
    pdf.set_font('helvetica', '', 12)
    pdf.multi_cell(0, 10, 'Target Journals: Nature Sustainability | The Lancet Planetary Health | ES&T', 0, 'C')
    pdf.ln(20)

    with open("Final_Nature_Manuscript_Extended.md", "r", encoding="utf-8") as f:
        content = f.read()
        # Replace problematic unicode
        content = content.replace('\u2013', '-').replace('\u2014', '-').replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
        content = content.replace('µg/m³', 'ug/m3').replace('³', '3').replace('²', '2')
        content = content.replace('$\Delta M$', 'Delta M').replace('$\Delta X$', 'Delta X').replace('$\beta$', 'beta')
        content = content.replace('$Y_t$', 'Yt').replace('$ws$', 'ws').replace('$Pop$', 'Pop')
        content = content.replace('$\phi_i$', 'phi_i').replace('$N$', 'N').replace('$S$', 'S').replace('$M$', 'M')
        content = content.replace('$f_x(S)$', 'fx(S)')
        lines = content.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            pdf.ln(5)
            continue
            
        if line.startswith('# '):
            continue
        elif line.startswith('## '):
            pdf.set_font('helvetica', 'B', 14)
            pdf.multi_cell(190, 10, line[3:], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font('helvetica', '', 12)
        elif line.startswith('### '):
            pdf.set_font('helvetica', 'B', 12)
            pdf.multi_cell(190, 10, line[4:], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font('helvetica', '', 12)
        elif line.startswith('#### '):
            pdf.set_font('helvetica', 'BI', 11)
            pdf.multi_cell(190, 8, line[5:], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font('helvetica', '', 12)
        elif line.startswith('!['):
            # Extract image path: ![alt](path)
            start = line.find('(') + 1
            end = line.find(')')
            img_path = line[start:end]
            if os.path.exists(img_path):
                # Center the image
                pdf.ln(5)
                # Max width 170 to keep some padding
                pdf.image(img_path, x=20, w=170)
                pdf.ln(5)
                # Caption (from alt text)
                alt_text = line[line.find('[')+1:line.find(']')]
                pdf.set_font('helvetica', 'I', 9)
                pdf.multi_cell(190, 5, alt_text, align='C', new_x="LMARGIN", new_y="NEXT")
                pdf.set_font('helvetica', '', 12)
                pdf.ln(5)
        elif line.startswith('|'):
            # Table handling - using fixed width
            pdf.set_font('courier', '', 8)
            pdf.multi_cell(190, 5, line, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font('helvetica', '', 12)
        else:
            pdf.multi_cell(190, 7, line, new_x="LMARGIN", new_y="NEXT")

    pdf.output("Final_Q1_Manuscript_30Pages.pdf")
    print("PDF Generated successfully with fpdf2.")

if __name__ == "__main__":
    generate_pdf()
