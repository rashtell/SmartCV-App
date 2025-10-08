from fpdf import FPDF


def export_to_pdf(content: str, filename: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", size=11)
    for line in content.split("\n"):
        if line.strip().startswith("**") and line.strip().endswith("**"):
            pdf.set_font("Helvetica", "B", 12)
            text = line.strip("* ")
            pdf.multi_cell(0, 6, text)
            pdf.set_font("Helvetica", size=11)
        else:
            pdf.multi_cell(0, 5, line)
    pdf.output(filename)
    return filename
