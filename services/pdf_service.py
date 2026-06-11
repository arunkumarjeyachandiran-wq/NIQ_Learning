from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table as RLTable, TableStyle

from Agents.Models import ReportOutput

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _safe_filename(label: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in label
    ).strip("_")[:120]


def export_report_pdf(topic: str, report: ReportOutput) -> str:
    """Export the structured report as a PDF file."""
    filename = f"{_safe_filename(topic)}.pdf"
    pdf_path = REPORT_DIR / filename

    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(report.title or topic, styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(report.executive_summary, styles["BodyText"]))
    elements.append(Spacer(1, 12))

    for section in report.sections:
        elements.append(Paragraph(section.heading, styles["Heading2"]))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(section.content, styles["BodyText"]))
        elements.append(Spacer(1, 10))

    if report.tables:
        elements.append(Paragraph("Tables", styles["Heading2"]))
        elements.append(Spacer(1, 6))
        for table in report.tables:
            elements.append(Paragraph(table.title, styles["Heading3"]))
            data = [table.headers] + table.rows
            report_table = RLTable(data, hAlign="LEFT")
            report_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4B6CB7")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), colors.gray),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ])
            )
            elements.append(report_table)
            elements.append(Spacer(1, 12))

    if report.charts:
        elements.append(Paragraph("Charts", styles["Heading2"]))
        elements.append(Spacer(1, 6))
        for chart in report.charts:
            elements.append(Paragraph(chart.title, styles["Heading3"]))
            elements.append(Paragraph(
                f"Type: {chart.chart_type}, X: {chart.x_label}, Y: {chart.y_label}",
                styles["BodyText"],
            ))
            elements.append(Spacer(1, 8))

    if report.images:
        elements.append(Paragraph("Images", styles["Heading2"]))
        elements.append(Spacer(1, 6))
        for image in report.images:
            elements.append(Paragraph(image.title, styles["Heading3"]))
            elements.append(Paragraph(
                f"Search query: {image.search_query}", styles["BodyText"]
            ))
            elements.append(Paragraph(image.reason, styles["BodyText"]))
            elements.append(Spacer(1, 8))

    elements.append(Spacer(1, 12))
    elements.append(Paragraph(report.conclusion, styles["BodyText"]))

    doc.build(elements)

    return str(pdf_path)
