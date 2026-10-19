"""Render the Markdown assignment report to a compact ten-page PDF."""
from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, Image, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle, Preformatted)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports" / "assignment_report.md"
OUTPUT = ROOT / "reports" / "assignment_report.pdf"
NAVY = colors.HexColor("#16324F")
TEAL = colors.HexColor("#167D8D")
PALE = colors.HexColor("#EAF2F5")


def inline(text: str) -> str:
    text = (text.replace("→", " then ").replace("—", "--").replace("–", "-")
            .replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"'))
    value = html.escape(text, quote=False)
    value = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r'<link href="\2" color="#167D8D">\1</link>', value)
    value = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", value)
    return value


def split_sections(lines: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    preamble: list[str] = []
    sections: dict[str, list[str]] = {}
    current = None
    for line in lines:
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        elif current is None:
            preamble.append(line)
        else:
            sections[current].append(line)
    return preamble, sections


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("TitleX", parent=base["Title"], fontName="Helvetica-Bold", fontSize=23, leading=27, textColor=NAVY, spaceAfter=9),
        "subtitle": ParagraphStyle("SubX", parent=base["Normal"], fontName="Helvetica", fontSize=12, leading=15, textColor=TEAL, spaceAfter=10),
        "h2": ParagraphStyle("H2X", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=14, leading=17, textColor=NAVY, spaceBefore=1, spaceAfter=8, keepWithNext=True),
        "h3": ParagraphStyle("H3X", parent=base["Heading3"], fontName="Helvetica-Bold", fontSize=9.3, leading=11, textColor=TEAL, spaceBefore=6, spaceAfter=3, keepWithNext=True),
        "body": ParagraphStyle("BodyX", parent=base["BodyText"], fontName="Helvetica", fontSize=8.3, leading=10.2, alignment=TA_LEFT, spaceAfter=5),
        "small": ParagraphStyle("SmallX", parent=base["BodyText"], fontName="Helvetica", fontSize=7.3, leading=8.8, spaceAfter=3),
        "quote": ParagraphStyle("QuoteX", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=8.2, leading=10, textColor=NAVY, backColor=PALE, borderColor=TEAL, borderWidth=0.6, borderPadding=6, leftIndent=4, rightIndent=4, spaceBefore=4, spaceAfter=7),
        "bullet": ParagraphStyle("BulletX", parent=base["BodyText"], fontName="Helvetica", fontSize=8.2, leading=10, leftIndent=13, firstLineIndent=-8, spaceAfter=2),
        "cell": ParagraphStyle("CellX", parent=base["BodyText"], fontName="Helvetica", fontSize=7, leading=8.3, spaceAfter=0),
        "code": ParagraphStyle("CodeX", parent=base["Code"], fontName="Courier", fontSize=6.3, leading=7.4, backColor=PALE, borderPadding=4, spaceBefore=2, spaceAfter=5),
    }


def parse_table(rows: list[str], st):
    matrix = []
    for row in rows:
        if re.match(r"^\|?\s*:?-{2,}", row):
            continue
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        matrix.append([Paragraph(inline(c), st["cell"]) for c in cells])
    if not matrix:
        return []
    n = max(map(len, matrix))
    widths = [7.05 * inch / n] * n
    table = Table(matrix, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#BCCBD3")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return [table, Spacer(1, 5)]


def parse_body(lines: list[str], st, add_eda_images=False):
    story = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            if code and code[0].strip() == "flowchart LR":
                story.append(Paragraph("<b>Architecture flow:</b> source tables → preparation and EDA → CV model comparison → MLflow and serialized pipeline → Docker image → Kubernetes service → FastAPI predictions; Prometheus scrapes metrics and Grafana presents a dashboard. GitHub Actions repeats linting, data preparation, tests, and training.", st["quote"]))
            else:
                story.append(Preformatted("\n".join(code), st["code"]))
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            story.extend(parse_table(rows, st))
            continue
        if line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), st["h3"]))
            i += 1
            continue
        if line.startswith("- [ ]"):
            story.append(Paragraph("&#9744; " + inline(line[5:].strip()), st["bullet"]))
            i += 1
            continue
        if line.startswith("- "):
            story.append(Paragraph("- " + inline(line[2:]), st["bullet"]))
            i += 1
            continue
        if line.startswith("> "):
            story.append(Paragraph(inline(line[2:]), st["quote"]))
            i += 1
            continue
        if add_eda_images and line.startswith("**Insert final EDA figures"):
            paths = [ROOT / "artifacts/eda/feature_histograms.png", ROOT / "artifacts/eda/correlation_heatmap.png", ROOT / "artifacts/eda/class_balance.png"]
            pics = []
            for path in paths:
                if path.exists():
                    pic = Image(str(path), width=3.38 * inch, height=2.45 * inch, kind="proportional")
                    pics.append(pic)
            if len(pics) >= 2:
                row = [pics[0], pics[1]]
                story.append(Table([row], colWidths=[3.48 * inch, 3.48 * inch], style=TableStyle([("VALIGN", (0,0),(-1,-1),"MIDDLE"), ("ALIGN",(0,0),(-1,-1),"CENTER")])) )
                if len(pics) == 3:
                    story.append(Table([[pics[2]]], colWidths=[7.0 * inch], style=TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER")])) )
            i += 1
            continue
        # Merge soft-wrapped source lines into one Markdown paragraph.
        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(("## ", "### ", "|", "- ", "> ", "```")):
            para.append(lines[i].strip())
            i += 1
        story.append(Paragraph(inline(" ".join(para)), st["body"]))
    return story


def header_footer(canvas, doc):
    canvas.saveState()
    w, h = letter
    canvas.setFillColor(NAVY)
    canvas.setFont("Helvetica", 7.2)
    canvas.drawString(doc.leftMargin, h - 0.45 * inch, "MLOps Assignment 01 | UCI Heart Disease")
    canvas.drawRightString(w - doc.rightMargin, h - 0.45 * inch, "AIMLCZG523")
    canvas.setStrokeColor(colors.HexColor("#BCCBD3"))
    canvas.line(doc.leftMargin, h - 0.53 * inch, w - doc.rightMargin, h - 0.53 * inch)
    canvas.setFillColor(colors.HexColor("#526675"))
    canvas.drawCentredString(w / 2, 0.38 * inch, f"{doc.page} / 10")
    canvas.restoreState()


def main():
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    preamble, sections = split_sections(lines)
    st = styles()
    doc = BaseDocTemplate(str(OUTPUT), pagesize=letter, leftMargin=0.72*inch, rightMargin=0.72*inch, topMargin=0.72*inch, bottomMargin=0.62*inch, title="End-to-End MLOps for UCI Heart Disease Classification", author="MLOps Assignment 01")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal", leftPadding=0, rightPadding=0, topPadding=0.08*inch, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=header_footer)])

    story = []
    h1 = next((line[2:].strip() for line in preamble if line.startswith("# ")), "MLOps Assignment Report")
    story.append(Spacer(1, 0.36*inch))
    story.append(Paragraph(inline(h1), st["title"]))
    story.append(Paragraph("End-to-end data, modeling, tracking, CI/CD, API, and deployment", st["subtitle"]))
    for line in preamble:
        line = line.strip()
        if line.startswith("**") or line.startswith("> "):
            story.append(Paragraph(inline(line[2:-2] if line.startswith("**") and line.endswith("**") else line[2:] if line.startswith("> ") else line), st["small"] if line.startswith("**") else st["quote"]))
    story.append(Spacer(1, 8))
    story.extend(parse_body(sections.get("Executive summary", []), st))
    story.append(PageBreak())

    page_sections = [
        ["1. Problem framing and source data"],
        ["2. Exploratory analysis"],
        ["3. Feature engineering and model development"],
        ["4. Experiment tracking and packaging"],
        ["5. CI and automated tests", "9. Reproducibility and setup"],
        ["6. Container and API interface"],
        ["7. Deployment architecture"],
        ["8. Monitoring, logging, and operational considerations"],
        ["10. Limitations, ethics, and next steps", "Submission evidence and references"],
    ]
    for idx, names in enumerate(page_sections):
        for name in names:
            if name in sections:
                story.append(Paragraph(inline(name), st["h2"]))
                story.extend(parse_body(sections[name], st, add_eda_images=(name == "2. Exploratory analysis")))
        if idx < len(page_sections) - 1:
            story.append(PageBreak())
    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    main()
