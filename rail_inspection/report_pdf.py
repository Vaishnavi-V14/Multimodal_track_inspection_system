"""Generate PDF inspection reports from detection logs."""

from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def detections_to_pdf_bytes(df: pd.DataFrame, title: str = "Rail Crack Detection Report") -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(
        Paragraph(
            f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC (local)",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    if df.empty:
        story.append(Paragraph("No detection records available.", styles["Normal"]))
        doc.build(story)
        buffer.seek(0)
        return buffer.read()

    total = len(df)
    critical = int((df["severity"] == "critical").sum()) if "severity" in df.columns else 0
    warning = int((df["severity"] == "warning").sum()) if "severity" in df.columns else 0
    avg_conf = float(df["confidence"].mean()) if "confidence" in df.columns else 0.0

    summary_data = [
        ["Metric", "Value"],
        ["Total events", str(total)],
        ["Critical", str(critical)],
        ["Warning", str(warning)],
        ["Avg confidence", f"{avg_conf:.4f}"],
    ]
    summary_tbl = Table(summary_data, colWidths=[3 * inch, 3 * inch])
    summary_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ]
        )
    )
    story.append(summary_tbl)
    story.append(Spacer(1, 0.25 * inch))

    display_cols = list(df.columns)[:14]
    table_rows = [display_cols]
    tail = df.tail(200)
    for _, row in tail.iterrows():
        table_rows.append([str(row.get(c, ""))[:80] for c in display_cols])

    detail_tbl = Table(table_rows, repeatRows=1)
    detail_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2e75b6")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
            ]
        )
    )
    story.append(Paragraph("<b>Recent detection log (last 200 rows)</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(detail_tbl)

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
