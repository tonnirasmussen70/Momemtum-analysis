from __future__ import annotations

from io import BytesIO

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


def format_pct(x):
    if pd.isna(x):
        return "-"
    return f"{x:.1%}"


def create_pdf(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), rightMargin=24, leftMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    story = [Paragraph("Momentumrapport MVP", styles["Title"]), Spacer(1, 12)]

    cols = ["Ticker", "Sector", "Weight", "1M", "3M", "6M", "12M", "Sharpe", "Sortino", "StopPct", "Signal"]
    visible = [c for c in cols if c in df.columns]
    table_data = [visible]
    for _, row in df.head(25).iterrows():
        table_data.append([
            format_pct(row[c]) if c in ["Weight", "1M", "3M", "6M", "12M", "StopPct"] else
            (f"{row[c]:.2f}" if c in ["Sharpe", "Sortino"] and pd.notna(row[c]) else str(row[c]))
            for c in visible
        ])

    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
    ]))
    story.append(table)
    doc.build(story)
    return buffer.getvalue()
