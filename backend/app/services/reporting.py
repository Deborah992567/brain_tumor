"""Professional PDF report generation.

Reports are titled *AI-Assisted Brain MRI Analysis Report*. They present the
AI prediction with its **model probability** (never framed as clinical
confidence), the full probability distribution, the model + version that
produced it, the uploaded MRI, the AI attention visualization, the supported
classes and the medical disclaimer. They are explicitly NOT medical diagnoses.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.core.config import settings
from app.core.constants import CLASSES, CONFIDENCE_LOW_THRESHOLD, display_label
from app.db.models import AnalysisRecord
from app.services.errors import ReportGenerationError

BRAND = "#0F766E"
TEXT = "#1F2937"
MUTED = "#6B7280"
RULE = "#D1D5DB"


def _iso_z(dt: datetime) -> str:
    return dt.isoformat() + "Z" if dt.tzinfo is None else dt.isoformat()


def _thumb_size(path: Path) -> tuple[int, int]:
    """Return a width/height keeping A4 margins in mind for embedded images."""
    from PIL import Image as PILImage

    with PILImage.open(path) as img:
        w, h = img.size
    max_w = 150 * mm
    max_h = 150 * mm
    scale = min(1.0, max_w / w, max_h / h)
    return int(w * scale), int(h * scale)


def _font_available(name: str) -> bool:
    try:
        pdfmetrics.getFont(name)
        return True
    except Exception:
        return False


def _register_helvetica_family() -> None:
    """Register Helvetica on zipdisk installs where reportlab ships no fonts."""
    for name, file in (
        ("Helvetica", "/System/Library/Fonts/Helvetica.ttc"),
        ("Helvetica-Bold", "/System/Library/Fonts/Helvetica.ttc"),
        ("Helvetica-Oblique", "/System/Library/Fonts/Helvetica.ttc"),
    ):
        try:
            if _font_available(name):
                continue
            pdfmetrics.registerFont(TTFont(name, file, subfontIndex=0))
        except Exception:
            continue


class ReportGenerator:
    """Generate a professional AI-assisted analysis report as a PDF."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or settings.report_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _register_helvetica_family()

    def _styles(self) -> dict:
        base = getSampleStyleSheet()
        body_font = "Helvetica" if _font_available("Helvetica") else "Helvetica"
        bold_font = "Helvetica-Bold" if _font_available("Helvetica-Bold") else "Helvetica"

        styles = {
            "title": ParagraphStyle(
                "title", parent=base["Title"], fontName=bold_font, fontSize=18,
                leading=22, textColor=colors.HexColor(TEXT), alignment=TA_LEFT,
                spaceAfter=2 * mm,
            ),
            "subtitle": ParagraphStyle(
                "subtitle", parent=base["Normal"], fontName=body_font, fontSize=10,
                leading=13, textColor=colors.HexColor(MUTED), spaceAfter=4 * mm,
            ),
            "section": ParagraphStyle(
                "section", parent=base["Heading2"], fontName=bold_font, fontSize=12,
                leading=15, textColor=colors.HexColor(BRAND), spaceBefore=5 * mm,
                spaceAfter=2 * mm, alignment=TA_LEFT,
            ),
            "body": ParagraphStyle(
                "body", parent=base["Normal"], fontName=body_font, fontSize=9.5,
                leading=13, textColor=colors.HexColor(TEXT),
            ),
            "value": ParagraphStyle(
                "value", parent=base["Normal"], fontName="Helvetica-Bold",
                fontSize=9.5, leading=13, textColor=colors.HexColor(TEXT),
            ),
            "disclaimer": ParagraphStyle(
                "disclaimer", parent=base["Normal"], fontName=body_font, fontSize=8.5,
                leading=12, textColor=colors.HexColor(MUTED),
            ),
        }
        return styles

    def generate(self, analysis: AnalysisRecord, overlay_bytes: bytes) -> Path:
        try:
            return self._build(analysis, overlay_bytes)
        except ReportGenerationError:
            raise
        except Exception as exc:
            raise ReportGenerationError(
                f"Report generation failed: {exc}"
            ) from exc

    def _build(self, analysis: AnalysisRecord, overlay_bytes: bytes) -> Path:
        styles = self._styles()
        report_path = self.output_dir / f"report_{analysis.id}.pdf"

        doc = SimpleDocTemplate(
            str(report_path),
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=16 * mm,
            bottomMargin=16 * mm,
            title=f"AI-Assisted Brain MRI Analysis Report - {analysis.id}",
            author=settings.report_company_name,
        )

        probabilities = json.loads(analysis.probabilities_json or "{}")
        max_prob = max(probabilities.values()) if probabilities else 0.0

        story: list = []
        story.append(Paragraph("AI-Assisted Brain MRI Analysis Report", styles["title"]))
        story.append(
            Paragraph(
                f"{settings.report_company_name} - Research and educational analysis output",
                styles["subtitle"],
            )
        )

        # Metadata block
        meta_rows = [
            ["Analysis ID", analysis.id],
            ["Date / time", _iso_z(analysis.created_at)],
            ["Input file", analysis.filename],
            ["Prediction", display_label(analysis.prediction_label)],
            ["Model probability", f"{analysis.confidence:.2%}"],
            ["Model", f"{analysis.model_name} v{analysis.model_version}"],
            ["Processing time", f"{analysis.processing_time_ms or 0} ms"],
        ]
        meta_table = Table(meta_rows, colWidths=[45 * mm, 100 * mm])
        meta_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                    ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor(MUTED)),
                    ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor(TEXT)),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("LINEBELOW", (1, 0), (1, -2), 0.25, colors.HexColor(RULE)),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(meta_table)

        # Probability distribution
        prob_rows = [["Class", "Probability"]]
        for class_key, prob in sorted(
            probabilities.items(), key=lambda kv: kv[1], reverse=True
        ):
            prob_rows.append([display_label(class_key), f"{prob:.2%}"])
        prob_table = Table(prob_rows, colWidths=[90 * mm, 55 * mm])
        prob_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(TEXT)),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E7F0EE")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(RULE)),
                    ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(Paragraph("Probability distribution", styles["section"]))
        story.append(prob_table)
        story.append(
            Paragraph(
                "Values are raw model (softmax) probabilities and are not calibrated; "
                "consider them relative scores, not clinical probabilities.",
                styles["disclaimer"],
            )
        )
        story.append(Spacer(1, 2 * mm))

        # Model scope / label space
        story.append(Paragraph("Model scope and limitations", styles["section"]))
        story.append(
            Paragraph(
                "The model recognizes exactly these four classes: "
                + ", ".join(display_label(c) for c in CLASSES)
                + ". Tumor types outside this label space (for example brain "
                "metastases) are not recognized, and an out-of-distribution image "
                "can still receive a high nominal probability for a wrong class.",
                styles["body"],
            )
        )
        story.append(Spacer(1, 2 * mm))

        # Images: MRI + attention overlay
        story.append(Paragraph("Imaging", styles["section"]))
        if overlay_bytes:
            try:
                overlay_path = self.output_dir / "report_tmp_overlay.png"
                overlay_path.write_bytes(overlay_bytes)
                w, h = _thumb_size(overlay_path)
                overlay = Image(str(overlay_path), width=w, height=h)
                story.append(
                    Paragraph("AI attention visualization", styles["body"])
                )
                story.append(overlay)
                story.append(
                    Spacer(1, 2 * mm)
                )
                story.append(
                    Paragraph(
                        "This visualization highlights image regions that influenced the "
                        "model's prediction. It is not a clinically validated tumor boundary.",
                        styles["disclaimer"],
                    )
                )
            except Exception:
                story.append(
                    Paragraph("(Attention visualization unavailable)", styles["body"])
                )

        story.append(Spacer(1, 6 * mm))

        # Interpretation guidance (low confidence / abstention)
        if analysis.low_confidence:
            story.append(Paragraph("Uncertainty notice", styles["section"]))
            story.append(
                Paragraph(
                    f"Classification is flagged as unavailable: the model probability "
                    f"({analysis.confidence:.1%}) is below the "
                    f"{CONFIDENCE_LOW_THRESHOLD:.0%} reliability threshold. This "
                    "result should not be used to draw a conclusion and must not "
                    "be treated as definitive.",
                    styles["body"],
                )
            )
            story.append(Spacer(1, 2 * mm))

        # Medical disclaimer
        story.append(Paragraph("Medical disclaimer", styles["section"]))
        story.append(
            Paragraph(
                settings.medical_disclaimer,
                styles["disclaimer"],
            )
        )
        story.append(
            Spacer(1, 3 * mm)
        )
        story.append(
            Paragraph(
                "AI-assisted analysis is provided for research and educational purposes. "
                "It does not replace a radiologist or a physician.",
                styles["disclaimer"],
            )
        )

        doc.build(story)
        # Remove temp overlay copy used for embedding
        tmp = self.output_dir / "report_tmp_overlay.png"
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return report_path