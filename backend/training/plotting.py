"""Deterministic chart renderers for evaluation artifacts.

Pillow + numpy only (no matplotlib), so the exact same inputs always produce
byte-identical PNGs. Used to generate the confusion matrix and calibration
artifacts committed under ``reports/model-evaluation/`` — which are only ever
written from real, observed values by the evaluation CLI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_TILE = 64            # confusion-matrix cell size in px
_PAD = 16
_LABEL_PAD = 140      # room for class labels on both axes
_FONT_SIZE = 14

Axis = tuple[int, int, int]  # RGB triplet
_BLACK: Axis = (24, 24, 28)
_WHITE: Axis = (250, 250, 250)
_GRID: Axis = (200, 202, 206)
_ACCENT: Axis = (38, 96, 148)


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _interp(a: Axis, b: Axis, t: float) -> Axis:
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _cell_color(value: float, maximum: float) -> Axis:
    t = (value / maximum) ** 0.6 if maximum > 0 else 0.0
    return _interp(_WHITE, _ACCENT, t)


def render_confusion_matrix(
    matrix: list[list[int]] | np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str = "Confusion matrix",
) -> Path:
    """Render a confusion matrix as a deterministic PNG; returns its path."""
    matrix = np.asarray(matrix, dtype=int)
    n = matrix.shape[0]
    if matrix.shape != (n, n):
        raise ValueError(f"Confusion matrix must be square, got {matrix.shape}.")

    width = _LABEL_PAD + n * _TILE + _PAD
    height = _PAD + 56 + _LABEL_PAD + n * _TILE + _LABEL_PAD
    image = Image.new("RGB", (width, height), _WHITE)
    draw = ImageDraw.Draw(image)
    font = _font()
    maximum = int(matrix.max()) if matrix.size else 1

    draw.text((_PAD, _PAD), title, fill=_BLACK, font=font)
    annotation = "Rows: actual class   Columns: predicted class"
    draw.text(
        (width - _PAD - draw.textlength(annotation, font=font), _PAD + 16),
        annotation,
        fill=(90, 94, 100),
        font=font,
    )

    top = 56 + _PAD + _LABEL_PAD
    left = _LABEL_PAD
    for r in range(n):
        for c in range(n):
            value = int(matrix[r, c])
            color = _cell_color(value, maximum)
            draw.rectangle(
                [left + c * _TILE, top + r * _TILE,
                 left + (c + 1) * _TILE, top + (r + 1) * _TILE],
                fill=color,
                outline=_GRID,
                width=1,
            )
            if n * n * maximum > 0 and value > maximum * 0.45 or color == _WHITE:
                text_color = _BLACK
            else:
                text_color = _WHITE
            label = str(value)
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text(
                (left + c * _TILE + (_TILE - tw) / 2, top + r * _TILE + (_TILE - th) / 2),
                label,
                fill=text_color,
                font=font,
            )

    # Axis labels.
    for r, name in enumerate(class_names):
        y = top + r * _TILE + _TILE / 2
        draw.text(
            (_LABEL_PAD - 12, y),
            name,
            fill=_BLACK,
            font=font,
            anchor="rm",
        )
    for c, name in enumerate(class_names):
        x = left + c * _TILE + _TILE / 2
        draw.text(
            (x, top + n * _TILE + 6),
            name,
            fill=_BLACK,
            font=font,
            anchor="mt",
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, "PNG")
    return output_path


def render_reliability_diagram(
    bins: list,
    output_path: Path,
    ece: float,
    title: str = "Reliability diagram",
) -> Path:
    """Render observed accuracy vs mean confidence per bin with a diagonal guide."""
    if not bins:
        raise ValueError("Reliability diagram requires at least one bin.")

    width, height = 640, 640
    margin = 40
    plot = (width - 2 * margin, height - 2 * margin)
    top = height - margin

    image = Image.new("RGB", (width, height + 28), _WHITE)
    draw = ImageDraw.Draw(image)
    font = _font()

    draw.text((margin, 8), f"{title} (ECE = {ece:.4f})", fill=_BLACK, font=font)

    # Diagonal: perfect calibration.
    draw.line(
        [margin, top, margin + plot[0], top - plot[1]],
        fill=_GRID,
        width=2,
    )

    for bin_info in bins:
        confidence = float(bin_info.confidence)
        accuracy = float(bin_info.accuracy)
        x = margin + confidence * plot[0]
        y = top - accuracy * plot[1]
        dx = max(8.0, plot[0] * (float(bin_info.bin_high) - float(bin_info.bin_low)))
        draw.rectangle([x - dx / 2, y, x + dx / 2, top], fill=_ACCENT, outline=_WHITE)
        draw.line([x, top, x, y], fill=_ACCENT, width=2)

    for i, tick in enumerate(np.linspace(0.0, 1.0, 6)):
        x = margin + tick * plot[0]
        y = top - tick * plot[1]
        draw.text((x - 10, top + 4), f"{tick:.1f}", fill=(90, 94, 100), font=font)
        draw.text((margin - 34, y - 6), f"{tick:.1f}", fill=(90, 94, 100), font=font)
        draw.line([x, top, x, top + 4], fill=_GRID)
        draw.line([margin - 4, y, margin, y], fill=_GRID)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, "PNG")
    return output_path


def render_training_history(
    history: dict,
    output_path: Path,
    title: str = "Training history",
) -> Path:
    """Render loss/accuracy curves from a recorded ``model.fit`` history."""
    required = {"loss", "val_loss"}
    if not required.issubset(history):
        raise ValueError("history must contain 'loss' and 'val_loss'.")

    metrics = ["loss", "accuracy"] if "accuracy" in history else ["loss"]
    panels = len(metrics)
    width, height = 700, 300 + 40 * panels
    margin = 44

    image = Image.new("RGB", (width, height), _WHITE)
    draw = ImageDraw.Draw(image)
    font = _font()
    draw.text((margin, 8), title, fill=_BLACK, font=font)

    for p, metric in enumerate(metrics):
        train = np.asarray(history[metric], dtype=np.float64)
        key = f"val_{metric}"
        val = np.asarray(history.get(key, train), dtype=np.float64)
        y0 = 46 + p * ((height - 46) / (panels + 0.2))
        panel_h = (height - 46) / (panels + 0.2) - 26
        combined = np.concatenate([train, val])
        vmin, vmax = float(combined.min()), float(combined.max())
        span = (vmax - vmin) or 1.0

        def _xy(values, eps=0.0):
            xs = np.linspace(margin, width - margin, len(values))
            ys = y0 + panel_h - (values - vmin) / span * panel_h
            return list(zip(xs.tolist(), ys.tolist()))

        draw.text((margin, y0 - 18), metric, fill=_BLACK, font=font)
        draw.line(_xy(train), fill=_ACCENT, width=2)
        draw.line(_xy(val, 0), fill=(200, 90, 60), width=2)
        draw.text((width - margin - 80, y0 - 18), "train/val", fill=(90, 94, 100), font=font)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, "PNG")
    return output_path