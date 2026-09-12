"""Unit tests for Grad-CAM overlay rendering and heatmap geometry.

Ensures the attention overlay is aligned to the submitted image (same
dimensions) and that the raw heatmap PNG is a valid file. Alignment is the
core correctness property: the heatmap and the displayed image must share
the same spatial extent so the overlay is not misleading.
"""

from __future__ import annotations

import io
import numpy as np
import pytest
from PIL import Image

from app.services.explainability import (
    render_attention_overlay,
    render_heatmap_png,
)


def test_overlay_matches_display_dimensions():
    rng = np.random.default_rng(3)
    display = rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)
    heatmap = rng.random((64, 64)).astype(np.float32)
    png = render_attention_overlay(display, heatmap)
    with Image.open(io.BytesIO(png)) as img:
        assert img.format == "PNG"
        assert img.size == (128, 96)


def test_overlay_accepts_square_model_heatmap():
    rng = np.random.default_rng(4)
    display = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    heatmap = np.zeros((64, 64), dtype=np.float32)
    heatmap[20:44, 20:44] = 1.0
    png = render_attention_overlay(display, heatmap)
    with Image.open(io.BytesIO(png)) as img:
        assert img.size == (224, 224)


def test_heatmap_png_is_valid_grayscale():
    heatmap = np.linspace(0.0, 1.0, 64 * 64).reshape(64, 64).astype(np.float32)
    png = render_heatmap_png(heatmap)
    with Image.open(io.BytesIO(png)) as img:
        assert img.format == "PNG"
        assert img.mode == "L"
        assert img.size == (64, 64)


def test_heatmap_values_clamped():
    rng = np.random.default_rng(5)
    heatmap = rng.normal(0.5, 1.0, (64, 64)).astype(np.float32)
    png = render_heatmap_png(heatmap)
    with Image.open(io.BytesIO(png)) as img:
        arr = np.asarray(img)
        assert arr.min() >= 0
        assert arr.max() <= 255