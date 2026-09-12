"""Explainable AI services.

Grad-CAM heatmaps are computed by the model backend; this service renders them
into an "AI Attention Visualization" overlay. The overlay highlights regions
that influenced the model prediction. It is NOT a clinically validated tumor
boundary and is never presented as segmentation.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

# A restrained scientific colormap (dark blue -> cyan -> green -> yellow -> red)
_COLORMAP: list[tuple[int, int, int]] = [
    (12, 27, 75),
    (10, 80, 130),
    (20, 130, 180),
    (40, 180, 150),
    (90, 210, 90),
    (210, 230, 70),
    (255, 160, 40),
    (255, 70, 40),
    (190, 20, 40),
]

_HUE_STOPS = np.linspace(0.0, 1.0, len(_COLORMAP))
_LUT = np.array(_COLORMAP, dtype=np.float32)  # (N, 3)

HEATMAP_SIZE = 224  # visualization resolution before resize (keeps quality)


def _apply_colormap(heatmap: np.ndarray) -> np.ndarray:
    """Map a [0,1] heatmap to an RGB array via the colormap LUT."""
    flat = heatmap.reshape(-1)
    indices = np.interp(flat, _HUE_STOPS, np.arange(len(_COLORMAP), dtype=np.float32))
    low = np.floor(indices).astype(np.int32)
    high = np.minimum(low + 1, len(_COLORMAP) - 1)
    frac = indices - low
    color = (_LUT[low] * (1.0 - frac)[:, None]) + (_LUT[high] * frac[:, None])
    return color.reshape(heatmap.shape[0], heatmap.shape[1], 3).astype(np.uint8)


def render_attention_overlay(
    display_rgb: np.ndarray,  # (H, W, 3) uint8
    heatmap: np.ndarray,      # (S, S) float32 in [0, 1]
    alpha: float = 0.55,
) -> bytes:
    """Return PNG bytes combining the original MRI with the attention heatmap."""
    img = Image.fromarray(display_rgb).convert("RGB")
    width, height = img.size

    hm_img = Image.fromarray(heatmap, mode="F").resize(
        (width, height), Image.Resampling.BILINEAR
    )
    hm_resized = np.asarray(hm_img, dtype=np.float32)

    colored = Image.fromarray(_apply_colormap(hm_resized)).convert("RGB")
    overlay = Image.blend(img, colored, alpha=alpha)

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    return buf.getvalue()


def render_heatmap_png(heatmap: np.ndarray) -> bytes:
    """Render the raw heatmap as a grayscale PNG (diagnostic/debug use)."""
    hm_img = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L")
    buf = io.BytesIO()
    hm_img.save(buf, format="PNG")
    return buf.getvalue()