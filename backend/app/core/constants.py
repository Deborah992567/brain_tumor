"""Shared class constants."""

from __future__ import annotations

CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary_tumor"]

DISPLAY_LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "no_tumor": "No Tumor",
    "pituitary_tumor": "Pituitary Tumor",
}

CONFIDENCE_LOW_THRESHOLD = 0.60


def display_label(class_key: str) -> str:
    return DISPLAY_LABELS.get(class_key, class_key.replace("_", " ").title())