"""Shared class constants."""

from __future__ import annotations

from app.core.preprocessing import PREPROCESSING_VERSION

CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary_tumor"]

DISPLAY_LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "no_tumor": "No Tumor",
    "pituitary_tumor": "Pituitary Tumor",
}

CONFIDENCE_LOW_THRESHOLD = 0.60

SUPPORTED_CLASSES_NOTE = (
    "This model was trained to recognize exactly four MRI classes: "
    f"{', '.join(DISPLAY_LABELS[c] for c in CLASSES)}. "
    "Other tumor types (for example brain metastases) are not part of its "
    "label space, so results for such images are unreliable even when the "
    "nominal probability is high."
)


def display_label(class_key: str) -> str:
    return DISPLAY_LABELS.get(class_key, class_key.replace("_", " ").title())


def preprocessing_version_info() -> str:
    return PREPROCESSING_VERSION