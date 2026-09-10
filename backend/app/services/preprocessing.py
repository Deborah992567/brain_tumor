"""Image preprocessing.

Preprocessing must match the preprocessing used during training. The legacy
model was trained with ``ImageDataGenerator(rescale=1./255)`` on images loaded
at ``target_size=(64, 64)`` (PIL bilinear resizing, RGB). This service
reproduces that procedure for any registered model.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.services.errors import ValidationError

PIL_LOAD_ERROR_HINT = (
    "Unable to analyze this image. Please upload a valid brain MRI image."
)


@dataclass
class PreparedImage:
    tensor: np.ndarray          # (1, H, W, 3) float32 in [0, 1], model input
    array_rgb: np.ndarray       # (H, W, 3) uint8 display copy
    mode: str                   # original mode, e.g. "RGB" or "L"
    width: int
    height: int


def open_image_from_bytes(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:  # Pillow raises various exceptions for corrupt data
        raise ValidationError(PIL_LOAD_ERROR_HINT) from exc
    if image is None:
        raise ValidationError(PIL_LOAD_ERROR_HINT)
    return image


def normalize_image(image: Image.Image, width: int, height: int) -> np.ndarray:
    """Convert to RGB, resize and rescale to [0, 1] exactly like training."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    return array / 255.0


def prepare_image(data: bytes, input_size: int = 64) -> PreparedImage:
    image = open_image_from_bytes(data)
    width, height = image.size
    rgb = image.convert("RGB") if image.mode != "RGB" else image
    display = np.asarray(rgb, dtype=np.uint8)
    tensor = normalize_image(image, input_size, input_size)
    tensor = np.expand_dims(tensor, axis=0)
    return PreparedImage(
        tensor=tensor,
        array_rgb=display,
        mode=image.mode,
        width=width,
        height=height,
    )