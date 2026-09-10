"""Image preprocessing for inference.

The canonical preprocessing pipeline lives in :mod:`app.core.preprocessing`
and is shared verbatim with the training pipeline. This service layers file
opening and the batched ``PreparedImage`` on top of it, so inference always
feeds the model the exact transformation it was trained with.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.core.preprocessing import add_batch_dimension, preprocess_image
from app.services.errors import ValidationError
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
    """RGB, bilinear resize and ``1/255`` rescale — the canonical pipeline.

    The model input is always square; ``prepare_image`` passes equal width and
    height. Non-square sizes fall back to the size derived from ``input_size``.
    """
    if width == height and width > 0:
        return preprocess_image(image, width)
    size = max(width, height) or 64
    return preprocess_image(image, size)


def prepare_image(data: bytes, input_size: int = 64) -> PreparedImage:
    image = open_image_from_bytes(data)
    width, height = image.size
    rgb = image.convert("RGB") if image.mode != "RGB" else image
    display = np.asarray(rgb, dtype=np.uint8)
    tensor = add_batch_dimension(normalize_image(image, input_size, input_size))
    return PreparedImage(
        tensor=tensor,
        array_rgb=display,
        mode=image.mode,
        width=width,
        height=height,
    )