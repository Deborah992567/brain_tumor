"""Single source of truth for model input preprocessing.

Both training and inference must apply exactly the same transformation so the
model's expected input distribution does not drift. This module is the one
place that defines the canonical pipeline:

    1. ensure the image is RGB (grayscale is broadcast to 3 channels)
    2. bilinear resize (PIL) to ``(input_size, input_size)``
    3. cast to ``float32`` and rescale by ``1 / 255``

The legacy model was trained with ``ImageDataGenerator(rescale=1/255)`` at
``target_size=(64, 64)`` (Keras' default interpolation is PIL bilinear and the
images were converted to RGB). ``PREPROCESSING_VERSION`` is recorded in the
model registry and in evaluation artifacts so results are traceable to this
exact definition.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

PREPROCESSING_VERSION = "1.0.0-255-bilinear-rgb"

MAX_CHANNEL = 255.0


def resize_rgb(image: Image.Image, input_size: int) -> Image.Image:
    """Ensure RGB and resize with PIL bilinear (matches Keras ``target_size``)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image.resize((input_size, input_size), Image.Resampling.BILINEAR)


def image_to_tensor(image: Image.Image) -> np.ndarray:
    """Convert an RGB image to a float32 tensor in ``[0, 1]`` via ``1/255``."""
    return np.asarray(image, dtype=np.float32) / MAX_CHANNEL


def preprocess_image(image: Image.Image, input_size: int) -> np.ndarray:
    """Canonical preprocessing for a PIL image: return ``(size, size, 3)`` float32."""
    return image_to_tensor(resize_rgb(image, input_size))


def preprocess_pixels(rgb: np.ndarray, input_size: int) -> np.ndarray:
    """Canonical preprocessing for an ``HxWx3`` uint8 array (training path).

    Produces exactly the same tensor as :func:`preprocess_image` for the same
    pixels, so the training pipeline and the inference service are guaranteed
    to feed identical inputs to the model.
    """
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected an HxWx3 RGB array.")
    image = Image.fromarray(np.asarray(arr, dtype=np.uint8), mode="RGB")
    return preprocess_image(image, input_size)


def add_batch_dimension(tensor: np.ndarray) -> np.ndarray:
    """Add the leading batch axis -> ``(1, size, size, 3)``."""
    return np.expand_dims(tensor, axis=0)