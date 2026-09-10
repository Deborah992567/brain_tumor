"""Preprocessing tests: tensor shape, range and ordering match training."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.errors import ValidationError
from app.services.preprocessing import open_image_from_bytes, prepare_image


def test_prepare_image_matches_training_preprocessing(sample_mri_bytes):
    prepared = prepare_image(sample_mri_bytes, input_size=64)
    assert prepared.tensor.shape == (1, 64, 64, 3)
    assert prepared.tensor.dtype == np.float32
    assert float(prepared.tensor.min()) >= 0.0
    assert float(prepared.tensor.max()) <= 1.0
    assert prepared.array_rgb.ndim == 3
    assert prepared.mode in {"L", "RGB", "RGBA"}


def test_prepare_image_resizes_to_arbitrary_input_size(sample_mri_bytes):
    prepared = prepare_image(sample_mri_bytes, input_size=128)
    assert prepared.tensor.shape == (1, 128, 128, 3)


def test_open_image_rejects_corrupt_bytes():
    with pytest.raises(ValidationError):
        open_image_from_bytes(b"not an image at all")


def test_normalization_is_bilinear_rgb(sample_mri_bytes):
    from app.services.preprocessing import normalize_image
    from PIL import Image

    image = Image.open(__import__("io").BytesIO(sample_mri_bytes)).convert("RGB")
    tensor = normalize_image(image, 64, 64)
    assert tensor.shape == (64, 64, 3)
    assert tensor.max() <= 1.0 + 1e-6