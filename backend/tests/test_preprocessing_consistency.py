"""Training/inference preprocessing consistency tests.

The single source of truth for preprocessing is ``app.core.preprocessing``.
Both the inference service (``app.services.preprocessing``) and the training
pipeline (``training.augmentation`` / ``training.data``) must produce byte-for-byte
identical model inputs for the same source image.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core.preprocessing import (
    PREPROCESSING_VERSION,
    preprocess_image,
    preprocess_pixels,
    resize_rgb,
)
from app.services.preprocessing import normalize_image
from training.data import load_images

REGISTRY = Path(__file__).resolve().parents[1] / "models" / "registry.json"


def _sample_image(size: int = 96, mode: str = "RGB") -> Image.Image:
    rng = np.random.default_rng(7)
    if mode == "L":
        array = rng.integers(0, 255, (size, size), dtype=np.uint8)
    else:
        array = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(array)


def _tensor(image: Image.Image, input_size: int = 64) -> np.ndarray:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    opened = Image.open(buf)
    return normalize_image(opened, input_size, input_size)


def test_inference_matches_shared_canonical_pipeline():
    image = _sample_image()
    inference_tensor = _tensor(image)
    shared_tensor = preprocess_image(image, 64)
    assert np.array_equal(inference_tensor, shared_tensor)


def test_training_pixels_match_inference_pipeline():
    """Preprocessing applied inside the training tf.data pipeline (via
    ``preprocess_pixels``) must match inference path for the same image."""
    image = _sample_image()
    pixels = np.asarray(resize_rgb(image, 64), dtype=np.uint8)
    from_training_path = preprocess_pixels(pixels, 64)
    from_inference_path = _tensor(image)
    assert np.array_equal(from_training_path, from_inference_path)
    assert from_training_path.dtype == np.float32


def test_grayscale_and_rgb_produce_same_distribution():
    """A grayscale MR slice is broadcast to 3 channels on both paths."""
    gray = _sample_image(mode="L")
    gray_tensor = preprocess_image(gray, 64)
    rgb_array = np.asarray(resize_rgb(gray, 64), dtype=np.uint8)
    gray_tensor_pixels = preprocess_pixels(rgb_array, 64)
    assert gray_tensor.shape == (64, 64, 3)
    assert np.array_equal(gray_tensor_pixels, gray_tensor)


def test_pixels_are_in_unit_range():
    image = _sample_image()
    tensor = preprocess_image(image, 64)
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_registry_declares_matching_preprocessing_version():
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    models = data.get("models", [])
    assert models, "registry has no models"
    assert models[0]["preprocessing_version"] == PREPROCESSING_VERSION


def test_data_loader_uses_bilinear_rgb_resize(tmp_path):
    """training.data.load_images applies the same canonical resizing before
    the augment pipeline scales to [0,1]."""
    (tmp_path / "Training").mkdir(parents=True, exist_ok=True)
    class_dirs = ["glioma", "meningioma", "no_tumor", "pituitary_tumor"]
    for label in class_dirs:
        folder = tmp_path / "Training" / label
        folder.mkdir(parents=True, exist_ok=True)
        _sample_image().save(folder / "a.png", format="PNG")
    dataset = load_images(tmp_path / "Training", input_size=64)
    assert dataset.x.shape == (4, 64, 64, 3)
    assert dataset.x.dtype == np.uint8


def test_preprocess_pixels_rejects_bad_shapes():
    with pytest.raises(ValueError):
        preprocess_pixels(np.zeros((64, 64), dtype=np.uint8), 64)
    with pytest.raises(ValueError):
        preprocess_pixels(np.zeros((64, 64, 1), dtype=np.uint8), 64)