"""Upload validation tests.

Validation must reject invalid input BEFORE any model runs, and it must never
allow an arbitrary prediction from a corrupt or unsupported image.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.services.errors import ValidationError
from app.services.validation import validate_upload


def _png_bytes(size=96, noise=True):
    if noise:
        array = np.random.default_rng(0).integers(0, 255, (size, size, 3), dtype=np.uint8)
        image = Image.fromarray(array)
    else:
        image = Image.new("RGB", (size, size), (128, 128, 128))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue(), "brain.png"


def test_valid_png_passes():
    data, name = _png_bytes()
    result = validate_upload(name, data)
    assert result.checksum
    assert result.width == 96
    assert result.height == 96
    assert result.extension == ".png"


def test_rejects_unsupported_extension():
    with pytest.raises(ValidationError) as exc:
        validate_upload("report.pdf", b"%PDF-1.4 fake")
    assert "Unsupported file format" in str(exc.value)


def test_rejects_empty_file():
    with pytest.raises(ValidationError):
        validate_upload("empty.png", b"")


def test_rejects_oversized_file(sample_mri_bytes):
    data, name = _png_bytes()
    from app.core.config import settings

    original = settings.max_upload_size_mb
    settings.max_upload_size_mb = 0
    try:
        with pytest.raises(ValidationError):
            validate_upload(name, data)
    finally:
        settings.max_upload_size_mb = original


def test_rejects_mime_mismatch():
    data, _ = _png_bytes()
    with pytest.raises(ValidationError):
        validate_upload("brain.txt", data)


def test_rejects_corrupt_image():
    with pytest.raises(ValidationError):
        validate_upload("broken.jpg", b"\xff\xd8\xff not really a jpeg")


def test_rejects_image_too_small():
    data, name = _png_bytes(size=16)
    with pytest.raises(ValidationError):
        validate_upload(name, data)


def test_rejects_effectively_blank_image():
    data, name = _png_bytes(size=96)
    blank = Image.new("RGB", (96, 96), (128, 128, 128))
    buf = io.BytesIO()
    blank.save(buf, format="PNG")
    with pytest.raises(ValidationError):
        validate_upload(name, buf.getvalue())


def test_rejects_bmp_content_under_png_name():
    buf = io.BytesIO()
    Image.new("RGB", (96, 96), (10, 10, 10)).save(buf, format="BMP")
    with pytest.raises(ValidationError):
        validate_upload("brain.png", buf.getvalue())