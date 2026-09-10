"""MRI upload validation.

Validation runs before any inference is attempted. If validation fails the
model is never executed and a meaningful error is returned.
"""

from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass, field

from PIL import Image

from app.core.config import settings
from app.services import preprocessing
from app.services.errors import ValidationError

ALLOWED_MIME_BY_EXT = {
    ".jpg": {"image/jpeg"},
    ".jpeg": {"image/jpeg"},
    ".png": {"image/png"},
    ".bmp": {"image/bmp"},
    ".tif": {"image/tiff"},
    ".tiff": {"image/tiff"},
}

MIN_DETECTABLE_ACTIVITY = 3.0  # standard deviation threshold for non-blank images
MIN_SUITABLE_DIMENSION = 128   # very small thumbnails are not useful MRI candidates


@dataclass
class ValidatedFile:
    bytes: bytes
    size: int
    extension: str
    mime: str
    sha256: str
    format: str = ""
    mode: str = ""
    width: int = 0
    height: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def checksum(self) -> str:
        return self.sha256


def _format_hint(extension: str) -> str:
    return f"Supported formats: {', '.join(x for x in ALLOWED_MIME_BY_EXT)}."


def validate_upload(filename: str, file_bytes: bytes) -> ValidatedFile:
    if not filename:
        raise ValidationError("No file was provided.")

    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_MIME_BY_EXT:
        raise ValidationError(
            f"Unsupported file format. {_format_hint(ext)}"
        )

    size = len(file_bytes)
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if size == 0:
        raise ValidationError("The uploaded file is empty.")
    if size > max_bytes:
        raise ValidationError(
            f"File exceeds the {settings.max_upload_size_mb} MB upload limit."
        )

    guard = hashlib.sha256()
    guard.update(file_bytes)
    checksum = guard.hexdigest()

    mime, _ = mimetypes.guess_type(filename)
    if mime and mime not in ALLOWED_MIME_BY_EXT[ext]:
        raise ValidationError(
            f"File contents do not match its extension. {_format_hint(ext)}"
        )

    image = preprocessing.open_image_from_bytes(file_bytes)
    im_format = (image.format or "").lower()
    if im_format and f"image/{im_format}" not in ALLOWED_MIME_BY_EXT[ext]:
        raise ValidationError(
            f"File contents do not match its extension ({image.format}). {_format_hint(ext)}"
        )

    width, height = image.size
    if width < settings.min_image_dimension or height < settings.min_image_dimension:
        raise ValidationError(
            f"Image is too small ({width}x{height}). Minimum 64x64 pixels is required."
        )

    mode = image.mode
    if mode not in {"L", "RGB", "RGBA", "I", "F"}:
        raise ValidationError("Unsupported image color mode.")

    warnings: list[str] = []
    if width < MIN_SUITABLE_DIMENSION or height < MIN_SUITABLE_DIMENSION:
        warnings.append(
            "The image is low resolution; analysis accuracy may be reduced."
        )

    # Basic suitability check: reject effectively blank frames. The downscale
    # uses a smoothing (LANCZOS) resample so striped/fine textures are averaged
    # rather than aliased to a near-constant frame by nearest-neighbour pick.
    try:
        gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
        import numpy as np

        if np.asarray(gray, dtype=np.float32).std() < MIN_DETECTABLE_ACTIVITY:
            raise ValidationError(
                "Unable to analyze this image. Please upload a valid brain MRI image."
            )
    except ValidationError:
        raise
    except Exception:
        raise ValidationError("Unable to analyze this image. Please upload a valid brain MRI image.")

    return ValidatedFile(
        bytes=file_bytes,
        size=size,
        extension=ext,
        mime=mime or f"image/{im_format}",
        sha256=checksum,
        format=im_format,
        mode=mode,
        width=width,
        height=height,
        warnings=warnings,
    )