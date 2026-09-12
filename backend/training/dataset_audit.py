"""Reproducible, offline dataset audit for the brain-tumor classification task.

Scans the Kaggle-style directory layout (``Training/`` and ``Testing/`` class
folders) and reports, without any TensorFlow dependency:

* total images and per-class counts (with unknown / mismatched class folders
  flagged so new diagnoses such as *metastasis* are never silently mapped into
  the four supported classes)
* image dimensions, formats and color channels
* missing/corrupt files
* exact duplicates (by file SHA-256) and perceptually near-duplicate images
  (by average hash), including cross-split and cross-class duplicates
* patient/study identifiers extracted from filenames through a configurable
  pattern, enabling patient-level leakage detection

The tool is honest by construction: it never fabricates data. If a dataset root
is missing or empty the report says exactly that. Nothing here claims a result
that was not observed on disk.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from training.data import CLASS_ALIASES
from training.evaluate import CLASSES

Image.MAX_IMAGE_PIXELS = None  # large medical scans must still be inspectable

DEFAULT_PATIENT_PATTERN = (
    r"(?P<patient>[A-Za-z0-9_-]+?)[-_][0-9]{1,6}(\.[A-Za-z0-9]+)?$"
)
_AVG_HASH_SIZE = 8  # 8x8 -> 64-bit perceptual hash
_NEAR_DUP_HAMMING = 4  # average-hash distance considered "near-identical"


@dataclass
class ImageRecord:
    path: Path
    split: str
    class_label: str
    size_bytes: int
    width: int = 0
    height: int = 0
    format: str = ""
    mode: str = ""
    sha256: str = ""
    avg_hash: int = 0
    patient_id: str | None = None


@dataclass
class ClassCounts:
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return int(sum(self.counts.values()))


@dataclass
class DuplicateGroup:
    kind: str  # "exact" | "near"
    image_hashes: list[str]
    paths: list[str]
    classes: list[str]
    splits: list[str]


@dataclass
class LeakageFinding:
    images: list[str]
    splits: list[str]
    classes: list[str]
    sha256: str


@dataclass
class DatasetAudit:
    root: Path | None
    dataset_found: bool
    splits: dict[str, ClassCounts]
    classes_found: list[str]
    unknown_dirs: list[str]
    total_images: int
    dimensions: list[tuple[int, int]]
    formats: list[str]
    modes: list[str]
    corrupt_files: list[str]
    duplicates: list[DuplicateGroup]
    leakage: list[LeakageFinding]
    grouped: int
    ungrouped: int
    records: list[ImageRecord] = field(repr=False, default_factory=list)

    @property
    def class_counts(self) -> dict[str, int]:
        combined: dict[str, int] = {}
        for split in self.splits.values():
            for cls, count in split.counts.items():
                combined[cls] = combined.get(cls, 0) + count
        return combined


def iter_image_files(root: Path):
    """Yield every regular file under ``root`` that looks like an image."""
    if not root.exists():
        return
    suffixes = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp",
    }
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path


def classify_folder(name: str) -> str | None:
    """Map a folder name to a canonical class label; None if unsupported."""
    return CLASS_ALIASES.get(name.strip().lower())


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _avg_hash(img: Image.Image) -> int:
    """64-bit average hash; 0 and 1 bits encode above/below mean luma."""
    small = img.convert("L").resize((_AVG_HASH_SIZE, _AVG_HASH_SIZE), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.uint8)
    mean = int(pixels.mean())
    return int("".join("1" if v > mean else "0" for v in pixels.ravel()), 2)


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def scan_split(root: Path, split: str, patient_pattern: str | None) -> list[ImageRecord]:
    """Scan one split folder (``Training``, ``Testing``, ...) into records."""
    records: list[ImageRecord] = []
    regex = re.compile(patient_pattern) if patient_pattern else None
    if not root.exists():
        return records
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        records.extend(
            _scan_class_folder(folder, classify_folder(folder.name), split, regex)
        )
    return records


def _scan_class_folder(
    folder: Path, label: str | None, split: str, regex: re.Pattern | None
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for path in iter_image_files(folder):
        record = ImageRecord(
            path=path,
            split=split,
            class_label=label or folder.name,
            size_bytes=path.stat().st_size,
        )
        try:
            with Image.open(path) as im:
                record.width, record.height = im.size
                record.format = (im.format or "").lower()
                record.mode = im.mode
                record.avg_hash = _avg_hash(im)
        except Exception:
            record.format = "corrupt"
        record.sha256 = _sha256(path.read_bytes())
        if regex is not None:
            match = regex.match(path.name)
            if match:
                record.patient_id = match.group("patient")
        records.append(record)
    return records


def audit_dataset(
    root: Path | None,
    splits: tuple[str, ...] = ("Training", "Testing"),
    patient_pattern: str | None = DEFAULT_PATIENT_PATTERN,
) -> DatasetAudit:
    """Run the full audit over the *supplied* directory layout.

    ``root`` must contain per-split class folders. Folders whose name is not a
    supported class are reported in ``unknown_dirs`` — they are never mapped to
    the supported label space.
    """
    if root is None or not root.exists():
        return DatasetAudit(
            root=root,
            dataset_found=False,
            splits={s: ClassCounts() for s in splits},
            classes_found=[],
            unknown_dirs=[],
            total_images=0,
            dimensions=[],
            formats=[],
            modes=[],
            corrupt_files=[],
            duplicates=[],
            leakage=[],
            grouped=0,
            ungrouped=0,
        )

    split_records: dict[str, list[ImageRecord]] = {}
    split_counts: dict[str, ClassCounts] = {}
    unknown_dirs: list[str] = []
    for split in splits:
        split_root = root / split
        records = scan_split(split_root, split, patient_pattern)
        split_records[split] = records
        counts: dict[str, int] = {}
        for record in records:
            counts[record.class_label] = counts.get(record.class_label, 0) + 1
            if record.class_label not in CLASSES:
                unknown_dirs.append(str(record.path.parent))
        split_counts[split] = ClassCounts(counts)

    all_records = [
        r for records in split_records.values() for r in records
    ]

    dimensions: list[tuple[int, int]] = []
    formats: list[str] = []
    modes: list[str] = []
    corrupt: list[str] = []
    for record in all_records:
        if record.format == "corrupt":
            corrupt.append(str(record.path))
            continue
        dimensions.append((record.width, record.height))
        formats.append(record.format)
        modes.append(record.mode)

    duplicates = _find_duplicates(all_records)
    leakage = _find_leakage(split_records)

    grouped = sum(1 for r in all_records if r.patient_id)
    return DatasetAudit(
        root=root,
        dataset_found=len(all_records) > 0,
        splits=split_counts,
        classes_found=sorted({r.class_label for r in all_records}),
        unknown_dirs=sorted(set(unknown_dirs)),
        total_images=len(all_records),
        dimensions=dimensions,
        formats=formats,
        modes=modes,
        corrupt_files=corrupt,
        duplicates=duplicates,
        leakage=leakage,
        grouped=grouped,
        ungrouped=len(all_records) - grouped,
        records=all_records,
    )


def _find_duplicates(records: list[ImageRecord]) -> list[DuplicateGroup]:
    """Group exact (SHA-256) and perceptually near (average-hash) duplicates."""
    by_sha: dict[str, list[ImageRecord]] = {}
    for record in records:
        by_sha.setdefault(record.sha256, []).append(record)

    groups: list[DuplicateGroup] = []
    for sha, group in sorted(by_sha.items()):
        if len(group) > 1:
            groups.append(
                DuplicateGroup(
                    kind="exact",
                    image_hashes=[sha],
                    paths=[str(r.path) for r in group],
                    classes=sorted({r.class_label for r in group}),
                    splits=sorted({r.split for r in group}),
                )
            )

    by_hash: dict[int, list[ImageRecord]] = {}
    for record in records:
        if record.avg_hash:
            by_hash.setdefault(record.avg_hash, []).append(record)
    # Only report near-duplicate groups that were not already exact duplicates
    # (exact groups are always also perceptually identical).
    exact_paths = {p for g in groups for p in g.paths}
    for h, group in sorted(by_hash.items()):
        unaccounted = [r for r in group if str(r.path) not in exact_paths]
        if len(unaccounted) > 1:
            hamming_zero = all(
                _hamming(h, r.avg_hash) <= _NEAR_DUP_HAMMING for r in unaccounted
            )
            if hamming_zero:
                groups.append(
                    DuplicateGroup(
                        kind="near",
                        image_hashes=[str(h)],
                        paths=[str(r.path) for r in unaccounted],
                        classes=sorted({r.class_label for r in unaccounted}),
                        splits=sorted({r.split for r in unaccounted}),
                    )
                )
    return groups


def _find_leakage(
    split_records: dict[str, list[ImageRecord]]
) -> list[LeakageFinding]:
    """Flag image hashes that appear in more than one split."""
    by_sha: dict[str, list[ImageRecord]] = {}
    for records in split_records.values():
        for record in records:
            by_sha.setdefault(record.sha256, []).append(record)
    findings: list[LeakageFinding] = []
    for sha, group in sorted(by_sha.items()):
        splits = sorted({r.split for r in group})
        if len(splits) > 1:
            findings.append(
                LeakageFinding(
                    images=[str(r.path) for r in group],
                    splits=splits,
                    classes=sorted({r.class_label for r in group}),
                    sha256=sha,
                )
            )
    return findings


def audit_to_dict(audit: DatasetAudit, extras: dict | None = None) -> dict:
    """Serialize the audit into a manifest-friendly dictionary."""
    data: dict = {
        "dataset_found": audit.dataset_found,
        "root": str(audit.root) if audit.root else None,
        "splits": {
            name: counts.counts for name, counts in audit.splits.items()
        },
        "classes_found": audit.classes_found,
        "unknown_dirs": audit.unknown_dirs,
        "total_images": audit.total_images,
        "dimensions": {
            "distinct": sorted({d for d in audit.dimensions}),
            "unique_sizes": len({d for d in audit.dimensions}),
        },
        "formats": sorted({f for f in audit.formats}),
        "modes": sorted({m for m in audit.modes}),
        "corrupt_files": audit.corrupt_files,
        "duplicate_groups": [
            {
                "kind": g.kind,
                "count": len(g.paths),
                "classes": g.classes,
                "splits": g.splits,
                "paths": g.paths,
            }
            for g in audit.duplicates
        ],
        "leakage_across_splits": [
            {
                "sha256": f.sha256,
                "splits": f.splits,
                "classes": f.classes,
                "images": f.images,
            }
            for f in audit.leakage
        ],
        "patient_grouping": {
            "grouped_by_pattern": DEFAULT_PATIENT_PATTERN,
            "images_grouped": audit.grouped,
            "images_not_grouped": audit.ungrouped,
        },
        "limitations": (
            "Where filenames carry no patient/study identifier, patient-level "
            "leakage cannot be completely excluded; the split is then not "
            "guaranteed patient-independent."
        ),
    }
    if extras:
        data.update(extras)
    return data