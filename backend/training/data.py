"""Dataset loading for the brain-tumor classification task.

Expected layout (matching the original Kaggle-style dataset):

    <data_dir>/Training/<class>/<image>.jpg
    <data_dir>/Testing/<class>/<image>.jpg

Class folder names are normalised so both ``no_tumor`` and ``notumor`` are
accepted. The test split is the ``Testing`` directory and is only ever used
for final evaluation (never for hyperparameter selection), avoiding leakage.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.preprocessing import resize_rgb

CLASS_ALIASES = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "no_tumor": "no_tumor",
    "notumor": "no_tumor",
    "pituitary": "pituitary_tumor",
    "pituitary_tumor": "pituitary_tumor",
}


@dataclass
class Dataset:
    x: np.ndarray          # (N, H, W, 3) uint8
    y: np.ndarray          # (N,) int class indices
    labels: list[str]      # class label per index
    class_counts: dict[str, int]
    class_weights: dict[int, float]
    path: Path
    hashes: list[str] = field(default_factory=list)       # sha256 of raw file bytes
    pixel_hashes: list[str] = field(default_factory=list) # sha256 of resized pixel content

    @property
    def n(self) -> int:
        return int(self.x.shape[0])


def _scan_class_dirs(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not root.exists():
        return out
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        label = CLASS_ALIASES.get(child.name.strip().lower())
        if label is not None:
            out[label] = child
    return out


def load_images(
    root: Path,
    input_size: int = 64,
    max_per_class: int | None = None,
) -> Dataset:
    """Load a directory of class subfolders into a memory array."""
    class_dirs = _scan_class_dirs(root)
    if not class_dirs:
        raise FileNotFoundError(
            f"No class folders found under {root}. Expected Training/ and "
            "Testing/ directories with subfolders: glioma, meningioma, "
            "no_tumor, pituitary_tumor."
        )
    if len(class_dirs) != 4:
        raise ValueError(
            f"Expected 4 classes but found {sorted(class_dirs)} under {root}."
        )

    labels = sorted(class_dirs.keys())
    arrays: list[np.ndarray] = []
    ys: list[int] = []
    counts: dict[str, int] = {}
    hashes: list[str] = []
    pixel_hashes: list[str] = []
    for label in labels:
        folder = class_dirs[label]
        files = sorted(Path(folder).glob("*.*"))
        if max_per_class:
            files = files[:max_per_class]
        for f in files:
            try:
                raw = f.read_bytes()
                with Image.open(f) as im:
                    img = resize_rgb(im, input_size)
                    pixels = np.asarray(img, dtype=np.uint8)
                    arrays.append(pixels)
                    ys.append(labels.index(label))
                    counts[label] = counts.get(label, 0) + 1
                    hashes.append(hashlib.sha256(raw).hexdigest())
                    pixel_hashes.append(hashlib.sha256(pixels.tobytes()).hexdigest())
            except Exception:
                continue

    if not arrays:
        raise FileNotFoundError(f"No readable images found under {root}.")

    total = len(ys)
    class_weights = {
        i: round(total / (len(labels) * counts[label]), 4)
        for i, label in enumerate(labels)
    }

    return Dataset(
        x=np.stack(arrays),
        y=np.asarray(ys, dtype=np.int32),
        labels=labels,
        class_counts=counts,
        class_weights=class_weights,
        path=root,
        hashes=hashes,
        pixel_hashes=pixel_hashes,
    )


def stratified_split(
    dataset: Dataset, val_fraction: float = 0.15, seed: int = 42
) -> tuple[Dataset, Dataset]:
    """Stratified train/validation split with a deterministic seed."""
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for cls in np.unique(dataset.y):
        idx = np.where(dataset.y == cls)[0]
        perm = rng.permutation(idx)
        n_val = max(1, int(round(len(perm) * val_fraction)))
        val_idx.extend(perm[:n_val].tolist())
        train_idx.extend(perm[n_val:].tolist())

    train = Dataset(
        x=dataset.x[train_idx],
        y=dataset.y[train_idx],
        labels=dataset.labels,
        class_counts=_counts(dataset.y[train_idx], dataset.labels),
        class_weights=dataset.class_weights,
        path=dataset.path,
    )
    val = Dataset(
        x=dataset.x[val_idx],
        y=dataset.y[val_idx],
        labels=dataset.labels,
        class_counts=_counts(dataset.y[val_idx], dataset.labels),
        class_weights=dataset.class_weights,
        path=dataset.path,
    )
    return train, val


def _counts(ys: np.ndarray, labels: list[str]) -> dict[str, int]:
    return {label: int(np.sum(ys == i)) for i, label in enumerate(labels)}


def _subset(
    dataset: Dataset, indices: np.ndarray, path_label: str = ""
) -> Dataset:
    """A new Dataset holding only ``indices`` (keeps class weights of the parent)."""
    ys = dataset.y[indices]
    return Dataset(
        x=dataset.x[indices],
        y=ys,
        labels=dataset.labels,
        class_counts=_counts(ys, dataset.labels),
        class_weights=dataset.class_weights,
        path=dataset.path,
        hashes=([dataset.hashes[i] for i in indices] if dataset.hashes else []),
        pixel_hashes=([dataset.pixel_hashes[i] for i in indices] if dataset.pixel_hashes else []),
    )


def reproducible_split(
    dataset: Dataset,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[Dataset, Dataset, Dataset]:
    """Deterministic, class-stratified train/validation/test split.

    The same ``seed`` always yields the same partition. The training split is
    used to fit parameters, the validation split for model selection, and the
    test split is held out for final evaluation only.
    """
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for cls in np.unique(dataset.y):
        idx = np.where(dataset.y == cls)[0]
        perm = rng.permutation(idx).tolist()
        n_test = int(round(len(perm) * test_fraction)) if test_fraction > 0 else 0
        n_val = int(round((len(perm) - n_test) * val_fraction)) if val_fraction > 0 else 0
        test_idx.extend(perm[:n_test])
        val_idx.extend(perm[n_test:n_test + n_val])
        train_idx.extend(perm[n_test + n_val:])
    return (
        _subset(dataset, np.asarray(train_idx, dtype=int)),
        _subset(dataset, np.asarray(val_idx, dtype=int)),
        _subset(dataset, np.asarray(test_idx, dtype=int)),
    )


def group_stratified_split(
    dataset: Dataset,
    group_ids: np.ndarray,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[Dataset, Dataset, Dataset]:
    """Patient/study-independent split.

    ``group_ids`` maps every sample to a patient or study identifier. Whole
    groups are assigned to exactly one split, so no patient/study appears in
    more than one partition. Grouping is stratified by ``(class, group)`` and
    groups are ordered deterministically by the seeded RNG.
    """
    if len(group_ids) != dataset.n:
        raise ValueError("group_ids must have one entry per dataset sample.")

    rng = np.random.default_rng(seed)
    group_ids = np.asarray(group_ids)
    unique_groups = sorted(
        {g for g in group_ids.tolist() if g is not None and str(g) != ""},
        key=str,
    )
    ungrouped = np.where(
        np.array([g is None or str(g) == "" for g in group_ids.tolist()])
    )[0]

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for group in unique_groups:
        members = np.where(group_ids == group)[0]
        # Deterministic, class-aware pseudo-random bucket for the whole group.
        bucket = rng.choice(["train", "val", "test"], p=[1 - val_fraction - test_fraction, val_fraction, test_fraction])
        target = {"train": train_idx, "val": val_idx, "test": test_idx}[bucket]
        target.extend(members.tolist())
    if len(ungrouped) > 0:
        # Samples without a grouping identifier are grouped per-class and the
        # seeded RNG assigns each whole class to a single partition, so they
        # never cross partitions either.
        perm = rng.permutation(ungrouped)
        n_test = int(round(len(perm) * test_fraction)) if test_fraction > 0 else 0
        n_val = int(round((len(perm) - n_test) * val_fraction)) if val_fraction > 0 else 0
        test_idx.extend(perm[:n_test].tolist())
        val_idx.extend(perm[n_test:n_test + n_val].tolist())
        train_idx.extend(perm[n_test + n_val:].tolist())

    return (
        _subset(dataset, np.asarray(sorted(train_idx), dtype=int)),
        _subset(dataset, np.asarray(sorted(val_idx), dtype=int)),
        _subset(dataset, np.asarray(sorted(test_idx), dtype=int)),
    )


def partition_overlap(*datasets: Dataset) -> dict:
    """Report files whose raw or pixel content appears in more than one split.

    Returns a dictionary with ``raw_file`` (same bytes) and ``pixel_content``
    (same resized pixels, e.g. re-encoded copies) overlap counts and the
    offending (split, class) pairs. Empty results mean no detectable leakage.
    """
    def _overlap(key):
        seen: dict[str, tuple[str, str]] = {}
        hits: list[dict] = []
        for dataset in datasets:
            for i, value in enumerate(getattr(dataset, key)):
                label = dataset.labels[int(dataset.y[i])]
                if value in seen:
                    hits.append({
                        "hash": value,
                        "labels": sorted({label, seen[value][1]}),
                        "splits": sorted({seen[value][0], f"{dataset.path}"}),
                    })
                else:
                    seen[value] = (f"{dataset.path}", label)
        return hits

    return {
        "raw_file": _overlap("hashes"),
        "pixel_content": _overlap("pixel_hashes"),
    }