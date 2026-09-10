"""Dataset loading for the brain-tumor classification task.

Expected layout (matching the original Kaggle-style dataset):

    <data_dir>/Training/<class>/<image>.jpg
    <data_dir>/Testing/<class>/<image>.jpg

Class folder names are normalised so both ``no_tumor`` and ``notumor`` are
accepted. The test split is the ``Testing`` directory and is only ever used
for final evaluation (never for hyperparameter selection), avoiding leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

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
    for label in labels:
        folder = class_dirs[label]
        files = sorted(Path(folder).glob("*.*"))
        if max_per_class:
            files = files[:max_per_class]
        for f in files:
            try:
                with Image.open(f) as im:
                    img = im.convert("RGB").resize(
                        (input_size, input_size), Image.Resampling.BILINEAR
                    )
                    arrays.append(np.asarray(img, dtype=np.uint8))
                    ys.append(labels.index(label))
                    counts[label] = counts.get(label, 0) + 1
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