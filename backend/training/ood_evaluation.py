"""OOD / abstention evaluation utilities (numpy only).

Self-contained scoring functions that operate on model probability vectors:

* ``maximum_softmax_probability`` (MSP) and predictive entropy as OOD scores
* ``should_abstain`` implementing the product's abstention decision rule
* rank-statistic AUROC for measuring how well an OOD score separates an
  in-distribution set from an out-of-distribution set
* ``evaluate_ood`` producing an honest, structured comparison

These are *measurement* tools. Whether the model actually separates supported
in-distribution MRI scans from unsupported ones can only be established by
running the CLI against real labelled in-/out-of-distribution image sets, which
documented in ``reports/model-evaluation/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.core.constants import CONFIDENCE_LOW_THRESHOLD

_EPS = 1e-12


def maximum_softmax_probability(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    return probabilities.max()


def predictive_entropy(probabilities: np.ndarray) -> float:
    """Shannon entropy of a probability vector over the supported classes."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    clipped = np.clip(probabilities, _EPS, 1.0)
    return float(-(clipped * np.log(clipped)).sum())


def normalized_entropy(probabilities: np.ndarray) -> float:
    n_classes = int(np.asarray(probabilities).size)
    if n_classes < 2:
        return 0.0
    return predictive_entropy(probabilities) / np.log(n_classes)


def should_abstain(
    probabilities: np.ndarray,
    threshold: float = CONFIDENCE_LOW_THRESHOLD,
) -> tuple[bool, str]:
    """Abstention decision: abstain when max model probability < threshold."""
    confidence = maximum_softmax_probability(probabilities)
    if confidence < threshold:
        return True, (
            f"Maximum model probability ({confidence:.1%}) is below the "
            f"{threshold:.0%} reliability threshold."
        )
    return False, ""


def score_distribution(probabilities: np.ndarray) -> dict[str, float]:
    """Describe the model-output distribution of a set of probability vectors."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    cards = probabilities.reshape(len(probabilities), -1)
    msp = cards.max(axis=1)
    entropy = np.array([predictive_entropy(row) for row in cards])
    return {
        "msp_mean": round(float(msp.mean()), 4),
        "msp_median": round(float(np.median(msp)), 4),
        "msp_p10": round(float(np.percentile(msp, 10)), 4),
        "msp_p90": round(float(np.percentile(msp, 90)), 4),
        "entropy_mean": round(float(entropy.mean()), 4),
        "entropy_median": round(float(np.median(entropy)), 4),
        "n": int(len(cards)),
    }


@dataclass
class OodSeparation:
    score: str
    auroc: float
    id_stats: dict
    ood_stats: dict
    abstention_rate_id: float
    abstention_rate_ood: float
    notes: list[str] = field(default_factory=list)


def _auroc(positive, negative) -> float:
    """Rank statistic: probability a random OOD sample scores above a random ID one."""
    if positive.size == 0 or negative.size == 0:
        return 0.0
    greater = float(np.sum(positive[:, None] > negative[None, :]))
    equal = float(np.sum(positive[:, None] == negative[None, :]))
    return round((greater + 0.5 * equal) / (positive.size * negative.size), 4)


def evaluate_ood(
    in_distribution_probs: np.ndarray,
    out_of_distribution_probs: np.ndarray | None,
    abstain_threshold: float = CONFIDENCE_LOW_THRESHOLD,
) -> dict:
    """Compare ID and (optionally) OOD probability vectors.

    Returns distributions of MSP and entropy for each set plus, when an OOD set
    is provided, the AUROC of ``1 - MSP`` and of ``entropy`` as OOD scores.
    Ran without an OOD set it only reports ID statistics and notes that
    separation cannot be established.
    """
    id_probs = np.asarray(in_distribution_probs, dtype=np.float64)
    id_flat = id_probs.reshape(len(id_probs), -1)
    id_msp = id_probs.max(axis=1) if id_probs.ndim == 2 else np.array([float(id_probs.max())])
    if id_probs.ndim == 1:
        id_flat = id_probs.reshape(1, -1)
    id_msp = id_flat.max(axis=1)
    id_entropy = np.array([predictive_entropy(row) for row in id_flat])

    result: dict = {
        "score": {"msp": "maximum softmax probability", "entropy": "predictive entropy"},
        "in_distribution": score_distribution(id_flat),
        "abstain_threshold": abstain_threshold,
        "abstention_rate_in_distribution": round(
            float(np.mean(id_msp < abstain_threshold)), 4
        ),
        "out_of_distribution": None,
        "separation_auroc": None,
        "notes": [],
    }

    if out_of_distribution_probs is None:
        result["notes"].append(
            "No out-of-distribution set was supplied, so score separation could "
            "not be measured. Run the CLI with --ood-root to estimate it."
        )
        return result

    ood_probs = np.asarray(out_of_distribution_probs, dtype=np.float64)
    ood_flat = ood_probs.reshape(len(ood_probs), -1)
    ood_msp = ood_flat.max(axis=1)
    ood_entropy = np.array([predictive_entropy(row) for row in ood_flat])

    result["out_of_distribution"] = score_distribution(ood_flat)
    result["abstention_rate_out_of_distribution"] = round(
        float(np.mean(ood_msp < abstain_threshold)), 4
    )
    result["separation_auroc"] = {
        # Treat OOD as the positive class: a high 1-MSP score should mark OOD.
        "ood_score_1_minus_msp": _auroc(1 - ood_msp, 1 - id_msp),
        "ood_score_entropy": _auroc(ood_entropy, id_entropy),
    }
    result["notes"].append(
        "AUROC describes how well each single score separates the supplied "
        "sets. Random noise alone is not a complete OOD evaluation: the sets "
        "must be representative of the intended deployment domain."
    )
    return result


def evaluate_ood_on_roots(
    model,
    in_distribution_root,
    out_of_distribution_root,
    input_size: int = 64,
    batch_size: int = 32,
    abstain_threshold: float = CONFIDENCE_LOW_THRESHOLD,
) -> dict:
    """Run the OOD comparison over two image roots using shared preprocessing.

    * ``in_distribution_root``: folders of the four supported classes.
    * ``out_of_distribution_root``: any image folder — unsupported tumor
      classes, non-MRI images, noise, etc. Losses nothing on unreadable files.

    If either root does not parse as a four-class dataset it is predicted
    image-by-image so unstructured / unsupported content can still be scored.
    """

    def _is_dataset_root(root) -> bool:
        return bool(
            root
            and root.exists()
            and len(_find_class_folders(root)) == 4
        )

    def _labels_mismatch_detail(root):
        return f"{root} (found {sorted(_find_class_folders(root))})"

    def _predict(root):
        if _is_dataset_root(root):
            dataset = load_images(root, input_size)
            feed = feature_pipeline(
                dataset.x, dataset.y, batch_size, input_size, augment=False
            )
            return model.predict(feed, verbose=0)
        if root and root.exists():
            return _predict_loose(root, model, input_size)
        raise FileNotFoundError(f"Dataset root not found: {root}")

    id_probs = _predict(in_distribution_root)
    if not _is_dataset_root(in_distribution_root):
        raise ValueError(
            "in-distribution root must contain the four supported classes; got "
            f"{_labels_mismatch_detail(in_distribution_root)}"
        )

    ood_probs = None
    if out_of_distribution_root is not None and out_of_distribution_root.exists():
        ood_probs = _predict_loose(out_of_distribution_root, model, input_size)

    return evaluate_ood(id_probs, ood_probs, abstain_threshold)


def _find_class_folders(root):
    from training.data import CLASS_ALIASES

    if not root.exists():
        return []
    return [
        child.name
        for child in sorted(root.iterdir())
        if child.is_dir() and child.name.strip().lower() in CLASS_ALIASES
    ]


def _predict_loose(root, model, input_size):
    """Predict a folder of arbitrary images (unsupported classes / non-MRI)."""
    import numpy as np
    from PIL import Image

    from app.core.preprocessing import add_batch_dimension, preprocess_image

    probs: list[np.ndarray] = []
    for path in sorted(root.rglob("*.*")):
        try:
            with Image.open(path) as im:
                tensor = add_batch_dimension(preprocess_image(im, input_size))
        except Exception:
            continue
        probs.append(np.asarray(model.predict(tensor, verbose=0)).reshape(-1))
    if not probs:
        raise FileNotFoundError(f"No readable images found under {root}.")
    return np.stack(probs)