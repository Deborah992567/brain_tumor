"""Model evaluation metrics computed on a held-out test set.

Provides classification metrics for the four brain-tumor classes:

* accuracy
* precision, recall, F1 (macro, weighted and per-class)
* sensitivity, specificity (per class)
* confusion matrix
* ROC-AUC (one-vs-rest)

Everything here avoids sklearn and relies only on numpy so it can run in
the same environment as the backend.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary_tumor"]


@dataclass
class PerClassMetrics:
    precision: float
    recall: float
    f1: float
    sensitivity: float
    specificity: float
    support: int


@dataclass
class EvaluationReport:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    confusion_matrix: list[list[int]]
    per_class: dict[str, PerClassMetrics] = field(default_factory=dict)
    roc_auc: dict[str, float] = field(default_factory=dict)
    n_samples: int = 0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["per_class"] = {
            k: asdict(v) for k, v in self.per_class.items()
        }
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(labels, predictions):
        matrix[true, pred] += 1
    return matrix


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def roc_auc_ovr(probabilities: np.ndarray, labels: np.ndarray, class_index: int) -> float:
    """One-vs-rest ROC-AUC by rank statistic (no sklearn)."""
    positives = probabilities[labels == class_index, class_index]
    negatives = probabilities[labels != class_index, class_index]
    if positives.size == 0 or negatives.size == 0:
        return 0.0
    # Probability a random positive ranks above a random negative (Mann-Whitney U).
    greater = float(np.sum(positives[:, None] > negatives[None, :]))
    equal = float(np.sum(positives[:, None] == negatives[None, :]))
    u_stat = greater + 0.5 * equal
    return round(u_stat / (positives.size * negatives.size), 4)


def evaluate(
    probabilities: np.ndarray,
    labels: np.ndarray,
    classes: list[str] | None = None,
) -> EvaluationReport:
    """Evaluate model outputs (N x C probabilities) against integer labels."""
    classes = classes or CLASSES
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    n_classes = len(classes)
    predictions = np.argmax(probabilities, axis=1)

    cm = confusion_matrix(labels, predictions, n_classes)
    n = int(len(labels))

    per_class: dict[str, PerClassMetrics] = {}
    for c in range(n_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum()) - tp
        fn = int(cm[c, :].sum()) - tp
        tn = n - (tp + fp + fn)

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)  # same as sensitivity
        f1 = _safe_div(2 * precision * recall, precision + recall)
        specificity = _safe_div(tn, tn + fp)

        per_class[classes[c]] = PerClassMetrics(
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            sensitivity=round(recall, 4),
            specificity=round(specificity, 4),
            support=int(cm[c, :].sum()),
        )

    precisions = [per_class[c].precision for c in classes]
    recalls = [per_class[c].recall for c in classes]
    f1s = [per_class[c].f1 for c in classes]
    supports = [per_class[c].support for c in classes]

    macro = lambda values: round(float(np.mean(values)), 4)
    weighted = lambda values: round(_safe_div(sum(v * s for v, s in zip(values, supports)), sum(supports)), 4)

    aucs = {classes[c]: roc_auc_ovr(probabilities, labels, c) for c in range(n_classes)}

    return EvaluationReport(
        accuracy=round(_safe_div(np.sum(predictions == labels), n), 4),
        precision_macro=macro(precisions),
        recall_macro=macro(recalls),
        f1_macro=macro(f1s),
        precision_weighted=weighted(precisions),
        recall_weighted=weighted(recalls),
        f1_weighted=weighted(f1s),
        confusion_matrix=cm.tolist(),
        per_class=per_class,
        roc_auc=aucs,
        n_samples=n,
    )