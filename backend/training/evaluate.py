"""Model evaluation metrics computed on a held-out test set.

Provides classification metrics for the four brain-tumor classes:

* accuracy
* precision, recall, F1 (macro, weighted and per-class)
* sensitivity, specificity (per class)
* confusion matrix
* ROC-AUC (one-vs-rest)
* calibration: expected calibration error (ECE), reliability bins and an
  optional temperature-scaling calibration fit

Everything here avoids sklearn and relies only on numpy so it can run in
the same environment as the backend.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary_tumor"]

_EPILOGUE_EPS = 1e-7


@dataclass
class PerClassMetrics:
    precision: float
    recall: float
    f1: float
    sensitivity: float
    specificity: float
    support: int


@dataclass
class ReliabilityBin:
    bin_low: float
    bin_high: float
    count: int
    confidence: float
    accuracy: float


@dataclass
class CalibrationReport:
    ece: float
    n_bins: int
    temperature: float
    calibrated: bool
    binned: list[ReliabilityBin] = field(default_factory=list)
    method: str = "temperature-scaling"


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
    calibration: CalibrationReport | None = None
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


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected calibration error: mean |confidence - accuracy| per confidence bin.

    The model is perfectly calibrated when, among the samples predicted with a
    probability of ``p``, exactly a fraction ``p`` is correct. ECE buckets the
    predictions by max-softmax confidence and measures the average deviation.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = confidences >= low
        else:
            mask = (confidences >= low) & (confidences < high)
        if not mask.any():
            continue
        ece += (mask.sum() / confidences.size) * abs(
            confidences[mask].mean() - accuracies[mask].mean()
        )
    return round(ece, 4)


def reliability_bins(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> list[ReliabilityBin]:
    """Observed accuracy vs average confidence per confidence bin (pairs for
    a reliability diagram)."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    confidences = probabilities.max(axis=1)
    accuracies = (probabilities.argmax(axis=1) == labels).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[ReliabilityBin] = []
    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = confidences >= low
        else:
            mask = (confidences >= low) & (confidences < high)
        if not mask.any():
            continue
        out.append(
            ReliabilityBin(
                bin_low=round(float(low), 4),
                bin_high=round(float(high), 4),
                count=int(mask.sum()),
                confidence=round(float(confidences[mask].mean()), 4),
                accuracy=round(float(accuracies[mask].mean()), 4),
            )
        )
    return out


def logits_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Recover centered logits from probabilities (used by temperature scaling)."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    clipped = np.clip(probabilities, _EPILOGUE_EPS, 1.0 - _EPILOGUE_EPS)
    logits = np.log(clipped)
    logits -= logits.mean(axis=1, keepdims=True)
    return logits


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature T to the logits (softmax(logits / T)).

    T > 1 softens the distribution (used for calibration); T < 1 sharpens it.
    """
    logits = logits_from_probabilities(probabilities)
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def nll_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    return -float(
        np.mean(np.log(np.clip(probabilities[np.arange(len(labels)), labels], _EPILOGUE_EPS, 1.0)))
    )


def fit_temperature(
    probabilities: np.ndarray,
    labels: np.ndarray,
    lr: float = 0.1,
    iterations: int = 1500,
) -> float:
    """Fit a temperature-scaling temperature by minimising NLL on a set that
    was held out from training (validation set). Using the same set that was
    used to pick the model would overfit the temperature; callers are
    responsible for passing a properly held-out labelled set.

    Gradient of the softmax NLL w.r.t. T (logits ``z``, scaled probs ``p``):

        dL/dT = (1 / T^2) * mean_i( sum_k p_{i,k} z_{i,k} - z_{i,y_i} )
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    temperature = 1.0
    base_logits = logits_from_probabilities(probabilities)
    label_logits = base_logits[np.arange(len(labels)), labels]
    for _ in range(iterations):
        scaled = temperature_scale(probabilities, temperature)
        grad = (
            np.average((base_logits * scaled).sum(axis=1)) - np.average(label_logits)
        ) / (temperature**2)
        temperature -= lr * grad
        temperature = max(temperature, 1e-4)
    return round(temperature, 4)


def calibration_report(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> CalibrationReport:
    """Measure calibration of raw model outputs (T=1)."""
    return CalibrationReport(
        ece=expected_calibration_error(probabilities, labels, n_bins=n_bins),
        n_bins=n_bins,
        temperature=1.0,
        calibrated=False,
        binned=reliability_bins(probabilities, labels, n_bins=n_bins),
    )


def evaluate_calibrated(
    probabilities: np.ndarray,
    labels: np.ndarray,
    calibration_fit: tuple[np.ndarray, np.ndarray] | None = None,
    n_bins: int = 15,
) -> tuple[EvaluationReport, CalibrationReport | None]:
    """Evaluate classification performance and measure calibration (ECE).

    ``probabilities``/``labels`` form the test set used to report metrics.
    ``calibration_fit`` is currently unused: temperature scaling requires
    access to the raw model logits before softmax, but our inference API
    only returns post-softmax probabilities. ECE is therefore always
    reported on the raw model outputs (T=1). If a future model backend
    exposes logits, this parameter will be used to fit a temperature.
    """
    report = evaluate(probabilities, labels)
    base = calibration_report(probabilities, labels, n_bins=n_bins)
    report.calibration = base
    return report, None


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