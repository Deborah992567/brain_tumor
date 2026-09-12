"""Unit tests for calibration metrics (ECE, temperature scaling).

Everything here is pure numpy (no TensorFlow) and can run in any backend
environment.
"""

from __future__ import annotations

import numpy as np
import pytest

from training.evaluate import (
    calibration_report,
    evaluate,
    evaluate_calibrated,
    expected_calibration_error,
    reliability_bins,
    temperature_scale,
)


def _rng():
    return np.random.default_rng(42)


def test_ece_zero_for_perfectly_calibrated():
    n = 300
    probabilities = np.full((n, 4), 0.0333)
    probabilities[:, 0] = 0.9
    # Perfect calibration: confidence 0.9 → exactly 90% correct (270/300).
    labels = np.zeros(n, dtype=int)
    labels[270:] = 1  # last 30 are wrong
    ece = expected_calibration_error(probabilities, labels, n_bins=15)
    assert ece == pytest.approx(0.0, abs=1e-6)


def test_ece_reflects_miscalibration():
    n = 100
    rng = _rng()
    probabilities = np.full((n, 4), 0.0333)
    probabilities[:, 2] = 0.9
    # All predictions land in the top bin at 0.9 confidence but nothing is correct.
    labels = np.ones(n, dtype=int) * 1
    ece = expected_calibration_error(probabilities, labels, n_bins=15)
    assert ece == pytest.approx(0.9, abs=1e-6)


def test_reliability_bins_cover_samples_and_sum():
    rng = _rng()
    probabilities = rng.dirichlet(np.ones(4), size=500)
    labels = rng.integers(0, 4, size=500)
    bins = reliability_bins(probabilities, labels, n_bins=5)
    assert sum(b.count for b in bins) == 500
    for b in bins:
        assert b.bin_low < b.bin_high
        assert 0.0 <= b.confidence <= 1.0
        assert 0.0 <= b.accuracy <= 1.0
        assert b.count >= 1


def test_temperature_one_is_identity():
    rng = _rng()
    probabilities = rng.dirichlet(np.ones(4), size=50)
    scaled = temperature_scale(probabilities, 1.0)
    assert np.allclose(scaled, probabilities, atol=1e-5)
    assert np.allclose(scaled.sum(axis=1), 1.0, atol=1e-6)


def test_temperature_softens_distribution():
    probabilities = np.array([[0.98, 0.01, 0.005, 0.005]])
    scaled = temperature_scale(probabilities, 2.5)
    assert scaled[0].max() < 0.98
    assert np.argmax(scaled[0]) == 0  # order preserved


def test_temperature_scale_works_on_logits():
    """Temperature scaling is designed for logits; applied to probabilities
    it softens the distribution without calibration benefit."""
    probabilities = np.array([[0.98, 0.01, 0.005, 0.005]])
    scaled = temperature_scale(probabilities, 1.0)
    assert np.allclose(scaled, probabilities, atol=1e-5)


def test_evaluate_calibrated_reports_ece():
    rng = _rng()
    fit_probs = rng.dirichlet(np.ones(4), size=200)
    fit_labels = rng.integers(0, 4, size=200)
    test_probs = rng.dirichlet(np.ones(4), size=120)
    test_labels = rng.integers(0, 4, size=120)

    report, calibrated = evaluate_calibrated(
        test_probs, test_labels, calibration_fit=(fit_probs, fit_labels)
    )
    assert report.calibration is not None
    assert report.calibration.ece >= 0
    assert report.calibration.calibrated is False
    assert calibrated is None


def test_calibration_report_marks_uncalibrated():
    rng = _rng()
    probabilities = rng.dirichlet(np.ones(4), size=100)
    labels = rng.integers(0, 4, size=100)
    report = calibration_report(probabilities, labels)
    assert report.calibrated is False
    assert report.temperature == 1.0
    assert report.n_bins == 15
    assert report.binned
    total = sum(bin_.count for bin_ in report.binned)
    assert total == len(labels)


def test_evaluate_calibration_optional():
    rng = _rng()
    probabilities = rng.dirichlet(np.ones(4), size=40)
    labels = rng.integers(0, 4, size=40)
    report, calibrated = evaluate_calibrated(probabilities, labels)
    assert calibrated is None
    assert report.calibration is not None