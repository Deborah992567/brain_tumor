"""Standalone model evaluation CLI (Phase 2).

Two honest layers:

* Artifact facts — always recorded, measured directly from the model file:
  architecture, parameter count, weight layers, input/output shapes, file size,
  keras version and single-image inference latency. These require no dataset.
* Performance metrics — recorded *only* when a labelled dataset is supplied.
  Without one the artifact explicitly says ``metrics: null`` and states why.

Usage (from the ``backend/`` directory):

    # artifact facts and latency without any dataset:
    python -m training.run_evaluation --model-path models/brain_tumor.h5 \\
        --output ../reports/model-evaluation/baseline-reproduction.json

    # full held-out evaluation when a labelled dataset is available:
    python -m training.run_evaluation --model-path models/brain_tumor.h5 \\
        --data-root /path/to/dataset \\
        --output ../reports/model-evaluation/final-model-evaluation.json

``--data-root`` must contain ``Training/`` and ``Testing/`` class folders for
the four supported classes. Every run also writes a confusion-matrix and a
reliability-diagram PNG (skipped when raw images of only one class exist).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from app.core.preprocessing import PREPROCESSING_VERSION
from training.data import load_images, reproducible_split
from training.evaluate import evaluate_calibrated
from training.augmentation import feature_pipeline
from training.model_spec import measure_latency, model_spec
from training.plotting import render_confusion_matrix, render_reliability_diagram


def _argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Trained model artifact (.h5/.keras).")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Optional dataset root with Training//Testing/.")
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Seed for the deterministic validation split.")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--output", type=Path,
                        default=Path("../reports/model-evaluation/evaluation.json"))
    parser.add_argument("--label", default="", help="Short artifact label.")
    parser.add_argument("--no-charts", action="store_true",
                        help="Skip PNG artifacts even when metrics exist.")
    return parser


def main() -> None:
    args = _argparse().parse_args()

    spec = model_spec(args.model_path, args.input_size)
    latency = measure_latency(args.model_path, args.input_size)
    from tensorflow import keras
    model = keras.models.load_model(str(args.model_path))

    base_path = args.output

    artifact: dict = {
        "label": args.label or str(args.model_path),
        "kind": "model-spec-only",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "model": spec,
        "latency_ms": latency,
        "preprocessing_version": PREPROCESSING_VERSION,
        "dataset": {
            "found": False,
            "note": (
                "No labelled dataset is present in this repository. Performance "
                "metrics are therefore not reported. Run this CLI with "
                "--data-root once labelled Training/ and Testing/ folders are "
                "available."
            ),
        },
        "metrics": None,
        "calibration": None,
        "artifacts": [],
    }

    dataset_report = None
    if args.data_root is not None:
        if not args.data_root.exists():
            raise FileNotFoundError(
                f"Dataset root not found: {args.data_root}. Refusing to invent "
                "metrics; pass a real root or run without --data-root."
            )
        dataset_report = _evaluate_with_dataset(
            model, args, spec, latency, base_path
        )
        artifact.update(dataset_report)
        artifact["kind"] = "held-out-evaluation"

    out = base_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2))
    print(f"Artifact written to {out}")
    if artifact.get("metrics"):
        print(json.dumps(artifact["metrics"], indent=2))
    else:
        print("metrics: null (no labelled dataset available)")


def _evaluate_with_dataset(model, args, spec, latency, base_path: Path) -> dict:
    train_root = args.data_root / "Training"
    test_root = args.data_root / "Testing"
    if not train_root.exists() or not test_root.exists():
        raise FileNotFoundError(
            "A dataset root must contain Training/ and Testing/ folders of the "
            "four supported classes."
        )

    train_full = load_images(train_root, args.input_size)
    test = load_images(test_root, args.input_size)

    train_ds, val_ds, _ = reproducible_split(
        train_full, val_fraction=args.val_fraction, test_fraction=0.0,
        seed=args.split_seed,
    )

    test_feed = feature_pipeline(test.x, test.y, args.batch_size, args.input_size, augment=False)
    test_probs = model.predict(test_feed, verbose=0)
    report, _ = evaluate_calibrated(test_probs, test.y)

    charts: list[str] = []
    if not args.no_charts and len(set(test.y.tolist())) > 1:
        cm_path = base_path.with_name(f"{base_path.stem}_confusion.png")
        render_confusion_matrix(report.confusion_matrix, test.labels, cm_path,
                                title="Held-out test confusion matrix")
        charts.append(str(cm_path))
        if report.calibration and report.calibration.binned:
            rel_path = base_path.with_name(f"{base_path.stem}_reliability.png")
            render_reliability_diagram(
                report.calibration.binned, rel_path, report.calibration.ece,
                title="Reliability diagram (held-out test)",
            )
            charts.append(str(rel_path))

    return {
        "kind": "held-out-evaluation",
        "model": spec,
        "latency_ms": latency,
        "preprocessing_version": PREPROCESSING_VERSION,
        "dataset": {
            "found": True,
            "root": str(args.data_root),
            "train_split": train_ds.class_counts,
            "validation_split": val_ds.class_counts,
            "held_out_test": test.class_counts,
            "split": {
                "strategy": "deterministic class-stratified split of Training/",
                "seed": args.split_seed,
                "val_fraction": args.val_fraction,
            },
            "leakage_note": (
                "Patient-level leakage cannot be completely excluded because "
                "this dataset provides no patient/study identifiers. Run the "
                "dataset audit (training.make_manifest) to re-check duplicates."
            ),
        },
        "metrics": report.to_dict(),
        "calibration": report.calibration.to_dict() if report.calibration else None,
        "artifacts": charts,
    }


if __name__ == "__main__":
    main()