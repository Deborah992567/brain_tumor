"""Standalone evaluation CLI for a trained model.

Runs the shared evaluation metrics (classification + calibration) on a model
artifact against a labelled dataset, and writes a JSON artifact under
``reports/model-evaluation/`` with the same schema produced by ``train.py`` so
every evaluation is directly comparable.

Usage (from the ``backend/`` directory, with the dataset available):

    python -m training.run_evaluation \
        --model-path models/brain_tumor.h5 \
        --data-root /path/to/braintumor \
        --input-size 64 \
        --output ../reports/model-evaluation/ev-1.0.0.json

``--data-root`` must contain a ``Testing/`` folder of class subfolders. If a
``Validation/`` folder is present it is used to fit a temperature; otherwise
only the uncalibrated expected calibration error (ECE) is reported.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from app.core.preprocessing import PREPROCESSING_VERSION
from training.data import load_images
from training.evaluate import evaluate_calibrated
from training.augmentation import feature_pipeline


def _argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Trained model artifact (.h5/.keras).")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Dataset root with Testing/(and optionally Validation/).")
    parser.add_argument("--input-size", type=int, default=64,
                        help="Model input size in pixels.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path,
                        default=Path("../../reports/model-evaluation/evaluation.json"))
    parser.add_argument("--label", default="",
                        help="Short label stored in the artifact, e.g. 'v1.0.0 held-out'.")
    return parser


def main() -> None:
    args = _argparse().parse_args()

    from tensorflow import keras
    model = keras.models.load_model(str(args.model_path))

    test = load_images(args.data_root / "Testing", args.input_size)

    test_features = feature_pipeline(
        test.x, test.y, args.batch_size, args.input_size, augment=False
    )
    test_probs = model.predict(test_features, verbose=0)

    report, _ = evaluate_calibrated(test_probs, test.y)

    artifact = {
        "label": args.label or str(args.model_path),
        "model_path": str(args.model_path),
        "preprocessing_version": PREPROCESSING_VERSION,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "split_notes": {
            "source": "labelled dataset provided by the operator",
            "held_out_test": f"{test.n} images",
            "patient_level_split": "unknown - patient IDs not available in the dataset",
        },
        "metrics": report.to_dict(),
        "calibration_note": (
            "Raw softmax probabilities are uncalibrated. ECE is measured on the "
            "model outputs (T=1). Temperature scaling is not available because "
            "the model inference API returns post-softmax probabilities rather "
            "than raw logits."
        ),
    }

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2))
    print(f"Evaluation artifact written to {out}")
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()