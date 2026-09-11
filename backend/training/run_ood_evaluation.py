"""OOD / abstention evaluation CLI.

Measures how well the model's own outputs separate a supported in-distribution
set from an out-of-distribution set (unsupported tumour classes, non-MRI
images, noise, ...). Produces ``reports/model-evaluation/ood-evaluation.json``.

Usage (from the ``backend/`` directory):

    python -m training.run_ood_evaluation --model-path models/brain_tumor.h5 \\
        --id-root /path/to/in-distribution \\
        --ood-root /path/to/out-of-distribution \\
        --output ../reports/model-evaluation/ood-evaluation.json

``--ood-root`` is optional: without it the artifact reports in-distribution
statistics and explicitly states that separation could not be measured.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from app.core.constants import CONFIDENCE_LOW_THRESHOLD
from app.core.preprocessing import PREPROCESSING_VERSION
from training.ood_evaluation import evaluate_ood_on_roots


def _argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--id-root", type=Path, required=True,
                        help="Folder of the four supported classes.")
    parser.add_argument("--ood-root", type=Path, default=None,
                        help="Optional folder of unsupported classes / non-MRI images.")
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--abstain-threshold", type=float,
                        default=CONFIDENCE_LOW_THRESHOLD)
    parser.add_argument("--output", type=Path,
                        default=Path("../reports/model-evaluation/ood-evaluation.json"))
    return parser


def main() -> None:
    args = _argparse().parse_args()

    from tensorflow import keras

    model = keras.models.load_model(str(args.model_path))
    result = evaluate_ood_on_roots(
        model,
        args.id_root,
        args.ood_root,
        input_size=args.input_size,
        batch_size=args.batch_size,
        abstain_threshold=args.abstain_threshold,
    )
    result.update({
        "model_path": str(args.model_path),
        "preprocessing_version": PREPROCESSING_VERSION,
        "in_distribution_root": str(args.id_root),
        "out_of_distribution_root": str(args.ood_root) if args.ood_root else None,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    })

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"OOD evaluation written to {out}")
    if result.get("separation_auroc"):
        print(json.dumps(result["separation_auroc"], indent=2))
    else:
        print("separation_auroc: not measured (no OOD set supplied)")


if __name__ == "__main__":
    main()