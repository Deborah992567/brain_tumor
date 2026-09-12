"""Dataset manifest generator (Phase 2).

Produces ``reports/model-evaluation/dataset-manifest.json`` describing whatever
the audit found on disk — including an explicit ``dataset_present: false`` when
no labelled data is available. It never fabricates counts, split sizes or
leakage results: every field is either observed or derived deterministically
from the supplied arguments.

Usage (from the ``backend/`` directory):

    python -m training.make_manifest [--data-root /path/to/dataset]
        [--split-seed 42] [--val-fraction 0.15] [--test-fraction 0.15]
        [--output ../reports/model-evaluation/dataset-manifest.json]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from app.core.preprocessing import PREPROCESSING_VERSION
from training.data import reproducible_split, load_images
from training.dataset_audit import (
    DEFAULT_PATIENT_PATTERN,
    audit_dataset,
    audit_to_dict,
)


def _argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Optional dataset root with Training//Testing folders.")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--patient-pattern", default=DEFAULT_PATIENT_PATTERN,
                        help="Regex with a 'patient' named group for grouping.")
    parser.add_argument("--output", type=Path,
                        default=Path("../reports/model-evaluation/dataset-manifest.json"))
    return parser


def main() -> None:
    args = _argparse().parse_args()

    audit = audit_dataset(args.data_root, patient_pattern=args.patient_pattern)

    split_plan: dict | None = None
    if audit.dataset_found and args.data_root:
        try:
            train_full = load_images(args.data_root / "Training", max_per_class=None)
            train_ds, val_ds, test_ds = reproducible_split(
                train_full,
                val_fraction=args.val_fraction,
                test_fraction=args.test_fraction,
                seed=args.split_seed,
            )
            split_plan = {
                "strategy": "deterministic class-stratified split (reproducible by seed)",
                "seed": args.split_seed,
                "val_fraction": args.val_fraction,
                "test_fraction": args.test_fraction,
                "train": train_ds.class_counts,
                "validation": val_ds.class_counts,
                "held_out_test": test_ds.class_counts,
            }
        except Exception as exc:  # data not loadable -> honest note
            split_plan = {"error": f"Could not compute split: {exc}"}

    manifest = audit_to_dict(
        audit,
        extras={
            "dataset_id": "brain-tumor-classification-4class",
            "dataset_version": "unavailable" if not audit.dataset_found else "provided-by-operator",
            "dataset_present": audit.dataset_found,
            "preprocessing_version": PREPROCESSING_VERSION,
            "split_plan": split_plan,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    # Make the "not present" story unambiguous.
    if not audit.dataset_found:
        manifest["limitations"] = (
            "No labelled dataset is present in this repository. The reproducible "
            "audit and split tooling is ready; the audit must be re-run with "
            "--data-root once a real dataset is supplied. Patient-level leakage "
            "cannot be excluded for a dataset that is absent, and retraining "
            "cannot be performed until one is available."
        )

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(f"Dataset manifest written to {out}")
    print(f"dataset_present: {audit.dataset_found} | total_images: {audit.total_images}")


if __name__ == "__main__":
    main()