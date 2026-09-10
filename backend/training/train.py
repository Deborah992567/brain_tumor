"""Training orchestration for the brain-tumor classifier.

Usage (from the ``backend/`` directory, with the dataset available):

    python -m training.train \
        --data-dir /path/to/dataset \
        --arch efficientnet_b0 \
        --input-size 128 \
        --epochs 30 \
        --result-name "EfficientNetB0" \
        --result-version "2.0.0" \
        --output models/efficientnet_b0.keras \
        --register

Validation is a stratified split of the *training* folder; the ``Testing``
folder is a held-out set used only once, after training, for the final
evaluation report. This prevents validation feedback loops and leakage.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from training.architectures import build
from training.augmentation import feature_pipeline
from training.data import load_images, stratified_split
from training.evaluate import evaluate

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "models"


def _argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Dataset root containing Training/ and Testing/ folders.")
    parser.add_argument("--arch", default="efficientnet_b0",
                        choices=sorted(_architectures()))
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--fine-tune-epochs", type=int, default=0,
                        help="Optional extra epochs with base layers unfrozen.")
    parser.add_argument("--result-name", default="EfficientNetB0")
    parser.add_argument("--result-version", default="2.0.0")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--register", action="store_true",
                        help="Register the trained model in models/registry.json.")
    parser.add_argument("--set-active", action="store_true",
                        help="Make the registered model the active one.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _architectures() -> set[str]:
    from training.architectures import ARCHITECTURES
    return ARCHITECTURES


def main() -> None:
    args = _argparse().parse_args()

    from tensorflow import keras

    model = build(
        args.arch,
        input_size=args.input_size,
        dropout_rate=args.dropout,
    )

    train_full = load_images(args.data_dir / "Training", args.input_size)
    train_ds, val_ds = stratified_split(train_full, val_fraction=0.15, seed=args.seed)

    print(
        f"Classes: {train_full.labels} | train: {train_ds.n} | "
        f"val: {val_ds.n} | class weights: {train_full.class_weights}"
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(args.output or DEFAULT_DIR / f"{args.result_name}.keras"),
            monitor="val_loss", save_best_only=True, verbose=1,
        ),
    ]

    train_ds_feed = feature_pipeline(
        train_ds.x, train_ds.y, args.batch_size, args.input_size, augment=True, seed=args.seed
    )
    val_ds_feed = feature_pipeline(
        val_ds.x, val_ds.y, args.batch_size, args.input_size, augment=False
    )

    model.fit(
        train_ds_feed,
        validation_data=val_ds_feed,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=train_full.class_weights,
    )

    if args.fine_tune_epochs > 0 and args.arch != "custom_cnn":
        base = model.layers[0]
        if hasattr(base, "trainable"):
            base.trainable = True
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate / 10),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            train_ds_feed, validation_data=val_ds_feed,
            epochs=args.fine_tune_epochs, callbacks=callbacks,
            class_weight=train_full.class_weights,
        )

    output_path = args.output or DEFAULT_DIR / f"{args.result_name}.keras"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))

    # Final evaluation on the held-out Testing set.
    test = load_images(args.data_dir / "Testing", args.input_size)
    test_features = feature_pipeline(
        test.x, test.y, args.batch_size, args.input_size, augment=False
    )
    probabilities = model.predict(test_features, verbose=0)
    report = evaluate(probabilities, test.y, test.labels)

    eval_path = output_path.with_suffix(".metrics.json")
    report_data = {
        "architecture": args.arch,
        "input_size": args.input_size,
        "dataset": f"Training={train_ds.n + val_ds.n} images, "
                   f"held-out Testing={test.n} images",
        "dataset_version": f"{train_ds.n + val_ds.n} train / {test.n} test",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": report.to_dict(),
        "training_args": vars(args),
    }
    eval_path.write_text(json.dumps(report_data, indent=2))
    print(f"Evaluation report written to {eval_path}")
    print(json.dumps(report.to_dict(), indent=2))

    if args.register:
        from training.register import register_model

        register_model(
            name=args.result_name,
            version=args.result_version,
            architecture=f"{args.arch} ({args.input_size}px)",
            file_path=str(output_path),
            input_size=args.input_size,
            dataset_version=report_data["dataset_version"],
            metrics=report.to_dict(),
            status="ready",
            description=(
                f"Trained with the improved pipeline ({args.arch}, "
                f"dropout={args.dropout}, stratified validation, held-out test evaluation)."
            ),
            set_active=args.set_active,
        )


if __name__ == "__main__":
    main()