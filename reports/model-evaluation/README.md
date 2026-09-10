# Model evaluation artifacts

This directory stores honest, traceable evaluation outputs for the bundled and
training-pipeline models. Files are produced by:

```bash
cd backend
python -m training.run_evaluation \
  --model-path models/<model>.h5 \
  --data-root /path/to/labelled/dataset \
  --input-size 64 \
  --output ../reports/model-evaluation/<artifact>.json \
  --label "<label>"
```

## What the bundled model (v1.0.0) can validate

The `Brain Tumor CNN` v1.0.0 shipped with this repo was trained on a legacy
Kaggle-style split (2870 train / 393 test images) and its recorded metrics are
training-time values (`train accuracy 0.9847`, `validation accuracy 0.7176`).
The original dataset is **not** distributed with this repository, so those
numbers cannot be independently re-verified here, and no ECE is available until
a labelled held-out set is supplied through the CLI above.

What *is* verifiable in this repository and is guaranteed by tests:

- **Preprocessing consistency.** Training and inference use one shared pipeline
  (RGB → PIL bilinear resize → `float32 / 255`), stamped as
  `preprocessing_version: "1.0.0-255-bilinear-rgb"`. See
  `backend/tests/test_preprocessing_consistency.py`.
- **Calibration honesty.** Raw softmax outputs are uncalibrated; the UI, PDF and
  API label them as *model probabilities*, never confidence. ECE measurement is
  available in `training/evaluate.py` (numpy-only) and used by the evaluation CLI.
- **Label space.** Exactly 4 classes — glioma, meningioma, no_tumor,
  pituitary_tumor. Types outside the space (e.g. brain metastases) are not
  recognized.

## Case study: external T1-contrast image with a metastasis

`Hirnmetastase_MRT-T1_KM.jpg` (a T1-weighted, contrast-enhanced MRI showing a
brain metastasis) produced a top class of **No Tumor at 0.9972** model
probability. Investigation conclusions:

1. **Root cause.** A brain metastasis is outside the model's 4-class label space.
   The model was only ever trained to choose among the 4 supported classes, so
   the softmax is forced to distribute probability over them for *any* input,
   including out-of-distribution ones. A high nominal probability of `No Tumor`
   is therefore not evidence of "no tumor".
2. **Preprocessing match.** The inference preprocessing reproduces the legacy
   training preprocessing (RGB, bilinear 64×64, `1/255`); this is not a
   preprocessing-drift issue.
3. **Remaining gaps.** No OOD detector or abstention trigger exists for
   in-label-space-but-unusual images; softmax probabilities are uncalibrated; no
   external-validation metric (accuracy or ECE) is available without the labelled
   dataset.

### How this case is handled by the application

- The API returns the full probability vector, the model identity, the supported
  classes and a note that probabilities are uncalibrated.
- The UI and PDF label results as *model probability* and state the supported
  classes, so a `No Tumor 0.9972` for a metastasis can be recognized as
  out-of-label-space rather than taken as a diagnosis.
- Below the 60% model-probability threshold the result is shown as
  *Classification unavailable* instead of a decision.

Fix the root cause properly by retraining with the pipeline (inclusion of
additional classes / an out-of-distribution rejection head) when a suitable
labelled dataset is available, and by running `training.run_evaluation` to
produce a measured ECE and per-class metrics that this repo can then display.