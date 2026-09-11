# Brain Tumor AI — AI-Assisted Brain MRI Analysis

A self-hosted web application that analyzes brain MRI images with a convolutional
neural network, classifies them into **glioma**, **meningioma**, **no tumor** or
**pituitary tumor**, explains its decision with a Grad-CAM attention overlay, and
generates a printable PDF report.

> **Medical disclaimer:** This AI-generated result is intended for research and
> educational assistance only and does not constitute a medical diagnosis. Always
> consult a qualified healthcare professional for medical interpretation.

---

## What this application is for

**Brain Tumor AI** is a self-hosted web application for *AI-assisted review of brain
MRI scans* in a research and education setting. An operator uploads a single axial,
coronal or sagittal brain MRI image and the app:

- runs a convolutional neural network that classifies it across the four supported
  classes — **glioma**, **meningioma**, **no tumor** and **pituitary tumor** — and
  returns the **full probability vector**, never just the top guess;
- explains the model's decision with a Grad-CAM **attention overlay** (which regions
  of the image drove the prediction), explicitly labeled as attention, never as tumor
  segmentation;
- generates a concise, printable **PDF report** with the probabilities, the attention
  map and a medical disclaimer;
- persists every analysis with the exact model version, inputs, outputs and processing
  time for a fully auditable record.

The application is designed to be **honest by construction**:

- probabilities are presented as *model probabilities*, never clinical confidence;
- results below a 60% model-probability threshold are marked **classification
  unavailable** (abstention) instead of being presented as a decision;
- the UI, API and reports always state the model's **supported label space**, so an
  out-of-label-space image (e.g. a brain metastasis) is not taken as a confident
  diagnosis;
- the bundled model's training-time metrics are labeled as such and never claimed as
  clinical performance.

Beyond inference, the repository ships reproducible **training and evaluation tooling**
so the demonstration model can be replaced with a properly validated one: one shared
preprocessing pipeline, deterministic dataset audits and splits, calibration (ECE) and
out-of-distribution evaluation, and JSON/PNG artifacts that state explicitly what could
and could not be measured (see [Evaluation tooling](#evaluation-tooling)).

### Who it is for

- **Researchers and ML engineers** retraining or replacing the model — the training
  pipeline, dataset audit, split tooling and evaluation CLIs are designed for them.
- **Radiology trainees and educators** studying model decisions — attention maps, the
  probability view and the packaged case study in `reports/model-evaluation/`.
- **Self-hosting administrators** — the whole stack (API + MariaDB) runs behind
  Docker Compose with sensible defaults.

### What it is *not*

- A diagnostic or clinical device. Reports are for research and education only and do
  not replace professional radiology reports.
- A general tumor detector. The label space is exactly four classes; other tumor types
  are unrecognized and may be pushed toward a wrong class with a high nominal
  probability.

## Key features

| Area | Details |
| --- | --- |
| Prediction API | `POST /api/v1/predictions` with layered validation, near-blank detection and rate limiting |
| Explainability | Grad-CAM heatmap overlay (Keras 3 compatible, raw tensor forward) |
| Reporting | ReportLab PDF with disclaimer, probability table and attention map |
| History | Paginated, searchable, filterable analysis log |
| Model management | Registry scan, metadata (incl. preprocessing version), activation switch (admin-gated) |
| Admin analytics | Prediction distribution, model inventory, recent activity |
| Frontend | Vite + React + TypeScript, light/dark theme, zero gradients |
| Deployment | Docker Compose (app + MariaDB), optional CI |

## Model reliability and validation

The application is honest about what its model can and cannot do:

- **Label space.** The model recognizes exactly four MRI classes — glioma, meningioma,
  no_tumor, pituitary_tumor. Brain tumor types outside that space (e.g. **brain
  metastases**) are *not* part of the label space, and an external image of such a
  lesion can be pushed to a high nominal probability for a wrong class. The UI and PDF
  report state the supported classes explicitly.
- **Calibration.** Raw softmax outputs are **not calibrated** — the inference API
  returns post-softmax probabilities (not logits), so ECE is always measured on raw
  outputs at T=1. The evaluation tooling (`training.run_evaluation`) computes ECE and
  reliability bins on a held-out set and carves a deterministic validation split from
  `Training/` for model selection; it does not fit a temperature.
- **Single preprocessing source of truth.** Training and inference share one canonical
  pipeline — RGB conversion, PIL bilinear resize to the model input size, `float32`
  scaling by `1/255` — defined in `app/core/preprocessing.py` and stamped into the
  registry and evaluation artifacts as `preprocessing_version`.
- **Abstention.** Results below the 60% model-probability threshold are marked
  "classification unavailable" in the UI instead of being presented as a decision.
- **Input validation.** Uploads are checked for extension, MIME type, size, corrupt
  decode, minimum dimension and near-blank content before the model ever runs.

> **Honesty note:** Metrics shown on the Models page are recorded at training time. The
> bundled `Brain Tumor CNN` is a small demonstration network (train accuracy 0.98 /
> validation accuracy 0.72 from the legacy notebook); these are **not** claimed as
> clinical performance and have not been re-verified on this repository's data.

## Evaluation tooling

All evaluation runs from the `backend/` directory and writes JSON + PNG artifacts
under `reports/model-evaluation/`. The tools are honest by design: when a required
labelled dataset is absent they say so explicitly and produce `metrics: null` rather
than inventing numbers.

### Dataset manifest and audit

```bash
cd backend
python -m training.make_manifest --data-root /path/to/dataset
```

Scans the Kaggle-style layout (`Training/` and `Testing/` class folders) and reports,
with no TensorFlow dependency:

- per-class image counts, dimensions, formats and color modes;
- **corrupt/unreadable files**;
- **exact duplicates** (file SHA-256) and **perceptually near-duplicate images**
  (8×8 average hash), including cross-split and cross-class duplicates;
- **cross-split leakage** — hashes seen in more than one split;
- **unknown class folders** (e.g. a `metastasis/` directory) flagged explicitly so new
  diagnoses are never silently mapped into the four supported classes;
- **patient/study identifiers** extracted from filenames (configurable regex) and how
  many images could / could not be grouped for patient-level leakage checks.

Without `--data-root` the tool writes a manifest with `dataset_present: false` and
states exactly why.

### Deterministic, leakage-checkable splits

`backend/training/data.py` provides:

- `reproducible_split` — class-stratified train/validation/test split; the same `seed`
  always yields the same partition (used by the evaluation CLIs).
- `group_stratified_split` — patient/study-independent split: whole groups (patient or
  study IDs) are each assigned to exactly one split.
- `partition_overlap` — checks raw-file and resized-pixel hashes for content appearing
  in more than one split (catches identical and re-encoded copies).

### Model evaluation (spec always, metrics only when a dataset is available)

```bash
# 1. Artifact facts + inference latency — no dataset required:
python -m training.run_evaluation --model-path models/<model>.h5 --label "model-spec"

# 2. Full held-out evaluation — requires a labelled dataset:
python -m training.run_evaluation --model-path models/<model>.h5 \
  --data-root /path/to/dataset --label "held-out v1.0.0"
```

- **Always recorded** (measured from the model file itself): architecture, parameter
  count, weight layers, input/output shapes, file size, Keras version and median
  single-image inference latency.
- **Only when `--data-root` is supplied** (must contain `Training/` and `Testing/`):
  accuracy/precision/recall/F1, per-class sensitivity/specificity, confusion matrix,
  ROC-AUC and **expected calibration error (ECE)** with reliability bins. A
  validation split for model selection / held-out calibration checks is carved
  deterministically from `Training/` (`--val-fraction`, `--split-seed`). Raw softmax
  outputs remain uncalibrated (the inference API returns post-softmax probabilities,
  not logits).
- Deterministic **PNG charts** (confusion matrix + reliability diagram) are written next
  to the JSON artifact (`--no-charts` to skip).

### Out-of-distribution / abstention evaluation

```bash
python -m training.run_ood_evaluation \
  --model-path models/<model>.h5 \
  --id-root  /path/to/in-distribution   # 4 supported classes
  --ood-root /path/to/out-of-distribution # any images: unsupported tumors, non-MRI...
```

Reports MSP / entropy score distributions and the AUROC that measures how well each
score separates the ID set from the OOD set, plus abstention rates (the 60% rule in
`app/core/constants.py`) on both sets. Without `--ood-root` it reports ID statistics
and states that separation could not be measured.

### Artifacts

`reports/model-evaluation/` contains the generated manifests and evaluation JSONs
plus a README that documents the bundled model's limitations and a worked
out-of-label-space case study (external T1-contrast metastasis pushed to
`No Tumor 0.9972`).

## Architecture

```
Browser (Vite SPA)
   │  HTTP/JSON + multipart uploads
   ▼
FastAPI (uvicorn) ------------► MariaDB / SQLite (SQLAlchemy ORM)
   │
   ├── Layered validation (ext, MIME, size, decode, dimension, near-blank)
   ├── Rate limiting (per IP per minute)
   ├── ModelManager ──► Keras CNN (.h5) ──► inference + Grad-CAM
   ├── ReportGenerator (ReportLab)
   └── Media endpoints (original image, attention overlay)
```

Components:

- `backend/app` — FastAPI application.
  - `app/api` — routers (health, predictions, history, models, reports, admin).
  - `app/core` — config, constants, logging, security, **shared preprocessing**.
  - `app/services` — errors, validation, preprocessing, inference, explainability,
    reporting, model management.
  - `app/db` — SQLAlchemy models and SQLite/MariaDB session management.
- `backend/training` — dataset loading, hashing and deterministic/reproducible splits,
  augmentation, architectures, training orchestrator, calibration/ECE evaluation,
  dataset audit + manifest (`make_manifest`), model-spec and held-out evaluation
  (`run_evaluation`), OOD/abstention evaluation (`run_ood_evaluation`), deterministic
  PNG charting, and registry registration.
- `frontend/src` — React pages, API client, theme provider, UI primitives.

## Tech stack

- **Backend:** Python 3.12, FastAPI, Uvicorn, Pydantic v2, SQLAlchemy 2, PyMySQL,
  TensorFlow 2 / Keras 3, Pillow, ReportLab.
- **Frontend:** React 18, React Router 6, Vite 5, TypeScript, Vitest + Testing Library.
- **Database:** SQLite (default, zero setup) or MariaDB/MySQL.

## Repository layout

```
backend/
  app/
    api/          HTTP routers
    core/         config, constants, logging, security
    db/           SQLAlchemy models + session
    models/       keras backend (inference + Grad-CAM)
    repositories/ data access helpers
    schemas/      pydantic schemas
    services/     business logic
    main.py       FastAPI application
  models/         model weights + registry.json
  training/       dataset/training/evaluation pipeline (splits, audit, OOD, evaluation)
  tests/          pytest suite (66 tests)
frontend/
  src/
    api/          typed API client
    components/   UI primitives
    pages/        routed views
    styles/       tokens + base + component CSS
    test/         vitest setup + tests
data/             runtime state (gitignored): DB, uploads, gradcam, reports
legacy/           original single-file code kept for reference
notebooks/        original notebooks
samples/          sample images
```

## Backend setup

Requires Python 3.12.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then edit as needed

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The database, `data/` directories and the model registry are created automatically on
startup. Interactive API docs: <http://127.0.0.1:8000/docs>.

### Backend tests

```bash
cd backend && source .venv/bin/activate
TF_CPP_MIN_LOG_LEVEL=3 python -m pytest -p no:cacheprovider
```

Run the integration tests with `python -m pytest -p no:cacheprovider -m integration`.
If TensorFlow is unavailable, deselect them: `python -m pytest -p no:cacheprovider -m "not integration"`.

## Frontend setup

Requires Node 20+.

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173 (proxies /api to 127.0.0.1:8000)
```

Build and tests:

```bash
npm run build      # type-check + production bundle
npm test           # vitest suite
```

### Theming

Light/dark/system theme persisted in `localStorage` (`btai-theme`), applied before first
paint to avoid flashing. The design system uses **solid colors only — no gradients**.

## Model management

- Models live in `backend/models/` — each `.h5` next to the filesystem, with metadata
  registered in `backend/models/registry.json` (including `preprocessing_version`,
  honest metric buckets and calibration state).
- On startup, `ModelManager` scans the directory, reconciles the registry against disk,
  and activates the registered default.
- Rescan: `POST /api/v1/models/reload` (admin).
- Switch active model: `POST /api/v1/models/{id}/activate` (admin).
- Admin endpoints require the `X-Admin-Key` header matching `ADMIN_API_KEY`.

## Training a replacement model

The labelled dataset (`Training/`, `Testing/` class folders) is **not** distributed
with this repo. When one is available, this is the full honest workflow. Everything
uses the exact same canonical preprocessing as inference
(`app/core/preprocessing.py`), and every artifact states precisely what was measured.

### 1. Audit and inspect the dataset

```bash
cd backend
python -m training.make_manifest --data-root /path/to/dataset \
  --split-seed 42 --val-fraction 0.15 --test-fraction 0.15
```

Review `reports/model-evaluation/dataset-manifest.json` for unknown class folders,
cross-split duplicates/leakage and how many images group into patients. Fix any leaks
before training.

### 2. Train

```bash
python -m training.train --data-dir /path/to/dataset \
  --arch custom_cnn --epochs 40 --batch-size 32 \
  --register --set-active
```

Architectures: `custom_cnn`, `efficientnet_b0`, `resnet50`, `mobilenet_v2`,
`densenet121`. The orchestrator writes honest metrics — including ECE measured on the
held-out `Testing/` set — to the registry; it never fakes numbers. Argument reference:
`python -m training.train --help`.

### 3. Held-out evaluation

```bash
python -m training.run_evaluation --model-path models/<model>.h5 \
  --data-root /path/to/dataset --label "held-out"
```

Writes `reports/model-evaluation/<name>.json` with metrics + ECE and deterministic
confusion-matrix / reliability-diagram PNGs. To record model-spec facts (params,
latency, shapes) without any dataset, simply omit `--data-root`.

### 4. Out-of-distribution behavior

```bash
python -m training.run_ood_evaluation --model-path models/<model>.h5 \
  --id-root /path/to/in-distribution \
  --ood-root /path/to/unsupported-or-non-mri-images
```

Confirms whether the model's probabilities actually separate supported from unsupported
images before it is trusted in front of users.

## API reference (all under `/api/v1`)

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness, database + active model status |
| POST | `/predictions` | Analyze an MRI (multipart `file`) → 201 |
| GET | `/predictions/{analysis_id}` | Fetch an analysis result |
| GET | `/predictions/{analysis_id}/image` | Original uploaded image |
| GET | `/predictions/{analysis_id}/gradcam` | AI attention overlay (PNG) |
| GET | `/history` | Paginated history (`page`, `page_size`, `search`, `prediction`, `date_from`, `date_to`) |
| GET | `/history/{analysis_id}` | Single history item |
| GET | `/models` | Registry models + active model |
| GET | `/models/{model_id}` | Model metadata incl. metrics + preprocessing version |
| POST | `/models/{model_id}/activate` | Set active model (admin) |
| POST | `/models/reload` | Rescan model dir (admin) |
| GET | `/reports` | Paginated report list |
| POST | `/reports` | Generate report for an analysis |
| GET | `/reports/{report_id}` | Report metadata |
| GET | `/reports/{report_id}/download` | Report PDF |
| GET | `/admin/analytics` | Admin analytics (admin) |

`POST /predictions` responds with the full class probabilities, `low_confidence`
(abstention flag), `supported_classes` (the model's label space), the model identity and
`model.processing_time_ms`; probabilities are labeled in the UI as **model probability**,
never confidence.

## Configuration (environment variables)

Copy `backend/.env.example` to `.env`. Key variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite:///./data/brain_tumor_ai.db` | SQLAlchemy URL (use `mysql+pymysql://…` for MariaDB) |
| `ADMIN_API_KEY` | *(empty)* | Enables admin endpoints; sent as `X-Admin-Key` |
| `CORS_ORIGINS` | `["http://localhost:5173","http://127.0.0.1:5173"]` | Allowed browser origins |
| `MAX_UPLOAD_SIZE_MB` | `10` | Upload size cap |
| `ALLOWED_IMAGE_EXTENSIONS` | JPG/PNG/BMP/TIFF | Accepted formats |
| `MIN_IMAGE_DIMENSION` | `64` | Minimum image side |
| `PREDICTION_RATE_LIMIT` | `6` | Predictions per minute per IP |
| `REPORT_RATE_LIMIT` | `10` | Reports per minute per IP |
| `HEALTH_RATE_LIMIT` | `30` | Health checks per minute per IP |
| `ACTIVE_MODEL_NAME` | `Brain Tumor CNN` | Default active model |
| `MODEL_TIMEOUT_SECONDS` | `60` | Inference timeout |

## Database

- **Development:** SQLite at `data/brain_tumor_ai.db` — zero external dependencies.
- **Production:** set `DATABASE_URL` to a MariaDB/MySQL instance (either external or via
  Docker Compose). Schema creation is automatic on startup. Never run the app with the
  default `SECRET_KEY` in production.

## Deployment

Run the full stack (API + MariaDB) with Docker Compose:

```bash
cp backend/.env.example .env   # adapt DATABASE_URL to the compose service
docker compose up --build
```

Frontend: `npm run build` and serve `frontend/dist` behind any static host, pointing the
API client at the backend (see the Settings page).

## Known limitations

- The bundled model is a **demonstration network**; it is not clinically validated.
- Its **label space is exactly four classes**; other tumor types (metastases, etc.) are
  unrecognized and can be misclassified with a high nominal probability.
- Softmax probabilities are **uncalibrated**; treat them as relative scores, not
  probabilities of ground truth.
- Grad-CAM is an **explanation aid**, not a tumor segmentation or diagnosis.
- Cross-scanner generalization has not been verified.
- The original training dataset is not included in this repository, so held-out
  metrics cannot be reproduced here.
- Reports do not replace professional radiology reports.

## Security notes

- No secrets are hardcoded; configure everything via `.env` (never committed).
- Admin endpoints only work when `ADMIN_API_KEY` is set.
- Rate limiting protects prediction, report and health endpoints.
- Local storage on the browser holds only UI preferences and the admin key.