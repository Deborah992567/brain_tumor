# Brain Tumor AI — AI-Assisted Brain MRI Analysis

A self-hosted web application that analyzes brain MRI images with a convolutional
neural network, classifies them into **glioma**, **meningioma**, **no tumor** or
**pituitary tumor**, explains its decision with a Grad-CAM attention overlay, and
generates a printable PDF report.

> **Medical disclaimer:** This AI-generated result is intended for research and
> educational assistance only and does not constitute a medical diagnosis. Always
> consult a qualified healthcare professional for medical interpretation.

---

## Purpose

- Classify a single axial, coronal or sagittal brain MRI scan across four classes.
- Show the **full probability distribution**, not only the top guess. Numbers are
  presented honestly as **model probabilities** — never as clinical confidence — and
  flagged as **classification unavailable** below a 60% trust threshold (abstention),
  or when the image is outside the model's 4-class label space.
- Explain the model's reasoning with an **AI attention visualization** (Grad-CAM),
  explicitly labeled as attention — never as tumor segmentation.
- Persist every analysis with exact model version, inputs, outputs and timing for
  full auditability.
- Generate a professional **PDF report** per analysis.
- Provide a training pipeline (with a single shared preprocessing pipeline and
  calibration metrics) to replace the demonstration model with a properly
  validated model.

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
- **Calibration.** Raw softmax outputs are **not calibrated**; the expected calibration
  error (ECE) is only measurable on a labelled held-out set, which is not bundled with
  the repository. The evaluation tooling (`training.run_evaluation`) computes ECE and
  reliability bins, and fits a temperature-scaling temperature on a separate held-out
  set when a `Validation/` folder is provided.
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
- `backend/training` — dataset loading, stratified split, augmentation, architectures,
  training orchestrator, calibration/ECE evaluation, `run_evaluation` CLI and registry
  registration.
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
  training/       dataset/training/evaluation pipeline
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

The dataset (`Training/`, `Testing/` class folders) is not distributed with this repo.
When available, use the pipeline — it uses the exact same canonical preprocessing as
inference and records calibration metrics:

```bash
cd backend

python -m training.train --data-dir /path/to/dataset \
  --arch custom_cnn --epochs 40 --batch-size 32 \
  --register --set-active
```

Architectures: `custom_cnn`, `efficientnet_b0`, `resnet50`, `mobilenet_v2`,
`densenet121`. The orchestrator writes honest metrics (including ECE and a temperature
fit from the held-out validation split) to the registry; it never fakes numbers.

Evaluate any trained artifact against a labelled `Testing/` set with the dedicated CLI,
which writes a comparable JSON artifact under `reports/model-evaluation/`:

```bash
cd backend

python -m training.run_evaluation \
  --model-path models/brain_tumor.h5 \
  --data-root /path/to/dataset \
  --input-size 64 \
  --output ../reports/model-evaluation/ev-1.0.0.json \
  --label "v1.0.0 held-out"
```

If a `Validation/` folder is present it is used to fit a temperature before reporting
the post-calibration ECE; otherwise only the uncalibrated ECE is reported.

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