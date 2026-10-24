# Hybrid SHAP + LLM Predictive Model Explainability Platform

A privacy-first, bare-metal Explainable AI (XAI) web platform that combines quantitative feature attributions from **SHAP (SHapley Additive exPlanations)** with grounded qualitative natural language interpretations powered by a local **Gemma** model running on **Ollama**.

Built with an asynchronous **FastAPI** backend and an interactive **React + Vite + TypeScript** frontend, this platform is designed for **zero cloud data egress** (GDPR Right to Explanation, FDA SaMD transparency) in sensitive tabular domains like clinical healthcare, loan underwriting, and policy analytics.

---

## Table of Contents

- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Role-Based Access Control (RBAC)](#role-based-access-control-rbac)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation & Local Setup](#installation--local-setup)
  - [1. Clone Repository & Setup Environment](#1-clone-repository--setup-environment)
  - [2. Ollama Local LLM Engine Setup](#2-ollama-local-llm-engine-setup)
  - [3. Backend Setup (FastAPI)](#3-backend-setup-fastapi)
  - [4. Frontend Setup (React + Vite)](#4-frontend-setup-react--vite)
- [API Reference](#api-reference)
- [Data Serialization & Grounding Contract](#data-serialization--grounding-contract)
- [Security Model & Threat Assumptions](#security-model--threat-assumptions)
- [Verification & Testing](#verification--testing)
- [Evaluation Support](#evaluation-support)
- [Deviations from SRS v1.0.1 / SDD v1.0.0](#deviations-from-srs-v101--sdd-v100)
- [Roadmap & Milestone Progression](#roadmap--milestone-progression)
- [License](#license)

---

## Key Features

1. **Dual-Layer Explainability Framework:**
   - **Quantitative Rigor:** Automated explainer factory selects `shap.TreeExplainer` for tree ensembles, explaining classifiers directly in probability space (interventional, against the model's reference dataset), with fallback to `shap.KernelExplainer` over a k-means summary of that dataset for black-box estimators.
   - **Consistent units:** every explanation reports `output_units` (probability, log-odds, target units) and the additivity gap `|E[f(x)] + Σφ − f(x)|`, so baseline, attributions and prediction are always on the same scale.
   - **Qualitative Grounding:** Top-$K$ positive and negative SHAP contributions are serialized into a strict JSON context and sent to a local Gemma instance via Ollama.
2. **Server-side grounding:** the LLM only ever sees SHAP values the server computed and stored. Chat requests reference an `explanation_id`; attribution values are never accepted from the client.
3. **Multi-turn follow-up Q&A** with full conversation history, streamed token-by-token over Server-Sent Events.
4. **Automated faithfulness check:** every LLM answer is scanned; each number must be a rounding of a value in the SHAP payload, and narratives are checked for the 4-part structure.
5. **Zero Cloud Egress by default:** a non-loopback `OLLAMA_BASE_URL` is refused unless `ALLOW_REMOTE_LLM=true` is set explicitly.
6. **Multi-Model Ownership & Discretionary Access Control (DAC):** grant, update, list and revoke `VIEWER` / `EXPLAINER` / `ADMIN` access by email; owners can delete models.
7. **Sandboxed storage & ingestion:** uploads are size-limited, quarantined under `storage/{models|datasets}/{user_id}/`, path-checked with `Path.is_relative_to`, and model files are deserialized in a disposable child process with a timeout. `.skops` (non-pickle) models are supported.
8. **Interactive client visualizations:** Plotly waterfall, force plot, global importance bar, beeswarm and dependence plots rendered from JSON.
9. **Audit trail & evaluation metrics:** every explanation and LLM answer is recorded with latency and faithfulness, with aggregate metrics and CSV export for the evaluation report.

---

## System Architecture

```
+-----------------------------------------------------------------------------------------+
|                                PRESENTATION LAYER (CLIENT)                              |
| React 18 + TypeScript + Vite (:3001)                                                    |
|  * Auth (JWT)  * Model / Dataset Upload  * Paginated Row Selector  * What-If Inputs     |
|  * Plotly Waterfall / Force / Beeswarm / Dependence  * Streaming Grounded Chat          |
|  * Sharing Panel  * Audit & Evaluation Metrics                                          |
+--------------------------------------------+--------------------------------------------+
                                             | HTTP / REST + SSE (Vite Dev Proxy -> :8001)
                                             v
+-----------------------------------------------------------------------------------------+
|                             APPLICATION & API LAYER (BACKEND)                           |
| FastAPI + Uvicorn Async Runtime (:8001)                                                 |
|  Auth & RBAC Gate | Model & Dataset Ingest | Inference | Explain / Chat | Audit         |
+--------------------------------------------+--------------------------------------------+
               +-----------------------------+-----------------------------+
               v                             v                             v
+---------------------------+  +--------------------------+  +----------------------------+
|     PERSISTENCE TIER      |  |   COMPUTE WORKER TIER    |  |     LLM RUNTIME TIER       |
| SQLite (aiosqlite)        |  | Thread-pool workers      |  | Local Ollama (:11434)      |
|  * users                  |  |  * sandboxed inspection  |  |  * Gemma                   |
|  * model_artifacts        |  |    subprocess            |  |  * Grounded JSON Prompter  |
|  * model_shares (DAC)     |  |  * LRU explainer cache   |  |  * SSE token streaming     |
|  * datasets               |  |  * shap.TreeExplainer    |  |                            |
|  * explanation_records    |  |  * shap.KernelExplainer  |  |                            |
|  * chat_messages          |  |                          |  |                            |
+---------------------------+  +--------------------------+  +----------------------------+
```

CPU-bound work (deserialization, SHAP, pandas) runs in a thread pool so the event loop stays responsive. Loaded estimators and explainers are cached per model; the cache key includes the file's modification time and the background dataset id.

---

## Role-Based Access Control (RBAC)

| Role | View Data & Predict | Run SHAP & LLM | Manage Sharing & Audit | Grant / Revoke ADMIN | Delete Model |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`VIEWER`** | :white_check_mark: | :x: | :x: | :x: | :x: |
| **`EXPLAINER`** | :white_check_mark: | :white_check_mark: | :x: | :x: | :x: |
| **`ADMIN`** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: | :x: |
| **`OWNER`** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |

Conversations are private to the user who ran the explanation. ADMIN and OWNER see all of them through the audit trail.

---

## Tech Stack

- **Backend:** Python 3.11+, FastAPI, Uvicorn, Pydantic v2, Pydantic-Settings
- **Machine Learning & XAI:** `shap` (v0.45+), `scikit-learn`, `joblib`, `skops`, `pandas`, `numpy`
- **LLM Integration:** Ollama (local daemon), Gemma, `httpx` (async, streaming)
- **Database & Security:** SQLite, SQLAlchemy 2.0 (AsyncIO with `aiosqlite`), `pwdlib` (bcrypt, work factor 12), PyJWT (HS256)
- **Frontend:** React 18, Vite, TypeScript, Tailwind CSS, Plotly.js, Lucide React

---

## Repository Structure

```text
c-box/
├── .env.example                     # Environment template
├── README.md
│
├── backend/
│   ├── requirements.txt             # Runtime dependencies
│   ├── requirements-dev.txt         # + pytest
│   ├── pytest.ini
│   ├── run.py                       # Local Uvicorn runner (BACKEND_HOST / BACKEND_PORT)
│   ├── init_db.py                   # Creates / upgrades the SQLite schema
│   │
│   ├── app/
│   │   ├── main.py                  # App entry point, CORS, lifespan (schema upgrade + backfill)
│   │   ├── config.py                # Settings, secret / zero-egress validation
│   │   ├── database.py              # Async engine, FK pragma, lightweight migrations
│   │   │
│   │   ├── api/v1/endpoints/
│   │   │   ├── auth.py              # Register, login, me
│   │   │   ├── models.py            # Upload, list, get, delete, sharing, background dataset
│   │   │   ├── datasets.py          # Upload + validation, list, paginated rows, delete
│   │   │   ├── predict.py           # Point inference (VIEWER+)
│   │   │   ├── explain.py           # Local / global SHAP, grounded chat (JSON + SSE)
│   │   │   └── audit.py             # Audit trail, evaluation metrics, CSV export
│   │   │
│   │   ├── core/
│   │   │   ├── security.py          # bcrypt hashing and JWT
│   │   │   ├── dependencies.py      # OAuth2 bearer guard
│   │   │   ├── permissions.py       # RBAC / DAC gates for models and datasets
│   │   │   ├── storage.py           # Path sandbox, filename sanitizing, size-limited uploads
│   │   │   ├── model_loader.py      # Extension policy, subprocess inspection
│   │   │   ├── _inspect_worker.py   # Child-process loader (joblib / skops)
│   │   │   ├── feature_alignment.py # Column validation, target detection, reordering
│   │   │   ├── shap_engine.py       # Explainer factory, unit detection, global beeswarm
│   │   │   ├── explainer_cache.py   # LRU cache of estimators / explainers
│   │   │   ├── instances.py         # Resolves manual inputs / dataset rows, cached engines
│   │   │   ├── inference.py         # Shared prediction helper
│   │   │   ├── llm_service.py       # Context payload, multi-turn prompts, Ollama streaming
│   │   │   ├── faithfulness.py      # Automated grounding check
│   │   │   └── maintenance.py       # Startup backfill of model metadata
│   │   │
│   │   ├── models/                  # user, artifact (+shares), dataset, explanation (+chat)
│   │   ├── schemas/                 # Pydantic contracts
│   │   └── storage/                 # Sandboxed models/ and datasets/
│   │
│   └── tests/                       # Engine, LLM, API, RBAC and security tests
│
├── frontend/
│   ├── vite.config.ts               # Dev server (:3001) and /api proxy (VITE_API_TARGET)
│   └── src/
│       ├── App.tsx
│       ├── components/
│       │   ├── upload/              # Model and dataset upload forms
│       │   ├── models/              # Model context card, sharing panel
│       │   ├── data/                # Paginated dataset table, prediction / what-if module
│       │   ├── visualizations/      # Waterfall, force, summary (bar / beeswarm / dependence)
│       │   ├── chat/                # Streaming grounded chat with faithfulness badges
│       │   └── audit/               # Audit trail and evaluation metrics
│       ├── services/api.ts          # Axios client + SSE stream reader
│       ├── types/api.ts
│       └── utils/format.ts
│
└── sample_data/                     # Sample model and matching evaluation CSV
```

---

## Prerequisites

1. **Python 3.11+**
2. **Node.js 18+** and **npm**
3. **Ollama** ([Download Ollama](https://ollama.com/download))

---

## Installation & Local Setup

### 1. Clone Repository & Setup Environment

```bash
git clone https://github.com/DocDim/c-box.git
cd c-box
cp .env.example .env
```

Generate a secret key and put it in `.env`. The backend refuses to start with the placeholder or with a key shorter than 32 characters:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

```env
SECRET_KEY="<paste generated key>"
DATABASE_URL="sqlite+aiosqlite:///./xai_app.db"
OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_MODEL="gemma:2b"
# ALLOW_REMOTE_LLM=true   # only if Ollama runs on another host (NFR-2 deviation)
```

All settings are listed in [backend/app/config.py](backend/app/config.py). Values in `.env` must use `KEY=value` syntax.

### 2. Ollama Local LLM Engine Setup

```bash
ollama serve
ollama pull gemma:2b        # larger variants (e.g. gemma2:9b) give noticeably more faithful narratives
curl http://localhost:11434/api/tags
```

### 3. Backend Setup (FastAPI)

```bash
cd backend
python -m venv venv
# Linux / macOS: source venv/bin/activate
# Windows PowerShell: .\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python run.py
```

The backend starts at `http://127.0.0.1:8001`; OpenAPI docs are at **`http://127.0.0.1:8001/docs`**.
On startup, existing SQLite databases are upgraded in place, and models uploaded before v1.1 are re-inspected so their task type and feature metadata are recorded.

### 4. Frontend Setup (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

The client runs at `http://localhost:3001`. Calls to `/api/v1` are proxied to `http://127.0.0.1:8001`; set `VITE_API_TARGET` to point the proxy elsewhere.

**UI layout.** The left column holds navigation plus the active model's metadata (role, estimator, task, classes, features, background dataset). The centre shows the current page. The right column holds the grounded chat for the current explanation.

| Page | Purpose |
| :--- | :--- |
| `#/models` | Register a model together with its background data; list, activate, share, delete. The owner reloads the background data under *Manage → Background data* |
| `#/predict` | Choose the model for the prediction, then predict a **single data point** from a form, or run a **batch** prediction on an evaluation dataset loaded on the page; explain any result with SHAP |
| `#/insights` | Global model behaviour from SHAP over the background data: importance, beeswarm, dependence (EXPLAINER+) |
| `#/audit` | Audit trail, evaluation metrics, CSV export (ADMIN / OWNER) |

---

## API Reference

All protected endpoints require an `Authorization: Bearer <JWT>` header.

### Authentication
- `POST /api/v1/auth/register` (`{ email, password }`; password 8 to 72 characters)
- `POST /api/v1/auth/login` (OAuth2 form `username`, `password`; returns a JWT)
- `GET /api/v1/auth/me`

### Models & Sharing
- `POST /api/v1/models/upload`: multipart `file` (`.skops`, `.joblib`, `.pkl`; max 250 MB), `name`, `domain_context`, optional `background_file` (CSV) and `target_column`. The background CSV is validated against the model's features and registered as its SHAP reference in the same transaction; if it is rejected, nothing is saved. Returns the detected estimator, task type, feature names and classes.
- `GET /api/v1/models/` · `GET /api/v1/models/{id}` · `DELETE /api/v1/models/{id}` (owner)
- `POST /api/v1/models/{id}/background` (multipart `file`, optional `target_column`; **owner only**): uploads or replaces the background data. The previous background dataset is deleted.
- `GET /api/v1/models/{id}/shares` · `POST /api/v1/models/{id}/share` (`{ user_email, role }`) · `DELETE /api/v1/models/{id}/share/{user_id}` (ADMIN+)

### Datasets
- `POST /api/v1/datasets/upload`: multipart `file`, optional `model_id`, `target_column`. Uploads **evaluation** data to predict on: validates columns against the model, auto-detects a single extra column as the target, and returns summary statistics and a preview. Any user who can run the model may load one. It never becomes background data.
- `GET /api/v1/datasets/?model_id=&purpose=background|evaluation` · `GET /api/v1/datasets/{id}/rows?offset=&limit=` · `DELETE /api/v1/datasets/{id}`

### Inference & Explainability
Instance requests take either `instance_features`, or `dataset_id` + `row_index` (optionally with `instance_features` overrides).
- `POST /api/v1/predict/` (VIEWER+): one data point.
- `POST /api/v1/predict/batch` (`{ model_id, dataset_id, offset, limit }`, VIEWER+): predicts every row and returns one page of rows, class counts or prediction statistics, and accuracy (classification) or MAE / RMSE / R² (regression) when the dataset has a target column.
- `GET /api/v1/predict/batch/export.csv?model_id=&dataset_id=`: the dataset with predictions and per-class probabilities appended.
- `POST /api/v1/explain/instance` (EXPLAINER+): stores and returns the explanation with its `explanation_id`.
- `GET /api/v1/explain/global/{model_id}?max_samples=&class_index=`: importance plus beeswarm / dependence data (N ≤ 300), computed on the model's background data.
- `POST /api/v1/explain/chat` (`{ explanation_id, question? }`): 4-part narrative when `question` is omitted, otherwise a follow-up with history.
- `POST /api/v1/explain/chat/stream`: same, as Server-Sent Events (`token`, `done`, `error`).
- `GET /api/v1/explain/{explanation_id}` · `GET /api/v1/explain/{explanation_id}/conversation`

### Audit (ADMIN / OWNER)
- `GET /api/v1/audit/models/{id}/explanations` · `GET /api/v1/audit/models/{id}/metrics` · `GET /api/v1/audit/models/{id}/export.csv`

---

## Background Data vs Evaluation Data

| | Background data | Evaluation data |
| :--- | :--- | :--- |
| Purpose | SHAP reference distribution (E[f(x)], Kernel k-means summary, interventional trees) and Global Insights | Rows to predict and explain |
| Loaded by | **Model owner only**: at registration, or replaced later on the Models page | Any user with access to the model, from the Predict page |
| Predictions | **Never.** `/predict`, `/predict/batch` and `/explain/instance` reject it | Single rows, batch, export |
| Lifecycle | One per model; replacing it deletes the old file; deleted with the model | Kept (unlinked) when the model is deleted |

## Data Serialization & Grounding Contract

The LLM receives a payload built from the stored explanation record:

```json
{
  "domain": "Clinical Cardiology",
  "task_type": "classification",
  "output_units": "probability",
  "units_note": "Baseline, contributions and final output are probabilities between 0 and 1 ...",
  "baseline_expectation": 0.354,
  "explained_output": 0.842,
  "top_positive_factors": [
    { "feature": "st_depression", "value": 2.3, "shap": 0.284 },
    { "feature": "max_heart_rate", "value": 112, "shap": 0.156 }
  ],
  "top_negative_factors": [{ "feature": "age", "value": 58, "shap": -0.02 }],
  "other_features_net_shap": 0.068,
  "predicted_class": "1",
  "explained_class": "1",
  "prediction_probability": 0.842,
  "class_probabilities": { "0": 0.158, "1": 0.842 }
}
```

The narrative uses four fixed headings: **Executive Summary**, **Upward Pressures**, **Mitigating Factors**, **Actionable Next Step**. Follow-up questions replay the full conversation behind the same context, so every answer is bounded by the same payload.

---

## Security Model & Threat Assumptions

- **Pickle artifacts (`.joblib`, `.pkl`) execute code when loaded.** Inspection runs in a disposable child process with a timeout, which isolates crashes and hangs but not OS privileges. Treat model owners as trusted, or set `ALLOW_PICKLE_MODELS=false` to accept only `.skops` files. `.skops` loading rejects types outside scikit-learn, NumPy, SciPy, XGBoost and LightGBM.
- **Zero egress:** prompts go only to a loopback Ollama unless `ALLOW_REMOTE_LLM=true`; the server logs a warning at startup when a remote host is in use.
- Passwords are hashed with bcrypt (work factor 12). JWTs are HS256-signed, and `SECRET_KEY` is required and validated.
- Uploaded files are size-limited, stored under per-user directories with sanitized names, and every file access is checked to resolve inside `STORAGE_DIR`.

---

## Verification & Testing

```bash
cd backend
python -m pytest
```

The suite (74 tests) covers:
- SHAP additivity for random forest, gradient boosting, logistic regression (Kernel) and regressor models, unit detection and global output shapes;
- prompt building, the faithfulness checker and Ollama streaming via a mocked transport;
- auth, ingestion limits, dataset validation, model + background registration (including rollback), owner-only background replacement, the ban on predicting background data, batch prediction metrics and export, the RBAC matrix, conversation privacy, deletion cascades, audit metrics, the path sandbox and config validation.

Tests use a temporary database and storage directory and never contact Ollama.

---

## Evaluation Support

- **Per-request telemetry:** SHAP time, LLM time-to-first-token and total latency are stored for every explanation and answer.
- **Faithfulness:** the rate of answers with zero unverified numbers, the share of verified numbers, 4-part structure compliance and key features cited are reported at `GET /audit/models/{id}/metrics` and in the audit panel.
- **Export:** CSV export gives one row per LLM answer for the SHAP-only vs SHAP + LLM study.

---

## Deviations from SRS v1.0.1 / SDD v1.0.0

| Item | Documents | Implementation | Reason |
| :--- | :--- | :--- | :--- |
| Ports | UI `:3000`, API `:8000` | UI `:3001`, API `:8001` (configurable) | Port conflicts on the development workstation |
| Model upload route | `/api/v1/models/upload` | same | – |
| Chat route | `/api/v1/explain/chat`, body `{model_id, instance_context, question}` | `/api/v1/explain/chat` (+ `/stream`), body `{explanation_id, question}` | Grounding on server-stored SHAP values only |
| Explain request field | `input_data` / `instance_features` | `instance_features`, or `dataset_id` + `row_index` | Row provenance in the audit trail |
| Model formats | `.joblib`, `.pkl` (proposal also `.h5`) | `.skops`, `.joblib`, `.pkl`; `.h5` out of scope | Safer non-pickle format; Keras is outside the tabular scope |
| Extra tables | users, model_artifacts, model_shares | + datasets, explanation_records, chat_messages | Background data, audit trail, multi-turn chat |
| LLM host | localhost:11434 | localhost by default; remote only with `ALLOW_REMOTE_LLM` | Interim remote GPU host |

---

## Roadmap & Milestone Progression

- [x] **Milestone 1:** Requirements Analysis & System Design (SRS v1.0.1 & SDD v1.0.0).
- [x] **Milestone 2:** User Authentication (JWT) & Discretionary Access Control (grant, list, revoke).
- [x] **Milestone 3:** Model & Dataset Ingestion, Estimator Introspection, Path Sandboxing, Subprocess Inspection.
- [x] **Milestone 4:** SHAP Engine (`TreeExplainer` & `KernelExplainer` with reference background, unit detection, global beeswarm).
- [x] **Milestone 5:** Grounded Gemma Prompt Pipeline, Multi-Turn Follow-Up Q&A, SSE Streaming, Faithfulness Checking.
- [x] **Milestone 6:** React + Vite Dashboard & Plotly Visualizations (waterfall, force, bar, beeswarm, dependence).
- [ ] **Milestone 7:** Usability Evaluation (SHAP-only vs SHAP + LLM study) & Statutory Transparency Audit. Tooling is in place (metrics, audit trail, CSV export); the user study is pending.

---

## License

This project is developed for academic research and software engineering evaluation at the Graduate School of Hood College. Distributed under the MIT License.
