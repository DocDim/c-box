# cbox: Hybrid SHAP + LLM Predictive Model Explainability Platform

A privacy-first Explainable AI (XAI) web platform that combines quantitative feature attributions from **SHAP (SHapley Additive exPlanations)** with grounded, plain-language explanations written by a local **Gemma** model running on **Ollama**.

It has an asynchronous **FastAPI** backend and a **React + Vite + TypeScript** frontend. It is designed for **zero cloud data egress** (GDPR Right to Explanation, FDA SaMD transparency) in sensitive tabular domains such as clinical healthcare, loan underwriting and policy analytics.

---

## Table of Contents

- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Role-Based Access Control (RBAC)](#role-based-access-control-rbac)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation & Local Setup](#installation--local-setup)
- [Configuration Reference](#configuration-reference)
- [Using the Application](#using-the-application)
- [Audience-Aware Prompt Layer](#audience-aware-prompt-layer)
- [API Reference](#api-reference)
- [Background Data vs Evaluation Data](#background-data-vs-evaluation-data)
- [Data Serialization & Grounding Contract](#data-serialization--grounding-contract)
- [Frontend Theming & Branding](#frontend-theming--branding)
- [Security Model & Threat Assumptions](#security-model--threat-assumptions)
- [Verification & Testing](#verification--testing)
- [Evaluation Support](#evaluation-support)
- [Deviations from SRS v1.0.1 / SDD v1.0.0](#deviations-from-srs-v101--sdd-v100)
- [Roadmap & Milestone Progression](#roadmap--milestone-progression)
- [License](#license)

---

## Key Features

1. **Two layers of explanation:**
   - **Quantitative:** an explainer factory picks `shap.TreeExplainer` for tree ensembles. It explains classifiers directly in probability space, interventionally, against the model's background dataset. Other estimators fall back to `shap.KernelExplainer` over a k-means summary of that dataset.
   - **Consistent units:** every explanation reports `output_units` (probability, log-odds, target units or raw margin) and the additivity gap `|E[f(x)] + Σφ − f(x)|`, so baseline, attributions and prediction are always on the same scale.
   - **Qualitative:** the top-K positive and negative SHAP contributions are serialized into a strict JSON context and sent to a local Gemma model through Ollama.
2. **Audience-aware narratives:** a prompt-engineering layer rewrites the explanation for a **decision-maker**, a **domain analyst** or a **data scientist**. Each audience gets its own persona, vocabulary, depth and section headings, and the prompts draw on business metadata the model owner supplies: a feature glossary, the outcome and class meanings, and the decision the model supports. See [Audience-Aware Prompt Layer](#audience-aware-prompt-layer).
3. **Server-side grounding:** the LLM only sees SHAP values the server computed and stored. Chat requests reference an `explanation_id`, and attribution values are never accepted from the client.
4. **Multi-turn follow-up Q&A** with full conversation history per audience, streamed token by token over Server-Sent Events.
5. **Automated faithfulness check:** every LLM answer is scanned. Each number must be a rounding of a value in the SHAP payload or the owner's metadata, narratives must contain the audience's four sections, and factors must appear in the section that matches the sign of their SHAP value.
6. **Zero cloud egress by default:** a non-loopback `OLLAMA_BASE_URL` is refused unless `ALLOW_REMOTE_LLM=true` is set explicitly.
7. **Multi-model ownership and discretionary access control (DAC):** owners grant, update, list and revoke `VIEWER`, `EXPLAINER` and `ADMIN` access by email, and can delete their models.
8. **Sandboxed storage and ingestion:** uploads are size-limited and quarantined under `storage/{models|datasets}/{user_id}/`, every path is checked with `Path.is_relative_to`, and model files are deserialized in a disposable child process with a timeout. `.skops` (non-pickle) models are supported.
9. **Prediction workflows:** single-row prediction from a form (with what-if edits), and batch prediction over an evaluation dataset with accuracy or regression metrics and CSV export.
10. **Interactive visualizations:** Plotly waterfall, force plot, global importance bar, beeswarm and dependence plots rendered from JSON, readable in both light and dark themes.
11. **Audit trail and evaluation metrics:** every explanation and LLM answer is recorded with its latency, audience and faithfulness result, with aggregate metrics and CSV export for the evaluation report.
12. **Branded, themeable UI:** the cbox logo, a three-colour brand palette and a light/dark switch that remembers each user's choice.

---

## System Architecture

```
+-----------------------------------------------------------------------------------------+
|                                PRESENTATION LAYER (CLIENT)                              |
| React 18 + TypeScript + Vite (:3001)                                                    |
|  * Auth (JWT)  * Model registration + business context  * Single / batch prediction    |
|  * Plotly waterfall / force / beeswarm / dependence  * Audience-aware streaming chat    |
|  * Sharing panel  * Audit & evaluation metrics  * Light / dark theme                   |
+--------------------------------------------+--------------------------------------------+
                                             | HTTP / REST + SSE (Vite dev proxy -> :8001)
                                             v
+-----------------------------------------------------------------------------------------+
|                             APPLICATION & API LAYER (BACKEND)                           |
| FastAPI + Uvicorn async runtime (:8001)                                                 |
|  Auth & RBAC gate | Model & dataset ingest | Inference | Explain / Chat | Audit         |
|                   Prompt-engineering layer (audience profiles + model metadata)         |
+--------------------------------------------+--------------------------------------------+
               +-----------------------------+-----------------------------+
               v                             v                             v
+---------------------------+  +--------------------------+  +----------------------------+
|     PERSISTENCE TIER      |  |   COMPUTE WORKER TIER    |  |     LLM RUNTIME TIER       |
| SQLite (aiosqlite)        |  | Thread-pool workers      |  | Local Ollama (:11434)      |
|  * users                  |  |  * sandboxed inspection  |  |  * Gemma                   |
|  * model_artifacts        |  |    subprocess            |  |  * Grounded JSON prompts   |
|    (+ business metadata)  |  |  * LRU explainer cache   |  |  * SSE token streaming     |
|  * model_shares (DAC)     |  |  * shap.TreeExplainer    |  |                            |
|  * datasets               |  |  * shap.KernelExplainer  |  |                            |
|  * explanation_records    |  |                          |  |                            |
|  * chat_messages          |  |                          |  |                            |
|    (+ audience)           |  |                          |  |                            |
+---------------------------+  +--------------------------+  +----------------------------+
```

CPU-bound work (deserialization, SHAP, pandas) runs in a thread pool so the event loop stays responsive. Loaded estimators and explainers are cached per model; the cache key includes the file's modification time and the background dataset id.

**Request flow for an explanation:**

1. The client asks for an explanation of a form input or a dataset row (`POST /explain/instance`).
2. The server computes SHAP values, stores them as an `ExplanationRecord` and returns its `explanation_id`.
3. The client opens the chat for that id and an audience (`POST /explain/chat/stream`).
4. The prompt layer combines the stored record, the model's business metadata and the audience profile into a system prompt, a JSON context and an instruction.
5. Gemma streams the answer. The faithfulness checker scores it, and the answer is stored with its audience and telemetry.

---

## Role-Based Access Control (RBAC)

| Role | View data & predict | Run SHAP & LLM | Manage sharing & audit | Grant / revoke ADMIN | Edit business context, background data, delete model |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`VIEWER`** | :white_check_mark: | :x: | :x: | :x: | :x: |
| **`EXPLAINER`** | :white_check_mark: | :white_check_mark: | :x: | :x: | :x: |
| **`ADMIN`** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: | :x: |
| **`OWNER`** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |

Conversations are private to the user who ran the explanation. ADMIN and OWNER see all of them through the audit trail.

These roles control **what a user may do**. The narrative **audience** (decision-maker, analyst, data scientist) is a separate, per-chat reading preference that any EXPLAINER+ user can switch freely.

---

## Tech Stack

- **Backend:** Python 3.11+, FastAPI, Uvicorn, Pydantic v2, Pydantic-Settings
- **Machine learning and XAI:** `shap` (v0.45+), `scikit-learn`, `joblib`, `skops`, `pandas`, `numpy`
- **LLM integration:** Ollama (local daemon), Gemma, `httpx` (async, streaming)
- **Database and security:** SQLite, SQLAlchemy 2.0 (AsyncIO with `aiosqlite`), `pwdlib` (bcrypt, work factor 12), PyJWT (HS256)
- **Frontend:** React 18, Vite 5, TypeScript, Tailwind CSS 3 (CSS-variable theme), Plotly.js, Lucide React, Axios

---

## Repository Structure

```text
c-box/
├── .env.example                     # Environment template (copy to .env)
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
│   │   ├── config.py                # Settings, secret and zero-egress validation
│   │   ├── database.py              # Async engine, FK pragma, lightweight in-place migrations
│   │   │
│   │   ├── api/v1/
│   │   │   ├── router.py            # Mounts every endpoint under /api/v1
│   │   │   └── endpoints/
│   │   │       ├── auth.py          # Register, login, me
│   │   │       ├── models.py        # Upload, list, get, delete, sharing, background data, business metadata
│   │   │       ├── datasets.py      # Upload + validation, list, paginated rows, delete
│   │   │       ├── predict.py       # Single and batch inference, batch CSV export (VIEWER+)
│   │   │       ├── explain.py       # Local / global SHAP, audience-aware grounded chat (JSON + SSE)
│   │   │       └── audit.py         # Audit trail, evaluation metrics, CSV export
│   │   │
│   │   ├── core/
│   │   │   ├── security.py          # bcrypt hashing and JWT
│   │   │   ├── dependencies.py      # OAuth2 bearer guard
│   │   │   ├── permissions.py       # RBAC / DAC gates for models and datasets
│   │   │   ├── storage.py           # Path sandbox, filename sanitizing, size-limited uploads
│   │   │   ├── model_loader.py      # Extension policy, subprocess inspection
│   │   │   ├── _inspect_worker.py   # Child-process loader (joblib / skops)
│   │   │   ├── dataset_ingest.py    # CSV ingestion shared by model registration and dataset upload
│   │   │   ├── feature_alignment.py # Column validation, target detection, reordering
│   │   │   ├── shap_engine.py       # Explainer factory, unit detection, global beeswarm
│   │   │   ├── explainer_cache.py   # LRU cache of estimators / explainers
│   │   │   ├── instances.py         # Resolves manual inputs / dataset rows, cached engines
│   │   │   ├── inference.py         # Shared prediction helper
│   │   │   ├── prompt_builder.py    # Prompt-engineering layer: audience profiles + model metadata
│   │   │   ├── llm_service.py       # Grounding payload, multi-turn messages, Ollama streaming
│   │   │   ├── faithfulness.py      # Automated grounding, structure and direction check
│   │   │   └── maintenance.py       # Startup backfill of model metadata
│   │   │
│   │   ├── models/                  # ORM: user, artifact (+shares), dataset, explanation (+chat)
│   │   ├── schemas/                 # Pydantic request / response contracts
│   │   └── storage/                 # Sandboxed models/ and datasets/ (created at runtime)
│   │
│   └── tests/                       # SHAP engine, LLM / prompt layer, API, RBAC and security tests
│
├── frontend/
│   ├── index.html                   # Favicon, no-flash theme script
│   ├── tailwind.config.js           # Brand + neutral palettes mapped to CSS variables
│   ├── vite.config.ts               # Dev server (:3001) and /api proxy (VITE_API_TARGET)
│   ├── public/
│   │   ├── logo.png                 # cbox logo (navbar, login page)
│   │   └── favicon.png              # Cube-only icon cropped from the logo
│   └── src/
│       ├── App.tsx                  # Session, active model, page routing, explanation state
│       ├── main.tsx
│       ├── index.css                # Light / dark palette variables
│       ├── pages/
│       │   ├── LoginPage.tsx        # Sign in / register
│       │   ├── ModelsPage.tsx       # Register, list, activate, manage models
│       │   └── PredictPage.tsx      # Single / batch prediction
│       ├── components/
│       │   ├── common/              # Navbar, ThemeToggle
│       │   ├── layout/              # Sidebar (navigation + model metadata), ChatSidebar
│       │   ├── models/              # RegisterModelForm, BackgroundDataCard, SharingPanel, ModelMetadataPanel
│       │   ├── predict/             # SinglePrediction (form + what-if), BatchPrediction
│       │   ├── visualizations/      # Waterfall, force, summary (bar / beeswarm / dependence)
│       │   ├── chat/                # Audience-aware streaming chat with faithfulness badges
│       │   └── audit/               # Audit trail and evaluation metrics
│       ├── hooks/
│       │   ├── useHashRoute.ts      # Minimal hash router (#/models, #/predict, ...)
│       │   └── useTheme.ts          # Light / dark theme store + chart colours
│       ├── services/api.ts          # Axios client + SSE stream reader
│       ├── types/api.ts             # API types
│       └── utils/                   # Formatting helpers, minimal markdown renderer
│
└── sample_data/
    ├── models/test_rf.joblib        # Sample Iris random-forest classifier
    └── datasets/                    # iris_1000_samples.csv (background), iris_eval.csv (evaluation), sample.py (generator)
```

---

## Prerequisites

1. **Python 3.11+**
2. **Node.js 18+** and **npm**
3. **Ollama** ([Download Ollama](https://ollama.com/download))

---

## Installation & Local Setup

### 1. Clone the repository and set up the environment

```bash
git clone https://github.com/DocDim/c-box.git
cd c-box
cp .env.example .env
```

Generate a secret key and put it in `.env`. The backend refuses to start with the placeholder or with a key shorter than 32 characters.

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

Values in `.env` must use `KEY=value` syntax. All settings are listed under [Configuration Reference](#configuration-reference).

### 2. Set up the Ollama local LLM

```bash
ollama serve
ollama pull gemma:2b        # larger variants (e.g. gemma2:9b) follow the audience headings and word limits more reliably
curl http://localhost:11434/api/tags
```

### 3. Set up the backend (FastAPI)

```bash
cd backend
python -m venv venv
# Linux / macOS: source venv/bin/activate
# Windows PowerShell: .\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python run.py
```

The backend starts at `http://127.0.0.1:8001`, with OpenAPI docs at **`http://127.0.0.1:8001/docs`**.

On startup, existing SQLite databases are upgraded in place: missing columns are added with `ALTER TABLE` (see `_ADDED_COLUMNS` in [database.py](backend/app/database.py)). Models uploaded before v1.1 are re-inspected so their task type and feature metadata are recorded. Restart the backend after pulling changes that add columns.

### 4. Set up the frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

The client runs at `http://localhost:3001`. Calls to `/api/v1` are proxied to `http://127.0.0.1:8001`; set `VITE_API_TARGET` to point the proxy elsewhere.

For a production build, run `npm run build` (type-checks, then bundles into `frontend/dist/`) and `npm run preview` to serve it locally.

> If a change to `tailwind.config.js` or `index.css` doesn't show up, restart `npm run dev` and hard-refresh the browser (Ctrl+Shift+R).

### 5. Try it with the sample data

1. Register an account on the login page.
2. On **Models**, register `sample_data/models/test_rf.joblib` with `sample_data/datasets/iris_1000_samples.csv` as its background dataset.
3. Optionally, open **Manage → Business Context for AI Narratives** and fill in the feature glossary, class meanings and decision context.
4. On **Predict → Batch (dataset)**, load `sample_data/datasets/iris_eval.csv`, run the batch and click **Explain** on a row.
5. Read the narrative in the right-hand chat and switch **Written for** between audiences.

---

## Configuration Reference

All settings live in [backend/app/config.py](backend/app/config.py) and can be overridden in `.env`.

| Setting | Default | Purpose |
| :--- | :--- | :--- |
| `SECRET_KEY` | *(required)* | JWT signing key, at least 32 characters |
| `ALGORITHM` / `ACCESS_TOKEN_EXPIRE_MINUTES` | `HS256` / `1440` | JWT algorithm and lifetime |
| `BCRYPT_ROUNDS` | `12` | Password hashing work factor |
| `DATABASE_URL` | `sqlite+aiosqlite:///./xai_app.db` | Database location |
| `BACKEND_HOST` / `BACKEND_PORT` | `127.0.0.1` / `8001` | Uvicorn bind address |
| `CORS_ORIGINS` | `http://localhost:3001`, `http://127.0.0.1:3001` | Allowed browser origins |
| `STORAGE_DIR` | `backend/app/storage` | Root of the upload sandbox |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | `http://localhost:11434` / `gemma:2b` | LLM endpoint and model |
| `OLLAMA_TIMEOUT_SECONDS` | `120` | LLM request timeout |
| `ALLOW_REMOTE_LLM` | `false` | Permit a non-loopback Ollama host |
| `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` | `0.2` / `768` | Generation settings |
| `LLM_TOP_K_FEATURES` | `5` | Upward and downward factors sent to the LLM |
| `MAX_MODEL_UPLOAD_MB` / `MAX_DATASET_UPLOAD_MB` | `250` / `100` | Upload size limits |
| `ALLOW_PICKLE_MODELS` | `true` | Accept `.joblib` / `.pkl` (set `false` for `.skops` only) |
| `MODEL_INSPECTION_TIMEOUT_SECONDS` | `60` | Child-process inspection timeout |
| `SHAP_BACKGROUND_MAX_ROWS` / `SHAP_KMEANS_K` | `100` / `30` | Background sampling and Kernel summary size |
| `SHAP_GLOBAL_MAX_SAMPLES` | `300` | Upper bound for global explanations |
| `SHAP_ADDITIVITY_TOLERANCE` | `1e-3` | Accepted additivity gap |
| `EXPLAINER_CACHE_SIZE` | `8` | Number of cached model explainers |

---

## Using the Application

**Layout.** The top bar shows the logo, the signed-in user, the light/dark switch and Logout. The left column holds navigation plus the active model's metadata (role, estimator, task, classes, features, background dataset). The centre shows the current page, and the right column holds the grounded chat for the current explanation.

| Page | Purpose |
| :--- | :--- |
| `#/models` | Register a model together with its background data; list, set active, predict, share and delete. Under **Manage**, the owner replaces the background data, manages sharing and edits the **business context** used by the AI narratives |
| `#/predict` | Choose the model, then predict a **single data point** from a form, or run a **batch** prediction on an evaluation dataset loaded on the page. Explain any result with SHAP |
| `#/insights` | Global model behaviour from SHAP over the background data: importance, beeswarm and dependence (EXPLAINER+) |
| `#/audit` | Audit trail, evaluation metrics and CSV export (ADMIN / OWNER) |

**Chat panel.** The **Written for** selector switches between *Decision-maker*, *Analyst* and *Data scientist*. Each audience keeps its own narrative and follow-up thread for the explanation. A narrative is generated the first time an audience is opened and reused afterwards. The last audience chosen is remembered in the browser. Every answer carries a faithfulness badge (verified numbers, factor directions, sections found).

---

## Audience-Aware Prompt Layer

[prompt_builder.py](backend/app/core/prompt_builder.py) builds every prompt from two inputs.

**1. The audience profile.** Each profile defines the reader persona, writing-style rules, four section headings with per-section instructions, and word limits.

| Audience | Headings | Style |
| :--- | :--- | :--- |
| **Decision-maker** (default) | Bottom Line · What Is Driving This Result · What Is Holding It Back · Recommended Action | Business names only, no SHAP or log-odds jargon, factor weight as strong / moderate / slight, about 220 words, one concrete action taken from the decision context |
| **Domain analyst** | Prediction Overview · Key Contributing Factors · Counterbalancing Factors · Points to Verify | Business name plus column name, actual values and effect sizes, domain interpretation clearly marked, about 320 words |
| **Data scientist** | Model Output · Positive Contributions · Negative Contributions · Diagnostics & Caveats | Exact column names and SHAP values with units, baseline relation, additivity gap, explainer type, SHAP limits, about 300 words |

**2. The model's business metadata.** The owner edits it under *Models → Manage → Business Context for AI Narratives*. All fields are optional.

| Field | Example | Used for |
| :--- | :--- | :--- |
| Feature glossary | `st_depression` → "ST segment depression on ECG" | `label` on every factor; decision-makers never see raw column names |
| Outcome description | "Risk that the patient has heart disease" | Framing the result |
| Class meanings | `1` → "high cardiac risk" | The prediction sentence, e.g. "class 1 (high cardiac risk)" |
| Decision context | "Refer to a cardiologist, or schedule routine follow-up" | The recommended action and points to verify |

All three audiences share the same grounding rules: use only the JSON context, quote only numbers it contains, keep SHAP signs, make no causal claims, and say so when the context can't answer. The metadata is wording context, never evidence about the case. Each section carries a semantic key (`executive_summary`, `upward_pressures`, `mitigating_factors`, `actionable_next_step`), so the faithfulness checker verifies structure and factor directions whatever the headings are.

To change a persona, its headings or its word limits, edit `PROFILES` in `prompt_builder.py`. To add an audience, add an `Audience` value and a profile, then add it to `AUDIENCES` in [ChatPanel.tsx](frontend/src/components/chat/ChatPanel.tsx).

---

## API Reference

All protected endpoints require an `Authorization: Bearer <JWT>` header. Interactive docs are at `/docs`.

### Authentication
- `POST /api/v1/auth/register` (`{ email, password }`; password 8 to 72 characters)
- `POST /api/v1/auth/login` (OAuth2 form `username`, `password`; returns a JWT)
- `GET /api/v1/auth/me`

### Models & Sharing
- `POST /api/v1/models/upload`: multipart `file` (`.skops`, `.joblib`, `.pkl`; max 250 MB), `name`, `domain_context`, optional `background_file` (CSV) and `target_column`. The background CSV is validated against the model's features and registered as its SHAP reference in the same transaction; if it is rejected, nothing is saved. Returns the detected estimator, task type, feature names and classes.
- `GET /api/v1/models/` · `GET /api/v1/models/{id}` · `DELETE /api/v1/models/{id}` (owner)
- `POST /api/v1/models/{id}/background` (multipart `file`, optional `target_column`; **owner only**): uploads or replaces the background data. The previous background dataset is deleted.
- `PUT /api/v1/models/{id}/metadata` (**owner only**): replaces the business context.
  ```json
  {
    "feature_glossary": { "st_depression": "ST segment depression on ECG" },
    "outcome_description": "Risk that the patient has heart disease",
    "class_descriptions": { "1": "high cardiac risk" },
    "decision_context": "Refer to a cardiologist, or schedule routine follow-up"
  }
  ```
  Blank entries are dropped. Feature and class keys must exist on the model, otherwise the request returns 422.
- `GET /api/v1/models/{id}/shares` · `POST /api/v1/models/{id}/share` (`{ user_email, role }`) · `DELETE /api/v1/models/{id}/share/{user_id}` (ADMIN+)

### Datasets
- `POST /api/v1/datasets/upload`: multipart `file`, optional `model_id` and `target_column`. Uploads **evaluation** data to predict on: it validates columns against the model, auto-detects a single extra column as the target, and returns summary statistics and a preview. Any user who can run the model may load one. It never becomes background data.
- `GET /api/v1/datasets/?model_id=&purpose=background|evaluation` · `GET /api/v1/datasets/{id}/rows?offset=&limit=` · `DELETE /api/v1/datasets/{id}`

### Inference & Explainability
Instance requests take either `instance_features`, or `dataset_id` + `row_index` (optionally with `instance_features` overrides for what-if analysis).
- `POST /api/v1/predict/` (VIEWER+): one data point.
- `POST /api/v1/predict/batch` (`{ model_id, dataset_id, offset, limit }`, VIEWER+): predicts every row and returns one page of rows, class counts or prediction statistics, and accuracy (classification) or MAE / RMSE / R² (regression) when the dataset has a target column.
- `GET /api/v1/predict/batch/export.csv?model_id=&dataset_id=`: the dataset with predictions and per-class probabilities appended.
- `POST /api/v1/explain/instance` (EXPLAINER+): stores and returns the explanation with its `explanation_id`.
- `GET /api/v1/explain/global/{model_id}?dataset_id=&max_samples=&class_index=`: importance plus beeswarm / dependence data (N ≤ 300), computed on the model's background data unless another dataset is named.
- `POST /api/v1/explain/chat` (`{ explanation_id, question?, audience? }`): the four-part narrative when `question` is omitted, otherwise a follow-up with that audience's history. `audience` is `decision_maker` (default), `domain_analyst` or `data_scientist`.
- `POST /api/v1/explain/chat/stream`: same, as Server-Sent Events (`token`, `done`, `error`).
- `GET /api/v1/explain/{explanation_id}` · `GET /api/v1/explain/{explanation_id}/conversation?audience=`

### Audit (ADMIN / OWNER)
- `GET /api/v1/audit/models/{id}/explanations` · `GET /api/v1/audit/models/{id}/metrics` · `GET /api/v1/audit/models/{id}/export.csv` (one row per LLM answer, including its audience)

---

## Background Data vs Evaluation Data

| | Background data | Evaluation data |
| :--- | :--- | :--- |
| Purpose | SHAP reference distribution (E[f(x)], Kernel k-means summary, interventional trees) and global insights | Rows to predict and explain |
| Loaded by | **Model owner only**: at registration, or replaced later on the Models page | Any user with access to the model, from the Predict page |
| Predictions | **Never.** `/predict`, `/predict/batch` and `/explain/instance` reject it | Single rows, batch, export |
| Lifecycle | One per model; replacing it deletes the old file; deleted with the model | Kept (unlinked) when the model is deleted |

---

## Data Serialization & Grounding Contract

The LLM receives a payload built from the stored explanation record, the model's metadata and the audience:

```json
{
  "domain": "Clinical Cardiology",
  "audience": "Decision-maker",
  "model_context": {
    "model_name": "Heart RF",
    "outcome_description": "Risk that the patient has heart disease",
    "decision_context": "Refer to a cardiologist, or schedule routine follow-up",
    "class_descriptions": { "1": "high cardiac risk" }
  },
  "task_type": "classification",
  "output_units": "probability",
  "units_note": "Baseline, contributions and final output are probabilities between 0 and 1 ...",
  "prediction_summary": "The model predicts class 1 (high cardiac risk) with probability 84.2%. ...",
  "baseline_expectation": 0.354,
  "explained_output": 0.842,
  "top_positive_factors": [
    { "feature": "st_depression", "label": "ST segment depression on ECG", "value": 2.3,
      "shap": 0.284, "effect": "raises the prediction", "shap_percentage_points": 28.4 }
  ],
  "top_negative_factors": [
    { "feature": "age", "label": "age", "value": 58, "shap": -0.02,
      "effect": "lowers the prediction", "shap_percentage_points": -2.0 }
  ],
  "other_features_net_shap": 0.068,
  "all_feature_names": ["st_depression", "max_heart_rate", "cholesterol", "age"],
  "predicted_class": "1",
  "explained_class": "1",
  "prediction_probability": 0.842,
  "class_probabilities": { "0": 0.158, "1": 0.842 }
}
```

- `prediction_summary` is a ready-made, numerically correct sentence, so small models restate it instead of converting units themselves.
- Each factor carries an explicit `effect`, so small models don't misread signs.
- The data-scientist payload also includes `explainer_type` and `additivity_gap`.

Follow-up questions replay the audience's full conversation behind the same context, so every answer is bounded by the same payload.

---

## Frontend Theming & Branding

- **Brand palette.** Three colours, defined as Tailwind colour scales (`50`–`950`) in [tailwind.config.js](frontend/tailwind.config.js); the `500` shade is the exact brand colour:

  | Token | Colour | Typical use |
  | :--- | :--- | :--- |
  | `primary` | `#009ddd` (blue) | Main actions, active tabs, links, Logout, Predict |
  | `secondary` | `#e48d12` (orange) | Manage, Run batch, background-data uploads, theme icon |
  | `accent` | `#00927e` (green) | Set active, sharing, success states |

  Neutral surfaces use `slate`; errors use `rose`.
- **Light / dark mode.** Every palette shade reads an RGB CSS variable defined in [index.css](frontend/src/index.css), and `<html class="light">` swaps the whole set. Existing utility classes therefore flip without per-component changes. [useTheme.ts](frontend/src/hooks/useTheme.ts) stores the choice in `localStorage`, falling back to the OS preference. A small script in `index.html` applies it before the first paint to avoid a flash. Plotly can't read CSS variables, so charts take their neutral colours from `CHART_COLORS` in the same hook.
- **SHAP colours.** Positive contributions are green, negative contributions blue; the beeswarm feature-value scale runs blue → orange → green.
- **Buttons.** Buttons use white text on a filled background: a brand colour for actions, `gray-600` for neutral controls (fixed in both themes), and `rose` for destructive actions.
- **Logo.** `public/logo.png` is shown unmodified in the navbar and on the login page. `public/favicon.png` is the cube cropped from it.

---

## Security Model & Threat Assumptions

- **Pickle artifacts (`.joblib`, `.pkl`) execute code when loaded.** Inspection runs in a disposable child process with a timeout, which isolates crashes and hangs but not OS privileges. Treat model owners as trusted, or set `ALLOW_PICKLE_MODELS=false` to accept only `.skops` files. `.skops` loading rejects types outside scikit-learn, NumPy, SciPy, XGBoost and LightGBM.
- **Zero egress:** prompts go only to a loopback Ollama unless `ALLOW_REMOTE_LLM=true`; the server logs a warning at startup when a remote host is in use.
- **Credentials:** passwords are hashed with bcrypt (work factor 12). JWTs are HS256-signed, and `SECRET_KEY` is required and validated.
- **Uploads:** files are size-limited and stored under per-user directories with sanitized names, and every file access is checked to resolve inside `STORAGE_DIR`.
- **Grounding:** the chat accepts only an `explanation_id`; SHAP values and context come from the server-side record, never from the client. Business metadata is editable only by the model owner, is length-limited, and is validated against the model's own feature and class names.

---

## Verification & Testing

```bash
cd backend
python -m pytest
```

The suite (89 tests) covers:
- SHAP additivity for random forest, gradient boosting, logistic regression (Kernel) and regressor models, plus unit detection and global output shapes;
- the prompt layer: per-audience personas, headings and word limits, metadata in the payload, data-scientist diagnostics;
- the faithfulness checker: rounding and percentages, sign flips, misplaced factors, audience headings, glossary labels, numbers from owner metadata;
- Ollama streaming and error mapping via a mocked transport;
- the API: auth, ingestion limits, dataset validation, model + background registration (including rollback), owner-only background replacement and metadata editing, the ban on predicting background data, batch prediction metrics and export, per-audience chat threads, the RBAC matrix, conversation privacy, deletion cascades, audit metrics, the path sandbox and config validation.

Tests use a temporary database and storage directory and never contact Ollama.

Frontend checks: `npx tsc --noEmit` (type-check) and `npm run build`.

---

## Evaluation Support

- **Per-request telemetry:** SHAP time, LLM time-to-first-token and total latency are stored for every explanation and answer.
- **Faithfulness:** the audit panel and `GET /audit/models/{id}/metrics` report:
  - the rate of answers with zero unverified numbers;
  - the share of verified numbers;
  - the rate of narratives with all four sections;
  - the direction-error rate;
  - the mean number of key features cited.
- **Export:** the CSV has one row per LLM answer, with its audience, for the SHAP-only vs SHAP + LLM study. It also lets you compare narratives across audiences.

---

## Deviations from SRS v1.0.1 / SDD v1.0.0

| Item | Documents | Implementation | Reason |
| :--- | :--- | :--- | :--- |
| Ports | UI `:3000`, API `:8000` | UI `:3001`, API `:8001` (configurable) | Port conflicts on the development workstation |
| Model upload route | `/api/v1/models/upload` | same | – |
| Chat route | `/api/v1/explain/chat`, body `{model_id, instance_context, question}` | `/api/v1/explain/chat` (+ `/stream`), body `{explanation_id, question, audience}` | Grounding on server-stored SHAP values only; audience-specific narratives |
| Narrative structure | Fixed four headings for a non-technical reader | Four sections per audience, each with its own headings | Decision-makers, analysts and data scientists need different depth and vocabulary |
| Explain request field | `input_data` / `instance_features` | `instance_features`, or `dataset_id` + `row_index` | Row provenance in the audit trail |
| Model formats | `.joblib`, `.pkl` (proposal also `.h5`) | `.skops`, `.joblib`, `.pkl`; `.h5` out of scope | Safer non-pickle format; Keras is outside the tabular scope |
| Extra tables / columns | users, model_artifacts, model_shares | + datasets, explanation_records, chat_messages; business metadata on model_artifacts; audience on chat_messages | Background data, audit trail, multi-turn chat, audience-aware prompts |
| LLM host | localhost:11434 | localhost by default; remote only with `ALLOW_REMOTE_LLM` | Interim remote GPU host |

---

## Roadmap & Milestone Progression

- [x] **Milestone 1:** Requirements analysis and system design (SRS v1.0.1 & SDD v1.0.0).
- [x] **Milestone 2:** User authentication (JWT) and discretionary access control (grant, list, revoke).
- [x] **Milestone 3:** Model and dataset ingestion, estimator introspection, path sandboxing, subprocess inspection.
- [x] **Milestone 4:** SHAP engine (`TreeExplainer` and `KernelExplainer` with reference background, unit detection, global beeswarm).
- [x] **Milestone 5:** Grounded Gemma prompt pipeline, multi-turn follow-up Q&A, SSE streaming, faithfulness checking.
- [x] **Milestone 6:** React + Vite dashboard and Plotly visualizations (waterfall, force, bar, beeswarm, dependence).
- [x] **Milestone 6.5:** Audience-aware prompt-engineering layer with owner-edited business metadata; branded light/dark UI.
- [ ] **Milestone 7:** Usability evaluation (SHAP-only vs SHAP + LLM study, including per-audience comparison) and statutory transparency audit. The tooling is in place (metrics, audit trail, CSV export with audience); the user study is pending.

---

## License

This project is developed for academic research and software engineering evaluation at the Graduate School of Hood College. Distributed under the MIT License.
