# Hybrid SHAP + LLM Predictive Model Explainability Platform

A privacy-first, bare-metal Explainable AI (XAI) web platform that combines quantitative feature attributions from **SHAP (SHapley Additive exPlanations)** with grounded qualitative natural language interpretations powered by a local **Gemma** model running on **Ollama**.

Built with an asynchronous **FastAPI** backend and an interactive **React + Vite + TypeScript** frontend, this platform operates with **zero cloud data egress**, ensuring full regulatory compliance (e.g., GDPR Right to Explanation, FDA SaMD guidelines) for sensitive tabular domains like clinical healthcare, loan underwriting, and policy analytics.

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
- [Verification & Testing](#verification--testing)
- [Roadmap & Milestone Progression](#roadmap--milestone-progression)
- [License](#license)

---

## Key Features

1. **Dual-Layer Explainability Framework:**
   - **Quantitative Rigor:** Automated explainer factory selects `shap.TreeExplainer` for high-speed tree ensembles (Random Forest, XGBoost, LightGBM) with fallback to `shap.KernelExplainer` for black-box estimators.
   - **Qualitative Grounding:** Serializes top-$K$ positive and negative SHAP feature contributions into strict JSON context injected into a local Gemma instance via Ollama.
2. **Zero Cloud Egress & Data Sovereignty:**
   - Model binaries, CSV datasets, and feature contributions are processed entirely on-premise on bare metal.
   - LLM queries are dispatched locally to `http://localhost:11434` without third-party API dependencies.
3. **Multi-Model Ownership & Discretionary Access Control (DAC):**
   - Users can upload and manage multiple machine learning models.
   - Owners can grant, update, or revoke access for other registered users by email with granular roles (`VIEWER`, `EXPLAINER`, `ADMIN`).
4. **Sandboxed Filesystem Storage:**
   - Model binaries (`.joblib`, `.pkl`) and datasets (`.csv`) are quarantined in tenant-isolated paths (`storage/models/{owner_id}/`) protected against directory traversal attacks.
5. **Interactive Client Visualizations:**
   - Frontend consumes pure JSON metrics to render dynamic interactive Waterfall, Force, and Global Summary beeswarm charts without relying on static image bitmaps.

---

## System Architecture

```
+-----------------------------------------------------------------------------------------+
|                                PRESENTATION LAYER (CLIENT)                              |
| React 18 + TypeScript + Vite (:3000)                                                    |
|  * Auth Context (JWT State)           * Model/Dataset Upload Form & Schema Inspector    |
|  * Tabular Data Grid & Row Selector   * Interactive Visualizer (Plotly Waterfall/Force) |
|  * Conversational Explanation Stream Card & Guided Follow-Up Chat Box                  |
+--------------------------------------------+--------------------------------------------+
                                             | HTTP / REST (Vite Dev Proxy -> :8000)
                                             v
+-----------------------------------------------------------------------------------------+
|                             APPLICATION & API LAYER (BACKEND)                           |
| FastAPI + Uvicorn Async Runtime (:8000)                                                 |
|  +--------------------+  +--------------------+  +------------------+  +--------------+ |
|  | Auth & Access Ctrl |  | Artifact Ingest    |  | Inference Engine |  | Explain/Chat | |
|  | (JWT, RBAC Gate)   |  | (Storage Scrubber) |  | (Predict Dispatch|  | (SHAP & LLM) | |
|  +---------+----------+  +---------+----------+  +--------+---------+  +-------+------+ |
|            |                       |                      |                    |        |
|            +-----------------------+----------------------+--------------------+        |
+--------------------------------------------+--------------------------------------------+
                                             |
               +-----------------------------+-----------------------------+
               v                                                           v
+---------------------------+  +--------------------------+  +----------------------------+
|     PERSISTENCE TIER      |  |   COMPUTE WORKER TIER    |  |     LLM RUNTIME TIER       |
| SQLite (aiosqlite)        |  | In-Process CPU Workers   |  | Local Ollama (:11434)      |
|  * users (Auth & Roles)   |  |  * joblib Model Loader   |  |  * Gemma 2B / 7B Model     |
|  * model_artifacts        |  |  * shap.TreeExplainer    |  |  * Grounded JSON Prompter  |
|  * model_shares (DAC)     |  |  * shap.KernelExplainer  |  |  * Zero-Egress Inference   |
+---------------------------+  +--------------------------+  +----------------------------+
```

---

## Role-Based Access Control (RBAC)

The system implements a discretionary role hierarchy governing access to model artifacts and inference operations:

| Role | View Metadata & Data Preview | Execute Inference | Run SHAP & LLM Narratives | Manage & Share Access | Delete / Transfer Artifact |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`VIEWER`** | :white_check_mark: | :white_check_mark: | :x: | :x: | :x: |
| **`EXPLAINER`** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: | :x: |
| **`ADMIN`** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: |
| **`OWNER`** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |

---

## Tech Stack

- **Backend:** Python 3.11+, FastAPI, Uvicorn, Pydantic v2, Pydantic-Settings
- **Machine Learning & XAI:** `shap` (v0.45+), `scikit-learn`, `joblib`, `pandas`, `numpy`
- **LLM Integration:** Ollama (Local Daemon), Gemma (2B / 7B parameter variants), `httpx` (async)
- **Database & Security:** SQLite, SQLAlchemy 2.0 (AsyncIO with `aiosqlite`), Passlib (Bcrypt), PyJWT (HS256)
- **Frontend:** React 18, Vite, TypeScript, Tailwind CSS, Plotly.js / Recharts, Lucide React

---

## Repository Structure

```text
xai-shap-llm/
├── .env.example                     # Environment template
├── .env                             # Active environment configuration
├── .gitignore                       # Git exclusion rules
├── README.md                        # Project documentation
│
├── backend/                         # FastAPI core backend service
│   ├── requirements.txt             # Python dependencies
│   ├── run.py                       # Local Uvicorn runner
│   │
│   ├── app/
│   │   ├── main.py                  # App entry point, CORS, and lifespan
│   │   ├── config.py                # Environment configurations & sandboxing
│   │   ├── database.py              # SQLAlchemy async engine & sessionmaker
│   │   │
│   │   ├── api/v1/                  # REST API v1 routing layer
│   │   │   ├── router.py            # Aggregator for v1 endpoints
│   │   │   └── endpoints/
│   │   │       ├── auth.py          # User registration, login, and profile
│   │   │       ├── upload.py        # Model and CSV dataset ingestion
│   │   │       ├── model_access.py  # Model access delegation & listing
│   │   │       ├── predict.py       # Point and batch inference
│   │   │       ├── explain.py       # Local and global SHAP computations
│   │   │       └── chat.py          # Gemma grounded narrative generation
│   │   │
│   │   ├── core/                    # Core business and algorithmic logic
│   │   │   ├── security.py          # Bcrypt hashing and JWT token handlers
│   │   │   ├── dependencies.py      # OAuth2 token authentication guards
│   │   │   ├── permissions.py       # RBAC / DAC access verification gate
│   │   │   ├── model_loader.py      # Safe model loading and inspection
│   │   │   ├── shap_engine.py       # Tree & Kernel explainer engine
│   │   │   └── llm_service.py       # Grounded Ollama prompt engine
│   │   │
│   │   ├── models/                  # SQLAlchemy ORM models
│   │   │   ├── user.py              # User account entity
│   │   │   └── artifact.py          # ModelArtifact and ModelShare entities
│   │   │
│   │   ├── schemas/                 # Pydantic validation schemas
│   │   │   ├── user.py              # Auth and profile contracts
│   │   │   ├── artifact.py          # Sharing and metadata contracts
│   │   │   ├── dataset.py           # Ingestion and preview contracts
│   │   │   ├── explain.py           # SHAP numerical payloads
│   │   │   └── chat.py              # LLM narrative and follow-up contracts
│   │   │
│   │   └── storage/                 # Sandboxed bare-metal storage
│   │       ├── models/              # Quarantined model binaries
│   │       └── datasets/            # Quarantined CSV datasets
│   │
│   └── tests/                       # Unit and integration test suites
│
├── frontend/                        # React + Vite client dashboard
│   ├── index.html                   # HTML entry point
│   ├── package.json                 # Node dependencies
│   ├── tsconfig.json                # TypeScript compiler config
│   ├── vite.config.ts               # Vite configuration & dev proxy
│   └── src/
│       ├── main.tsx                 # React entry point
│       ├── App.tsx                  # Pipeline navigation
│       ├── components/              # Modular UI components
│       ├── services/api.ts          # Axios / Fetch client with JWT interceptor
│       └── types/api.ts             # TypeScript interfaces
│
└── sample_data/                     # Baseline evaluation artifacts
    ├── models/                      # Sample trained models
    └── datasets/                    # Matching evaluation CSVs
```

---

## Prerequisites

1. **Python 3.11+** installed locally.
2. **Node.js (v18+)** and **npm** installed.
3. **Ollama** installed on your workstation ([Download Ollama](https://ollama.com/download)).

---

## Installation & Local Setup

### 1. Clone Repository & Setup Environment

```bash
git clone https://github.com/your-username/xai-shap-llm.git
cd xai-shap-llm

# Copy environment variables
cp .env.example .env
```

Ensure `.env` matches your local configuration:

```env
PROJECT_NAME="SHAP-LLM Explainability Platform"
DEBUG=True
SECRET_KEY="generate-a-secure-random-secret-key-here"
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=1440
DATABASE_URL="sqlite+aiosqlite:///./xai_app.db"
OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_MODEL="gemma"
```

---

### 2. Ollama Local LLM Engine Setup

Start the Ollama daemon and pull the Gemma model:

```bash
# Start Ollama service (if not running as a system service)
ollama serve

# In a separate terminal, pull the model
ollama pull gemma
```

Verify that the local endpoint responds:

```bash
curl http://localhost:11434/api/tags
```

---

### 3. Backend Setup (FastAPI)

Navigate to the `backend/` directory, create a virtual environment, and install requirements:

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux / macOS:
source venv/bin/activate
# On Windows CMD:
venv\Scripts\activate
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Start the application server
python run.py
```

The backend starts at `http://127.0.0.1:8000`.  
Explore interactive OpenAPI documentation at **`http://127.0.0.1:8000/docs`**.

---

### 4. Frontend Setup (React + Vite)

Open a new terminal window, navigate to `frontend/`, and initialize dependencies:

```bash
cd frontend

# Install dependencies
npm install

# Start Vite development server
npm run dev
```

The client will be running at `http://localhost:5173`. Any API calls to `/api/v1` are automatically proxied to `http://127.0.0.1:8000`.

---

## API Reference

All protected endpoints require an `Authorization: Bearer <JWT>` header.

### Authentication Endpoints
- `POST /api/v1/auth/register`: Register new account (`{ email, password }`).
- `POST /api/v1/auth/login`: OAuth2 password form (`username`, `password`); returns Bearer JWT.
- `GET /api/v1/auth/me`: Retrieve profile of currently authenticated user.

### Model & Dataset Ingestion
- `POST /api/v1/upload/model`: Upload `.joblib` / `.pkl` binary, model name, and domain context (`multipart/form-data`).
- `POST /api/v1/upload/dataset`: Upload `.csv` dataset; returns columns, shape, and 5-row schema preview.

### Access Sharing (DAC)
- `GET /api/v1/models/`: List all models owned by the user or shared with them.
- `POST /api/v1/models/{id}/share`: Grant access to another user (`{ user_email, role }`). Requires `OWNER` or `ADMIN` role.

### Inference & Explainability
- `POST /api/v1/predict/{model_id}`: Execute point prediction on a single feature dictionary.
- `POST /api/v1/explain/instance`: Compute local base value, prediction, and per-feature SHAP attributions.
- `POST /api/v1/explain/global`: Compute mean absolute global SHAP values across evaluation samples.
- `POST /api/v1/explain/chat`: Stream or generate structured 4-part natural language narrative via local Gemma.

---

## Data Serialization & Grounding Contract

To prevent LLM hallucination of numerical metrics, SHAP values are extracted and serialized into a deterministic JSON context:

```json
{
  "domain": "Clinical Healthcare - Cardiology",
  "baseline_expectation": 0.354,
  "final_prediction": 0.842,
  "top_positive_factors": [
    { "feature": "st_depression", "actual_value": 2.3, "shap_value": 0.284 },
    { "feature": "max_heart_rate", "actual_value": 112, "shap_value": 0.156 }
  ],
  "top_negative_factors": [
    { "feature": "age", "actual_value": 58, "shap_value": -0.020 }
  ]
}
```

The Ollama Gemma engine is conditioned with a strict system prompt to formulate a structured 4-part explanation:
1. **Executive Summary:** Prediction relative to baseline reference.
2. **Upward Drivers:** Factors increasing the score.
3. **Mitigating Drivers:** Factors reducing the score.
4. **Actionable Recommendations:** Grounded, domain-specific next steps.

---

## Verification & Testing

Run unit tests from the `backend/` directory:

```bash
cd backend
python -m pytest tests/ -v
```

---

## Roadmap & Milestone Progression

- [x] **Milestone 1:** Requirements Analysis & System Design (SRS v1.0.1 & SDD v1.0.0 completed).
- [x] **Milestone 2:** User Authentication (JWT) & Discretionary Access Control (Model sharing).
- [x] **Milestone 3:** Model Ingestion, Estimator Introspection & Path Sandboxing.
- [ ] **Milestone 4:** SHAP Engine Integration (`TreeExplainer` & `KernelExplainer`).
- [ ] **Milestone 5:** Grounded Gemma Prompt Pipeline & Follow-Up Q&A via Ollama.
- [ ] **Milestone 6:** React + Vite Dashboard & Plotly Interactive Visualizations.
- [ ] **Milestone 7:** Usability Evaluation & Statutory Transparency Audit (GDPR/FDA compliance).

---

## License

This project is developed for academic research and software engineering evaluation at the Graduate School of Hood College. Distributed under the MIT License.