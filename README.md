---
title: OC P8 Credit Scoring API
emoji: 💳
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

[![Python][python-badge]][python-url]
[![FastAPI][fastapi-badge]][fastapi-url]
[![CI][ci-badge]][ci-url]
[![uv][uv-badge]][uv-url]
[![License: Internal][license-badge]](#license)

<br />
<div align="center">
  <h1 align="center">OC P8 — Credit Scoring API</h1>
  <p align="center">
    Production-grade FastAPI wrapper around the LightGBM credit scoring model
    trained in OC_P6. Built for <em>Prêt à Dépenser</em>'s Crédit Express department:
    real-time default risk prediction for loan officers.
    <br />
    <a href="http://localhost:8000/docs"><strong>Swagger UI (local) »</strong></a>
    <br />
    <br />
    <a href="#getting-started">Quick Start</a>
    ·
    <a href="#cicd-pipeline">CI/CD</a>
    ·
    <a href="#roadmap">Roadmap</a>
  </p>
</div>

---

## Table of Contents

- [About The Project](#about-the-project)
- [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [One-time offline setup](#one-time-offline-setup)
  - [Run the API](#run-the-api)
  - [Tests](#tests)
- [Usage](#usage)
- [Architecture](#architecture)
- [CI/CD Pipeline](#cicd-pipeline)
  - [Jobs overview](#jobs-overview)
  - [Job: test](#job-test)
  - [Job: deploy](#job-deploy)
  - [Required secrets](#required-secrets)
- [Docker](#docker)
- [Project Layout](#project-layout)
- [Roadmap](#roadmap)
- [License](#license)

---

## About The Project

The **Credit Scoring API** exposes a single `POST /predict` endpoint. Given a loan application (`SK_ID_CURR` + 120 raw `application_train` fields), it returns:

- `probability_default` — model score between 0 and 1
- `decision` — `"REFUSED"` if `proba ≥ 0.33`, `"GRANTED"` otherwise
- `threshold`, `model_version`, `client_known` — explainability metadata

The threshold **0.33** is optimised for an asymmetric cost function (10 × false negatives + false positives), meaning the model is intentionally conservative: missing a bad borrower costs 10× more than wrongly refusing a good one.

---

## Built With

[![Python][python-badge]][python-url]
[![FastAPI][fastapi-badge]][fastapi-url]
[![LightGBM][lightgbm-badge]][lightgbm-url]
[![uv][uv-badge]][uv-url]
[![Docker][docker-badge]][docker-url]
[![GitHub Actions][gha-badge]][gha-url]

---

## Getting Started

### Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.12 | [python.org](https://www.python.org/downloads/) |
| uv | latest | `pip install uv` |
| Docker | any | [docs.docker.com](https://docs.docker.com/get-docker/) |
| OC_P6 data | — | `C:/Users/Kevin/projects/OC_P6/data/` |

### One-time offline setup

Generate all runtime artefacts (feature store parquet + metadata JSONs):

```powershell
uv sync
uv run python scripts/build_feature_store.py
uv run python scripts/build_no_history_template.py
```

This creates:

| Artefact | Size | Description |
|----------|------|-------------|
| `data/features_store.parquet` | ~200 MB | Pre-computed bureau / prev / POS / CC / install aggregates |
| `models/feature_names.json` | ~30 KB | Canonical 768-column order |
| `models/app_train_columns.json` | ~50 KB | Spec for the 122 raw input columns |
| `models/app_train_categories.json` | ~5 KB | Categorical vocabulary for one-hot encoding |
| `models/app_train_binary_mappings.json` | <1 KB | Factorize codes for binary columns |
| `models/no_history_template.json` | ~30 KB | Default values for unknown clients |

### Run the API

```powershell
uv run uvicorn api.main:app --reload
```

| Endpoint | URL |
|----------|-----|
| Swagger UI | http://127.0.0.1:8000/docs |
| Health check | http://127.0.0.1:8000/health |
| Model info | http://127.0.0.1:8000/model/info |

### Tests

```powershell
# All tests + coverage report
uv run pytest --cov=api --cov-report=term-missing

# Unit tests only
uv run pytest tests/unit/

# Integration tests only
uv run pytest tests/integration/

# A specific module
uv run pytest tests/unit/test_ratios.py -v
```

Current coverage: **98%** across 45 tests. Coverage gate enforced at **80%** in CI.

#### Test strategy

All tests run **without the real 5 GB training data**. The `conftest.py` fixture `synthetic_artefacts_dir` generates a complete but minimal artefact set in `tmp_path` at test time:

- A 2-row `features_store.parquet` (clients 100002 and 100003)
- Minimal JSON vocabularies (13 categorical columns, 3 binary columns)
- A `FakeModel` whose probability is deterministically driven by `AMT_INCOME_TOTAL / AMT_CREDIT`, so tests can exercise both `GRANTED` and `REFUSED` branches reliably
- A realistic 122-field `VALID_PAYLOAD` dict reused across all test modules

The `patched_settings` fixture redirects all `api.settings` paths to `tmp_path` via env vars + `importlib.reload()`, so no monkey-patching of internal state is needed.

#### Unit tests

| File | Module tested | What is verified |
|------|--------------|-----------------|
| `test_ratios.py` | `api/ratios.py` | 5 ratio formulas (known values, division-by-zero → NaN, NaN propagation, input immutability) |
| `test_inputs_transform.py` | `api/inputs_transform.py` | One-hot fix (all training categories emitted on a single row), binary factorization, `DAYS_EMPLOYED` sentinel → NaN, unknown category → all-zero |
| `test_predictor.py` | `api/predictor.py` | Threshold loaded from `model_info.json`, fallback to default, GRANTED/REFUSED boundary at exactly `threshold`, 1D and 3D prediction shape handling |
| `test_inference_assembler.py` | `api/inference_assembler.py` | Known client pulls parquet aggregates, unknown client uses template (counts=0, NaN), column order matches `feature_names`, inf values scrubbed to NaN |
| `test_schemas.py` | `api/schemas.py` | Pydantic range guards: negative income rejected, age < 18 and > 70 rejected, unknown contract type rejected, extra fields rejected, optional fields accept `null` |

#### Integration tests

`tests/integration/test_api.py` boots the full FastAPI app via `TestClient` with synthetic artefacts:

| Test | What is verified |
|------|-----------------|
| `test_health_endpoint` | `GET /health` returns 200 with `model_version` |
| `test_swagger_docs_available` | `GET /docs` returns 200 (brief requirement) |
| `test_openapi_schema` | `/openapi.json` exposes the `/predict` path |
| `test_model_info_endpoint` | `GET /model/info` returns threshold, version, n_features |
| `test_predict_known_client` | Full predict flow for a client in the feature store: `client_known=true`, decision in `{GRANTED, REFUSED}`, proba ∈ [0, 1] |
| `test_predict_unknown_client` | Unknown `SK_ID_CURR` → `client_known=false`, no crash |
| `test_predict_rejects_negative_income` | HTTP 422 on invalid input (Pydantic guard) |
| `test_predict_rejects_missing_required_field` | HTTP 422 when `DAYS_BIRTH` is absent |
| `test_predict_rejects_extra_field` | HTTP 422 on field injection attempt |
| `test_predict_rejects_unknown_contract_type` | HTTP 422 on out-of-vocabulary enum value |

---

## Usage

Send a POST request to `/predict` with a JSON body containing `SK_ID_CURR` and the 120 `application_train` fields:

```powershell
curl -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"SK_ID_CURR": 100001, "AMT_INCOME_TOTAL": 202500.0, ...}'
```

Example response:

```json
{
  "sk_id_curr": 100001,
  "probability_default": 0.1523,
  "decision": "GRANTED",
  "threshold": 0.33,
  "model_version": "2",
  "client_known": true
}
```

`decision: "GRANTED"` = loan **granted** · `decision: "REFUSED"` = loan **refused**

You can also use the interactive **Swagger UI** at `/docs` → `POST /predict` → **Try it out**.

---

## Architecture

```
JSON {SK_ID_CURR + 120 raw application_train fields}
        ▼
   Pydantic validation (122 ranged fields)
        ▼
   ┌─────────────────────────┐
   │ Known SK_ID_CURR ?      │
   └────┬────────────────┬───┘
   yes ▼                no ▼
  feature_store     no_history_template
  parquet lookup    (counts=0, NaN)
        │                  │
        └────────┬─────────┘
                 ▼
   transform app_train inputs (factorize + one-hot
   with training categories) + 5 derived ratios
                 ▼
   reindex to feature_names → 768 cols
                 ▼
   model.predict_proba()[:, 1]
                 ▼
   decision = proba ≥ 0.33  (business threshold optimised
                              for 10*FN + FP cost)
                 ▼
   {sk_id_curr, probability_default, decision,
    threshold, model_version, client_known}
```

**Two-case inference flow:**

| Case | Trigger | Aggregate source |
|------|---------|-----------------|
| **Known client** | `SK_ID_CURR` found in `features_store.parquet` | Pre-computed bureau / prev / POS / CC / install |
| **Unknown client** | `SK_ID_CURR` not found | `no_history_template.json` (counts=0, rest NaN) |

The unknown-client path preserves LightGBM's training-time NaN signal ("no historical data") rather than imputing fictitious medians.

### Data layer — code/data separation

The 235 MB `features_store.parquet` is **not bundled** in the Docker image. It lives in a companion HF Dataset repo (`KLEB38/oc-p8-features`) and is fetched at the API's first cold start via `huggingface_hub.hf_hub_download`, then cached on disk. This follows HF's recommended pattern: Spaces hold code, Datasets hold data.

| Layer | Repo | Content |
|-------|------|---------|
| Code + small artefacts | `KLEB38/OC_P8` (Space, Docker) | `api/`, `models/*.json`, `models/model.joblib` |
| Large data | `KLEB38/oc-p8-features` (Dataset) | `features_store.parquet` (235 MB, LFS) |

The local path (`data/features_store.parquet`) takes precedence — the HF download only fires when the file is absent (Space cold start). Configurable via `OC_P8_HF_DATASET_REPO_ID` and `OC_P8_HF_DATASET_FILENAME`.

---

## CI/CD Pipeline

The pipeline is defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) and runs on **GitHub Actions**. It is composed of two sequential jobs.

```
push to main / pull request
        │
        ▼
   ┌─────────┐
   │  test   │  ← runs on every push and PR
   └────┬────┘
        │ success + workflow_dispatch
        ▼
   ┌─────────┐
   │  deploy │  ← manual trigger only
   └─────────┘
```

### Jobs overview

| Job | Trigger | Runner | Purpose |
|-----|---------|--------|---------|
| `test` | Every push / PR to `main` | `ubuntu-latest` | Lint + tests + coverage |
| `deploy` | Manual `workflow_dispatch` (after `test`) | `ubuntu-latest` | Push repo to Hugging Face Space |

---

### Job: test

**Trigger:** every `push` and `pull_request` targeting `main`.

**Steps:**

1. **Checkout** — fetches the full source tree (`actions/checkout@v6`)
2. **Set up Python 3.12** — installs the exact Python version (`actions/setup-python@v6`)
3. **Install uv** — installs the uv package manager (`astral-sh/setup-uv@v8.1.0`)
4. **Install dependencies** — `uv sync --frozen` (respects the lockfile, no version drift)
5. **Lint (ruff)** — `uv run ruff check api feature_engineering tests`
   - Fails fast if any style/quality issue is found
   - Checked directories: `api/`, `feature_engineering/`, `tests/`
6. **Run tests with coverage** — `uv run pytest --cov=api --cov-report=xml --cov-fail-under=80`
   - Minimum coverage gate: **80%** — pipeline fails below this threshold
   - Generates `coverage.xml` for downstream reporting
7. **Upload coverage artifact** — uploads `coverage.xml` with 30-day retention (always runs, even if tests fail)

**Failure policy:** any failing step blocks the `deploy` job downstream.

---

### Job: deploy

**Trigger:** manual via **Actions → Run workflow** (`workflow_dispatch`). Requires `test` to have succeeded.

**Steps:**

1. **Checkout** — full history (`fetch-depth: 0`) so git can push all commits
2. **Push to Hugging Face Space** — force-pushes the `main` branch to the HF Space remote
Hugging Face Spaces automatically rebuilds the Docker container from the `Dockerfile` when the branch is updated.

---

### Required secrets

Configure these in **GitHub → Settings → Secrets and variables → Actions**:

| Secret | Required for | Description |
|--------|-------------|-------------|
| `HF_TOKEN` | `deploy` (GitHub Actions) | Hugging Face write token ([huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)) |
| `HF_TOKEN` | Space runtime (optional) | Only required if the Dataset `KLEB38/oc-p8-features` is **private**. Set as a Space secret in HF → Space settings → Variables and secrets. |

---

## Docker

Build and run locally:

```powershell
docker build -t oc-p8-api .
docker run -p 7860:7860 oc-p8-api
curl http://127.0.0.1:7860/health
```

The Docker image bundles only the code and the small JSON/joblib artefacts under `models/`. The `Dockerfile` fails fast at build time if any of those are missing. The 235 MB `features_store.parquet` is **not** in the image — it is downloaded at startup from the companion HF Dataset (see [Data layer](#data-layer--codedata-separation)).

---

## Project Layout

```
api/                      # Runtime — bundled in Docker image
  main.py                 # FastAPI app + lifespan model loading
  predictor.py            # Model + threshold wrapper
  schemas.py              # Pydantic — 122 hand-crafted fields with ranges
  inputs_transform.py     # Single-row app_train transform (one-hot fix)
  ratios.py               # 5 derived ratio formulas
  inference_assembler.py  # Branch known/unknown + reindex to 768 cols
  settings.py             # Paths resolved from env vars with defaults

feature_engineering/      # Offline ONLY — not imported by the API
  aggregations.py         # 5 aggregation funcs (bureau, prev, POS, CC, install)
  orchestrator.py         # merge_files() — full training dataframe build

scripts/                  # Offline maintenance scripts
  build_feature_store.py
  build_no_history_template.py
  export_model.py         # Imports model.joblib from OC_P6 MLflow registry
  smoke_test_model.py
  check_registry.py
  upload_data_to_hf.py    # One-shot upload of features_store.parquet to HF Dataset

tests/
  conftest.py             # Synthetic fixtures — no real data needed
  unit/                   # Per-module unit tests
  integration/            # FastAPI TestClient end-to-end tests

models/                   # model.joblib + JSON metadata (committed to git)
data/                     # features_store.parquet (gitignored — fetched from HF Dataset at runtime)
.github/workflows/
  ci.yml                  # 2-job CI/CD pipeline (test + manual deploy)
Dockerfile
pyproject.toml
```

---

## Monitoring & Data Drift (Étape 3)

Every `/predict` call is logged to a Supabase PostgreSQL table
(`predictions_log`) for production observability and drift analysis.

### Stack

| Component | Purpose | Location |
|---|---|---|
| Supabase Postgres | Storage for prediction logs | `database/` |
| `api/logger.py` | Synchronous insert on every request, best-effort | `api/` |
| Evidently | Generates a feature-drift HTML report | `scripts/generate_drift_report.py` |
| Streamlit dashboard | Reads Supabase + embeds Evidently HTML | `dashboard/`, deployed at `KLEB38/OC_P8_monitoring` |

### Schema (`predictions_log`)

One row per request. Metadata in proper columns; the 121 raw inputs and
the 768 engineered features are kept in JSONB to absorb PostgreSQL's
63-char identifier limit and stay flexible if the feature pipeline evolves.

```
id (uuid) | timestamp (tz) | sk_id_curr | client_known | latency_ms
status_code | error_message | raw_input (jsonb) | features (jsonb)
probability_default | decision | threshold | model_version
top_shap (jsonb, nullable) | ground_truth (nullable)
```

### Initial setup (once)

```powershell
# .env file: DATABASE_URL=postgresql://...   (gitignored, never commit)
uv run python -m database.setup --create
```

A separate table `predictions_log_test` is created in the same DB for
CI/integration tests — production data is never polluted.

### Generate the drift report

```powershell
# 1. Build the frozen reference (10k stratified rows from training)
uv run python scripts/build_reference_dataset.py --upload

# 2. Compare last 30 days of prod vs reference
uv run python scripts/generate_drift_report.py --days 30
# -> dashboard/static/drift_report.html
```

### Run the dashboard locally

```powershell
$env:DATABASE_URL = "postgresql://..."
cd dashboard && uv run streamlit run app.py
# http://localhost:8501
```

### Deploy the dashboard

```powershell
$env:HF_TOKEN = "hf_..."
uv run python scripts/deploy_dashboard.py
```

`DATABASE_URL` must be set as a Space secret on `KLEB38/OC_P8_monitoring`.
A read-only Supabase role is recommended.

### Interpretation guide

- **Ops tab** — Total volume, error rate, latency p50/p95 by hour,
  `probability_default` histogram split by decision.
- **Drift tab** — embedded Evidently report. Watch the
  *"Share of Drifted Features"* indicator (>30 % typically warrants
  retraining or threshold revision).
- **Business tab** — GRANTED / REFUSED ratio, known / unknown client mix,
  last 50 raw calls.

---

## Roadmap

- [x] **Étape 3** — Supabase logging + Evidently drift + Streamlit dashboard
- [ ] **Étape 4** — Profiling, ONNX export, latency optimisation
- [ ] Top-N SHAP contributors in `PredictionResponse` for loan officer explainability
- [ ] Concept drift via `ground_truth` feedback collection

---

## License

Internal project — Prêt à Dépenser MLOps formation OpenClassrooms.

---

<!-- BADGE LINKS -->
[python-badge]: https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/
[fastapi-badge]: https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white
[fastapi-url]: https://fastapi.tiangolo.com/
[lightgbm-badge]: https://img.shields.io/badge/LightGBM-4.x-2E8B57?style=for-the-badge
[lightgbm-url]: https://lightgbm.readthedocs.io/
[uv-badge]: https://img.shields.io/badge/uv-package%20manager-DE5FE9?style=for-the-badge
[uv-url]: https://docs.astral.sh/uv/
[docker-badge]: https://img.shields.io/badge/Docker-container-2496ED?style=for-the-badge&logo=docker&logoColor=white
[docker-url]: https://www.docker.com/
[gha-badge]: https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF?style=for-the-badge&logo=githubactions&logoColor=white
[gha-url]: https://github.com/features/actions
[ci-badge]: https://img.shields.io/github/actions/workflow/status/KLEB38/OC_P8/ci.yml?branch=main&style=for-the-badge&label=CI
[ci-url]: https://github.com/KLEB38/OC_P8/actions
[license-badge]: https://img.shields.io/badge/license-internal-lightgrey?style=for-the-badge
