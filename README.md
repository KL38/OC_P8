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
    Production-grade FastAPI wrapper around the XGBoost credit scoring model
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
- `decision` — `true` (loan refused) if `proba ≥ 0.33`, `false` (loan granted) otherwise
- `threshold`, `model_version`, `client_known` — explainability metadata

The threshold **0.33** is optimised for an asymmetric cost function (10 × false negatives + false positives), meaning the model is intentionally conservative: missing a bad borrower costs 10× more than wrongly refusing a good one.

---

## Built With

[![Python][python-badge]][python-url]
[![FastAPI][fastapi-badge]][fastapi-url]
[![XGBoost][xgboost-badge]][xgboost-url]
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
  "decision": false,
  "threshold": 0.33,
  "model_version": "xgb-v1.0",
  "client_known": true
}
```

`decision: false` = loan **granted** · `decision: true` = loan **refused**

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

The unknown-client path preserves XGBoost's training-time NaN signal ("no historical data") rather than imputing fictitious medians.

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
| `HF_TOKEN` | `deploy` | Hugging Face write token ([huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)) |

---

## Docker

Build and run locally:

```powershell
docker build -t oc-p8-api .
docker run -p 7860:7860 oc-p8-api
curl http://127.0.0.1:8000/health
```

The `Dockerfile` validates that all runtime artefacts are present at build time — if you forgot to run `build_feature_store.py`, the build fails fast with a clear message.

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

tests/
  conftest.py             # Synthetic fixtures — no real data needed
  unit/                   # Per-module unit tests
  integration/            # FastAPI TestClient end-to-end tests

models/                   # model.joblib + JSON metadata (committed to git)
data/                     # features_store.parquet (gitignored — too large)
.github/workflows/
  ci.yml                  # 3-job CI/CD pipeline
Dockerfile
pyproject.toml
```

---

## Roadmap

- [ ] **Étape 3** — Structured request logging + Evidently AI drift report + Streamlit monitoring dashboard
- [ ] **Étape 4** — Profiling, ONNX export, latency optimisation
- [ ] Top-N SHAP contributors in `PredictionResponse` for loan officer explainability

---

## License

Internal project — Prêt à Dépenser MLOps formation OpenClassrooms.

---

<!-- BADGE LINKS -->
[python-badge]: https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/
[fastapi-badge]: https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white
[fastapi-url]: https://fastapi.tiangolo.com/
[xgboost-badge]: https://img.shields.io/badge/XGBoost-2.x-F7931E?style=for-the-badge
[xgboost-url]: https://xgboost.readthedocs.io/
[uv-badge]: https://img.shields.io/badge/uv-package%20manager-DE5FE9?style=for-the-badge
[uv-url]: https://docs.astral.sh/uv/
[docker-badge]: https://img.shields.io/badge/Docker-container-2496ED?style=for-the-badge&logo=docker&logoColor=white
[docker-url]: https://www.docker.com/
[gha-badge]: https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF?style=for-the-badge&logo=githubactions&logoColor=white
[gha-url]: https://github.com/features/actions
[ci-badge]: https://img.shields.io/github/actions/workflow/status/KLEB38/OC_P8/ci.yml?branch=main&style=for-the-badge&label=CI
[ci-url]: https://github.com/KLEB38/OC_P8/actions
[license-badge]: https://img.shields.io/badge/license-internal-lightgrey?style=for-the-badge
