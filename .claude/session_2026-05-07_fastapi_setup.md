# Session 2026-05-07 — FastAPI MLOps setup

> Point d'entrée pour reprendre le projet en session future. Capture le plan
> validé, les fichiers livrés, les déviations vs plan, et les prochaines étapes.

## Contexte projet

OC_P8 — déploiement du modèle de scoring crédit (LightGBM, `lgbm_credit_scoring v2`)
entraîné dans OC_P6. Brief : API FastAPI + Docker + CI/CD + (étapes ultérieures :
drift monitoring + optimisation). Le feature engineering vivait dans
`notebooks/EDA.ipynb` et a été industrialisé.

## Plan validé (extrait du plan complet)

Plan complet conservé : `C:\Users\Kevin\.claude\plans\tu-es-un-mlops-gleaming-shamir.md`

### Décisions actées avec l'utilisateur

1. **JSON input = toutes les 122 colonnes brutes d'application_train**
   (pas un top-20). Raison : drift monitoring (étape 3 brief) a besoin
   d'observer l'intégralité de l'espace input.

2. **Deux cas à l'inférence** :
   - **Cas 1 (client connu)** : `SK_ID_CURR` ∈ `features_store.parquet` →
     lookup des ~600 features agrégées pré-calculées.
   - **Cas 2 (client inconnu)** : template "no-history"
     - Counts (`BURO_COUNT`, `INSTAL_COUNT`, `POS_COUNT`, `CC_COUNT`) → 0
     - Autres agrégats (means, sums, etc.) → NaN
     - Justification : LightGBM a été entraîné avec NaN comme **signal**
       informatif ("pas d'historique"). Imputer une médiane crée du
       train/serve skew.
   - **Pas de SimpleImputer** — point clé sur lequel on s'est challengé.

3. **Pydantic schema hand-crafted** avec ranges métier (FLAG ∈ {0,1},
   DAYS_BIRTH ∈ [-25550, -6570] = 18-70 ans, etc.).

4. **Threshold business 0.33** (lu de model_info.json, optimisé pour
   `10*FN + FP`).

5. **Séparation offline / runtime stricte** :
   - `feature_engineering/` (offline) : 5 fonctions d'agrégation lourdes,
     `merge_files()`. Pas importé par `api/`.
   - `api/` (runtime) : juste la transformation app_train + ratios + lookup
     parquet. Image Docker légère.

6. **One-hot pitfall** résolu via `pd.Categorical(values, categories=KNOWN)`
   avant `get_dummies()` — sinon une seule ligne ne génère qu'une colonne
   par catégorielle au lieu de N. `models/app_train_categories.json` stocke
   le vocabulaire exhaustif.

7. **Binary mappings capturés** (`models/app_train_binary_mappings.json`)
   pour reproduire exactement l'ordre `pd.factorize()` du training (ordre de
   première apparition, pas alphabétique).

8. **CI/CD** : GitHub Actions → pytest (gate 80%) → docker build → push
   ghcr.io → deploy manuel HuggingFace Space.

### Hors scope cette session

- Top-N SHAP contributors dans `PredictionResponse` (explicabilité)
- Étape 3 brief : logging structuré + Evidently drift + Streamlit dashboard
- Étape 4 brief : profiling, ONNX, optimisation latence

---

## Ce qui a été livré

### Structure créée

```
OC_P8/
├── api/                              ← runtime (bundled in Docker)
│   ├── __init__.py
│   ├── main.py                       ← FastAPI app + lifespan loading
│   ├── schemas.py                    ← Pydantic 121 champs (SK_ID_CURR + 120)
│   ├── predictor.py                  ← Singleton model + threshold
│   ├── inputs_transform.py           ← Transform app_train (one-hot fix)
│   ├── ratios.py                     ← 5 ratios dérivés
│   ├── inference_assembler.py        ← Branche connu/inconnu + reindex
│   └── settings.py                   ← Paths via env vars
├── feature_engineering/              ← offline ONLY
│   ├── __init__.py
│   ├── aggregations.py               ← 5 fonctions d'agrégation lourdes
│   └── orchestrator.py               ← merge_files() + app_train_clean()
├── scripts/
│   ├── build_feature_store.py        ← NEW — génère parquet + 4 JSONs
│   ├── build_no_history_template.py  ← NEW — template counts=0 / NaN
│   ├── export_model.py               ← existant
│   ├── check_registry.py             ← existant
│   └── smoke_test_model.py           ← existant
├── tests/
│   ├── conftest.py                   ← Fixtures synthétiques
│   ├── unit/
│   │   ├── test_ratios.py            (5 tests)
│   │   ├── test_inputs_transform.py  (6 tests)
│   │   ├── test_inference_assembler.py (6 tests)
│   │   ├── test_predictor.py         (8 tests)
│   │   └── test_schemas.py           (10 tests)
│   └── integration/
│       └── test_api.py               (10 tests, FastAPI TestClient)
├── models/                           ← model.joblib + JSONs (committed)
├── data/                             ← features_store.parquet (gitignored)
├── Dockerfile                        ← NEW
├── .dockerignore                     ← NEW
├── .gitignore                        ← MAJ — exclut parquet, garde JSONs
├── .github/workflows/ci.yml          ← NEW — 3 jobs (test/build/deploy)
├── pyproject.toml                    ← MAJ — fastapi, uvicorn, pyarrow,
│                                       pydantic + dev: pytest, pytest-cov,
│                                       httpx, ruff
└── README.md                         ← NEW — architecture + run instructions
```

### Tests : 45/45 passent, 97.9 % de couverture sur `api/`

```
api/__init__.py                100%
api/inference_assembler.py      95%
api/inputs_transform.py        100%
api/main.py                     89%
api/predictor.py               100%
api/ratios.py                  100%
api/schemas.py                 100%
api/settings.py                100%
TOTAL                           98%  (gate fixé à 80%)
```

### Artefacts à générer (TODO côté utilisateur)

Ces commandes n'ont **pas** été exécutées dans cette session car elles
nécessitent les données OC_P6 (~5 GB) :

```powershell
uv run python scripts/build_feature_store.py
uv run python scripts/build_no_history_template.py
```

Outputs attendus :
- `data/features_store.parquet` (~200 MB)
- `models/feature_names.json` (768 colonnes ordonnées)
- `models/app_train_columns.json` (122 cols meta)
- `models/app_train_categories.json` (vocab catégoriel)
- `models/app_train_binary_mappings.json` (codes factorize)
- `models/no_history_template.json` (template cas 2)

---

## Déviations vs plan

### 1. Naming `feature_pipeline.py` → `inputs_transform.py`
**Plan** : `api/feature_pipeline.py` (8 fonctions)
**Réel** : `api/inputs_transform.py` + `api/ratios.py` séparément
**Raison** : suite à la simplification (les 5 fonctions d'agrégation lourdes
sont sorties de `api/`), le module runtime ne contient plus que la transfo
app_train + les ratios. Plus clair de les séparer.

### 2. Ajout `app_train_binary_mappings.json` (pas dans le plan initial)
**Raison** : découvert pendant l'implémentation que `pd.factorize()` assigne
les codes par **ordre de première apparition** dans les données, pas
alphabétique. Coder en dur `{"M": 0, "F": 1}` était fragile. Le mapping est
maintenant capturé pendant `build_feature_store.py`.

### 3. Schéma Pydantic = 121 champs au lieu de 122
**Plan** : 122 champs.
**Réel** : 121 champs (SK_ID_CURR + 120 features).
**Raison** : TARGET est exclu (c'est la cible à prédire, pas un input).
La colonne reste dans application_train.csv mais ne fait pas partie du JSON
input. Cohérent avec le brief.

### 4. `apply_derived_ratios()` est dans le runtime, pas dans le orchestrator
**Plan** : ratios calculés dans `app_train_clean()` (offline).
**Réel** : duplication maîtrisée — le offline `app_train_clean()` les calcule
(pour le training) ET `api/ratios.py` les recalcule (pour l'inférence après
l'override JSON). C'était nécessaire pour que les ratios reflètent les inputs
JSON et non les valeurs stockées.

### 5. Dockerfile sanity check ajouté
**Plan** : juste COPY + uvicorn.
**Réel** : ajouté un `RUN test -f ...` qui fait échouer le build si les
artefacts JSON/parquet sont manquants. Évite de découvrir le problème au
runtime.

### 6. Coverage 97.9 % au lieu du minimum 80 %
**Plan** : ≥80 %.
**Réel** : dépassé sans effort particulier. Gate CI reste à 80 %.

### Aucune déviation majeure
Le design (deux cas, no-history template, séparation offline/runtime,
one-hot fix, hand-crafted Pydantic) est exactement celui du plan validé.

---

## Prochaines étapes (à reprendre à la prochaine session)

### Immédiat (à faire avant la prochaine session)
1. **Générer les artefacts** :
   ```powershell
   uv run python scripts/build_feature_store.py
   uv run python scripts/build_no_history_template.py
   ```
2. **Smoke test API** :
   ```powershell
   uv run uvicorn api.main:app --reload
   # Puis tester via http://127.0.0.1:8000/docs
   ```
3. **Build Docker en local** :
   ```powershell
   docker build -t oc-p8-api .
   docker run -p 8000:8000 oc-p8-api
   ```
4. **Push GitHub** pour déclencher CI.

### À la prochaine session (Étape 3 brief — drift monitoring)

Décisions à prendre :
- **Stockage des logs prod** : Supabase (PostgreSQL) ? SQLite local ?
  fichiers JSON ?
- **Drift detection** : Evidently AI vs NannyML ?
- **Dashboard** : Streamlit (suggéré dans le brief) ? HuggingFace Space
  séparé ?
- **Logging structuré** : structlog ? loguru ? logging stdlib + JSON
  formatter ?

Travaux à prévoir :
- Middleware FastAPI qui logue chaque requête (timestamp, inputs, output,
  latency) dans le storage choisi
- Script `scripts/build_drift_report.py` qui prend les logs prod + le
  training set comme reference, génère un rapport Evidently HTML
- Dashboard Streamlit qui affiche : distribution des scores, latence p50/p95,
  taux d'erreur, top features driftées

### Plus tard (Étape 4 brief — optimisation)
- Profiling cProfile / line_profiler sur `assemble()` + `predict()`
- Conversion ONNX du modèle si latence > seuil acceptable
- Quantification des poids (LightGBM le supporte mal directement → ONNX
  intermédiaire)
- Comparaison avant/après en métriques + ROC-AUC

### Améliorations API (priorité moyenne)
- **Top-N SHAP contributors** dans `PredictionResponse` :
  `TreeExplainer.shap_values()` sur la prédiction → top-3 features
  contributives → JSON enrichi pour l'agent commercial.
- **GET /clients/{sk_id_curr}** route debug/admin pour vérifier la présence
  d'un client dans le store.
- **Authentification JWT** (pas demandée explicitement par le brief mais
  attendu en prod).

---

## Choix techniques notables (pour future référence)

### Modèle chargé via mlflow.pyfunc puis joblib.dump
- Le modèle exporté de OC_P6 est `MLflow PyFuncModel`
- `predictor._predict_proba()` gère 2 shapes : `(n,)` ou `(n, 2)`
- Si LightGBM natif est ré-extrait plus tard, refactor de `_predict_proba`
  trivial.

### Tests : isolation totale du code prod
- `conftest.py` génère synthetic_artefacts_dir avec mini-parquet, mini-JSONs,
  FakeModel.joblib
- `patched_settings` fixture utilise `monkeypatch` pour pointer les env vars
  vers le tmp_path
- L'app FastAPI est rechargée (`importlib.reload`) à chaque test integration
  pour repartir sur des artefacts neufs.

### Sécurité Pydantic
- `model_config = ConfigDict(extra="forbid")` rejette les champs inconnus
  → couvre l'injection de données
- Tous les `Optional[float]` ont des bornes `Field(ge=, le=)` pour éviter
  des outliers ridicules

### Déploiement HuggingFace
- Le job `deploy` du workflow CI est en `workflow_dispatch` (manuel) pour
  éviter les push accidentels
- Skip gracieux si `HF_TOKEN` / `HF_SPACE` ne sont pas configurés en
  secrets — le workflow ne casse pas, il échoue silencieusement avec un log

---

## Bugs corrigés en route

1. **Test `test_unknown_client_uses_no_history_template`** :
   `np.isnan` ne fonctionnait pas car les `None` JSON restaient en object
   dtype → fix dans `InferenceArtefacts.load` qui convertit None → np.nan.

2. **Test `test_inf_values_replaced_with_nan`** :
   `np.isinf(features.values)` échouait car le DataFrame contient des
   `pd.Int64` (extension type) → fix : test ciblé sur les colonnes ratios.

3. **Ruff warnings** : 4 unused imports / unused f-strings auto-fixés dans
   scripts.

---

## Session du soir 2026-05-07 — refonte du déploiement HF

Après le setup initial du matin, plusieurs corrections sur la chaîne de
déploiement HF Space en se basant sur les retours du runner CI et sur les
docs HF officielles.

### CI/CD pipeline simplifié

- **Suppression du job `build`** (push image GHCR) : l'image Docker est
  buildée directement par HF Space à partir du `Dockerfile`, donc le job
  de build GitHub était redondant.
- Pipeline réduit à 2 jobs : `test` (à chaque push/PR) → `deploy`
  (`workflow_dispatch` manuel).
- `lfs: true` ajouté aux `actions/checkout` (par défaut, checkout ne pull
  pas les blobs LFS — mais en pratique pas de fichier LFS dans le repo
  GitHub pour l'instant).
- Job `deploy` reste sur `needs: test` : trade-off accepté que les tests
  re-tournent au moment du déploiement manuel (sécurité > 30s de runner).

### Switch git push → `huggingface_hub.upload_folder` pour le deploy

- Tentative initiale `git push --force https://...@huggingface.co/...`
  rejetée par HF avec `pre-receive hook declined` car `models/model.joblib`
  > 10 Mo n'était pas en LFS.
- Tentative suivante avec `git lfs migrate import` en CI : a fonctionné
  pour le model.joblib mais pose des problèmes connus (cf OC_P5) liés au
  routage des objets LFS entre serveur GitHub et HF.
- **Solution retenue** : `huggingface_hub.upload_folder` qui passe par
  l'API REST HF et gère le routage LFS automatiquement côté serveur.
  Plus simple, plus robuste, c'est le pattern recommandé par HF.

### Frontmatter HF Space dans le README

HF a refusé le premier build avec `Missing configuration in README`. Ajout
du YAML frontmatter en tête du README :

```yaml
---
title: OC P8 Credit Scoring API
emoji: 💳
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
```

Note : `app_port: 7860` (pas 8000) car le Dockerfile expose 7860 et c'est
le port par défaut HF Docker Spaces. Le `port` du `docker run` local doit
être aligné : `docker run -p 7860:7860`.

### Le piège HF Spaces avec gros fichiers (235 Mo parquet)

**Le problème central de la session.** Le Dockerfile copiait
`data/features_store.parquet` (235 Mo, gitignored) → fail sur HF car le
fichier n'existe pas dans le contexte de build.

**Tentatives infructueuses** :
1. `huggingface_hub.upload_file` avec `repo_type="space"` → API retourne
   un OID de commit valide, mais le commit est **vide** côté objets.
   Diagnostic via `list_repo_files` post-upload : le parquet n'apparaît pas.
2. Ajout d'un `.gitattributes` taggant `*.parquet` en LFS dans le même
   commit → même résultat, file silencieusement non-persisté.
3. Warning HF explicite : *"It seems that you are about to commit a data
   file to a space repository. If you are trying to upload a dataset,
   please set repo_type='dataset'"*. Pas une suggestion : c'est en fait
   une **protection serveur silencieuse** non documentée — les uploads
   >10 Mo dans un Space sont rejetés sans erreur claire.

**Solution retenue : pattern code/data séparés (officiel HF)**

| Layer | Repo | Content |
|-------|------|---------|
| Code + small artefacts | `KLEB38/OC_P8` (Space, Docker) | `api/`, `models/*.json`, `models/model.joblib` |
| Large data | `KLEB38/oc-p8-features` (Dataset) | `features_store.parquet` (235 Mo, LFS auto) |

Le Space télécharge le parquet au cold start via `hf_hub_download(...,
repo_type="dataset")`, qui met en cache localement → pas de re-download
aux boot suivants.

### Modifs concrètes pour appliquer le pattern

1. **`pyproject.toml`** : `huggingface-hub>=1.14.0` déplacée de `dev` →
   deps prod (utilisée au runtime maintenant, plus juste pour le script
   d'upload).
2. **`api/settings.py`** : ajout de `HF_DATASET_REPO_ID` (défaut
   `"KLEB38/oc-p8-features"`) et `HF_DATASET_FILENAME` (défaut
   `"features_store.parquet"`), tous deux configurables via env vars
   `OC_P8_HF_DATASET_REPO_ID` / `OC_P8_HF_DATASET_FILENAME`.
3. **`api/main.py`** : nouvelle fonction `_resolve_feature_store_path()`
   appelée dans le lifespan. Logique :
   - Si `settings.FEATURE_STORE_PATH.exists()` → retourne ce path
     (cas dev local + cas tests via `patched_settings`).
   - Sinon → `hf_hub_download(...)` et retourne le chemin du cache HF.
   - Import lazy de `huggingface_hub` dans la fonction (pas à
     l'import-time du module) pour ne pas ralentir l'app si la lib n'est
     pas dispo.
4. **`Dockerfile`** : suppression de `COPY data/ ./data/` et du
   `test -f data/features_store.parquet`. Image Docker plus légère.
5. **`scripts/upload_data_to_hf.py`** : cible désormais
   `KLEB38/oc-p8-features` avec `repo_type="dataset"`. Documente en
   prérequis la création manuelle du Dataset sur
   https://huggingface.co/new-dataset.
6. **`README.md`** : nouvelle section "Data layer — code/data
   separation" sous Architecture, mise à jour Docker, Required secrets
   (HF_TOKEN runtime optionnel si Dataset privé), project layout.

### Tests

- 45/45 passent post-refacto, couverture 97 % (gate 80 % CI OK).
- Le lifespan ne déclenche **pas** de download HF en test : la fixture
  `patched_settings` set `OC_P8_FEATURE_STORE_PATH` vers un parquet
  synthétique dans `tmp_path` → la condition `.exists()` est True.

### Bug collatéral résolu pendant la session

`test_unexpected_prediction_shape_raises` et
`test_1d_array_prediction_supported` échouaient avec `PicklingError:
Can't pickle <class '<locals>.WeirdModel'>`. Cause : les classes étaient
définies à l'intérieur des fonctions de test, et `pickle` (utilisé par
`joblib.dump`) ne peut sérialiser que les classes accessibles par
import-path. Fix : déplacer `_WeirdModel` et `_FlatModel` au niveau
module dans `tests/unit/test_predictor.py`.

### Étape utilisateur restante

Côté Kevin (manuel, une seule fois) :

1. Créer le Dataset `KLEB38/oc-p8-features` sur
   https://huggingface.co/new-dataset (public recommandé pour éviter
   `HF_TOKEN` au runtime).
2. `$env:HF_TOKEN = "hf_xxx..."; uv run python scripts/upload_data_to_hf.py`
3. Vérifier la présence du parquet sur HF Dataset.
4. Re-trigger le `workflow_dispatch` du CI → build Docker passe, Space
   démarre, lifespan télécharge le parquet une fois et le cache.

---

## Session 2026-05-08 — Code review complète + Tier 1 cleanup

Quatre subagents lancés en parallèle (python-reviewer, security-reviewer,
code-reviewer, architect) sur l'ensemble du repo. Tests post-cleanup :
**45/45 pass, ruff clean, coverage 96.56 %**.

### Fix critique de la session : libgomp.so.1 sur HF Space

`OSError: libgomp.so.1: cannot open shared object file` au cold start.
LightGBM dépend d'OpenMP au runtime, absent de `python:3.12-slim`.

**Faux ami `packages.txt`** : ce fichier n'est lu par HF Spaces que pour
`sdk: gradio` ou `sdk: streamlit`. Pour `sdk: docker`, **il est ignoré** —
les paquets système doivent être installés dans le `Dockerfile` via
`apt-get`.

Fix appliqué (Dockerfile) :

```dockerfile
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
```

`packages.txt` supprimé du repo.

### Tier 1 — fixes appliqués (12 items, zéro changement comportemental)

| Fichier | Fix |
|---------|-----|
| `api/inference_assembler.py:93` | drop `inplace=True` après `.copy()` |
| `api/main.py:115-117` | comment FR retiré, return type, tag `meta` |
| `api/predictor.py` + `api/schemas.py` | `Decision` dédupliqué (DRY) |
| `api/schemas.py` | `Optional[X]` → `X \| None` (PEP 604, Pydantic v2) |
| `feature_engineering/aggregations.py:201-202` | `.apply(lambda)` → `.clip(lower=0)` |
| `feature_engineering/orchestrator.py` | `print()` → `logger.info()` |
| `scripts/build_no_history_template.py` | `list(...)`, type widening `\| None` |
| `scripts/export_model.py:55` | guard `if mv.run_id is None` (mypy) |
| `scripts/export_model.py` | emojis `✅⚠️` → `[OK]`/`[WARN]` |
| `api/settings.py:42-44` | comment expliquant le seuil 0.33 |
| `README.md` | XGBoost → LightGBM (5 occurrences + badge) |
| `README.md` | exemple `decision: "GRANTED"` au lieu de `false` |

### Tier 2 — suggestions reportées (à attaquer plus tard si besoin)

#### A. Split deps prod / offline → image Docker -700 MB à -1 GB

`pyproject.toml` actuellement liste comme deps prod : `mlflow`, `optuna`,
`shap`, `xgboost`, `jupyter`, `ipykernel`, `matplotlib`, `seaborn`,
`plotly`. Aucune n'est importée par `api/`. Plan :

```toml
[project]
dependencies = [
    "fastapi>=0.136.1",
    "huggingface-hub>=1.14.0",
    "joblib",            # ajouter explicitement
    "lightgbm>=4.0.0",
    "numpy>=2.4.3",
    "pandas==2.3.3",
    "pyarrow>=23.0.1",
    "pydantic>=2.13.3",
    "uvicorn[standard]>=0.46.0",
]

[dependency-groups]
dev = ["httpx", "pytest", "pytest-cov", "ruff", "mypy"]
offline = [
    "mlflow>=2.18,<3", "optuna", "shap", "xgboost", "matplotlib",
    "seaborn", "plotly", "jupyter", "ipykernel",
]
```

Dockerfile reste sur `uv sync --frozen --no-dev` (le groupe offline est
exclu par défaut sauf si activé).

**Risque** : faible. Vérifier que `api/predictor.py` n'importe pas mlflow
au runtime (à confirmer — le modèle est désérialisé via joblib mais
`joblib.load` peut tirer mlflow lors de l'unpickle si le modèle est un
`PyFuncModel`). Si oui → alternative : option B ci-dessous.

#### B. Re-export modèle en LightGBM Booster natif → 3-5× faster + supprime mlflow

Actuellement `models/model.joblib` est un `mlflow.pyfunc.PyFuncModel`
contenant le Booster. À l'inférence on appelle `model.predict(df)` qui
re-route via le wrapper PyFunc.

Plan :

```python
# Dans OC_P6, à côté de la sauvegarde MLflow :
booster = lgbm_model.booster_   # ou .estimator.booster_ si pipeline
booster.save_model("models/model.txt")
```

Côté API (`predictor.py`) :

```python
import lightgbm as lgb
model = lgb.Booster(model_file=str(model_path))
proba = model.predict(features.values)[0]
```

**Bénéfice** : élimine mlflow du runtime (déblocage du A), latence
3-5× plus faible sur single-row, image plus légère.

**Risque** : moyen. Demande un test de parité (probabilité identique
±1e-9 entre l'ancien et le nouveau modèle sur un échantillon de 1000
clients).

#### C. Multi-stage Dockerfile + pin base image

```dockerfile
FROM python:3.12.10-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

FROM python:3.12.10-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /app
COPY api/ ./api/
COPY models/ ./models/
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

**Bénéfice** : -100-200 MB sur l'image finale (pas de cache uv,
pas de build-essential), reproductibilité (digest pinning possible
en remplaçant `:3.12.10-slim` par `@sha256:...`).

**Risque** : minimal.

#### D. DataFrame downcast au cold start → RAM -40-60 %

Le parquet (235 MB sur disque) → ~1.0-1.8 GB en RAM avec dtypes par défaut
(float64, int64). Conversion :

```python
# Dans InferenceArtefacts.load() :
feature_store = pd.read_parquet(feature_store_path)
for col in feature_store.select_dtypes(include="float64").columns:
    feature_store[col] = feature_store[col].astype("float32")
for col in feature_store.select_dtypes(include="int64").columns:
    feature_store[col] = pd.to_numeric(feature_store[col], downcast="integer")
```

**Bénéfice** : RAM 600-900 MB au lieu de 1-1.8 GB, `.loc` plus rapide.

**Risque** : faible si LightGBM a été entraîné sur float32 (à vérifier ;
si entraîné float64, faible perte de précision sur les agrégats — à
mesurer via test de parité).

#### E. Précompute numpy template → latence -5 à -15 ms par requête

Le hot path actuel fait `pd.DataFrame([raw])` + `pd.Categorical` × 14 +
`get_dummies` + `concat` + `reindex(768)` + `replace(inf, nan)` pour
chaque requête. Plan : remplacer par une copie de template numpy pré-aligné
sur `feature_names`, avec un mapping `name → index` calculé une fois au boot.

**Bénéfice** : élimine pandas du hot path, latence p50 probablement de
20-40 ms à 3-8 ms.

**Risque** : moyen. Demande un golden-file test exhaustif vs assemble()
actuel (sortie strictement identique sur ≥100 cas couvrant les deux
branches connu/inconnu, toutes les catégorielles).

#### F. Sécurité advisory (non bloquant pour formation OC)

- **Rate limiting** : `slowapi` per-IP sur `/predict` (10 req/min).
- **CORS** : `CORSMiddleware` avec `allow_origins=["https://...your-frontend..."]`.
- **API key** : header `X-API-Key` via dépendance FastAPI si besoin
  d'auth simple.
- **Non-root user dans Dockerfile** : `RUN useradd -m appuser && USER appuser`
  (HF Spaces enforce déjà un user namespace, mais defense-in-depth).

À faire **uniquement** si l'API doit être exposée à du trafic réel.

### Findings non-bloquants laissés tels quels

- **Chemins absolus** dans `scripts/build_feature_store.py:41`,
  `scripts/export_model.py:27-30`, `scripts/check_registry.py:4` —
  scripts personnels one-shot, pas de gain immédiat à env-var-iser.
- **CI action versions** (`actions/checkout@v6`, `setup-python@v6`,
  `upload-artifact@v7`) — flagged par code-reviewer comme non-existantes,
  mais le CI tourne, donc soit elles existent, soit GitHub résout
  silencieusement. Non-bloquant.
- **Duplication ratios** offline / runtime (orchestrator vs `api/ratios.py`)
  — intentionnelle et confirmée par 2 reviewers (le runtime a besoin du
  scrub `inf → NaN` que le offline n'a pas).
- **`joblib.load` pickle-based** — modèle baked-in à l'image build,
  pas un vecteur d'attaque exploitable.
