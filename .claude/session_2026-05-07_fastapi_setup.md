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
