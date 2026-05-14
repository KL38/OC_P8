"""Shared pytest fixtures.

Builds a minimal but realistic set of artefacts in tmp_path so tests run
without requiring the actual training data (~5 GB of CSVs).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


# A tiny but valid app_train categories vocabulary — enough to exercise the
# one-hot logic without listing all 58 ORGANIZATION_TYPE values.
SYNTHETIC_CATEGORIES = {
    "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"],
    "NAME_TYPE_SUITE": ["Family", "Unaccompanied"],
    "NAME_INCOME_TYPE": ["Pensioner", "State servant", "Working"],
    "NAME_EDUCATION_TYPE": [
        "Higher education",
        "Lower secondary",
        "Secondary / secondary special",
    ],
    "NAME_FAMILY_STATUS": ["Civil marriage", "Married", "Single / not married"],
    "NAME_HOUSING_TYPE": ["House / apartment", "Rented apartment", "With parents"],
    "OCCUPATION_TYPE": ["Accountants", "Core staff", "Laborers", "Managers"],
    "WEEKDAY_APPR_PROCESS_START": ["MONDAY", "TUESDAY", "WEDNESDAY"],
    "ORGANIZATION_TYPE": ["Business Entity Type 3", "Government", "School", "XNA"],
    "FONDKAPREMONT_MODE": ["not specified", "reg oper account"],
    "HOUSETYPE_MODE": ["block of flats", "terraced house"],
    "WALLSMATERIAL_MODE": ["Block", "Panel", "Stone, brick"],
    "EMERGENCYSTATE_MODE": ["No", "Yes"],
}

SYNTHETIC_BINARY_MAPPINGS = {
    "CODE_GENDER": {"M": 0, "F": 1},
    "FLAG_OWN_CAR": {"N": 0, "Y": 1},
    "FLAG_OWN_REALTY": {"Y": 0, "N": 1},
}


# A realistic, valid payload accepted by api.schemas.PredictionRequest.
VALID_PAYLOAD: dict[str, Any] = {
    "SK_ID_CURR": 100002,
    "NAME_CONTRACT_TYPE": "Cash loans",
    "CODE_GENDER": "M",
    "FLAG_OWN_CAR": "N",
    "FLAG_OWN_REALTY": "Y",
    "CNT_CHILDREN": 0,
    "AMT_INCOME_TOTAL": 202500.0,
    "AMT_CREDIT": 406597.5,
    "AMT_ANNUITY": 24700.5,
    "AMT_GOODS_PRICE": 351000.0,
    "NAME_TYPE_SUITE": "Unaccompanied",
    "NAME_INCOME_TYPE": "Working",
    "NAME_EDUCATION_TYPE": "Secondary / secondary special",
    "NAME_FAMILY_STATUS": "Single / not married",
    "NAME_HOUSING_TYPE": "House / apartment",
    "REGION_POPULATION_RELATIVE": 0.018801,
    "DAYS_BIRTH": -9461,
    "DAYS_EMPLOYED": -637,
    "DAYS_REGISTRATION": -3648.0,
    "DAYS_ID_PUBLISH": -2120,
    "OWN_CAR_AGE": None,
    "FLAG_MOBIL": 1,
    "FLAG_EMP_PHONE": 1,
    "FLAG_WORK_PHONE": 0,
    "FLAG_CONT_MOBILE": 1,
    "FLAG_PHONE": 1,
    "FLAG_EMAIL": 0,
    "OCCUPATION_TYPE": "Laborers",
    "CNT_FAM_MEMBERS": 1.0,
    "REGION_RATING_CLIENT": 2,
    "REGION_RATING_CLIENT_W_CITY": 2,
    "WEEKDAY_APPR_PROCESS_START": "WEDNESDAY",
    "HOUR_APPR_PROCESS_START": 10,
    "REG_REGION_NOT_LIVE_REGION": 0,
    "REG_REGION_NOT_WORK_REGION": 0,
    "LIVE_REGION_NOT_WORK_REGION": 0,
    "REG_CITY_NOT_LIVE_CITY": 0,
    "REG_CITY_NOT_WORK_CITY": 0,
    "LIVE_CITY_NOT_WORK_CITY": 0,
    "ORGANIZATION_TYPE": "Business Entity Type 3",
    "EXT_SOURCE_1": 0.0830,
    "EXT_SOURCE_2": 0.2629,
    "EXT_SOURCE_3": 0.1393,
    "APARTMENTS_AVG": 0.0247,
    "BASEMENTAREA_AVG": 0.0369,
    "YEARS_BEGINEXPLUATATION_AVG": 0.9722,
    "YEARS_BUILD_AVG": 0.6192,
    "COMMONAREA_AVG": 0.0143,
    "ELEVATORS_AVG": 0.0,
    "ENTRANCES_AVG": 0.0690,
    "FLOORSMAX_AVG": 0.0833,
    "FLOORSMIN_AVG": 0.125,
    "LANDAREA_AVG": 0.0369,
    "LIVINGAPARTMENTS_AVG": 0.0205,
    "LIVINGAREA_AVG": 0.0193,
    "NONLIVINGAPARTMENTS_AVG": 0.0,
    "NONLIVINGAREA_AVG": 0.0,
    "APARTMENTS_MODE": 0.0252,
    "BASEMENTAREA_MODE": 0.0383,
    "YEARS_BEGINEXPLUATATION_MODE": 0.9722,
    "YEARS_BUILD_MODE": 0.6341,
    "COMMONAREA_MODE": 0.0144,
    "ELEVATORS_MODE": 0.0,
    "ENTRANCES_MODE": 0.0690,
    "FLOORSMAX_MODE": 0.0833,
    "FLOORSMIN_MODE": 0.125,
    "LANDAREA_MODE": 0.0377,
    "LIVINGAPARTMENTS_MODE": 0.022,
    "LIVINGAREA_MODE": 0.0198,
    "NONLIVINGAPARTMENTS_MODE": 0.0,
    "NONLIVINGAREA_MODE": 0.0,
    "APARTMENTS_MEDI": 0.025,
    "BASEMENTAREA_MEDI": 0.0369,
    "YEARS_BEGINEXPLUATATION_MEDI": 0.9722,
    "YEARS_BUILD_MEDI": 0.6243,
    "COMMONAREA_MEDI": 0.0144,
    "ELEVATORS_MEDI": 0.0,
    "ENTRANCES_MEDI": 0.069,
    "FLOORSMAX_MEDI": 0.0833,
    "FLOORSMIN_MEDI": 0.125,
    "LANDAREA_MEDI": 0.0377,
    "LIVINGAPARTMENTS_MEDI": 0.0205,
    "LIVINGAREA_MEDI": 0.0198,
    "NONLIVINGAPARTMENTS_MEDI": 0.0,
    "NONLIVINGAREA_MEDI": 0.0,
    "FONDKAPREMONT_MODE": "reg oper account",
    "HOUSETYPE_MODE": "block of flats",
    "TOTALAREA_MODE": 0.0149,
    "WALLSMATERIAL_MODE": "Stone, brick",
    "EMERGENCYSTATE_MODE": "No",
    "OBS_30_CNT_SOCIAL_CIRCLE": 2.0,
    "DEF_30_CNT_SOCIAL_CIRCLE": 2.0,
    "OBS_60_CNT_SOCIAL_CIRCLE": 2.0,
    "DEF_60_CNT_SOCIAL_CIRCLE": 2.0,
    "DAYS_LAST_PHONE_CHANGE": -1134.0,
    "FLAG_DOCUMENT_2": 0,
    "FLAG_DOCUMENT_3": 1,
    "FLAG_DOCUMENT_4": 0,
    "FLAG_DOCUMENT_5": 0,
    "FLAG_DOCUMENT_6": 0,
    "FLAG_DOCUMENT_7": 0,
    "FLAG_DOCUMENT_8": 0,
    "FLAG_DOCUMENT_9": 0,
    "FLAG_DOCUMENT_10": 0,
    "FLAG_DOCUMENT_11": 0,
    "FLAG_DOCUMENT_12": 0,
    "FLAG_DOCUMENT_13": 0,
    "FLAG_DOCUMENT_14": 0,
    "FLAG_DOCUMENT_15": 0,
    "FLAG_DOCUMENT_16": 0,
    "FLAG_DOCUMENT_17": 0,
    "FLAG_DOCUMENT_18": 0,
    "FLAG_DOCUMENT_19": 0,
    "FLAG_DOCUMENT_20": 0,
    "FLAG_DOCUMENT_21": 0,
    "AMT_REQ_CREDIT_BUREAU_HOUR": 0.0,
    "AMT_REQ_CREDIT_BUREAU_DAY": 0.0,
    "AMT_REQ_CREDIT_BUREAU_WEEK": 0.0,
    "AMT_REQ_CREDIT_BUREAU_MON": 0.0,
    "AMT_REQ_CREDIT_BUREAU_QRT": 0.0,
    "AMT_REQ_CREDIT_BUREAU_YEAR": 1.0,
}


def _build_fake_predict_fn(feature_names: list[str]):
    """Build a predict_fn that drives proba from AMT_INCOME_TOTAL / AMT_CREDIT.

    Replaces the legacy FakeModel(predict_proba) used before the ONNX
    migration. Operates on the numpy array passed by CreditScoringPredictor,
    looking up positions via feature_names so tests can deterministically
    exercise both GRANTED and REFUSED branches.
    """
    income_idx = feature_names.index("AMT_INCOME_TOTAL")
    credit_idx = feature_names.index("AMT_CREDIT")

    def predict_fn(arr: np.ndarray) -> np.ndarray:
        # arr is (1, n_features) float32 — NaN values may exist for skipped fields.
        income = float(arr[0, income_idx]) if not np.isnan(arr[0, income_idx]) else 0.0
        credit_raw = arr[0, credit_idx]
        credit = float(credit_raw) if not np.isnan(credit_raw) else 1.0
        ratio = income / max(credit, 1.0)
        proba_default = max(0.0, min(1.0, 1.0 - ratio))
        return np.array([[1 - proba_default, proba_default]])

    return predict_fn


@pytest.fixture
def synthetic_feature_names() -> list[str]:
    """Build a stable, ordered list of feature names matching the synthetic
    artefacts. Mixes a handful of app_train one-hot columns + ratios + a few
    aggregate columns (counts and a generic NaN-aggregate)."""
    return [
        # Numeric/binary fields kept as-is by transform_app_train_inputs
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "CNT_CHILDREN",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "CNT_FAM_MEMBERS",
        # Derived ratios
        "DAYS_EMPLOYED_PERC",
        "INCOME_CREDIT_PERC",
        "INCOME_PER_PERSON",
        "ANNUITY_INCOME_PERC",
        "PAYMENT_RATE",
        # Sample aggregate columns
        "BURO_DAYS_CREDIT_MEAN",
        "POS_COUNT",
        "INSTAL_COUNT",
        "CC_COUNT",
        "PREV_AMT_ANNUITY_MEAN",
    ]


@pytest.fixture
def synthetic_artefacts_dir(tmp_path: Path, synthetic_feature_names: list[str]) -> Path:
    """Create a fully populated artefacts directory under tmp_path/.

    Layout mirrors the project root so we can override settings paths
    individually.
    """
    models_dir = tmp_path / "models"
    data_dir = tmp_path / "data"
    models_dir.mkdir()
    data_dir.mkdir()

    # 1. feature_names.json
    (models_dir / "feature_names.json").write_text(json.dumps(synthetic_feature_names))

    # 2. app_train_categories.json (multi-cat only)
    (models_dir / "app_train_categories.json").write_text(json.dumps(SYNTHETIC_CATEGORIES))

    # 3. app_train_binary_mappings.json
    (models_dir / "app_train_binary_mappings.json").write_text(
        json.dumps(SYNTHETIC_BINARY_MAPPINGS)
    )

    # 4. no_history_template.json — counts → 0, others → null (NaN once loaded)
    template = {
        "BURO_DAYS_CREDIT_MEAN": None,
        "POS_COUNT": 0,
        "INSTAL_COUNT": 0,
        "CC_COUNT": 0,
        "PREV_AMT_ANNUITY_MEAN": None,
    }
    (models_dir / "no_history_template.json").write_text(json.dumps(template))

    # 5. Mini features_store.parquet — 2 known clients with realistic aggregates
    feature_store = pd.DataFrame(
        {
            "BURO_DAYS_CREDIT_MEAN": [-1234.5, -890.2],
            "POS_COUNT": [12, 5],
            "INSTAL_COUNT": [25, 10],
            "CC_COUNT": [3, 0],
            "PREV_AMT_ANNUITY_MEAN": [15000.0, 8000.0],
        },
        index=pd.Index([100002, 100003], name="SK_ID_CURR"),
    )
    feature_store.to_parquet(data_dir / "features_store.parquet")

    # 6. model.onnx — placeholder. The real ONNX session is bypassed by the
    # patched_settings fixture (it monkey-patches CreditScoringPredictor.load
    # to inject a deterministic fake predict_fn), so this file just needs to
    # exist for paths that os.stat() it. Tests do NOT run actual ONNX
    # inference; that's covered by the live model in CI smoke tests.
    (models_dir / "model.onnx").write_bytes(b"")

    # 7. model_info.json — minimal subset used by predictor + main routes
    (models_dir / "model_info.json").write_text(
        json.dumps(
            {
                "model_name": "fake_test_model",
                "version": "test-1",
                "metrics": {"best_threshold_mean": 0.33},
            }
        )
    )

    return tmp_path


@pytest.fixture
def patched_settings(
    monkeypatch,
    synthetic_artefacts_dir: Path,
    synthetic_feature_names: list[str],
) -> None:
    """Point api.settings at the synthetic artefacts (re-import safe).

    Also replaces CreditScoringPredictor.load with an override that injects a
    deterministic fake predict_fn — this avoids needing a real ONNX file
    keyed to the 23 synthetic feature columns.
    """
    base = synthetic_artefacts_dir
    # Force the prediction logger to no-op for unit/integration FastAPI tests
    # so they don't accidentally hit a developer's local DATABASE_URL.
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("OC_P8_MODEL_PATH", str(base / "models" / "model.onnx"))
    monkeypatch.setenv("OC_P8_MODEL_INFO_PATH", str(base / "models" / "model_info.json"))
    monkeypatch.setenv(
        "OC_P8_FEATURE_NAMES_PATH", str(base / "models" / "feature_names.json")
    )
    monkeypatch.setenv(
        "OC_P8_APP_TRAIN_CATEGORIES_PATH",
        str(base / "models" / "app_train_categories.json"),
    )
    monkeypatch.setenv(
        "OC_P8_APP_TRAIN_BINARY_MAPPINGS_PATH",
        str(base / "models" / "app_train_binary_mappings.json"),
    )
    monkeypatch.setenv(
        "OC_P8_NO_HISTORY_TEMPLATE_PATH",
        str(base / "models" / "no_history_template.json"),
    )
    monkeypatch.setenv(
        "OC_P8_FEATURE_STORE_PATH", str(base / "data" / "features_store.parquet")
    )

    # Force a fresh import of api.settings so it picks up the env vars.
    import importlib

    import api.settings as s

    importlib.reload(s)

    # Also reset the lazy DB engine so a previous test's connection isn't
    # leaked into this one (important when DATABASE_URL was set elsewhere).
    from api import db

    db.reset_engine()

    # Replace CreditScoringPredictor.load() with a stub that injects the
    # deterministic fake predict_fn keyed to the synthetic feature order.
    # Avoids the need for a real .onnx file in unit/integration tests.
    # Reuses resolve_threshold_and_version() so the parsing rules stay
    # consistent with production code.
    from api.predictor import CreditScoringPredictor, resolve_threshold_and_version

    fake_predict_fn = _build_fake_predict_fn(synthetic_feature_names)

    def fake_load(
        cls,
        model_path: Path,
        model_info_path: Path,
        default_threshold: float,
    ) -> "CreditScoringPredictor":
        threshold, version = resolve_threshold_and_version(
            model_info_path, default_threshold
        )
        return cls(
            predict_fn=fake_predict_fn,
            threshold=threshold,
            model_version=version,
        )

    monkeypatch.setattr(CreditScoringPredictor, "load", classmethod(fake_load))


@pytest.fixture
def valid_payload() -> dict[str, Any]:
    """Return a fresh copy of the canonical valid request payload."""
    return dict(VALID_PAYLOAD)
