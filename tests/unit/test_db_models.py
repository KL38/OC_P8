"""Schema-shape tests for the prediction log table.

We don't connect to a real DB here — just verify the SQLAlchemy Table object
contains the columns we expect, with the right types and nullability.
"""

from __future__ import annotations

from sqlalchemy import MetaData

from database.models import (
    PROD_TABLE_NAME,
    TEST_TABLE_NAME,
    build_predictions_log_table,
)


def test_prod_and_test_tables_have_identical_schemas() -> None:
    prod = build_predictions_log_table(PROD_TABLE_NAME, MetaData())
    tst = build_predictions_log_table(TEST_TABLE_NAME, MetaData())

    prod_cols = {c.name: (c.type.__class__, c.nullable) for c in prod.columns}
    test_cols = {c.name: (c.type.__class__, c.nullable) for c in tst.columns}

    assert prod_cols == test_cols
    assert prod.name == "predictions_log"
    assert tst.name == "predictions_log_test"


def test_table_contains_expected_columns() -> None:
    table = build_predictions_log_table("predictions_log", MetaData())
    names = {c.name for c in table.columns}
    expected = {
        "id",
        "timestamp",
        "sk_id_curr",
        "client_known",
        "latency_ms",
        "feature_assembly_ms",
        "inference_ms",
        "inference_cpu_ms",
        "plumbing_ms",
        "db_log_ms",
        "status_code",
        "error_message",
        "raw_input",
        "features",
        "probability_default",
        "decision",
        "threshold",
        "model_version",
        "top_shap",
        "ground_truth",
    }
    assert expected <= names


def test_required_columns_are_not_null() -> None:
    table = build_predictions_log_table("predictions_log", MetaData())
    cols = {c.name: c for c in table.columns}

    assert cols["sk_id_curr"].nullable is False
    assert cols["decision"].nullable is False
    assert cols["raw_input"].nullable is False
    assert cols["features"].nullable is False
    # Error rows (status_code != 200) may have NULL proba.
    assert cols["probability_default"].nullable is True
    # Ground truth and top_shap are post-hoc data.
    assert cols["ground_truth"].nullable is True
    assert cols["top_shap"].nullable is True
    # Fine-grained timings (étape 4) are nullable — error rows leave them NULL.
    assert cols["feature_assembly_ms"].nullable is True
    assert cols["inference_ms"].nullable is True
    assert cols["inference_cpu_ms"].nullable is True
    assert cols["plumbing_ms"].nullable is True
    assert cols["db_log_ms"].nullable is True
