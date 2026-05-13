"""Unit tests for api.logger.

We never connect to a real database — the engine and connection are
replaced by mocks. The point is to verify that:
- log_prediction silently no-ops when no engine is configured;
- it scrubs NaN/Inf to None in the JSONB payloads;
- it builds an INSERT with the right values;
- it never raises, even when the DB fails.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from api import db, logger as api_logger


@pytest.fixture(autouse=True)
def _reset_engine():
    db.reset_engine()
    yield
    db.reset_engine()


def _call(features: pd.DataFrame | None = None, **overrides):
    defaults = dict(
        sk_id_curr=100002,
        client_known=True,
        raw_input={"CODE_GENDER": "M", "AMT_INCOME_TOTAL": 200_000.0},
        features=features if features is not None else pd.DataFrame([{"EXT_SOURCE_1": 0.3}]),
        probability_default=0.27,
        decision="GRANTED",
        threshold=0.33,
        model_version="test-1",
        latency_ms=42,
    )
    defaults.update(overrides)
    api_logger.log_prediction(**defaults)


def test_noop_when_engine_unset() -> None:
    """No engine = no DB call = no error."""
    _call()  # must not raise


def test_insert_payload_contains_expected_fields(monkeypatch) -> None:
    fake_engine = MagicMock()
    captured = {}

    class _Ctx:
        def __enter__(self):
            return _Conn()

        def __exit__(self, *a):
            return False

    class _Conn:
        def execute(self, stmt):
            captured["values"] = stmt.compile().params

    fake_engine.begin.return_value = _Ctx()
    monkeypatch.setattr(db, "_engine", fake_engine)

    _call(
        features=pd.DataFrame([{"EXT_SOURCE_1": 0.42, "FOO": np.nan, "BAR": np.inf}]),
        feature_assembly_ms=12.5,
        inference_ms=3.2,
        inference_cpu_ms=2.9,
    )

    values = captured["values"]
    assert values["sk_id_curr"] == 100002
    assert values["decision"] == "GRANTED"
    assert values["probability_default"] == 0.27
    assert values["client_known"] is True
    assert values["features"]["EXT_SOURCE_1"] == pytest.approx(0.42)
    # NaN/Inf scrubbed to None for JSONB compatibility
    assert values["features"]["FOO"] is None
    assert values["features"]["BAR"] is None
    # Fine-grained timings propagated through to the insert payload.
    assert values["feature_assembly_ms"] == pytest.approx(12.5)
    assert values["inference_ms"] == pytest.approx(3.2)
    assert values["inference_cpu_ms"] == pytest.approx(2.9)


def test_db_failure_is_swallowed(monkeypatch, caplog) -> None:
    fake_engine = MagicMock()
    fake_engine.begin.side_effect = RuntimeError("connection refused")
    monkeypatch.setattr(db, "_engine", fake_engine)

    _call()  # must not raise
    assert any("Failed to log prediction" in rec.message for rec in caplog.records)


def test_error_path_logs_status_and_message(monkeypatch) -> None:
    fake_engine = MagicMock()
    captured = {}

    class _Ctx:
        def __enter__(self):
            return _Conn()

        def __exit__(self, *a):
            return False

    class _Conn:
        def execute(self, stmt):
            captured["values"] = stmt.compile().params

    fake_engine.begin.return_value = _Ctx()
    monkeypatch.setattr(db, "_engine", fake_engine)

    api_logger.log_prediction(
        sk_id_curr=999,
        client_known=False,
        raw_input={"CODE_GENDER": "F"},
        features=None,
        probability_default=None,
        decision=None,
        threshold=0.33,
        model_version="test-1",
        latency_ms=5,
        status_code=500,
        error_message="boom",
    )

    values = captured["values"]
    assert values["status_code"] == 500
    assert values["error_message"] == "boom"
    assert values["decision"] == "ERROR"
    assert values["features"] == {}
    # Error rows leave timing breakdown NULL (defaults to None when omitted).
    assert values["feature_assembly_ms"] is None
    assert values["inference_ms"] is None
    assert values["inference_cpu_ms"] is None
