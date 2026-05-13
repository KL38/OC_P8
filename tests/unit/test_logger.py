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


def _build_mock_engine(captured: dict, monkeypatch, fake_id: int = 42) -> None:
    """Wire a MagicMock engine that captures INSERT and UPDATE payloads.

    log_prediction now performs two roundtrips:
      1. INSERT ... RETURNING id  -> mock returns fake_id from scalar_one()
      2. UPDATE ... SET db_log_ms WHERE id = X
    The mock distinguishes them by inspecting the compiled SQL string.
    """
    fake_engine = MagicMock()

    class _Result:
        def scalar_one(self):
            return fake_id

    class _Conn:
        def execute(self, stmt):
            compiled = stmt.compile()
            params = compiled.params
            if "INSERT" in str(compiled).upper():
                captured["insert"] = params
            else:
                captured["update"] = params
            return _Result()

    class _Ctx:
        def __enter__(self):
            return _Conn()

        def __exit__(self, *a):
            return False

    fake_engine.begin.return_value = _Ctx()
    monkeypatch.setattr(db, "_engine", fake_engine)


def test_insert_payload_contains_expected_fields(monkeypatch) -> None:
    captured: dict = {}
    _build_mock_engine(captured, monkeypatch)

    _call(
        features=pd.DataFrame([{"EXT_SOURCE_1": 0.42, "FOO": np.nan, "BAR": np.inf}]),
        feature_assembly_ms=12,
        inference_ms=3,
        inference_cpu_ms=3,
        plumbing_ms=1,
    )

    values = captured["insert"]
    assert values["sk_id_curr"] == 100002
    assert values["decision"] == "GRANTED"
    assert values["probability_default"] == 0.27
    assert values["client_known"] is True
    assert values["features"]["EXT_SOURCE_1"] == pytest.approx(0.42)
    # NaN/Inf scrubbed to None for JSONB compatibility
    assert values["features"]["FOO"] is None
    assert values["features"]["BAR"] is None
    # Fine-grained timings propagated through to the insert payload.
    assert values["feature_assembly_ms"] == 12
    assert values["inference_ms"] == 3
    assert values["inference_cpu_ms"] == 3
    assert values["plumbing_ms"] == 1
    # db_log_ms is filled in by the follow-up UPDATE, not the INSERT.
    assert values["db_log_ms"] is None
    # And the UPDATE carries an int db_log_ms.
    assert "update" in captured
    assert isinstance(captured["update"]["db_log_ms"], int)
    assert captured["update"]["db_log_ms"] >= 0


def test_db_failure_is_swallowed(monkeypatch, caplog) -> None:
    fake_engine = MagicMock()
    fake_engine.begin.side_effect = RuntimeError("connection refused")
    monkeypatch.setattr(db, "_engine", fake_engine)

    _call()  # must not raise
    assert any("Failed to log prediction" in rec.message for rec in caplog.records)


def test_error_path_logs_status_and_message(monkeypatch) -> None:
    captured: dict = {}
    _build_mock_engine(captured, monkeypatch)

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

    values = captured["insert"]
    assert values["status_code"] == 500
    assert values["error_message"] == "boom"
    assert values["decision"] == "ERROR"
    assert values["features"] == {}
    # Error rows leave timing breakdown NULL (defaults to None when omitted).
    assert values["feature_assembly_ms"] is None
    assert values["inference_ms"] is None
    assert values["inference_cpu_ms"] is None
    assert values["plumbing_ms"] is None
    # db_log_ms is still measured on the error path — the INSERT still happened.
    assert "update" in captured
    assert isinstance(captured["update"]["db_log_ms"], int)
