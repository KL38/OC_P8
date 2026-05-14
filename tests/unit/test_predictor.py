"""Unit tests for CreditScoringPredictor.

The predictor is duck-typed on a `predict_fn` callable, so these tests inject
fake functions directly rather than round-tripping through a real .onnx file.
The full load() path (ONNX session + model_info.json) is exercised by the
integration tests with a real model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from api.predictor import CreditScoringPredictor


def _fixed_predict_fn(proba: float):
    """Build a predict_fn that always returns the given positive-class proba."""

    def fn(arr: np.ndarray) -> np.ndarray:
        return np.array([[1 - proba, proba]])

    return fn


class _FakeOnnxSession:
    """Stand-in for ort.InferenceSession used to test the load() contract.

    Exposes the same get_inputs/get_outputs/run surface as a real session
    without requiring a real .onnx file. Used by tests that exercise
    CreditScoringPredictor.load — keeps the session-wiring detail
    out of every test body.
    """

    def get_inputs(self):
        return [type("X", (), {"name": "input"})()]

    def get_outputs(self):
        return [
            type("L", (), {"name": "label"})(),
            type("P", (), {"name": "probabilities"})(),
        ]

    def run(self, _outputs, _feed):
        return [np.array([[0.6, 0.4]])]


@pytest.fixture
def patch_onnx_session(monkeypatch):
    """Replace ort.InferenceSession with the fake so load() can be unit-tested."""
    monkeypatch.setattr(
        "api.predictor.ort.InferenceSession", lambda *_a, **_kw: _FakeOnnxSession()
    )


def _build(proba: float, threshold: float) -> CreditScoringPredictor:
    return CreditScoringPredictor(
        predict_fn=_fixed_predict_fn(proba),
        threshold=threshold,
        model_version="test",
    )


def _make_info(path: Path, threshold: float | None = 0.33) -> Path:
    metrics = {"best_threshold_mean": threshold} if threshold is not None else {}
    path.write_text(json.dumps({"version": "test-1", "metrics": metrics}))
    return path


def test_threshold_property_round_trip():
    assert _build(proba=0.1, threshold=0.42).threshold == 0.42


def test_model_version_property_round_trip():
    pred = CreditScoringPredictor(
        predict_fn=_fixed_predict_fn(0.0),
        threshold=0.5,
        model_version="abc-123",
    )
    assert pred.model_version == "abc-123"


@pytest.mark.parametrize(
    "proba, expected",
    [
        (0.10, "GRANTED"),   # below threshold
        (0.50, "REFUSED"),   # above threshold
        (0.33, "REFUSED"),   # boundary: proba >= threshold → REFUSED
    ],
)
def test_decision_logic(proba, expected):
    p, decision = _build(proba=proba, threshold=0.33).predict(pd.DataFrame([{"x": 1.0}]))
    assert p == pytest.approx(proba)
    assert decision == expected


def test_proba_is_continuous_not_label():
    """Regression: predict() must surface the proba, not a class label."""
    proba, _ = _build(proba=0.27, threshold=0.5).predict(pd.DataFrame([{"x": 1.0}]))
    assert proba == pytest.approx(0.27)


def test_features_are_converted_to_float32():
    """The predict_fn must receive a float32 numpy array (ONNX requirement)."""
    captured: dict[str, np.ndarray] = {}

    def capturing_fn(arr: np.ndarray) -> np.ndarray:
        captured["arr"] = arr
        return np.array([[0.7, 0.3]])

    pred = CreditScoringPredictor(
        predict_fn=capturing_fn, threshold=0.5, model_version="v1"
    )
    pred.predict(pd.DataFrame([{"a": 1.0, "b": 2.0}]))

    assert captured["arr"].dtype == np.float32
    assert captured["arr"].shape == (1, 2)


def test_load_reads_threshold_from_model_info(tmp_path, patch_onnx_session):
    """load() must resolve threshold from model_info.json without touching ONNX."""
    info_path = _make_info(tmp_path / "info.json", threshold=0.42)

    pred = CreditScoringPredictor.load(
        model_path=tmp_path / "dummy.onnx",
        model_info_path=info_path,
        default_threshold=0.5,
    )
    assert pred.threshold == 0.42
    assert pred.model_version == "test-1"


def test_load_falls_back_to_default_threshold(tmp_path, patch_onnx_session):
    info_path = _make_info(tmp_path / "info.json", threshold=None)

    pred = CreditScoringPredictor.load(
        model_path=tmp_path / "dummy.onnx",
        model_info_path=info_path,
        default_threshold=0.7,
    )
    assert pred.threshold == 0.7
