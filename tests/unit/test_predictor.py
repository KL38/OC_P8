"""Unit tests for CreditScoringPredictor."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from api.predictor import CreditScoringPredictor


class _FixedProbModel:
    """Fake sklearn-style classifier exposing predict_proba."""

    def __init__(self, proba: float) -> None:
        self.proba = proba

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([[1 - self.proba, self.proba]])


def _write_model_info(path: Path, threshold: float, version: str = "test-1") -> None:
    path.write_text(
        json.dumps(
            {
                "model_name": "fake",
                "version": version,
                "metrics": {"best_threshold_mean": threshold},
            }
        )
    )


def _build(tmp_path: Path, proba: float, threshold: float) -> CreditScoringPredictor:
    model_path = tmp_path / "model.joblib"
    info_path = tmp_path / "info.json"
    joblib.dump(_FixedProbModel(proba), model_path)
    _write_model_info(info_path, threshold)
    return CreditScoringPredictor.load(
        model_path=model_path, model_info_path=info_path, default_threshold=0.5
    )


def test_threshold_loaded_from_model_info(tmp_path):
    pred = _build(tmp_path, proba=0.1, threshold=0.42)
    assert pred.threshold == 0.42


def test_threshold_falls_back_to_default(tmp_path):
    """If model_info lacks best_threshold_mean, default applies."""
    info = tmp_path / "info.json"
    info.write_text(json.dumps({"model_name": "fake", "version": "1", "metrics": {}}))
    model = tmp_path / "model.joblib"
    joblib.dump(_FixedProbModel(0.0), model)
    pred = CreditScoringPredictor.load(model, info, default_threshold=0.7)
    assert pred.threshold == 0.7


def test_decision_granted_below_threshold(tmp_path):
    pred = _build(tmp_path, proba=0.1, threshold=0.33)
    proba, decision = pred.predict(pd.DataFrame([{}]))
    assert proba == pytest.approx(0.1)
    assert decision == "GRANTED"


def test_decision_refused_above_threshold(tmp_path):
    pred = _build(tmp_path, proba=0.5, threshold=0.33)
    proba, decision = pred.predict(pd.DataFrame([{}]))
    assert decision == "REFUSED"


def test_decision_at_threshold_is_refused(tmp_path):
    """proba >= threshold → REFUSED (boundary test)."""
    pred = _build(tmp_path, proba=0.33, threshold=0.33)
    _, decision = pred.predict(pd.DataFrame([{}]))
    assert decision == "REFUSED"


def test_proba_in_unit_interval(tmp_path):
    pred = _build(tmp_path, proba=0.42, threshold=0.5)
    proba, _ = pred.predict(pd.DataFrame([{}]))
    assert 0.0 <= proba <= 1.0


def test_proba_is_continuous_not_label(tmp_path):
    """Regression: predict() must return predict_proba output, not class labels."""
    pred = _build(tmp_path, proba=0.27, threshold=0.5)
    proba, _ = pred.predict(pd.DataFrame([{}]))
    assert proba == pytest.approx(0.27)
    assert proba not in (0.0, 1.0)


def test_pyfunc_wrapper_is_unwrapped(tmp_path):
    """A PyFunc-style wrapper exposing get_raw_model() must be unwrapped on load."""

    class _PyFuncLike:
        def __init__(self, inner):
            self._inner = inner

        def get_raw_model(self):
            return self._inner

        # PyFunc.predict() would return labels — must NOT be called.
        def predict(self, df):
            raise AssertionError("PyFunc.predict() should not be called")

    info = tmp_path / "info.json"
    _write_model_info(info, 0.5)
    model = tmp_path / "model.joblib"
    joblib.dump(_PyFuncLike(_FixedProbModel(0.42)), model)
    pred = CreditScoringPredictor.load(model, info, default_threshold=0.5)
    proba, _ = pred.predict(pd.DataFrame([{}]))
    assert proba == pytest.approx(0.42)
