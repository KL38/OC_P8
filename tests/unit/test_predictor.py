"""Unit tests for CreditScoringPredictor."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from api.predictor import CreditScoringPredictor


class _FakeClassifier:
    """Stand-in sklearn-style classifier: returns a fixed positive-class proba."""

    def __init__(self, proba: float) -> None:
        self.proba = proba

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([[1 - self.proba, self.proba]])


def _make_info(path: Path, threshold: float | None = 0.33) -> Path:
    metrics = {"best_threshold_mean": threshold} if threshold is not None else {}
    path.write_text(json.dumps({"version": "test-1", "metrics": metrics}))
    return path


def _build(tmp_path: Path, proba: float, threshold: float) -> CreditScoringPredictor:
    model_path = tmp_path / "model.joblib"
    joblib.dump(_FakeClassifier(proba), model_path)
    return CreditScoringPredictor.load(
        model_path=model_path,
        model_info_path=_make_info(tmp_path / "info.json", threshold),
        default_threshold=0.5,
    )


def test_threshold_loaded_from_model_info(tmp_path):
    assert _build(tmp_path, proba=0.1, threshold=0.42).threshold == 0.42


def test_threshold_falls_back_to_default(tmp_path):
    """If model_info lacks best_threshold_mean, default applies."""
    joblib.dump(_FakeClassifier(0.0), tmp_path / "model.joblib")
    pred = CreditScoringPredictor.load(
        model_path=tmp_path / "model.joblib",
        model_info_path=_make_info(tmp_path / "info.json", threshold=None),
        default_threshold=0.7,
    )
    assert pred.threshold == 0.7


@pytest.mark.parametrize(
    "proba, expected",
    [
        (0.10, "GRANTED"),   # below threshold
        (0.50, "REFUSED"),   # above threshold
        (0.33, "REFUSED"),   # boundary: proba >= threshold → REFUSED
    ],
)
def test_decision_logic(tmp_path, proba, expected):
    p, decision = _build(tmp_path, proba=proba, threshold=0.33).predict(pd.DataFrame([{}]))
    assert p == pytest.approx(proba)
    assert decision == expected


def test_proba_is_continuous_not_label(tmp_path):
    """Regression: predict() must surface predict_proba output, not class labels."""
    proba, _ = _build(tmp_path, proba=0.27, threshold=0.5).predict(pd.DataFrame([{}]))
    assert proba == pytest.approx(0.27)


def test_pyfunc_wrapper_is_unwrapped():
    """A model exposing get_raw_model() (MLflow PyFunc) is unwrapped at init."""

    class _PyFuncLike:
        def __init__(self, inner):
            self._inner = inner

        def get_raw_model(self):
            return self._inner

        def predict(self, df):  # PyFunc would return labels — must NOT be called
            raise AssertionError("PyFunc.predict() must not be called")

    pred = CreditScoringPredictor(
        model=_PyFuncLike(_FakeClassifier(0.42)),
        threshold=0.5,
        model_version="v1",
    )
    proba, _ = pred.predict(pd.DataFrame([{}]))
    assert proba == pytest.approx(0.42)
