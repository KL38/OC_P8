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
    def __init__(self, proba: float) -> None:
        self.proba = proba

    def predict(self, df: pd.DataFrame) -> np.ndarray:
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


def test_unexpected_prediction_shape_raises(tmp_path):
    class WeirdModel:
        def predict(self, df):
            return np.array([[[0.1, 0.9]]])  # 3D — invalid

    info = tmp_path / "info.json"
    _write_model_info(info, 0.5)
    model = tmp_path / "model.joblib"
    joblib.dump(WeirdModel(), model)
    pred = CreditScoringPredictor.load(model, info, default_threshold=0.5)
    with pytest.raises(ValueError, match="Unexpected prediction shape"):
        pred.predict(pd.DataFrame([{}]))


def test_1d_array_prediction_supported(tmp_path):
    """Some PyFunc wrappers return a 1D probability array — supported too."""

    class FlatModel:
        def predict(self, df):
            return np.array([0.6])

    info = tmp_path / "info.json"
    _write_model_info(info, 0.5)
    model = tmp_path / "model.joblib"
    joblib.dump(FlatModel(), model)
    pred = CreditScoringPredictor.load(model, info, default_threshold=0.5)
    proba, decision = pred.predict(pd.DataFrame([{}]))
    assert proba == pytest.approx(0.6)
    assert decision == "REFUSED"
