"""Model loading and prediction wrapper.

Loaded once at API startup (see api.main:lifespan) — never per-request.
Wraps the joblib MLflow PyFuncModel so callers don't have to know about the
internal layout. Threshold is read from model_info.json with a fallback to
the value in api.settings.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from api.schemas import Decision


class CreditScoringPredictor:
    """Singleton-style wrapper. Build once via load(), reuse for every request."""

    def __init__(self, model, threshold: float, model_version: str) -> None:
        self._model = model
        self._threshold = threshold
        self._model_version = model_version

    @classmethod
    def load(
        cls,
        model_path: Path,
        model_info_path: Path,
        default_threshold: float,
    ) -> "CreditScoringPredictor":
        loaded = joblib.load(model_path)
        # MLflow PyFunc wraps the sklearn model; unwrap so we can call
        # predict_proba (PyFunc.predict() returns class labels, not probas).
        model = loaded.get_raw_model() if hasattr(loaded, "get_raw_model") else loaded

        info = json.loads(model_info_path.read_text())
        metrics = info.get("metrics", {})
        threshold = float(metrics.get("best_threshold_mean", default_threshold))
        version = str(info.get("version", "unknown"))

        return cls(model=model, threshold=threshold, model_version=version)

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def model_version(self) -> str:
        return self._model_version

    def predict(self, features: pd.DataFrame) -> tuple[float, Decision]:
        """Return (probability_of_default, decision)."""
        proba = self._predict_proba(features)
        decision: Decision = "REFUSED" if proba >= self._threshold else "GRANTED"
        return proba, decision

    def _predict_proba(self, features: pd.DataFrame) -> float:
        """Extract the positive-class probability from the underlying model."""
        proba = self._model.predict_proba(features)
        return float(proba[0, 1])
