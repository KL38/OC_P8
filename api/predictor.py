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
        # MLflow PyFunc wraps the sklearn model; unwrap so we can call
        # predict_proba (PyFunc.predict() returns class labels, not probas).
        self._model = model.get_raw_model() if hasattr(model, "get_raw_model") else model
        self._threshold = threshold
        self._model_version = model_version

    @classmethod
    def load(
        cls,
        model_path: Path,
        model_info_path: Path,
        default_threshold: float,
    ) -> "CreditScoringPredictor":
        info = json.loads(model_info_path.read_text())
        threshold = float(info.get("metrics", {}).get("best_threshold_mean", default_threshold))
        return cls(
            model=joblib.load(model_path),
            threshold=threshold,
            model_version=str(info.get("version", "unknown")),
        )

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def model_version(self) -> str:
        return self._model_version

    def predict(self, features: pd.DataFrame) -> tuple[float, Decision]:
        """Return (probability_of_default, decision)."""
        proba = float(self._model.predict_proba(features)[0, 1])
        decision: Decision = "REFUSED" if proba >= self._threshold else "GRANTED"
        return proba, decision
