"""Model loading and prediction wrapper.

Loaded once at API startup (see api.main:lifespan) — never per-request.
Wraps an ONNX Runtime session (converted from the original LightGBM model via
scripts/export_to_onnx.py). Threshold is read from model_info.json with a
fallback to the value in api.settings.

The constructor is duck-typed (predict_fn callable) so tests can inject a
fake without round-tripping through a real .onnx file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime as ort
import pandas as pd

from api.schemas import Decision

# Takes a (n, n_features) float32 array and returns a (n, 2) probability matrix
# in column order [prob_class_0, prob_class_1].
PredictFn = Callable[[np.ndarray], np.ndarray]


def resolve_threshold_and_version(
    model_info_path: Path, default_threshold: float
) -> tuple[float, str]:
    """Parse threshold + version from model_info.json.

    Extracted so test fixtures can reuse the exact same parsing rules as the
    production loader, instead of re-implementing the dict navigation.
    """
    info = json.loads(model_info_path.read_text())
    threshold = float(
        info.get("metrics", {}).get("best_threshold_mean", default_threshold)
    )
    version = str(info.get("version", "unknown"))
    return threshold, version


class CreditScoringPredictor:
    """Singleton-style wrapper. Build once via load(), reuse for every request."""

    def __init__(
        self,
        predict_fn: PredictFn,
        threshold: float,
        model_version: str,
    ) -> None:
        self._predict_fn = predict_fn
        self._threshold = threshold
        self._model_version = model_version

    @classmethod
    def load(
        cls,
        model_path: Path,
        model_info_path: Path,
        default_threshold: float,
    ) -> "CreditScoringPredictor":
        threshold, version = resolve_threshold_and_version(
            model_info_path, default_threshold
        )

        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        # The LightGBM→ONNX graph emits two outputs: labels (idx 0) and the
        # (n, 2) probability matrix (idx 1) — we want the latter.
        proba_output_name = session.get_outputs()[1].name

        def predict_fn(arr: np.ndarray) -> np.ndarray:
            return session.run([proba_output_name], {input_name: arr})[0]

        return cls(
            predict_fn=predict_fn,
            threshold=threshold,
            model_version=version,
        )

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def model_version(self) -> str:
        return self._model_version

    def predict(self, features: pd.DataFrame) -> tuple[float, Decision]:
        """Return (probability_of_default, decision)."""
        # ONNX Runtime requires float32 contiguous arrays. The column order is
        # already aligned upstream by InferenceArtefacts.feature_names via
        # reindex(columns=...) in assemble().
        arr = features.to_numpy(dtype=np.float32)
        proba = float(self._predict_fn(arr)[0, 1])
        decision: Decision = "REFUSED" if proba >= self._threshold else "GRANTED"
        return proba, decision
