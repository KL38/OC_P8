"""Synchronous prediction logger.

Writes one row to the configured predictions table for every ``/predict``
call (success or failure). Insert failures are logged but never propagated:
the API client has already received its prediction by the time we log, so
losing observability is preferable to losing the response.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd
from sqlalchemy import MetaData, Table, insert

from api import db, settings
from database.models import build_predictions_log_table

logger = logging.getLogger(__name__)


def _jsonify_features(features: pd.DataFrame) -> dict[str, Any]:
    """Convert a single-row features DataFrame to a JSON-safe dict.

    NaN/Inf are not valid JSON — they are coerced to None so PostgreSQL
    receives ``null``. The 768 features mix floats, ints, and pandas
    extension types; ``.to_dict()`` plus this scrub handles all of them.
    """
    record = features.iloc[0].to_dict()
    cleaned: dict[str, Any] = {}
    for key, value in record.items():
        if value is None:
            cleaned[key] = None
        elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            cleaned[key] = None
        else:
            try:
                cleaned[key] = value.item()  # numpy scalar -> Python scalar
            except AttributeError:
                cleaned[key] = value
    return cleaned


def _jsonify_raw_input(raw_input: dict[str, Any]) -> dict[str, Any]:
    """Same NaN/Inf scrub for the raw Pydantic dump."""
    cleaned: dict[str, Any] = {}
    for key, value in raw_input.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            cleaned[key] = None
        else:
            cleaned[key] = value
    return cleaned


def _get_table() -> Table | None:
    """Resolve the active Table bound to the configured engine."""
    engine = db.get_engine()
    if engine is None:
        return None
    metadata = MetaData()
    return build_predictions_log_table(settings.PREDICTIONS_TABLE, metadata)


def log_prediction(
    *,
    sk_id_curr: int,
    client_known: bool,
    raw_input: dict[str, Any],
    features: pd.DataFrame | None,
    probability_default: float | None,
    decision: str | None,
    threshold: float,
    model_version: str,
    latency_ms: int,
    status_code: int = 200,
    error_message: str | None = None,
    top_shap: dict[str, Any] | None = None,
) -> None:
    """Insert one prediction record. Best-effort: never raises."""
    table = _get_table()
    if table is None:
        return  # logging disabled, nothing to do

    payload = {
        "sk_id_curr": sk_id_curr,
        "client_known": client_known,
        "latency_ms": latency_ms,
        "status_code": status_code,
        "error_message": error_message,
        "raw_input": _jsonify_raw_input(raw_input),
        "features": _jsonify_features(features) if features is not None else {},
        "probability_default": probability_default,
        "decision": decision or "ERROR",
        "threshold": threshold,
        "model_version": model_version,
        "top_shap": top_shap,
    }

    engine = db.get_engine()
    assert engine is not None  # _get_table() returned a Table -> engine is set
    try:
        with engine.begin() as conn:
            conn.execute(insert(table).values(**payload))
    except Exception as exc:  # noqa: BLE001 — best-effort logger
        logger.warning(
            "Failed to log prediction for sk_id=%s: %s", sk_id_curr, exc
        )
