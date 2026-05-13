"""Synchronous prediction logger.

Writes one row to the configured predictions table for every ``/predict``
call (success or failure). Insert failures are logged but never propagated:
the API client has already received its prediction by the time we log, so
losing observability is preferable to losing the response.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import pandas as pd
from sqlalchemy import MetaData, Table, insert, update

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
    feature_assembly_ms: int | None = None,
    inference_ms: int | None = None,
    inference_cpu_ms: int | None = None,
    plumbing_ms: int | None = None,
    status_code: int = 200,
    error_message: str | None = None,
    top_shap: dict[str, Any] | None = None,
) -> None:
    """Insert one prediction record. Best-effort: never raises.

    Measures the INSERT wall-clock itself as ``db_log_ms`` and writes it on
    the same row — so the full server-side budget can be reconstructed as
    ``latency_ms + db_log_ms``.
    """
    table = _get_table()
    if table is None:
        return  # logging disabled, nothing to do

    payload = {
        "sk_id_curr": sk_id_curr,
        "client_known": client_known,
        "latency_ms": latency_ms,
        "feature_assembly_ms": feature_assembly_ms,
        "inference_ms": inference_ms,
        "inference_cpu_ms": inference_cpu_ms,
        "plumbing_ms": plumbing_ms,
        # Filled in after the INSERT completes below; included in payload now
        # for column ordering but overwritten before send.
        "db_log_ms": None,
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
    if engine is None:
        # _get_table() returned a Table only because the engine was set when
        # it was called. A race / re-entrancy between the two calls would
        # land here; rather than rely on an ``assert`` (stripped under -O),
        # fail closed: skip the insert silently. The client already has its
        # prediction.
        return
    # Self-describing rows: we want each prediction to carry the wall-clock
    # cost of its own INSERT, so the dashboard can decompose latency into
    # handler + DB log without external joins. The pattern is INSERT with
    # RETURNING id, measure, then UPDATE that exact id. Cost: 2 round-trips
    # per logged prediction — acceptable for monitoring at low QPS, and a
    # documented trade-off in the étape 4 report. If the UPDATE fails after
    # the INSERT succeeded, the row remains in the DB with ``db_log_ms``
    # NULL — acceptable, dashboard PERCENTILE_CONT ignores NULLs natively.
    try:
        t_insert = time.perf_counter()
        with engine.begin() as conn:
            inserted_id = conn.execute(
                insert(table).values(**payload).returning(table.c.id)
            ).scalar_one()
        db_log_ms = round((time.perf_counter() - t_insert) * 1000.0)
        with engine.begin() as conn:
            conn.execute(
                update(table)
                .where(table.c.id == inserted_id)
                .values(db_log_ms=db_log_ms)
            )
    except Exception as exc:  # noqa: BLE001 — best-effort logger
        logger.warning(
            "Failed to log prediction for sk_id=%s: %s", sk_id_curr, exc
        )
