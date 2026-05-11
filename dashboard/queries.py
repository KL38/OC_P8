"""Supabase read-only helpers for the monitoring dashboard.

All queries hit the ``predictions_log`` table (production data) and never
touch ``predictions_log_test``. Connection is cached by Streamlit so we
don't reopen a pool on every interaction.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st
from sqlalchemy import Engine, create_engine, text

PROD_TABLE = "predictions_log"


@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Configure it as a Space secret "
            "(read-only role recommended)."
        )
    return create_engine(url, pool_size=2, max_overflow=2, pool_pre_ping=True, future=True)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_recent(days: int) -> pd.DataFrame:
    """Wide DataFrame of recent rows. JSONB features stay as Python dicts."""
    since = datetime.now(tz=timezone.utc) - timedelta(days=days)
    sql = text(
        f"""
        SELECT timestamp, sk_id_curr, client_known, latency_ms, status_code,
               error_message, probability_default, decision, threshold,
               model_version, features
        FROM {PROD_TABLE}
        WHERE timestamp >= :since
        ORDER BY timestamp DESC
        """
    )
    with get_engine().connect() as conn:
        df = pd.read_sql(sql, conn, params={"since": since})
    return df


@st.cache_data(ttl=60, show_spinner=False)
def fetch_summary(days: int) -> dict:
    """Aggregate KPIs computed in SQL to keep the dashboard responsive."""
    since = datetime.now(tz=timezone.utc) - timedelta(days=days)
    sql = text(
        f"""
        SELECT
            COUNT(*)                                                AS total,
            SUM(CASE WHEN status_code = 200 THEN 1 ELSE 0 END)      AS ok,
            SUM(CASE WHEN status_code != 200 THEN 1 ELSE 0 END)     AS errors,
            SUM(CASE WHEN decision = 'GRANTED' THEN 1 ELSE 0 END)   AS granted,
            SUM(CASE WHEN decision = 'REFUSED' THEN 1 ELSE 0 END)   AS refused,
            PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY latency_ms) AS p50,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95,
            AVG(probability_default)                                AS avg_proba
        FROM {PROD_TABLE}
        WHERE timestamp >= :since
        """
    )
    with get_engine().connect() as conn:
        row = conn.execute(sql, {"since": since}).mappings().one()
    return dict(row)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_volume_by_hour(days: int) -> pd.DataFrame:
    since = datetime.now(tz=timezone.utc) - timedelta(days=days)
    sql = text(
        f"""
        SELECT date_trunc('hour', timestamp) AS hour,
               COUNT(*)                       AS total,
               SUM(CASE WHEN status_code != 200 THEN 1 ELSE 0 END) AS errors,
               PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY latency_ms) AS p50,
               PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95
        FROM {PROD_TABLE}
        WHERE timestamp >= :since
        GROUP BY 1
        ORDER BY 1
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(sql, conn, params={"since": since})
