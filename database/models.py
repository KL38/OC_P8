"""SQLAlchemy models for the prediction monitoring database.

A single Table definition is parametrised by name so the production table
(``predictions_log``) and the CI test table (``predictions_log_test``) share
exactly the same schema without duplication.

The 768 engineered features are stored in a JSONB column rather than as
individual columns — three feature names exceed PostgreSQL's 63-character
identifier limit, and a flat JSONB also keeps the schema stable when the
feature engineering pipeline evolves.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

PROD_TABLE_NAME = "predictions_log"
TEST_TABLE_NAME = "predictions_log_test"


def build_predictions_log_table(name: str, metadata: MetaData) -> Table:
    """Return a Table object with the prediction log schema, bound to ``metadata``.

    Args:
        name: physical table name in PostgreSQL.
        metadata: SQLAlchemy MetaData to attach the Table to. Pass a fresh
            MetaData per table to avoid duplicate-key collisions in tests.

    Returns:
        Table ready to be created via ``metadata.create_all(engine)``.
    """
    return Table(
        name,
        metadata,
        # Identity
        Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=func.gen_random_uuid(),
        ),
        Column(
            "timestamp",
            DateTime(timezone=True),
            nullable=False,
            server_default=func.now(),
            index=True,
        ),
        Column("sk_id_curr", Integer, nullable=False, index=True),
        Column("client_known", Boolean, nullable=False),
        # Operational metrics
        Column("latency_ms", Integer, nullable=False),
        Column("status_code", Integer, nullable=False, server_default="200"),
        Column("error_message", Text, nullable=True),
        # Payloads
        Column("raw_input", JSONB, nullable=False),
        Column("features", JSONB, nullable=False),
        # Model output. Nullable so error rows (status != 200) can carry NULL.
        Column("probability_default", Float, nullable=True),
        Column("decision", String(16), nullable=False, index=True),
        Column("threshold", Float, nullable=False),
        Column("model_version", String(64), nullable=False),
        Column("top_shap", JSONB, nullable=True),
        # Ground truth — filled post-hoc when business feedback arrives
        Column("ground_truth", Integer, nullable=True),
    )
