"""Generate the Evidently data-drift HTML report.

Reads:
- Reference: ``data/reference_dataset.parquet`` (training sample, frozen)
- Current: last N days of production predictions from Supabase

Writes an HTML report consumable by the Streamlit dashboard.

Usage:
    uv run python scripts/generate_drift_report.py --days 30
    uv run python scripts/generate_drift_report.py --output dashboard/static/drift_report.html
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger("scripts.generate_drift_report")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

DEFAULT_REFERENCE = Path("data/reference_dataset.parquet")
DEFAULT_OUTPUT = Path("dashboard/static/drift_report.html")
DEFAULT_FEATURE_NAMES = Path("models/feature_names.json")


def _load_reference(path: Path, feature_names: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    keep = [c for c in feature_names if c in df.columns]
    missing = set(feature_names) - set(df.columns)
    if missing:
        logger.warning(
            "%d feature(s) absent from reference dataset (drift skipped on those)",
            len(missing),
        )
    return df[keep].copy()


def _load_current(database_url: str, days: int, feature_names: list[str]) -> pd.DataFrame:
    """Pull JSONB features back into a wide DataFrame for Evidently."""
    since = datetime.now(tz=timezone.utc) - timedelta(days=days)
    engine = create_engine(database_url, future=True)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT features FROM predictions_log "
                "WHERE timestamp >= :since AND status_code = 200"
            ),
            {"since": since},
        ).all()
    if not rows:
        raise SystemExit(
            f"No prediction rows in the last {days} days — generate some traffic first."
        )
    flat = pd.json_normalize([r.features for r in rows])
    keep = [c for c in feature_names if c in flat.columns]
    logger.info("Loaded %d production rows, %d features available", len(flat), len(keep))
    return flat[keep].copy()


def _build_report(reference: pd.DataFrame, current: pd.DataFrame):
    """Import Evidently lazily so the script still works when only inspecting."""
    try:
        # Evidently >= 0.5 stable API.
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError as exc:
        raise SystemExit(
            "evidently not installed. Add it to dashboard/requirements.txt "
            "or `uv pip install 'evidently>=0.5,<0.7'`."
        ) from exc

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--feature-names", type=Path, default=DEFAULT_FEATURE_NAMES)
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    args = parser.parse_args()

    if not args.database_url:
        # Allow loading from database/.env for local runs.
        try:
            from dotenv import load_dotenv

            load_dotenv(Path("database/.env"))
            args.database_url = os.getenv("DATABASE_URL")
        except ImportError:
            pass
    if not args.database_url:
        raise SystemExit("DATABASE_URL is required (env var or database/.env).")

    feature_names = json.loads(args.feature_names.read_text())
    reference = _load_reference(args.reference, feature_names)
    current = _load_current(args.database_url, args.days, feature_names)

    # Align columns: drop any column not in BOTH frames (Evidently expects parity)
    common = [c for c in reference.columns if c in current.columns]
    reference = reference[common]
    current = current[common]
    logger.info("Comparing on %d common features", len(common))

    report = _build_report(reference, current)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(args.output))
    logger.info("Saved drift report to %s", args.output)


if __name__ == "__main__":
    main()
