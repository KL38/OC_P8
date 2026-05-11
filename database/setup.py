"""CLI utility to create or truncate the prediction log tables.

Usage:
    uv run python -m database.setup --create           # both prod + test
    uv run python -m database.setup --create-test      # test table only
    uv run python -m database.setup --truncate-test    # cleanup CI artefacts

DATABASE_URL is read from the environment (or ``database/.env`` for local
runs). The script is idempotent — re-running ``--create`` is a no-op when
the tables already exist.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from sqlalchemy import MetaData, create_engine, text

from database.models import (
    PROD_TABLE_NAME,
    TEST_TABLE_NAME,
    build_predictions_log_table,
)

logger = logging.getLogger("database.setup")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")


def _load_dotenv_fallback() -> None:
    """Load database/.env if python-dotenv is installed and the file exists."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        logger.warning("python-dotenv not installed; relying on shell env only")
        return
    load_dotenv(env_path)


def _resolve_database_url(cli_url: str | None) -> str:
    if cli_url:
        return cli_url
    url = os.getenv("DATABASE_URL") or os.getenv("TEST_DATABASE_URL")
    if not url:
        raise SystemExit(
            "DATABASE_URL not set. Export it or pass --database-url. "
            "Local devs can populate database/.env."
        )
    return url


def create_tables(url: str, *, include_prod: bool, include_test: bool) -> None:
    engine = create_engine(url)
    metadata = MetaData()
    if include_prod:
        build_predictions_log_table(PROD_TABLE_NAME, metadata)
    if include_test:
        build_predictions_log_table(TEST_TABLE_NAME, metadata)
    metadata.create_all(engine, checkfirst=True)
    created = ", ".join(metadata.tables.keys())
    logger.info("Tables ensured: %s", created)


def truncate_test(url: str) -> None:
    """Wipe the test table. Hard-coded to TEST_TABLE_NAME — production is never touched."""
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {TEST_TABLE_NAME} RESTART IDENTITY"))
    logger.info("Truncated %s", TEST_TABLE_NAME)


def main(argv: list[str] | None = None) -> int:
    _load_dotenv_fallback()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", help="Override env DATABASE_URL")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create", action="store_true", help="Create both tables")
    group.add_argument("--create-test", action="store_true", help="Create test table only")
    group.add_argument(
        "--truncate-test", action="store_true", help="Empty the test table"
    )
    args = parser.parse_args(argv)

    url = _resolve_database_url(args.database_url)
    if args.create:
        create_tables(url, include_prod=True, include_test=True)
    elif args.create_test:
        create_tables(url, include_prod=False, include_test=True)
    elif args.truncate_test:
        truncate_test(url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
