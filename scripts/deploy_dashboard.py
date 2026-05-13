"""One-shot deploy of dashboard/ to HF Space KLEB38/OC_P8_monitoring.

Same pattern as the API deploy (cf .github/workflows/ci.yml).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("scripts.deploy_dashboard")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = REPO_ROOT / "dashboard"
REPO_ID = os.getenv("OC_P8_MONITORING_REPO", "KLEB38/OC_P8_monitoring")


def _load_env_file() -> None:
    """Load ``database/.env`` if present so HF_TOKEN can live there alongside
    DATABASE_URL. Silent no-op if python-dotenv is missing or the file is
    absent (CI uses real env vars and never ships the .env)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = REPO_ROOT / "database" / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def main() -> int:
    _load_env_file()
    token = os.getenv("HF_TOKEN")
    if not token:
        logger.error(
            "HF_TOKEN not set; aborting. "
            "Add it to database/.env or export it before running."
        )
        return 1
    if not DASHBOARD_DIR.exists():
        logger.error("Dashboard folder missing: %s", DASHBOARD_DIR)
        return 1

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    logger.info("Uploading %s -> %s", DASHBOARD_DIR, REPO_ID)
    api.upload_folder(
        folder_path=str(DASHBOARD_DIR),
        repo_id=REPO_ID,
        repo_type="space",
        ignore_patterns=["__pycache__", "*.pyc", ".pytest_cache", ".venv"],
    )
    logger.info("Dashboard deployed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
