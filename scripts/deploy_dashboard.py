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

DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "dashboard"
REPO_ID = os.getenv("OC_P8_MONITORING_REPO", "KLEB38/OC_P8_monitoring")


def main() -> int:
    token = os.getenv("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN not set; aborting.")
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
