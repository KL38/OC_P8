"""One-shot upload of large data artefacts to the HF Space.

Run this once, locally, after building the feature store. The parquet is
gitignored (200 MB) and excluded from the CI deploy job's `ignore_patterns`,
so HF keeps it between subsequent deploys without re-uploading every time.

Usage (PowerShell):
    $env:HF_TOKEN = "hf_xxx..."
    uv run python scripts/upload_data_to_hf.py
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "KLEB38/OC_P8"
PARQUET_PATH = Path("data/features_store.parquet")


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN env var is required")

    if not PARQUET_PATH.exists():
        raise SystemExit(
            f"{PARQUET_PATH} not found — run "
            "`uv run python scripts/build_feature_store.py` first."
        )

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(PARQUET_PATH),
        path_in_repo=str(PARQUET_PATH).replace("\\", "/"),
        repo_id=REPO_ID,
        repo_type="space",
    )
    print(f"Uploaded {PARQUET_PATH} to https://huggingface.co/spaces/{REPO_ID}")


if __name__ == "__main__":
    main()
