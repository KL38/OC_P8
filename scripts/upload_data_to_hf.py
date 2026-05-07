"""Upload the feature store parquet to the companion HF Dataset repo.

HF Spaces silently drop files larger than 10 MB on direct upload. The
recommended pattern is therefore to host large data files in a Dataset
repo and have the Space download them at runtime via `hf_hub_download`.

PREREQUISITE — create the Dataset once on https://huggingface.co/new-dataset
    Owner: KLEB38
    Name: oc-p8-features
    Visibility: public (or private + HF_TOKEN secret on the Space)

Usage (PowerShell):
    $env:HF_TOKEN = "hf_xxx..."
    uv run python scripts/upload_data_to_hf.py
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO_ID = "KLEB38/oc-p8-features"
REPO_TYPE = "dataset"
PARQUET_PATH = Path("data/features_store.parquet")


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN env var is required")

    if not PARQUET_PATH.exists():
        raise SystemExit(
            f"{PARQUET_PATH} not found - run "
            "`uv run python scripts/build_feature_store.py` first."
        )

    size_mb = PARQUET_PATH.stat().st_size / (1024 * 1024)
    print(f"Local file: {PARQUET_PATH} ({size_mb:.1f} MB)")

    api = HfApi(token=token)
    print(f"Authenticated as: {api.whoami()['name']}")

    print(f"\nUploading to {REPO_TYPE} repo: {REPO_ID}")
    commit_info = api.create_commit(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        operations=[
            CommitOperationAdd(
                path_in_repo="features_store.parquet",
                path_or_fileobj=str(PARQUET_PATH),
            ),
        ],
        commit_message="Update features store parquet",
    )
    print(f"Commit OID: {commit_info.oid}")
    print(f"Commit URL: {commit_info.commit_url}")

    print("\nFiles AFTER upload:")
    for f in api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE):
        marker = "  <-- TARGET" if f == "features_store.parquet" else ""
        print(f"  - {f}{marker}")


if __name__ == "__main__":
    main()
